"""The worker half of the deploy gate — run the engines in the deployed image.

    modal run scripts/worker_check.py

`scripts/smoke.py` checks the **API** container over HTTP. It structurally cannot
see the worker: §4 gives that container its own image, and the only channel
between the two is a job row. So every question this file asks — do BTC's weights
load, is Beat This!'s checkpoint really baked in, are ffmpeg and yt-dlp on the
PATH — is one `/healthz` answers `ok` to while being wrong.

That gap is not theoretical. Three defects shipped through a fully green smoke
run, each invisible on a laptop and each fatal in the image:

1. `beat_this` requires `torchaudio` unpinned, so it resolved to the **CUDA**
   wheel and could not import (`libcudart.so.13`). macOS has no CUDA variant.
2. BTC's `utils/mir_eval_modules` imports **`mir_eval`**, a dependency of the
   *checkout* that nothing in this repo names — so it is invisible to a grep, and
   present on any machine that has run the bench.
3. PyTorch 2.6 flipped `torch.load`'s `weights_only` default, and BTC's 2019
   checkpoint stores numpy scalars. Older local torch, newer image torch.

All three failed **in the chord stage**, behind `canAnalyze: true`,
`engines: {chords: [btc, ...]}` and `is_ready: true` — because registration
checks that a dependency and an adapter module exist, not that the engine runs.
And YouTube's bot check hides all of it, since fetch fails first: you would chase
the fetch stage, get through, and then meet these one at a time. (Chasing it with
cookies would not even have got you that far — the check is per egress IP and
cookies made no measurable difference. `scripts/real_song_check.py` is the gate
that covers fetch, on real audio.)

Synthesized audio, not a recording: this asks "do the engines load and produce a
grid", not "are they accurate" — the §8 benchmark already answered that, and a
deploy gate should not depend on a third party serving a video.
"""

import modal

from modal_app import worker_image

app = modal.App("rosetta-dechorder-workercheck")

# `worker_image` mounts the `app` package but not `modal_app` itself — and the
# container entrypoint imports *this* module, whose first act is to import it.
# Without this the function crash-loops on ModuleNotFoundError.
check_image = worker_image.add_local_python_source("modal_app")

SAMPLE_RATE = 22050
BPM = 120.0
BEATS = 24


def _reference_audio(np):
    """~12 s of 4/4 at 120 BPM: sustained triads for the chord engine, a
    percussive click on every beat for the beat tracker and onset detector."""
    beat_s = 60.0 / BPM
    total = int(SAMPLE_RATE * beat_s * BEATS)
    t = np.arange(total) / SAMPLE_RATE
    audio = np.zeros(total, dtype=np.float32)

    triads = [(261.63, 329.63, 392.00), (349.23, 440.00, 523.25),
              (392.00, 493.88, 587.33), (220.00, 261.63, 329.63)]
    for bar in range(BEATS // 4):
        start = int(bar * 4 * beat_s * SAMPLE_RATE)
        end = min(int((bar + 1) * 4 * beat_s * SAMPLE_RATE), total)
        for freq in triads[bar % len(triads)]:
            audio[start:end] += 0.2 * np.sin(2 * np.pi * freq * t[start:end]).astype(np.float32)
    for beat in range(BEATS):
        start = int(beat * beat_s * SAMPLE_RATE)
        end = min(start + int(0.02 * SAMPLE_RATE), total)
        envelope = np.exp(-np.linspace(0, 12, end - start)).astype(np.float32)
        noise = np.random.default_rng(beat).standard_normal(end - start).astype(np.float32)
        audio[start:end] += 0.5 * envelope * noise
    return np.clip(audio, -1.0, 1.0)


@app.function(image=check_image, timeout=600, memory=4096)
def check() -> dict:
    import shutil
    import subprocess

    import numpy as np

    from app.analysis import engines
    from app.config import load_settings

    settings = load_settings()
    result = {
        "available": engines.available(),
        "isReady": engines.is_ready(settings),
        "configured": {"chords": settings.chord_engine, "beats": settings.beat_tracker},
    }

    audio = _reference_audio(np)
    for label, build, run in (
        ("chords", engines.build_chord_engine, lambda e: e.analyze(audio, SAMPLE_RATE)),
        ("beats", engines.build_beat_tracker, lambda e: e.track(audio, SAMPLE_RATE)),
        ("onsets", engines.build_onset_detector, lambda e: e.detect(audio, SAMPLE_RATE)),
    ):
        try:
            engine = build(settings)
            if engine is None:
                result[label] = {"engine": None, "note": "none registered (optional)"}
                continue
            out = run(engine)
            info = {"engine": getattr(engine, "name", "?"),
                    "version": str(getattr(engine, "version", "?"))}
            if label == "beats":
                info.update(beats=len(out.beats_ms), downbeats=len(out.downbeats_ms),
                            bpm=round(out.bpm, 1), usable=out.is_usable)
            else:
                info["count"] = len(out)
                if label == "chords" and out:
                    info["sample"] = [span.label for span in out[:6]]
                    info["meanConfidence"] = round(
                        sum(span.confidence for span in out) / len(out), 3)
            result[label] = info
        except Exception as exc:  # noqa: BLE001 — reporting, not handling
            result[label] = {"ERROR": f"{type(exc).__name__}: {exc}"}

    # The fetch stage's binaries. Never reached while the bot check stands, so
    # nothing else in the deployment would notice them missing.
    #
    # `ffprobe` is here for the upload path (`app/analysis/file_source.py`),
    # which calls it to read a file's duration *before* the §3 gate. It ships in
    # Debian's `ffmpeg` package, so it is present today by luck rather than by
    # declaration — and `ModalJobRunner.can_accept_uploads()` answers True for
    # the worker without asking, so nothing else in the deployment would notice
    # if a future base image split the two.
    for binary, flag in (("ffmpeg", "-version"), ("ffprobe", "-version"),
                         ("yt-dlp", "--version")):
        path = shutil.which(binary)
        version = None
        if path:
            try:
                done = subprocess.run([path, flag], capture_output=True, text=True, timeout=30)
                lines = (done.stdout or done.stderr or "").splitlines()
                version = lines[0][:60] if lines else "(no output)"
            except Exception as exc:  # noqa: BLE001
                version = f"present but failed: {exc}"
        result[binary] = {"path": path, "version": version}

    return result


def _verdict(report: dict) -> int:
    """Exit non-zero on anything that would fail every job, so this is usable as
    a deploy gate exactly like `scripts/smoke.py`."""
    failures = []
    if not report.get("isReady"):
        failures.append("engines.is_ready() is False")
    for label in ("chords", "beats"):          # onsets are optional by design
        entry = report.get(label) or {}
        if "ERROR" in entry:
            failures.append(f"{label}: {entry['ERROR']}")
        elif not entry.get("engine"):
            failures.append(f"{label}: no engine built")
    beats = report.get("beats") or {}
    if beats.get("engine") and not beats.get("usable"):
        failures.append("beat grid unusable — fewer than two downbeats")
    chords = report.get("chords") or {}
    if chords.get("engine") and not chords.get("count"):
        failures.append("chord engine returned no spans")
    for binary in ("ffmpeg", "ffprobe", "yt-dlp"):
        if not (report.get(binary) or {}).get("path"):
            failures.append(f"{binary} missing from the worker image")

    for failure in failures:
        print(f"[ FAIL ] {failure}")
    if failures:
        print(f"\nFAILED — {len(failures)} check(s)")
        return 1
    print("\nAll worker checks passed")
    return 0


@app.local_entrypoint()
def main():
    import json
    import sys

    report = check.remote()
    print(json.dumps(report, indent=2, default=str))
    sys.exit(_verdict(report))
