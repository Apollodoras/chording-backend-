"""The fetch/decode implementation — §8 step 4, behind `fetch.py`'s seam.

`fetch.py` states the contract and the reason for it; this is the half that
touches a recording, and it exists **only in the worker image**. If this module
is importable from the API container, the split that makes §2's blast-radius
argument true has already been lost.

Two calls, kept apart on purpose:

- **`probe`** runs yt-dlp with `download=False`. Metadata only — duration, title,
  channel id — because §3's blocklist is per-video *and* per-channel and the
  10-minute cap is a rejection, so both must be decidable before a byte of audio
  moves. The pipeline calls `gate()` between the two.
- **`decode`** fetches the audio stream into the caller's scratch directory and
  pipes it through ffmpeg to mono float32 at 22.05 kHz. It writes nothing outside
  that directory and returns no paths — only samples, which die with the worker.

**Everything is bounded.** A fetch stage with no limits is a worker that hangs on
one pathological video and takes its container's whole timeout with it, so:
socket timeout and retry count on yt-dlp, a wall-clock timeout on each
subprocess, a hard ceiling on the downloaded file, and a re-check of the real
duration after decoding — because the metadata cap is only as honest as the
metadata, and `duration` is a field a video can simply not have.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from ..errors import AnalysisError, EgressBlocked, MediaUrlRefused, VideoUnavailable
from .types import PCM, VideoMeta

log = logging.getLogger("chords.fetch.ytdlp")

_SAMPLE_RATE = 22050

# Wall-clock ceilings, and **defaults only** — `config.Settings` owns the real
# numbers so that the three of them plus the DSP reserve can be checked against the
# job deadline as one budget (see the long note there). They were constants here,
# and constants in three files is how the set came to sum to 405 s inside a
# container that was killed at 300 s and a job deadline of 180.
_PROBE_TIMEOUT_S = 45
_FETCH_TIMEOUT_S = 120
_DECODE_TIMEOUT_S = 90

# A 10-minute song at a sane bitrate is a few MB. 256 MB means something is
# wrong — a mislabelled duration, a live stream, a video that is not a song —
# and downloading it is how a worker with a memory cap dies instead of erroring.
_MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024

# Substrings yt-dlp puts in its stderr for the "can't have it" family. These are
# a *player-visible* outcome (§16.3's `video_unavailable`), not a bug, and must
# not be reported as "something went wrong on our side".
_UNAVAILABLE_MARKERS = (
    "video unavailable", "private video", "removed by the uploader",
    "not available in your country", "blocked it in your country",
    "sign in to confirm your age", "age-restricted", "members-only",
    "this live event", "has been terminated", "account associated with this video",
    "unable to extract", "no video formats", "requested format is not available",
)

# The bot check, kept apart from the family above. To the player it is the same
# calm "can't have this one" outcome, so it stays refundable and is not dressed
# up as our crash — but operationally it is the opposite of a video being
# private: nothing is wrong with the video, and the *next* video may well fail
# the same way. That deserves a log line an operator can alert on, which is why
# it doesn't just sit in the tuple above being silently indistinguishable.
#
# It is **per egress IP, not per account**, which was measured rather than
# assumed: a fan-out of ten containers over ten distinct Modal IPs resolved the
# same three ordinary uploads on two of them — 20%, and *identical* with and
# without cookies (6/30 in both arms). So this is not "all or nothing" across the
# deployment, and it is not something a credential fixes.
_BOT_CHECK_MARKERS = (
    "sign in to confirm you're not a bot",
    "sign in to confirm you’re not a bot",
    "confirm you're not a bot",
)

# The *download* refusing an URL the *probe* was given happily. See
# `MediaUrlRefused`, whose docstring carries both mechanisms this has now been
# measured to have — a media URL fetched from an address other than the one that
# resolved it (fixed by `_sticky_proxy`), and a media URL resolved through a
# player client YouTube declines to serve (fixed by `_PLAYER_CLIENT`). Matched
# only on the fetch, never on the probe: a 403 there is a different animal, and
# marking it retryable would buy cold starts for videos that fail identically
# forever.
_MEDIA_REFUSED_MARKERS = (
    "http error 403",
    "403: forbidden",
)

# Which yt-dlp **player client** resolves the media URL, and as of 2026-08-17 the
# difference between an analysis and a 403 on every video in the corpus.
#
# This file used to say, in `__init__`, that the player client was measured and
# is not a lever. That measurement was real and it is still true *about the thing
# it measured*: against the **bot check**, six Modal IPs × eight clients came back
# all-or-nothing per IP — the clean address served every client, the blocked ones
# refused every client. The check is on the address, so the client cannot move it.
#
# The 403 on the media URL is a different failure that happens to be reachable
# from the same place, and there the client is the only thing that matters.
# Measured in the worker image, one sticky session per attempt, on three videos —
# an ordinary upload, an official label upload, and a Beatles remaster:
#
#     client            36X3wecT2z8    8ui9umU0C2g    CGj85pVzRJs
#     default (vr)      403            bot check      bot check
#     android           **ok (18)**    **ok (18)**    **ok (18)**
#     android_music     403            403            403
#     android_creator   403            403            403
#     tv_embedded       403            403            403
#     web / web_embedded   no format available on any of the three
#     tv                DRM protected
#
# So: every client that offers the audio-only `140` stream is refused the bytes,
# and the one client that is served hands back progressive `18` instead. That is
# the whole reason `-f` still ends in a `/best` fallback — under this client there
# is no audio-only format to prefer, and the pipeline throws the video away in
# ffmpeg a moment later. The cost is transfer, not correctness: ~11 MB for a
# four-minute song against ~2 MB for `140`, paid once per song ever because the
# map is cached and shared (§5.3).
#
# Overridable, because the one certainty here is that this moves: YouTube retires
# clients and yt-dlp adds them. `CHORDS_YTDLP_PLAYER_CLIENT=""` restores yt-dlp's
# own default, which is the setting this had for the first two weeks of August and
# which returns 403 today. `scripts/real_song_check.py` is where a regression
# shows up, and it is the only gate that can see one: no unit test reaches
# YouTube, and the failure is invisible to every green suite.
_PLAYER_CLIENT = "android"

# How long a sticky egress session is held. It has to outlive one **analysis** —
# probe plus fetch, so `probe_timeout_s + fetch_timeout_s`, comfortably inside ten
# minutes at the configured 45 + 120 — and every analysis asks for a new one, so a
# longer lifetime would not make any single download more reliable, it would just
# keep reusing an address after it stopped being needed.
_PROXY_SESSION_LIFETIME_M = 10


class FetchError(AnalysisError):
    """A fetch or decode failed for a reason that is ours, not the video's."""

    def __init__(self, message: str = "That video couldn’t be downloaded."):
        super().__init__(message)


def _is_bot_check(stderr: str) -> bool:
    return any(marker in (stderr or "").lower() for marker in _BOT_CHECK_MARKERS)


def _is_media_refused(stderr: str) -> bool:
    return any(marker in (stderr or "").lower() for marker in _MEDIA_REFUSED_MARKERS)


def _sticky_proxy(raw: str, session: str) -> str:
    """`raw` with `session` pinned onto the credential.

    The pool is bought rotating, and rotating is still what we want *between*
    jobs — every retry has to be a fresh draw or `EgressBlocked` has nothing to
    retry into. What broke the download is that it also rotated *within* one
    fetch. This asks for both: one address for the life of a single analysis, a
    new one next time.

    IPRoyal takes its pool directives in the **password** (`pass_session-<id>_
    lifetime-<n>m`); other vendors put them in the username, so this is
    deliberately the one vendor-shaped thing in the module rather than a general
    "proxy session" abstraction over a pool we do not have. A credential that
    already names a session is left exactly alone — that is an operator pinning
    one on purpose, and silently appending a second directive would be a strange
    way to repay them.

    String surgery rather than `urlsplit`/`urlunsplit` on purpose: `urlsplit`
    returns userinfo still percent-encoded, so rebuilding through `quote` would
    double-encode a password containing a `@` or `:` — and the suffix added here
    is URL-safe by construction, so appending it to the raw text is both simpler
    and lossless.
    """
    scheme, sep, rest = raw.partition("://")
    if not sep or "@" not in rest:
        return raw
    userinfo, _, hostport = rest.rpartition("@")
    if ":" not in userinfo:
        return raw
    user, _, password = userinfo.partition(":")
    if "_session-" in password:
        return raw
    password = f"{password}_session-{session}_lifetime-{_PROXY_SESSION_LIFETIME_M}m"
    return f"{scheme}://{user}:{password}@{hostport}"


def _looks_unavailable(stderr: str) -> bool:
    lowered = (stderr or "").lower()
    if _is_bot_check(lowered):
        log.error(
            "yt-dlp hit YouTube's bot check — nothing is wrong with this video. "
            "It is per egress IP, so a retry on a fresh container may well "
            "succeed; measured hit rate on Modal was ~20%% of IPs. Cookies do "
            "NOT fix it (measured: identical with and without) — the lever is "
            "egress: set CHORDS_YTDLP_PROXY to a rotating residential pool. If "
            "it is already set, this line means the pool itself was refused."
        )
        return True
    return any(marker in lowered for marker in _UNAVAILABLE_MARKERS)


def _unavailable_error(stderr: str) -> VideoUnavailable:
    """The right flavour of "can't have this one".

    Identical on the wire either way; the difference is only whether a *retry on
    a different container* has any chance of helping. See `EgressBlocked`.
    """
    return EgressBlocked() if _is_bot_check(stderr) else VideoUnavailable()


def _run(command: list[str], *, timeout: int, cwd: Path | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout, cwd=str(cwd) if cwd else None,
                              check=False)
    except subprocess.TimeoutExpired as exc:
        raise FetchError("That video took too long to download — try a shorter one.") from exc
    except FileNotFoundError as exc:
        # Wrong image: this is the API container, or ffmpeg is missing from the
        # worker. Loud, because it is a deployment error and not a video problem.
        raise FetchError(f"Required tool missing from this image: {command[0]}") from exc


class YtDlpSource:
    """`VideoSource` — yt-dlp for the bytes, ffmpeg for the samples."""

    name = "yt-dlp"

    def __init__(self, settings=None) -> None:
        self._settings = settings
        self._ytdlp = [sys.executable, "-m", "yt_dlp"]
        self._ffmpeg = os.environ.get("CHORDS_FFMPEG", "ffmpeg")
        self._probe_timeout = int(getattr(settings, "probe_timeout_s", _PROBE_TIMEOUT_S))
        self._fetch_timeout = int(getattr(settings, "fetch_timeout_s", _FETCH_TIMEOUT_S))
        self._decode_timeout = int(getattr(settings, "decode_timeout_s", _DECODE_TIMEOUT_S))
        # The proxy session this source is currently using, and the *cached probe
        # result* that was resolved through it. See `_common_args`.
        self._session: str | None = None
        self._probed: tuple[str, str] | None = None
        # Optional cookies, and deliberately unset in this deployment: measured
        # against the bot check they made no difference at all (see
        # `_BOT_CHECK_MARKERS`). Kept because they remain yt-dlp's supported
        # mechanism for age-restricted and members-only video, and because the
        # picture changes the moment egress is not a datacentre IP — not because
        # they are the answer to a bot check. See `_cookie_file` for why this is
        # accepted as *content* and not only as a path.
        self._cookies_path = os.environ.get("CHORDS_YTDLP_COOKIES") or None
        self._cookies_data = os.environ.get("CHORDS_YTDLP_COOKIES_CONTENT") or None
        self._materialized: str | None = None
        # Which player client resolves the media URL — see `_PLAYER_CLIENT` for
        # the measurement, and for why "the player client is not a lever" was
        # true of the bot check and false of the 403 that replaced it.
        self._player_client = os.environ.get(
            "CHORDS_YTDLP_PLAYER_CLIENT", _PLAYER_CLIENT).strip() or None
        # **The actual lever on the bot check**, and the only one measurement
        # supports. Cookies changed nothing (6/30 both arms) and neither did the
        # player client (six IPs × eight clients: the clean IP served all eight,
        # the blocked ones refused all eight) — that last clause is about the bot
        # check *only*, and `_PLAYER_CLIENT` is the failure it does not cover.
        # The check is on the *address*, so the fix is to leave from somewhere
        # else — a **rotating residential** proxy, as
        # `scheme://user:pass@host:port`.
        #
        # Rotating and residential are both load-bearing words. A *static*
        # datacentre address — which is what a cloud provider's static-egress
        # feature sells you, Modal's `modal.Proxy` included — is the exact
        # fingerprint this check exists to catch: one address fetching audio all
        # day gets flagged once and then fails forever, with no fresh draw to
        # retry into. That is strictly worse than the unproxied default.
        #
        # Unset is still a legitimate state — it costs money, so it is the
        # owner's call — but it is no longer the *expected* one. When it is set,
        # the first attempt is meant to work, and `modal_app` spends a much
        # smaller retry budget accordingly (see `egress`).
        self._proxy = os.environ.get("CHORDS_YTDLP_PROXY") or None

    @property
    def egress(self) -> str:
        """`"proxy"` or `"direct"` — which way this container leaves the network.

        Read by two callers that both need it and neither of which can see the
        environment variable: `/healthz`, so an operator can tell a deployment
        that is paying for clean egress from one that is drawing lots; and the
        Modal worker, which sizes its retry budget from it. Twelve attempts is
        the right number for a 1-in-6 lottery and pure waste when the first
        attempt is bought and paid for.

        A property rather than a field on the `VideoSource` protocol: only this
        source leaves the network at all, and `FileSource` should not have to
        answer a question about egress it does not have.
        """
        return "proxy" if self._proxy else "direct"

    @property
    def version(self) -> str:
        """The yt-dlp version, resolved once and cached.

        Persisted on every stored map (§5.3): yt-dlp changes behaviour often
        enough that "which version fetched this" is a question worth being able
        to answer when a cached map turns out to be wrong.
        """
        cached = getattr(self, "_version", None)
        if cached is None:
            result = _run(self._ytdlp + ["--version"], timeout=30)
            cached = (result.stdout or "unknown").strip() or "unknown"
            self._version = cached
        return cached

    def _cookie_file(self) -> str | None:
        """The cookies file to hand yt-dlp, materializing one if we were given
        content instead of a path.

        A path alone is unusable on the deployment this actually runs on. Modal
        delivers secrets as **environment variables**, and the worker mounts no
        Volume by design (§4) — so there is no filesystem anywhere in that
        container that a cookies *file* could have been placed on. Supporting
        only `CHORDS_YTDLP_COOKIES` meant this could not be configured in
        production at all.

        Nothing here is currently set: cookies were tried against the bot check,
        measured, and removed. This exists for age-restricted and members-only
        video, and against the day egress stops being a datacentre IP.

        Written 0600 into the container's own tmp, once per process. Not into the
        §2.1 scratch root: that directory is destroyed after every job and this
        is needed by `probe`, which runs before one exists. It is a credential
        rather than audio, so the invariant it must respect is "never leaves this
        container" — and a Modal container's filesystem dies with it.
        """
        if self._cookies_path:
            return self._cookies_path
        if not self._cookies_data:
            return None
        if self._materialized is None:
            import tempfile
            handle, path = tempfile.mkstemp(prefix="yt-cookies-", suffix=".txt")
            with os.fdopen(handle, "w") as target:
                target.write(self._cookies_data)
            os.chmod(path, 0o600)
            self._materialized = path
            # Removed when the process ends. A Modal container's filesystem dies
            # with it, so this is belt-and-braces there — but this source also runs
            # under `ThreadJobRunner` in a long-lived local process and in the
            # bench harness, and a 0600 file full of session cookies that outlives
            # the run that needed it is a credential left lying around.
            atexit.register(self._discard_cookie_file)
            log.info("cookies: materialized from CHORDS_YTDLP_COOKIES_CONTENT")
        return self._materialized

    def _discard_cookie_file(self) -> None:
        path, self._materialized = self._materialized, None
        if not path:
            return
        try:
            os.unlink(path)
        except OSError:      # pragma: no cover - cleanup, never fatal
            pass

    # -- egress session ------------------------------------------------------

    def _new_session(self) -> str:
        """Start a fresh egress session. One per **analysis**, not per subprocess.

        `probe` opens one and `decode` reuses it, and that is a correctness fix
        rather than a saving. Two things follow from it:

        - **The address that cleared the bot check is the one that downloads.** The
          check is per egress IP at roughly 1-in-6, and a job needs *both* calls to
          get through. Drawing independently meant two chances to be refused —
          `1 − p²` rather than `1 − p` — so the old comment here ("two draws are
          better than one") had it backwards: independent draws are better when
          either one will do, and worse when you need both.
        - **A googlevideo media URL is bound to the IP that resolved it.** With
          separate sessions the metadata was resolved on one address and the bytes
          requested from another, which is exactly the 403 `MediaUrlRefused`
          documents on label-owned uploads.

        Rotation between jobs is untouched: a retry lands on a new container, or on
        a re-probe here, and either way asks the pool for a new address.
        """
        self._session = uuid.uuid4().hex[:12]
        return self._session

    def _common_args(self) -> list[str]:
        args = ["--no-warnings", "--no-playlist", "--socket-timeout", "15",
                "--retries", "2", "--no-progress"]
        if self._player_client:
            # On both calls, not just the download. The probe is what resolves the
            # formats the download then asks for, so a probe on one client and a
            # fetch on another is the same shape of mistake as a probe on one IP
            # and a fetch on another — it picks a format the second client was
            # never offered, and `_UNAVAILABLE_MARKERS` reports "requested format
            # is not available", which reads as a fact about the video.
            args += ["--extractor-args", f"youtube:player_client={self._player_client}"]
        cookies = self._cookie_file()
        if cookies:
            args += ["--cookies", cookies]
        if self._proxy:
            args += ["--proxy", _sticky_proxy(self._proxy, self._session or self._new_session())]
        return args

    # -- metadata only ------------------------------------------------------

    def probe(self, video_id: str) -> VideoMeta:
        # A new session per analysis, opened here because `probe` is the first
        # thing an analysis does. `decode` inherits it — see `_new_session`.
        self._new_session()
        result = _run(
            self._ytdlp + self._common_args() + [
                "--skip-download", "--dump-single-json",
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            timeout=self._probe_timeout,
        )
        if result.returncode != 0:
            if _looks_unavailable(result.stderr):
                raise _unavailable_error(result.stderr)
            log.warning("probe failed for %s: %s", video_id, (result.stderr or "").strip()[:400])
            raise FetchError("That video couldn’t be looked up.")

        try:
            info = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise FetchError("That video’s details couldn’t be read.") from exc

        if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming"}:
            raise VideoUnavailable("That’s a live stream — try a recorded song.")

        duration = info.get("duration")
        if duration is None:
            # No duration means the length cap cannot be enforced, and §18's cap
            # is not advisory. Refuse rather than fetch something unbounded.
            raise VideoUnavailable("That video’s length couldn’t be determined.")

        return VideoMeta(
            video_id=video_id,
            title=(info.get("title") or video_id).strip(),
            duration_s=float(duration),
            channel_id=info.get("channel_id") or info.get("uploader_id"),
        )

    # -- the only code that holds audio -------------------------------------

    def decode(self, video_id: str, workdir: Path, *,
               sample_rate: int = _SAMPLE_RATE) -> tuple[PCM, int]:
        """Fetch and decode into `workdir`, which the caller destroys.

        `workdir` is a `scratch.scratch()` directory. Nothing here writes outside
        it and nothing returns a path — the samples are the only thing that
        leaves, and they live in the worker's memory until it dies.
        """
        import numpy as np

        workdir = Path(workdir)
        media = self._fetch(video_id, workdir)
        wav = workdir / "decoded.wav"

        result = _run([
            self._ffmpeg, "-y", "-loglevel", "error",
            "-i", str(media),
            "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
            str(wav),
        ], timeout=self._decode_timeout)
        if result.returncode != 0 or not wav.is_file():
            log.warning("decode failed for %s: %s", video_id, (result.stderr or "").strip()[:400])
            raise FetchError("That video’s audio couldn’t be decoded.")

        # Freed as soon as it is redundant. The scratch teardown would remove it
        # anyway; dropping it here keeps the peak footprint to one copy, which is
        # what the worker's memory cap is sized against.
        try:
            media.unlink()
        except OSError:
            pass

        pcm = _read_wav(np, wav)
        try:
            wav.unlink()
        except OSError:
            pass

        if pcm.size == 0:
            raise FetchError("That video had no audio track.")

        # The cap, re-checked against what was actually decoded. `probe`'s
        # duration is metadata, and metadata can be wrong or absent; this is the
        # number that is true. Belt and braces on §18's ceiling.
        limit = getattr(self._settings, "max_video_seconds", None)
        if limit and pcm.size / sample_rate > limit * 1.1:
            from ..errors import VideoTooLong
            raise VideoTooLong(int(limit))

        return pcm, sample_rate

    def _fetch(self, video_id: str, workdir: Path) -> Path:
        """Download the smallest usable audio-only stream into `workdir`."""
        template = str(workdir / "media.%(ext)s")
        result = _run(
            self._ytdlp + self._common_args() + [
                # Audio-only, preferring the smallest acceptable stream: the
                # pipeline resamples to 22.05 kHz mono regardless, so a
                # high-bitrate download buys nothing but transfer time.
                #
                # The trailing `/best` is not decoration. Under the player client
                # this has to use (see `_PLAYER_CLIENT`) YouTube offers no
                # audio-only format at all, so every fetch lands here and takes
                # progressive `18` — video included, thrown away by ffmpeg one
                # step later. ~11 MB rather than ~2 MB, once per song ever.
                "-f", "worstaudio[abr>=64]/bestaudio/best",
                "--max-filesize", str(_MAX_DOWNLOAD_BYTES),
                "-o", template,
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            timeout=self._fetch_timeout,
            cwd=workdir,
        )
        if result.returncode != 0:
            if _looks_unavailable(result.stderr):
                raise _unavailable_error(result.stderr)
            if _is_media_refused(result.stderr):
                log.error(
                    "yt-dlp resolved a media URL and was then refused it (403) — "
                    "nothing is wrong with this video, and the probe for it "
                    "succeeded. Two causes, in the order worth checking: (1) the "
                    "player client is one YouTube will not serve bytes to, which "
                    "fails for *every* video and is what CHORDS_YTDLP_PLAYER_"
                    "CLIENT exists for — currently %s; (2) a googlevideo URL is "
                    "bound to the IP that resolved it, so per-request proxy "
                    "rotation gets refused at the media endpoint, and "
                    "CHORDS_YTDLP_PROXY must support session pinning (see "
                    "`_sticky_proxy`). A retry lands on a fresh container and "
                    "draws a fresh sticky session, which settles (2) and cannot "
                    "touch (1): if every video fails, it is (1).",
                    self._player_client or "yt-dlp's own default",
                )
                raise MediaUrlRefused()
            log.warning("fetch failed for %s: %s", video_id, (result.stderr or "").strip()[:400])
            raise FetchError()

        candidates = sorted(p for p in workdir.iterdir() if p.name.startswith("media"))
        if not candidates:
            # yt-dlp exits 0 when --max-filesize aborts the download.
            raise FetchError("That video’s audio was too large to download.")
        return candidates[0]


def _read_wav(np, path: Path):
    """16-bit PCM WAV → mono float32 in [-1, 1].

    The stdlib `wave` module rather than soundfile/librosa: ffmpeg has already
    produced exactly the format we asked for, so this needs no format guessing,
    and it keeps one more C library out of the image that handles recordings.
    """
    import wave

    with wave.open(str(path)) as source:
        frames = source.readframes(source.getnframes())
    if not frames:
        return np.zeros(0, dtype="float32")
    return np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0


def available() -> bool:
    """Whether this image can actually fetch — used by `build_source`.

    `find_spec` rather than an import: this runs on the API container's startup
    path, and that container is defined by *not* being able to touch audio (§4).
    Importing yt-dlp to discover that we shouldn't have would be a small
    contradiction, and it is the kind that a later refactor turns into a real
    one. Checked presence, never loaded.
    """
    import importlib.util

    if shutil.which(os.environ.get("CHORDS_FFMPEG", "ffmpeg")) is None:
        return False
    try:
        return importlib.util.find_spec("yt_dlp") is not None
    except (ImportError, ValueError):
        return False
