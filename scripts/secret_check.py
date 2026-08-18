"""What each Modal secret actually contains — key names, never credential values.

    modal run scripts/secret_check.py

The gap this fills is the one `/healthz` is explicit about and cannot close. The
API container reports `fetch`/`egress` for **itself**, and on the deployed
two-container shape that is correctly `unconfigured`/`null` (§4: the API has no
audio stack and no proxy credential). So there is no way, from outside, to tell a
worker that is paying for clean egress from one that is drawing lottery tickets —
and the honest `null` reads exactly like a missing setting. That ambiguity cost a
debugging session: `"fetch": "unconfigured"` was read as "the proxy was never
configured" when it means "this container was never supposed to fetch".

This asks each secret's own environment instead, which is the only place the
answer lives. It audits **both** secrets, because the two failures that actually
bite are cross-container ones: a credential in the image that should not hold it,
and a knob set on one side of the split whose invariant is enforced on the other.

**Credential values are never printed.** Every key is reported present/absent;
`CHORDS_YTDLP_PROXY` additionally as its scheme and host — enough to tell a
rotating residential pool from a datacentre one, with the userinfo stripped. A
diagnostic that makes you paste a secret to read it is a diagnostic people run on
their screen-shared laptop.

`TUNABLES` is the deliberate exception and the reason this script grew a second
half. Deadlines, ratios and feature flags are not secrets — they are *operating
configuration that overrides a code default from outside the repo*, which is the
one class of setting a green test suite structurally cannot see. `.env.example`
documents a budget whose terms must add up:

    probe + fetch + decode + dsp_reserve ≤ deadline < worker timeout < job lease

`tests/test_deployment.py` asserts that on the **defaults**. A stale
`CHORDS_JOB_DEADLINE_S=180` left in a secret from before 2026-08-17 breaks the
chain in production while every test stays green and every value stays
individually plausible — the failure being a slow-but-succeeding fetch killed
with no terminal status. So the values of these keys are printed and the chain is
re-checked against what the secret really carries, not against config.py.

Deliberately NOT on `worker_image`: this inspects the *secrets*, not the images,
so it runs on a slim container that starts in seconds rather than pulling torch.
`scripts/worker_check.py` is the one that needs the real image, because it runs
the engines.
"""

import modal

app = modal.App("rosetta-dechorder-secretcheck")

image = modal.Image.debian_slim(python_version="3.11")

# The worker timeout lives in `modal_app.py`, which this slim image deliberately
# does not mount (importing it would pull the image definitions). It is the one
# term of the budget that is code rather than configuration, so it is restated
# here and `tests/test_deployment.py` pins the two together.
WORKER_TIMEOUT_S = 600
JOB_LEASE_S = 900

# Keys whose *values* are printed. Nothing here is a credential; everything here
# can silently override a code default, which is exactly why it has to be read
# from the secret rather than from the repo. Anything not in this dict is
# reported present/absent only.
TUNABLES = {
    "CHORDS_JOB_DEADLINE_S", "CHORDS_PROBE_TIMEOUT_S", "CHORDS_FETCH_TIMEOUT_S",
    "CHORDS_DECODE_TIMEOUT_S", "CHORDS_DSP_RESERVE_S", "CHORDS_MAX_VIDEO_SECONDS",
    "CHORDS_ANALYSIS_ENABLED", "CHORDS_CHORD_ENGINE", "CHORDS_BEAT_TRACKER",
    "CHORDS_ONSET_DETECTOR", "CHORDS_STRUCTURE_PROBE", "CHORDS_CONFIDENCE_FLOOR",
    "CHORDS_THEORY_CONSENSUS", "CHORDS_THEORY_VOCABULARY", "CHORDS_THEORY_BELIEF",
    "CHORDS_THEORY_TEMPO_OCTAVE", "CHORDS_TRUSTED_PROXY_HOPS", "CHORDS_SCALE_OUT",
    "CHORDS_DAILY_QUOTA", "CHORDS_RATE_LIMIT_PER_MIN", "CHORDS_RATE_LIMIT_IP_PER_MIN",
    "CHORDS_RATE_LIMIT_POLL_PER_MIN", "CHORDS_RATE_LIMIT_WINDOW_S",
    "CHORDS_REQUIRE_AUTH", "CHORDS_SCRATCH_ROOT", "CHORDS_CORS_ORIGINS",
}

# Everything each secret is supposed to be able to read, and what each absence
# costs. Absent is a legitimate state for most of these — the point is to say so
# out loud rather than leave it to be inferred from a null somewhere else.
WORKER_EXPECTED = {
    "CHORDS_DATABASE_URL":
        "REQUIRED — without it `build_store(role=ROLE_WORKER)` refuses to start "
        "rather than write job rows to an ephemeral SQLite the API never reads",
    "CHORDS_YTDLP_PROXY":
        "the only measured lever on YouTube's bot check. Unset ⇒ every fetch is "
        "a ~1-in-6 draw on a Modal datacentre IP, paid for in cold starts",
    "CHORDS_ANALYSIS_ENABLED": "optional kill switch (default on)",
    "CHORDS_MAX_VIDEO_SECONDS": "optional length cap (default 600)",
    "CHORDS_YTDLP_COOKIES_CONTENT":
        "optional, and deliberately unset — measured worthless against the bot "
        "check; kept for age-restricted video only",
}

API_EXPECTED = {
    "CHORDS_DATABASE_URL":
        "REQUIRED on the deployed shape — SQLite in the API image is a per-"
        "container file, so two containers disagree about every job. It is also "
        "what `CHORDS_SCALE_OUT` is gated on",
    "FIREBASE_PROJECT_ID":
        "REQUIRED — without it auth falls back to a mode that is not Firebase, "
        "and `/healthz` says so in `auth`",
    "FIREBASE_SERVICE_ACCOUNT_JSON":
        "REQUIRED for the admin/takedown path and for verifying tokens",
    "CHORDS_ADMIN_TOKEN":
        "unset ⇒ the admin routes answer 503 and there is no takedown lever (§3)",
    "CHORDS_DAILY_QUOTA": "optional per-uid cap (default 10)",
    "CHORDS_RATE_LIMIT_PER_MIN": "optional (0 = off)",
    "CHORDS_RATE_LIMIT_IP_PER_MIN": "optional (0 = off)",
}

# Credentials each side must NOT have. §19.2 in two assertions: the worker
# authenticates nobody, so a Firebase key there is blast radius bought for
# nothing — and the API never fetches, so a proxy credential there is a paid
# secret sitting in the internet-facing container for no purpose.
WORKER_FORBIDDEN = ["FIREBASE_SERVICE_ACCOUNT_JSON", "FIREBASE_PROJECT_ID",
                    "CHORDS_ADMIN_TOKEN", "CHORDS_DEV_TOKEN"]
API_FORBIDDEN = ["CHORDS_YTDLP_PROXY", "CHORDS_YTDLP_COOKIES_CONTENT"]


def _redacted_proxy(raw: str) -> str:
    """`scheme://host:port` — the shape of the egress, with the credentials cut.

    Enough to answer the question that matters ("is this a residential pool or
    somebody's datacentre static IP?") and nothing more. Parsed rather than
    regex-trimmed so a malformed value reports as malformed instead of leaking
    the part that failed to match.
    """
    from urllib.parse import urlsplit
    try:
        parts = urlsplit(raw)
        if not parts.scheme or not parts.hostname:
            return "set, but not a parseable proxy URL"
        port = f":{parts.port}" if parts.port else ""
        auth = " (with credentials)" if parts.username else " (no credentials)"
        return f"{parts.scheme}://{parts.hostname}{port}{auth}"
    except ValueError:
        return "set, but not a parseable proxy URL"


def _inspect(expected: list, forbidden: list) -> dict:
    """Runs inside the container. Returns names and tunable values only."""
    import os

    return {
        "present": {k: bool(os.environ.get(k)) for k in expected},
        "proxy": (_redacted_proxy(os.environ["CHORDS_YTDLP_PROXY"])
                  if os.environ.get("CHORDS_YTDLP_PROXY") else None),
        "forbidden": [k for k in forbidden if os.environ.get(k)],
        "tunables": {k: os.environ[k] for k in sorted(TUNABLES) if os.environ.get(k)},
        # Everything else the secret carries, by name. Catches the opposite
        # failure from the one above: a key set under a typo'd name reads as
        # "absent" in `present` and as noise here, and only the second one is
        # visible.
        "other_chords_keys": sorted(
            k for k in os.environ
            if k.startswith("CHORDS_") and k not in expected and k not in TUNABLES
        ),
    }


@app.function(image=image, secrets=[modal.Secret.from_name("chords-worker-secrets")],
              timeout=60)
def worker_report() -> dict:
    return _inspect(list(WORKER_EXPECTED), WORKER_FORBIDDEN)


@app.function(image=image, secrets=[modal.Secret.from_name("chords-secrets")],
              timeout=60)
def api_report() -> dict:
    return _inspect(list(API_EXPECTED), API_FORBIDDEN)


def _budget_verdict(tunables: dict) -> list[str]:
    """Re-derive `.env.example`'s chain from what the secret actually carries.

    The defaults are the ones in `app/config.py`; a key absent from the secret
    means the default applies, which is the case worth stating explicitly rather
    than skipping — "unset" is how the chain is *supposed* to hold.
    """
    defaults = {"CHORDS_PROBE_TIMEOUT_S": 45.0, "CHORDS_FETCH_TIMEOUT_S": 120.0,
                "CHORDS_DECODE_TIMEOUT_S": 90.0, "CHORDS_DSP_RESERVE_S": 180.0,
                "CHORDS_JOB_DEADLINE_S": 450.0}
    values, overridden = {}, []
    for key, default in defaults.items():
        raw = tunables.get(key)
        if raw is None:
            values[key] = default
            continue
        try:
            values[key] = float(raw)
        except ValueError:
            return [f"!! {key}={raw!r} is not a number — load_settings() will "
                    f"raise on container start"]
        overridden.append(key)

    stages = sum(values[k] for k in defaults if k != "CHORDS_JOB_DEADLINE_S")
    deadline = values["CHORDS_JOB_DEADLINE_S"]
    lines = [f"stages {stages:.0f}s ≤ deadline {deadline:.0f}s "
             f"< worker timeout {WORKER_TIMEOUT_S}s < job lease {JOB_LEASE_S}s"]
    if overridden:
        lines.append("overridden by the secret: " + ", ".join(sorted(overridden)))
    else:
        lines.append("every term is the code default (nothing overrides it here)")

    if stages > deadline:
        lines.append(f"!! the stage ceilings sum to {stages:.0f}s, which the "
                     f"{deadline:.0f}s deadline cannot contain — a fetch that is "
                     f"merely slow gets killed instead of finishing")
    elif not deadline < WORKER_TIMEOUT_S < JOB_LEASE_S:
        lines.append("!! the deadline no longer fires before the container "
                     "timeout — a killed container writes no terminal status, so "
                     "the job hangs until the lease reaper finds it")
    else:
        lines.append("[ ok ] the chain holds")
    return lines


def _print(title: str, expected: dict, result: dict, *, forbidden_note: str,
           ok_note: str, egress: bool) -> None:
    print(f"\n{title}\n")
    for key, why in expected.items():
        mark = "set" if result["present"][key] else "unset"
        print(f"  [{mark:>5}] {key}")
        if not result["present"][key]:
            print(f"          {why}")

    if egress:
        proxy = result["proxy"]
        print(f"\n  egress: {'proxy — ' + proxy if proxy else 'DIRECT (no proxy configured)'}")

    if result["tunables"]:
        print("\n  operating configuration (values, because none of these are secrets):")
        for key, value in result["tunables"].items():
            print(f"    {key} = {value}")

    if result["other_chords_keys"]:
        print("\n  also set (not expected here — check for a typo):")
        for key in result["other_chords_keys"]:
            print(f"    {key}")

    if result["forbidden"]:
        print(f"\n  !! {forbidden_note}")
        for key in result["forbidden"]:
            print(f"    {key}")
    else:
        print(f"\n  [ ok ] {ok_note}")


@app.local_entrypoint()
def main():
    worker = worker_report.remote()
    api = api_report.remote()

    _print("chords-worker-secrets — what the worker can actually read",
           WORKER_EXPECTED, worker,
           forbidden_note=("auth credentials found in the WORKER secret — these "
                           "belong only in chords-secrets:"),
           ok_note="no auth credentials in the worker secret (§19.2)",
           egress=True)

    _print("chords-secrets — what the API container can actually read",
           API_EXPECTED, api,
           forbidden_note=("fetch credentials found in the API secret — the API "
                           "image has no audio stack and never fetches, so this "
                           "is a paid secret in the internet-facing container "
                           "for no purpose:"),
           ok_note="no fetch credentials in the API secret (§4)",
           egress=False)

    print("\nthe job time budget, as the deployment really carries it\n")
    # The worker is where a job's clock actually runs, so its secret is the one
    # that decides — but a stale override on either side is worth seeing, and the
    # API is what reports the deadline back to a client.
    for scope, result in (("worker", worker), ("api", api)):
        print(f"  {scope}:")
        for line in _budget_verdict(result["tunables"]):
            print(f"    {line}")
    print()
