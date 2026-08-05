"""What the worker's secret actually contains — key names only, never values.

    modal run scripts/worker_env_check.py

The gap this fills is the one `/healthz` is explicit about and cannot close. The
API container reports `fetch`/`egress` for **itself**, and on the deployed
two-container shape that is correctly `unconfigured`/`null` (§4: the API has no
audio stack and no proxy credential). So there is no way, from outside, to tell a
worker that is paying for clean egress from one that is drawing lottery tickets —
and the honest `null` reads exactly like a missing setting. That ambiguity cost a
debugging session: `"fetch": "unconfigured"` was read as "the proxy was never
configured" when it means "this container was never supposed to fetch".

This asks the worker's own environment instead, which is the only place the
answer lives.

**Values are never printed.** Every key is reported as present/absent, and
`CHORDS_YTDLP_PROXY` additionally as its scheme and host — enough to tell a
rotating residential pool from a datacentre one, with the credentials in the
userinfo stripped. A diagnostic that makes you paste a secret to read it is a
diagnostic people run on their screen-shared laptop.

Deliberately NOT on `worker_image`: this inspects the *secret*, not the image, so
it runs on a slim container that starts in seconds rather than pulling torch.
`scripts/worker_check.py` is the one that needs the real image, because it runs
the engines.
"""

import modal

app = modal.App("rosetta-dechorder-envcheck")

image = modal.Image.debian_slim(python_version="3.11")
worker_secrets = [modal.Secret.from_name("chords-worker-secrets")]

# Everything the worker is supposed to be able to read, and what each absence
# costs. Absent is a legitimate state for most of these — the point is to say so
# out loud rather than leave it to be inferred from a null somewhere else.
EXPECTED = {
    "CHORDS_DATABASE_URL":
        "REQUIRED — without it the worker writes job rows to its own ephemeral "
        "SQLite and the API never sees the job finish",
    "CHORDS_YTDLP_PROXY":
        "the only measured lever on YouTube's bot check. Unset ⇒ every fetch is "
        "a ~1-in-6 draw on a Modal datacentre IP, paid for in cold starts",
    "CHORDS_ANALYSIS_ENABLED": "optional kill switch (default on)",
    "CHORDS_MAX_VIDEO_SECONDS": "optional length cap (default 600)",
    "CHORDS_YTDLP_COOKIES_CONTENT":
        "optional, and deliberately unset — measured worthless against the bot "
        "check; kept for age-restricted video only",
}

# Credentials the worker must NOT have. §19.2 in one assertion: it authenticates
# nobody, so a Firebase key here is blast radius bought for nothing.
FORBIDDEN = ["FIREBASE_SERVICE_ACCOUNT_JSON", "FIREBASE_PROJECT_ID",
             "CHORDS_ADMIN_TOKEN", "CHORDS_DEV_TOKEN"]


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


@app.function(image=image, secrets=worker_secrets, timeout=60)
def report() -> dict:
    import os

    present = {k: (k in os.environ and bool(os.environ[k])) for k in EXPECTED}
    proxy = os.environ.get("CHORDS_YTDLP_PROXY") or ""
    return {
        "present": present,
        "proxy": _redacted_proxy(proxy) if proxy else None,
        "forbidden": [k for k in FORBIDDEN if os.environ.get(k)],
        # Everything else the secret carries, by name. Catches the opposite
        # failure from the one above: a key set under a typo'd name reads as
        # "absent" in `present` and as noise here, and only the second one is
        # visible.
        "other_chords_keys": sorted(
            k for k in os.environ
            if k.startswith("CHORDS_") and k not in EXPECTED
        ),
    }


@app.local_entrypoint()
def main():
    result = report.remote()

    print("\nchords-worker-secrets — what the worker can actually read\n")
    for key, why in EXPECTED.items():
        mark = "set" if result["present"][key] else "unset"
        print(f"  [{mark:>5}] {key}")
        if not result["present"][key]:
            print(f"          {why}")

    proxy = result["proxy"]
    print(f"\n  egress: {'proxy — ' + proxy if proxy else 'DIRECT (no proxy configured)'}")

    if result["other_chords_keys"]:
        print("\n  also set (not in the expected list — check for a typo):")
        for key in result["other_chords_keys"]:
            print(f"    {key}")

    if result["forbidden"]:
        print("\n  !! auth credentials found in the WORKER secret — these belong "
              "only in chords-secrets:")
        for key in result["forbidden"]:
            print(f"    {key}")
    else:
        print("\n  [ ok ] no auth credentials in the worker secret (§19.2)")
    print()
