"""Post-deploy smoke test — the proof that a deployment actually works.

    export CHORDS_BASE_URL=https://…modal.run
    python scripts/smoke.py                      # config half: no token needed
    export CHORDS_ID_TOKEN=<a Firebase ID token>
    python scripts/smoke.py --video QDYfEBY9NM4  # the whole thing, for real

Two halves, because they fail for different reasons and at different times.

**The config half** reads `/healthz` and checks the answers against what a
production deployment must look like. Every check here corresponds to a way this
service can come up *looking* healthy and be wrong: serving 401s to everyone
because the Firebase credential didn't parse, pinned to a SQLite file that dies
with the container because `CHORDS_DATABASE_URL` wasn't passed at deploy time,
or unable to accept an analysis because the API container answered a question
about its own capabilities instead of the worker's.

**The analysis half** runs one real video end to end and then asks for it again.
The second request is the interesting one: §16.4 says a cache hit is free, and
the only way to know it is free is to read the quota before and after. A quota
that ticks on a cache hit is §2.6 broken — a song already in the Library costing
the player something to play.

Exit code is 0 only if every check passed, so this is usable as a deploy gate.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx

BASE_URL = os.environ.get("CHORDS_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
ID_TOKEN = os.environ.get("CHORDS_ID_TOKEN", "")

PASS, FAIL, WARN = "  ok  ", " FAIL ", " warn "
_failed = 0
_warned = 0


def check(label: str, ok: bool, detail: str = "", *, info: str = "", fatal: bool = True) -> bool:
    """Record and print one check.

    `detail` is the *diagnosis*, shown only when the check does not pass — a
    remedy printed next to a green tick is noise at best, and at worst reads as
    the failure it was written to describe. `info` is the reverse: a value worth
    seeing when things are fine.

    `fatal=False` marks a finding that is a deliberate choice in some deployments
    (an absent rate limit is a decision) and must not fail the gate on its own.
    """
    global _failed, _warned
    if ok:
        mark, note = PASS, info
    elif fatal:
        mark, note = FAIL, detail
        _failed += 1
    else:
        mark, note = WARN, detail
        _warned += 1
    print(f"[{mark}] {label}" + (f" — {note}" if note else ""))
    return ok


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def get(path: str, *, token: str | None = None, timeout: float = 30.0) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.get(f"{BASE_URL}{path}", headers=headers, timeout=timeout)


def post(path: str, body: dict, *, token: str, timeout: float = 30.0) -> httpx.Response:
    return httpx.post(f"{BASE_URL}{path}", json=body, timeout=timeout,
                      headers={"Authorization": f"Bearer {token}"})


# --- the config half --------------------------------------------------------

def check_health(*, expect_production: bool) -> dict:
    section(f"/healthz — {BASE_URL}")
    try:
        response = get("/healthz")
    except httpx.HTTPError as exc:
        check("reachable", False, str(exc))
        return {}

    if not check("reachable", response.status_code == 200, f"HTTP {response.status_code}"):
        return {}

    body = response.json()
    check("status ok", body.get("status") == "ok", str(body.get("status")))

    # The store's ping, not its construction — a Postgres pool builds lazily, so
    # an unreachable database looks fine until the first request that needs a row.
    check("database reachable", body.get("db") == "ok",
          f"store says {body.get('db')!r}")

    # THE check for this service. The API container has no audio stack and no
    # engines on purpose (§4); `canAnalyze` asks the runner — which is what
    # actually does the work — whether a new analysis would be accepted.
    check("accepts new analyses (canAnalyze)", body.get("canAnalyze") is True,
          "the deployment cannot start a job — check the worker wiring")

    check("kill switch is off", body.get("analysis") == "enabled",
          f"CHORDS_ANALYSIS_ENABLED says {body.get('analysis')}")
    check("admin/takedown configured", body.get("admin") == "configured",
          "CHORDS_ADMIN_TOKEN is unset — §3 takedowns cannot be served")
    check("video length cap", bool(body.get("maxVideoSeconds")),
          "unset", info=f"{body.get('maxVideoSeconds')}s")

    if expect_production:
        # Refusing a dev token in production is the difference between an
        # authenticated service and an open one.
        check("auth is Firebase, not the dev token",
              body.get("auth") not in {"dev-token", "open", "none"},
              f"auth mode is {body.get('auth')!r} — set CHORDS_REQUIRE_AUTH=1",
              info=f"mode {body.get('auth')!r}")
        # SQLite here means CHORDS_DATABASE_URL was not passed at `modal deploy`
        # time: the deployment is pinned to one container and its writes live on
        # a disk that is not the database anyone else reads.
        check("store is Postgres", body.get("store") == "postgres",
              f"store is {body.get('store')!r} — pass CHORDS_DATABASE_URL to `modal deploy`")
        limits = body.get("rateLimit") or {}
        check("per-IP rate limit set", bool(limits.get("ipPerMin")),
              "CHORDS_RATE_LIMIT_IP_PER_MIN=0 — an unauthenticated flood is uncapped",
              info=f"{limits.get('ipPerMin')}/min", fatal=False)
        # Egress, *only when this container can see it*. On the two-container
        # deployment it cannot: fetching lives in the worker and the API image
        # has `source=None` by design (§4), so `egress` is null here no matter
        # how the worker is configured — and the proxy credential deliberately
        # is not in this container's secret, because a container with no use for
        # a credential should not hold one.
        #
        # Asserting through that would be the exact failure this gate exists to
        # catch, in reverse: a red mark on a correctly-configured deployment,
        # which teaches an operator to ignore the gate. Same lesson as
        # `canAnalyze` — ask the thing that knows, or say you cannot see.
        egress = body.get("egress")
        if egress is None:
            print("[ info ] egress — not visible from the API container; "
                  "check CHORDS_YTDLP_PROXY on the worker secret")
        else:
            # Not fatal: unproxied is a real deployment, it just is not a *good*
            # one. It means roughly five jobs in six start by losing YouTube's
            # per-IP lottery and clawing back through cold-start retries, so the
            # success rate follows Google's datacentre policy rather than
            # anything in this repo. Worth a warning, because the failure it
            # predicts arrives as "analysis got flaky", not as an outage.
            check("egress is proxied", egress == "proxy",
                  "CHORDS_YTDLP_PROXY is unset — every fetch draws at the bot "
                  "check (~1 in 6 IPs pass) and pays for the misses in cold starts",
                  info="rotating residential proxy", fatal=False)

    return body


# --- the analysis half ------------------------------------------------------

def quota(token: str) -> tuple[int, int]:
    body = get("/v1/me", token=token).json()
    used = body.get("quota", {}).get("used", -1)
    limit = body.get("quota", {}).get("limit", -1)
    return used, limit


def check_analysis(video_id: str, token: str, *, poll_timeout_s: float) -> None:
    section(f"one real analysis — {video_id}")

    identity = get("/v1/me", token=token)
    if not check("authenticated", identity.status_code == 200,
                 f"HTTP {identity.status_code} {identity.text[:160]}"):
        return
    me = identity.json()
    check("email verified", me.get("emailVerified") is True,
          "an unverified password account has a quota of zero by design (§16.4)")
    check("analysis advertised to the client", me.get("analysisEnabled") is True,
          "the app hides the analyze affordance on this")

    used_before, limit = quota(token)
    print(f"        quota before: {used_before}/{limit}")

    started = time.monotonic()
    response = post("/v1/analyze", {"videoId": video_id}, token=token)
    if response.status_code == 200:
        check("analysis accepted", True, info="already cached — served inline")
        _check_payload(response.json())
        _check_cache_is_free(video_id, token, used_before)
        return
    if not check("analysis accepted", response.status_code == 202,
                 f"HTTP {response.status_code} {response.text[:200]}"):
        return

    job_id = response.json()["jobId"]
    print(f"        job {job_id}")

    last = None
    while time.monotonic() - started < poll_timeout_s:
        time.sleep(2.0)
        poll = get(f"/v1/analyze/{job_id}", token=token)
        if poll.status_code != 200:
            check("poll succeeded", False, f"HTTP {poll.status_code} {poll.text[:200]}")
            return
        body = poll.json()
        state = (body.get("status"), round(body.get("progress", 0), 2))
        if state != last:
            print(f"        {body.get('status'):<12} {body.get('progress', 0):.0%}")
            last = state
        if body.get("status") == "ready":
            check(f"analysis finished in {time.monotonic() - started:.0f}s", True)
            _check_payload(body)
            _check_cache_is_free(video_id, token, used_before)
            return
        if body.get("status") in {"failed", "blocked", "unavailable"}:
            check("analysis finished", False,
                  f"{body.get('status')}: {body.get('message')} [{body.get('code')}]")
            return

    check("analysis finished", False, f"still {last} after {poll_timeout_s:.0f}s")


def _check_payload(body: dict) -> None:
    """The envelope the app imports. Shape only — the app's own importer is the
    real judge (§16.5's contract fixtures), and this is a deploy check, not a
    second implementation of the lint."""
    song = body.get("song")
    if not check("song payload present", isinstance(song, dict)):
        return
    check("payload is v2", song.get("version") == 2, f"version {song.get('version')!r}")
    sections = song.get("sections") or []
    check("has sections", bool(sections), "none — the payload has no playable content",
          info=f"{len(sections)} section(s)")
    title = (song.get("metadata") or {}).get("title") or song.get("title") or "?"
    tempo = (song.get("metadata") or {}).get("tempo") or song.get("tempo")
    print(f"        '{title}'  tempo={tempo}  sections={len(sections)}")

    sync = body.get("videoSync")
    if sync is None:
        # Withheld below the confidence floor (§13.3) — a correct outcome, and
        # the song still plays self-paced, which is Phase 1's whole promise.
        check("videoSync sidecar", False,
              "withheld — low confidence (§13.3); the song still plays self-paced",
              fatal=False)
    else:
        anchors = sync.get("beatAnchors") or []
        check("videoSync sidecar", bool(anchors), "present but carries no anchors",
              info=f"{len(anchors)} anchors")


def _check_cache_is_free(video_id: str, token: str, used_before: int) -> None:
    """§16.4 and §2.6: the same video again must cost nothing.

    This is the check worth having in a deploy gate. A cache hit that charges is
    invisible in every functional test — everything still works — and it means a
    song already in the player's Library costs them something to play, which §2.6
    says this feature may never do.
    """
    again = post("/v1/analyze", {"videoId": video_id}, token=token)
    check("second request is a cache hit", again.status_code == 200,
          f"HTTP {again.status_code} — expected 200 with the result inline")

    used_after, limit = quota(token)
    check("cache hit did not charge the player", used_after == used_before,
          f"quota went {used_before} → {used_after} of {limit}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", help="run one real analysis end to end (needs CHORDS_ID_TOKEN)")
    parser.add_argument("--local", action="store_true",
                        help="relax the production-only checks (dev token, Postgres, limits)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="how long to poll before calling it stuck (default 300s)")
    args = parser.parse_args()

    check_health(expect_production=not args.local)

    if args.video:
        if not ID_TOKEN:
            check("CHORDS_ID_TOKEN set", False,
                  "needed for the analysis half — a Firebase ID token for a verified account")
        else:
            check_analysis(args.video, ID_TOKEN, poll_timeout_s=args.timeout)
    else:
        print("\n(config half only — pass --video <id> with CHORDS_ID_TOKEN "
              "to run a real analysis)")

    print()
    if _failed:
        print(f"FAILED — {_failed} check(s) failed"
              + (f", {_warned} warning(s)" if _warned else ""))
        return 1
    print("All checks passed" + (f" ({_warned} warning(s))" if _warned else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
