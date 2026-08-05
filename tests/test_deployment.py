"""The deployed shape, which is not the shape every other test runs in.

`modal_app.py` cannot be imported by the suite — `modal` is not a dependency of
this package, and adding it would put the deploy tool in the audio-free CI job
for the sake of one import. So the properties that file depends on are asserted
here, on the classes it actually uses.

Every one of these covers something that was broken and could not have been seen
locally, because the failure only exists in the two-container deployment:

- the API container has **no audio stack and no engines by design** (§4). Asking
  whether *it* can fetch or analyze, to decide whether an analysis can run, is a
  question that answers "no" on a perfectly healthy production deployment.
- a worker container is killed from outside — Modal's `timeout=`, an OOM at the
  memory cap, a reclaim — and `run_job`'s terminal-status guarantee, which holds
  for every exception it can catch, does not hold for a SIGKILL.
- Modal delivers secrets as environment variables, and the worker mounts no
  Volume, so a cookies **path** is unconfigurable there.
"""

from __future__ import annotations

import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.analysis import engines
from app.jobs import JobRunner, RemoteJobRunner
from app.main import create_app
from app.store import (
    STATUS_ANALYZING,
    STATUS_FAILED,
    STATUS_READY,
    SQLiteStore,
)

AUTH = {"Authorization": "Bearer dev-token"}
VIDEO = "dQw4w9WgXcQ"


class SpawningRunner(RemoteJobRunner):
    """What `ModalJobRunner` is, minus Modal: dispatch elsewhere, record it."""

    def __init__(self, settings, store, source=None):
        super().__init__(settings, store, source)
        self.spawned: list[str] = []

    def submit(self, *, job_id: str, video_id: str, difficulty: str, uid: str) -> None:
        self.spawned.append(job_id)


class FailingRunner(RemoteJobRunner):
    def submit(self, *, job_id: str, video_id: str, difficulty: str, uid: str) -> None:
        raise RuntimeError("modal refused the spawn")


@pytest.fixture
def no_engines():
    """The API image's real registry: empty. Nothing is installed over there."""
    registries = (engines._CHORD_ENGINES, engines._BEAT_TRACKERS,
                  engines._ONSET_DETECTORS, engines._STRUCTURE_PROBES)
    saved = [dict(registry) for registry in registries]
    for registry in registries:
        registry.clear()
    yield
    for registry, contents in zip(registries, saved):
        registry.update(contents)


@pytest.fixture
def api_container(settings, store, no_engines):
    """`fastapi_app()` from `modal_app.py`, as exactly as this suite can build it:
    no source, no engines, and a runner that dispatches to a worker."""
    runner = SpawningRunner(settings, store)
    app = create_app(settings, store=store, source=None, runner=runner)
    with TestClient(app) as client:
        client.runner = runner
        yield client


# --- the API container can start an analysis it cannot itself perform --------

def test_api_container_accepts_an_analysis_without_its_own_audio_stack(api_container):
    """The regression that made the whole deployment inert.

    Gating on `app.state.source is None or not engines.is_ready(settings)` reads
    like "can this service analyze", and in the deployment it means "can this
    *container* analyze" — which §4 guarantees is no. Every uncached analysis
    answered 503 while the worker sat there able to do the job.
    """
    response = api_container.post("/v1/analyze", json={"videoId": VIDEO}, headers=AUTH)

    assert response.status_code == 202, response.json()
    assert response.json()["status"] == "queued"
    assert api_container.runner.spawned == [response.json()["jobId"]]


def test_api_container_advertises_analysis_as_available(api_container):
    """`/v1/me` gates the app's analyze affordance, so it must answer the same
    question `POST /v1/analyze` does. It used to read the local engine registry
    and hide the feature on a working deployment."""
    body = api_container.get("/v1/me", headers=AUTH).json()
    assert body["analysisEnabled"] is True


def test_healthz_separates_container_capability_from_service_capability(api_container):
    """Both facts, neither one lying. The API container reporting `fetch:
    unconfigured` and no engines is §4 working — `canAnalyze` is the field that
    answers "would a new analysis be accepted"."""
    body = api_container.get("/healthz").json()

    assert body["fetch"] == "unconfigured"
    assert body["engines"] == {"chords": [], "beats": [], "onsets": [], "structure": []}
    assert body["enginesReady"] is False
    assert body["canAnalyze"] is True
    assert body["db"] == "ok"


def test_inline_runner_with_no_source_still_refuses(settings, store, no_engines):
    """The other direction: a single-container deployment that genuinely cannot
    analyze must still answer a clean 503 rather than queue work nothing runs."""
    app = create_app(settings, store=store, source=None,
                     runner=JobRunner(settings, store, None))
    with TestClient(app) as client:
        response = client.post("/v1/analyze", json={"videoId": VIDEO}, headers=AUTH)
        assert response.status_code == 503
        assert response.json()["code"] == "feature_disabled"
        assert client.get("/healthz").json()["canAnalyze"] is False


def test_kill_switch_still_wins_over_a_capable_runner(settings, store, no_engines):
    """§3's switch is not "are the engines there" — it must stop a deployment
    that is fully able to run the job."""
    app = create_app(replace(settings, analysis_enabled=False), store=store,
                     source=None, runner=SpawningRunner(settings, store))
    with TestClient(app) as client:
        response = client.post("/v1/analyze", json={"videoId": VIDEO}, headers=AUTH)
        assert response.status_code == 503
        assert response.json()["code"] == "feature_disabled"


# --- a dispatch that fails must not be paid for ------------------------------

def test_failed_dispatch_refunds_and_terminates_the_job(settings, store, no_engines):
    """Modal refusing a spawn charged the player for a job row nothing would ever
    pick up — and `active_job_for` then handed that dead id to everyone else who
    asked for the same video."""
    app = create_app(settings, store=store, source=None,
                     runner=FailingRunner(settings, store, None))
    with TestClient(app) as client:
        response = client.post("/v1/analyze", json={"videoId": VIDEO}, headers=AUTH)

    assert response.status_code == 503
    assert store.usage_today("local-dev") == 0, "charged for a job that never ran"

    job = store.active_job_for(VIDEO, "normal")
    assert job is None, "a dead job is still blocking its video"


# --- an abandoned job must not block its video forever -----------------------

def _abandon(store: SQLiteStore, *, job_id: str, uid: str, age_s: float) -> None:
    """A job row left mid-flight `age_s` ago — a SIGKILLed worker, in the only
    state one leaves behind."""
    store.create_job(job_id=job_id, uid=uid, video_id=VIDEO, difficulty="normal")
    store.update_job(job_id, status=STATUS_ANALYZING, progress=0.35)
    stale = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                          time.gmtime(time.time() - age_s))
    with store._cursor(write=True) as cur:
        cur.execute("UPDATE jobs SET updated_at = ? WHERE job_id = ?", (stale, job_id))


def test_abandoned_job_stops_blocking_its_video(store):
    """The compounding half of the bug: `active_job_for` joins new requests to an
    in-flight job, and a killed worker's row is 'in flight' forever. One dead
    container made one video permanently un-analyzable."""
    _abandon(store, job_id="dead", uid="u1", age_s=store._JOB_LEASE_S + 60)

    assert store.active_job_for(VIDEO, "normal") is None


def test_job_within_its_lease_is_still_joined(store):
    """The behaviour that must survive the fix: a slow-but-alive job is exactly
    what `active_job_for` exists to share."""
    _abandon(store, job_id="alive", uid="u1", age_s=30)

    active = store.active_job_for(VIDEO, "normal")
    assert active is not None and active.job_id == "alive"


def test_reaper_fails_and_refunds_an_abandoned_job(store):
    """The poller's side. Without a terminal status the client watches
    `analyzing` until it gives up, and the row is never prunable — `prune_jobs`
    only collects rows that already reached one."""
    store.try_record_use("u1", 5)
    _abandon(store, job_id="dead", uid="u1", age_s=store._JOB_LEASE_S + 60)

    assert store.reap_stale_jobs(older_than_s=store._JOB_LEASE_S) == 1

    job = store.get_job("dead")
    assert job.status == STATUS_FAILED
    assert job.error_code == "analysis_failed"
    assert job.error_message
    assert store.usage_today("u1") == 0, "player paid for an analysis they never got"


def test_reaper_leaves_a_live_job_alone(store):
    store.try_record_use("u1", 5)
    _abandon(store, job_id="alive", uid="u1", age_s=30)

    assert store.reap_stale_jobs(older_than_s=store._JOB_LEASE_S) == 0
    assert store.get_job("alive").status == STATUS_ANALYZING
    assert store.usage_today("u1") == 1


def test_reaper_never_refunds_a_job_that_delivered(store):
    """A worker that was merely slow and finished between the reaper's read and
    its write must keep its result — and must not be refunded for it."""
    store.try_record_use("u1", 5)
    _abandon(store, job_id="slow", uid="u1", age_s=store._JOB_LEASE_S + 60)
    store.update_job("slow", status=STATUS_READY, progress=1.0)

    assert store.reap_stale_jobs(older_than_s=store._JOB_LEASE_S) == 0
    assert store.get_job("slow").status == STATUS_READY
    assert store.usage_today("u1") == 1


def test_reaped_job_is_then_prunable(store):
    """The two housekeepers meet: reaping is what makes an abandoned row eligible
    for the TTL sweep that was previously unable to see it."""
    _abandon(store, job_id="dead", uid="u1", age_s=store._JOB_LEASE_S + 60)

    assert store.prune_jobs(older_than_s=0) == 0, "prune should not touch a live status"
    store.reap_stale_jobs(older_than_s=store._JOB_LEASE_S)
    assert store.prune_jobs(older_than_s=0) == 1


# --- the bot-check escape hatch has to be configurable where it runs ---------

def test_cookies_are_accepted_as_content_not_only_as_a_path(monkeypatch, tmp_path):
    """Modal delivers secrets as env vars and the worker mounts no Volume, so a
    path has nothing to point at. Supporting only `CHORDS_YTDLP_COOKIES` made the
    documented mitigation for the most likely production failure unconfigurable
    in production."""
    from app.analysis.ytdlp_source import YtDlpSource

    monkeypatch.delenv("CHORDS_YTDLP_COOKIES", raising=False)
    monkeypatch.setenv("CHORDS_YTDLP_COOKIES_CONTENT", "# Netscape HTTP Cookie File\n")

    source = YtDlpSource()
    args = source._common_args()

    assert "--cookies" in args
    path = args[args.index("--cookies") + 1]
    with open(path) as handle:
        assert handle.read() == "# Netscape HTTP Cookie File\n"


def test_an_explicit_cookies_path_is_used_as_given(monkeypatch, tmp_path):
    from app.analysis.ytdlp_source import YtDlpSource

    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("CHORDS_YTDLP_COOKIES", str(cookies))
    monkeypatch.setenv("CHORDS_YTDLP_COOKIES_CONTENT", "ignored")

    args = YtDlpSource()._common_args()
    assert args[args.index("--cookies") + 1] == str(cookies)


def test_no_cookies_configured_passes_no_flag(monkeypatch):
    from app.analysis.ytdlp_source import YtDlpSource

    monkeypatch.delenv("CHORDS_YTDLP_COOKIES", raising=False)
    monkeypatch.delenv("CHORDS_YTDLP_COOKIES_CONTENT", raising=False)

    assert "--cookies" not in YtDlpSource()._common_args()


def test_bot_check_is_reported_as_unavailable_and_logged(monkeypatch, caplog):
    """To the player it is the same calm outcome as a private video — and it is
    refundable for the same reason. To an operator it is the opposite: nothing is
    wrong with the video, and every video is about to fail the same way."""
    from app.analysis import ytdlp_source

    with caplog.at_level("ERROR"):
        assert ytdlp_source._looks_unavailable(
            "ERROR: [youtube] abc: Sign in to confirm you're not a bot"
        )
    assert "bot check" in caplog.text.lower()


# --- the health check must not be green in front of a dead database ----------

def test_healthz_reports_an_unreachable_store(api_container, monkeypatch):
    def explode():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(api_container.app.state.store, "ping", explode)
    body = api_container.get("/healthz").json()

    assert body["db"] == "unavailable"


# ---------------------------------------------------------------------------
# The image's package list is a duplicate of pyproject's, and duplicates drift
# ---------------------------------------------------------------------------

def _requirement_names(lines) -> set[str]:
    """Distribution names from requirement strings, normalised (PEP 503).

    Extras and version specifiers are dropped: `uvicorn[standard]>=0.30` and
    `uvicorn>=0.31` are the same *dependency*, and this check is about one list
    forgetting a package the other has, not about pinning.
    """
    import re

    names = set()
    for line in lines:
        match = re.match(r"^\s*([A-Za-z0-9._-]+)", line)
        if match:
            names.add(re.sub(r"[-_.]+", "-", match.group(1)).lower())
    return names


def test_the_modal_image_installs_everything_pyproject_declares():
    """`modal_app.py`'s BASE_PACKAGES must not fall behind pyproject.

    Modal builds its images from an explicit package list rather than from the
    local project, so that list is a hand-maintained copy of
    `[project].dependencies` — and the copy is invisible to every other check
    here. `modal_app.py` cannot be imported (see the module docstring), so the
    list is read out of the source.

    This is not hygiene. When `python-multipart` was added to pyproject and not
    to BASE_PACKAGES, the suite stayed green, `modal deploy` **succeeded**, and
    the API container then died at import — `create_app()` runs at module scope,
    so FastAPI's "Form data requires python-multipart" fired while registering
    `POST /v1/analyze/upload` and the ASGI app never built at all. Not a degraded
    upload path: no `/healthz`, no cached maps, nothing.
    """
    import ast
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    declared = _requirement_names(
        tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    )

    tree = ast.parse((root / "modal_app.py").read_text())
    packages = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "BASE_PACKAGES" for t in node.targets
        ):
            packages = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    assert packages is not None, "BASE_PACKAGES not found in modal_app.py"

    missing = declared - _requirement_names(packages)
    assert not missing, (
        f"pyproject declares {sorted(missing)} but modal_app.py's BASE_PACKAGES does not — "
        f"the deployed image would be built without them, and the API container dies at "
        f"import rather than starting degraded"
    )


def test_a_proxied_worker_retries_less_than_an_unproxied_one():
    """The two egress retry budgets must not collapse into one number.

    They encode different things. Unproxied, a retry is the *mitigation*: each
    attempt is an independent draw at a fresh Modal IP, ~1 in 6 clears YouTube's
    bot check, and twelve of them is what gets a job to ~89%. Proxied, the first
    attempt is bought and paid for, so the same twelve buys nothing and spends
    eleven cold starts arriving at a failure that was never going to clear.

    Read out of the source for the same reason as BASE_PACKAGES above:
    `modal_app.py` cannot be imported by this suite. The relationship is pinned
    rather than the values — 12 and 3 are judgement calls and should stay
    tunable; "proxied costs at least as much as unproxied" is the bug.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "modal_app.py").read_text())

    budgets: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("MAX_EGRESS_ATTEMPTS"):
                    budgets[target.id] = node.value.value

    assert set(budgets) == {"MAX_EGRESS_ATTEMPTS_DIRECT", "MAX_EGRESS_ATTEMPTS_PROXIED"}, (
        f"expected exactly the two egress budgets, found {sorted(budgets)} — a single "
        f"MAX_EGRESS_ATTEMPTS means one of the two deployments is being charged the "
        f"other's retry policy"
    )
    assert budgets["MAX_EGRESS_ATTEMPTS_PROXIED"] < budgets["MAX_EGRESS_ATTEMPTS_DIRECT"]
    # Insurance, not abolition: a rotating pool can still miss, and a proxied
    # deployment that never retries fails songs the second attempt would have got.
    assert budgets["MAX_EGRESS_ATTEMPTS_PROXIED"] >= 2
