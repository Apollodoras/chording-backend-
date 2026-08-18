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


def test_the_container_pin_is_lifted_by_a_flag_not_by_a_credential():
    """`CHORDS_SCALE_OUT=1`, not the Postgres DSN.

    The pin exists because SQLite on a network volume tolerates one writer, and
    only the deploy shell can say whether the secret holds a DSN. Spelling that
    assertion as the DSN *itself* meant a live database password had to be pasted
    into a shell — into history, into scrollback — to communicate one bit. It got
    skipped, exactly as that ask deserves, and the deployment ran pinned to a
    single API container while `/healthz` reported `store: "postgres"` and looked
    entirely fine.

    Read out of the source rather than imported, for the reason in this module's
    docstring. Both spellings must remain live: dropping `CHORDS_DATABASE_URL`
    would silently re-pin any runbook or CI job still using it, which is the same
    invisible regression in the other direction.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "modal_app.py").read_text()
    pin = next(line for line in source.splitlines() if line.startswith("_SCALE_OUT"))
    assert "CHORDS_SCALE_OUT" in pin, (
        "the pin must be liftable without putting a database credential in the "
        "deploy shell"
    )
    assert "CHORDS_DATABASE_URL" in pin, (
        "the old spelling must keep working, or an existing runbook silently "
        "re-pins the deployment to one container"
    )


def test_the_deploy_says_out_loud_which_container_shape_it_chose():
    """The pin's failure mode is being invisible, not being wrong.

    A pinned API container behaves correctly under any load one person can
    generate; it is discovered later, by a queue. `modal deploy` runs this module
    locally, so the deploy shell is the only place the answer exists and the only
    place it can be reported.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "modal_app.py").read_text()
    assert "print(" in source and "PINNED TO 1" in source


# --- the time budget --------------------------------------------------------
#
# Four numbers in three files describe one job's wall clock, and they used to
# contradict each other in every direction. Each was defensible alone; the set was
# not, and the failure it produced was the worst-shaped one available — a
# *successful but slow* fetch SIGKILLed with no terminal status written, leaving the
# player watching a spinner until the 900 s lease reaper noticed.

def _source_constant(relative_path: str, name: str):
    """One module-level `NAME = <literal>` out of a file in the repo root.

    Read rather than imported for the reason the whole module gives: `modal` is not
    a dependency of this package, and both files this is used on import it at
    module level.
    """
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / relative_path
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value.value
    raise AssertionError(f"{relative_path} no longer defines {name}")


def _modal_constant(name: str):
    return _source_constant("modal_app.py", name)


def test_the_stage_ceilings_fit_inside_the_job_deadline():
    """The deadline has to be the thing that fires, not the container timeout.

    `probe + fetch + decode + DSP reserve` must fit, because only the deadline can
    write a terminal status, a message the player can read, and a refund. If the
    stages can outlast it, the job is still running when something outside kills it
    — and a SIGKILLed container writes nothing at all.

    This was 45 + 240 + 120 = 405 s of subprocess budget against a 180 s deadline.
    """
    from app.config import Settings

    settings = Settings()
    assert settings.stage_budget_s <= settings.job_deadline_s, (
        f"the stages may spend {settings.stage_budget_s}s but the job is killed at "
        f"{settings.job_deadline_s}s — a slow stage would outlive the only check that "
        f"can answer the player"
    )


def test_the_worker_timeout_sits_outside_the_job_deadline():
    """And the lease sits outside that. Read outward, each layer is the backstop
    for the one inside it:

        stages ≤ job deadline < container timeout < job lease

    The container timeout was 300 s against a 405 s stage budget, so Modal killed
    workers that were working. The lease has to be larger again because a container
    killed at its timeout writes nothing, and the reaper is the only thing left.
    """
    from app.config import Settings
    from app.store import Store

    settings = Settings()
    worker_timeout = _modal_constant("WORKER_TIMEOUT_S")

    assert settings.job_deadline_s < worker_timeout, (
        f"the worker is killed at {worker_timeout}s but its own deadline is "
        f"{settings.job_deadline_s}s — the kill wins, and a kill writes no status"
    )
    assert worker_timeout < Store._JOB_LEASE_S, (
        f"the lease ({Store._JOB_LEASE_S}s) must outlast the container timeout "
        f"({worker_timeout}s), or the reaper collects jobs that are still running"
    )


def test_the_secret_checker_restates_the_budget_correctly():
    """`scripts/secret_check.py` re-derives the chain from what a secret carries,
    and to do that on a slim image it restates two constants it cannot import.

    The whole point of that script is to catch a value that drifted out of sight;
    a diagnostic whose own copy of the ceiling has drifted would report a healthy
    chain while the deployment breaks, which is the failure it exists to prevent.
    """
    from app.store import Store

    restated = lambda name: _source_constant("scripts/secret_check.py", name)  # noqa: E731

    assert restated("WORKER_TIMEOUT_S") == _modal_constant("WORKER_TIMEOUT_S")
    assert restated("JOB_LEASE_S") == Store._JOB_LEASE_S


def test_a_timed_out_job_is_refunded():
    """The deadline is ours to size; the video did nothing wrong.

    It used to raise a bare `AnalysisError`, whose code is `analysis_failed` — a
    code deliberately *outside* `REFUNDABLE_CODES` because it means the analysis
    genuinely ran and genuinely failed. So a player whose job we killed for taking
    too long paid a daily analysis for our capacity planning.
    """
    from app.errors import AnalysisTimeout
    from app.jobs import REFUNDABLE_CODES

    assert AnalysisTimeout().code in REFUNDABLE_CODES


# --- the worker cannot use SQLite -------------------------------------------

def test_a_remote_worker_refuses_a_sqlite_store():
    """The two-container shape only works on Postgres, and nothing enforced it.

    The worker calls `build_store` in *its own container*. `db_path` is relative and
    the `chords-data` Volume is mounted on the API function only, so SQLite there
    means a brand-new database file on a disk that dies with the call: `analyzing`,
    `ready`, and the finished map all written where nothing will ever read them.
    From the API's side every job sits at `queued` until the lease reaper fails it —
    a total outage of the feature, behind a `/healthz` that is green in *both*
    containers, because individually each one is fine.

    The `MAX_CONTAINERS = 1` pin does not cover this: the worker is a separate
    container whether or not the API is pinned.
    """
    from app.config import Settings
    from app.store import ROLE_WORKER, StoreUnusable, build_store

    with pytest.raises(StoreUnusable) as raised:
        build_store(Settings(db_path="chords.sqlite3"), role=ROLE_WORKER)
    # The message has to name the remedy: this surfaces in a worker's log, and the
    # operator reading it is the only audience it will ever have.
    assert "CHORDS_DATABASE_URL" in str(raised.value)


def test_the_api_container_still_gets_sqlite(tmp_path):
    """The refusal is about the worker, not about SQLite. Local dev, CI and a
    single-container deployment are all legitimate and all use it."""
    from app.config import Settings
    from app.store import SQLiteStore, build_store

    store = build_store(Settings(db_path=str(tmp_path / "t.sqlite3")))
    assert isinstance(store, SQLiteStore)
    store.close()


def test_the_worker_function_asks_for_the_worker_role():
    """The guard above is only worth having if the worker actually passes the flag.

    Read out of the source, like the other `modal_app.py` assertions: an
    `analysis_worker` that called plain `build_store()` would be exactly as broken
    as before, and the guard would sit there looking like protection.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "modal_app.py").read_text()
    worker = source.split("def analysis_worker(", 1)[1].split("\ndef ", 1)[0]
    assert "role=ROLE_WORKER" in worker, (
        "analysis_worker must build its store with role=ROLE_WORKER, or the SQLite "
        "guard never runs in the one place it exists for"
    )


def test_a_materialized_cookie_file_is_removed(monkeypatch):
    """It is a credential, and it was never unlinked.

    A Modal container's filesystem dies with it, so this is belt-and-braces there —
    but this source also runs under `ThreadJobRunner` in a long-lived local process
    and in the bench harness, and a 0600 file full of session cookies that outlives
    the run that needed it is a credential left lying around.
    """
    import os

    from app.analysis import ytdlp_source

    monkeypatch.setenv("CHORDS_YTDLP_COOKIES_CONTENT", "# Netscape HTTP Cookie File\n")
    monkeypatch.delenv("CHORDS_YTDLP_COOKIES", raising=False)

    source = ytdlp_source.YtDlpSource()
    path = source._cookie_file()
    assert path and os.path.isfile(path)

    source._discard_cookie_file()
    assert not os.path.exists(path)
    # Idempotent: the atexit hook fires even after an explicit discard.
    source._discard_cookie_file()


def test_the_cookie_file_cleanup_is_registered_at_exit(monkeypatch):
    """A process that never calls `_discard_cookie_file` explicitly — which is every
    process — still has to lose the file."""
    import atexit

    from app.analysis import ytdlp_source

    registered = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: registered.append(fn) or fn)
    monkeypatch.setenv("CHORDS_YTDLP_COOKIES_CONTENT", "# Netscape HTTP Cookie File\n")
    monkeypatch.delenv("CHORDS_YTDLP_COOKIES", raising=False)

    source = ytdlp_source.YtDlpSource()
    try:
        source._cookie_file()
        assert source._discard_cookie_file in registered
    finally:
        source._discard_cookie_file()


def test_a_cookies_path_given_by_the_operator_is_never_unlinked(monkeypatch, tmp_path):
    """We materialized it, we remove it. A path the operator supplied is theirs."""
    from app.analysis import ytdlp_source

    given = tmp_path / "cookies.txt"
    given.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("CHORDS_YTDLP_COOKIES", str(given))
    monkeypatch.delenv("CHORDS_YTDLP_COOKIES_CONTENT", raising=False)

    source = ytdlp_source.YtDlpSource()
    assert source._cookie_file() == str(given)
    source._discard_cookie_file()
    assert given.is_file()
