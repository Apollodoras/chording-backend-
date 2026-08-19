"""YouTube's bot check is a property of the egress IP, and the job survives it.

Measured 2026-08-04: six Modal containers, eight yt-dlp player clients each. One
IP answered every client; five refused every client. So the check keys on the
*address*, not on how the client dresses up — which makes it the one analysis
failure where "run this somewhere else" is a real strategy, and makes the
per-client impersonation knobs a dead end worth not adding.

Two things follow, and this file pins both:

- The failure must stay **indistinguishable on the wire** (`EgressBlocked` is a
  `VideoUnavailable`), because the player does not care whose IP it was and the
  iOS client already renders that case calmly.
- It must stay **distinguishable in-process**, because the worker acts on it —
  retiring its container and re-spawning — and acting on a plain
  `VideoUnavailable` would mean retrying private and deleted videos forever.
"""

from __future__ import annotations

import pytest

from app.analysis import engines, ytdlp_source
from app.errors import (CODE_VIDEO_UNAVAILABLE, EgressBlocked, MediaUrlRefused,
                        VideoUnavailable)
from app.jobs import OUTCOME_EGRESS_BLOCKED, OUTCOME_FAILED, run_job
from app.store import STATUS_QUEUED, STATUS_UNAVAILABLE
from tests.conftest import FakeBeatTracker, FakeChordEngine, FakeOnsetDetector, FakeSource

VIDEO = "dQw4w9WgXcQ"


@pytest.fixture(autouse=True)
def fake_engines():
    """`run_job` builds its engines before it runs the pipeline, so an
    unregistered name fails the job as a crash and every assertion below would
    pass for the wrong reason. The `settings` fixture names these; register them.
    """
    engines.register_chord_engine("fake-chords", FakeChordEngine)
    engines.register_beat_tracker("fake-beats", FakeBeatTracker)
    engines.register_onset_detector("fake-onsets", FakeOnsetDetector)
    yield


BOT_CHECK_STDERR = (
    "ERROR: [youtube] dQw4w9WgXcQ: Sign in to confirm you're not a bot. "
    "Use --cookies-from-browser or --cookies for the authentication."
)
PRIVATE_STDERR = "ERROR: [youtube] dQw4w9WgXcQ: Private video. Sign in if you've been granted access"


# --- telling the two apart ---------------------------------------------------

def test_the_bot_check_is_its_own_error_but_still_reads_as_unavailable():
    error = ytdlp_source._unavailable_error(BOT_CHECK_STDERR)

    assert isinstance(error, EgressBlocked)
    # The whole point of the subclass: every existing handler still catches it,
    # and the client sees the identical code, status and message.
    assert isinstance(error, VideoUnavailable)
    assert error.code == CODE_VIDEO_UNAVAILABLE
    assert error.status == 422
    assert error.message == VideoUnavailable().message


def test_an_actually_unavailable_video_is_not_mistaken_for_an_egress_block():
    """The expensive direction of the mistake: retrying a private video on twelve
    containers costs twelve cold starts and cannot ever succeed."""
    error = ytdlp_source._unavailable_error(PRIVATE_STDERR)

    assert isinstance(error, VideoUnavailable)
    assert not isinstance(error, EgressBlocked)


# --- what the job does about it ---------------------------------------------

def test_a_retryable_egress_block_leaves_the_job_alive(settings, store):
    """No terminal status, no error, no refund — the next container inherits the
    row. A client polling through the retry must never see the job fail and then
    un-fail."""
    store.create_job(job_id="j1", uid="u1", video_id=VIDEO)
    store.try_record_use("u1", 5)
    used_before = store.usage_today("u1")

    outcome = run_job(job_id="j1", video_id=VIDEO, uid="u1",
                      settings=settings, store=store,
                      source=FakeSource(error=EgressBlocked()),
                      may_retry_elsewhere=True)

    assert outcome == OUTCOME_EGRESS_BLOCKED
    job = store.get_job("j1")
    assert job.status == STATUS_QUEUED
    assert job.error_code is None
    assert store.usage_today("u1") == used_before, "a retryable attempt must not refund"


def test_the_last_attempt_still_reports_the_failure(settings, store):
    """The half that makes the retry safe: whoever runs the final attempt passes
    `may_retry_elsewhere=False`, and the job lands somewhere terminal instead of
    sitting in `queued` forever."""
    store.create_job(job_id="j2", uid="u1", video_id=VIDEO)
    store.try_record_use("u1", 5)

    outcome = run_job(job_id="j2", video_id=VIDEO, uid="u1",
                      settings=settings, store=store,
                      source=FakeSource(error=EgressBlocked()),
                      may_retry_elsewhere=False)

    assert outcome == OUTCOME_FAILED
    job = store.get_job("j2")
    assert job.status == STATUS_UNAVAILABLE
    assert job.error_code == CODE_VIDEO_UNAVAILABLE
    assert store.usage_today("u1") == 0, "a video we could not fetch is refundable"


def test_a_plain_unavailable_video_is_terminal_even_when_retries_are_allowed(settings, store):
    store.create_job(job_id="j3", uid="u1", video_id=VIDEO)

    outcome = run_job(job_id="j3", video_id=VIDEO, uid="u1",
                      settings=settings, store=store,
                      source=FakeSource(error=VideoUnavailable()),
                      may_retry_elsewhere=True)

    assert outcome == OUTCOME_FAILED
    assert store.get_job("j3").status == STATUS_UNAVAILABLE


# --- the proxy knob ----------------------------------------------------------

def _proxy_arg(args: list[str]) -> str:
    return args[args.index("--proxy") + 1]


def test_the_proxy_is_passed_to_yt_dlp_when_configured(monkeypatch):
    """The lever the measurement actually points at. Inert until set, so the
    default deployment is unchanged."""
    monkeypatch.setenv("CHORDS_YTDLP_PROXY", "http://user:pass@proxy.example:8080")
    args = ytdlp_source.YtDlpSource()._common_args()

    assert "--proxy" in args
    proxy = _proxy_arg(args)
    # Same pool, same credential, same endpoint — only the session is added.
    assert proxy.startswith("http://user:pass_session-")
    assert proxy.endswith("@proxy.example:8080")


def test_no_proxy_argument_when_unset(monkeypatch):
    monkeypatch.delenv("CHORDS_YTDLP_PROXY", raising=False)
    assert "--proxy" not in ytdlp_source.YtDlpSource()._common_args()


# --- the sticky session ------------------------------------------------------

def test_one_session_serves_a_whole_analysis(monkeypatch):
    """Sticky across probe **and** fetch, fresh between analyses.

    Both halves matter, and the unit is the *analysis*, not the subprocess.

    One address for the whole analysis buys two things. A googlevideo media URL is
    bound to the IP that resolved it, so resolving on one address and downloading
    from another is the 403 `MediaUrlRefused` documents. And the bot check is per
    IP: a job needs both calls to get through, so two independent draws is two
    chances to be refused (`1 − p²`), not one.

    A *new* address next time is what gives `EgressBlocked` somewhere to retry
    into. Pinning one session in the credential would buy the first and destroy the
    second.
    """
    monkeypatch.setenv("CHORDS_YTDLP_PROXY", "http://user:pass@proxy.example:8080")
    source = ytdlp_source.YtDlpSource()

    source._new_session()
    probe_args = _proxy_arg(source._common_args())
    fetch_args = _proxy_arg(source._common_args())
    assert probe_args == fetch_args, "probe and fetch must leave from one address"
    assert "_lifetime-10m" in probe_args

    source._new_session()
    assert _proxy_arg(source._common_args()) != probe_args, \
        "a second analysis must draw a fresh address, or a retry retries nothing"


def test_a_session_the_operator_pinned_is_left_alone(monkeypatch):
    """Appending a second `_session-` directive to a credential that already names
    one is a strange way to repay someone who did it deliberately."""
    pinned = "http://user:pass_session-abc123_lifetime-30m@proxy.example:8080"
    monkeypatch.setenv("CHORDS_YTDLP_PROXY", pinned)

    assert _proxy_arg(ytdlp_source.YtDlpSource()._common_args()) == pinned


@pytest.mark.parametrize("raw", [
    "http://proxy.example:8080",              # no credential at all
    "socks5://proxy.example:1080",
])
def test_a_credentialless_proxy_is_passed_through_untouched(monkeypatch, raw):
    """There is nowhere to put a session directive, and inventing a userinfo
    section would turn a working proxy into a failing one."""
    monkeypatch.setenv("CHORDS_YTDLP_PROXY", raw)

    assert _proxy_arg(ytdlp_source.YtDlpSource()._common_args()) == raw


def test_a_password_with_reserved_characters_is_not_double_encoded(monkeypatch):
    """`urlsplit` hands back userinfo still percent-encoded, so rebuilding the URL
    through `quote` would turn `%40` into `%2540` and break authentication."""
    monkeypatch.setenv("CHORDS_YTDLP_PROXY", "http://user:p%40ss@proxy.example:8080")

    proxy = _proxy_arg(ytdlp_source.YtDlpSource()._common_args())

    assert proxy.startswith("http://user:p%40ss_session-")
    assert "%2540" not in proxy


# --- the download refused an URL the probe was given ------------------------

MEDIA_403_STDERR = "ERROR: unable to download video data: HTTP Error 403: Forbidden"


def test_a_refused_media_url_is_retryable_rather_than_terminal():
    """The defect this replaced: a 403 raised a bare `FetchError`, which is
    terminal — the job failed, unrefunded and un-retried, on a video whose own
    probe had just succeeded and which a second container would have served."""
    assert ytdlp_source._is_media_refused(MEDIA_403_STDERR)

    error = MediaUrlRefused()
    assert isinstance(error, EgressBlocked), "the worker must retry this elsewhere"
    # ...while staying the same calm, refundable outcome on the wire.
    assert isinstance(error, VideoUnavailable)
    assert error.code == CODE_VIDEO_UNAVAILABLE
    assert error.message == VideoUnavailable().message


def test_a_retryable_media_refusal_leaves_the_job_alive(settings, store):
    store.create_job(job_id="j4", uid="u1", video_id=VIDEO)
    store.try_record_use("u1", 5)
    used_before = store.usage_today("u1")

    outcome = run_job(job_id="j4", video_id=VIDEO, uid="u1",
                      settings=settings, store=store,
                      source=FakeSource(error=MediaUrlRefused()),
                      may_retry_elsewhere=True)

    assert outcome == OUTCOME_EGRESS_BLOCKED
    assert store.get_job("j4").status == STATUS_QUEUED
    assert store.usage_today("u1") == used_before


def test_an_ordinary_fetch_failure_is_still_terminal():
    """The 403 marker must not swallow the general case — a fetch that failed for
    a reason a fresh container cannot change should not cost three cold starts."""
    assert not ytdlp_source._is_media_refused("ERROR: unable to download video data")
    assert not ytdlp_source._is_media_refused("ERROR: HTTP Error 404: Not Found")


# --- egress, reported ---------------------------------------------------------

def test_the_source_reports_which_way_it_leaves(monkeypatch):
    """Two callers read this and neither can see the environment variable: the
    worker sizes its retry budget from it, and `/healthz` shows an operator
    whether the deployment is paying for clean egress or drawing lots."""
    monkeypatch.delenv("CHORDS_YTDLP_PROXY", raising=False)
    assert ytdlp_source.YtDlpSource().egress == "direct"

    monkeypatch.setenv("CHORDS_YTDLP_PROXY", "http://user:pass@proxy.example:8080")
    assert ytdlp_source.YtDlpSource().egress == "proxy"


def test_healthz_reports_egress_when_this_container_can_fetch(settings, store):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import FakeSource

    app = create_app(settings, store=store, source=FakeSource())
    with TestClient(app) as client:
        assert client.get("/healthz").json()["egress"] == "direct"


def test_healthz_reports_no_egress_at_all_on_a_container_that_cannot_fetch(settings, store):
    """`None`, not `"direct"`. The API container has no audio stack by design
    (§4), so it has no egress posture — and answering `"direct"` would invent a
    reassuring-sounding fact about a container that never leaves the network."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(settings, store=store, source=None)
    with TestClient(app) as client:
        assert client.get("/healthz").json()["egress"] is None


# --- the player client -------------------------------------------------------

def _extractor_arg(args: list[str]) -> str | None:
    if "--extractor-args" not in args:
        return None
    return args[args.index("--extractor-args") + 1]


def test_the_player_client_is_pinned_on_probe_and_fetch(monkeypatch):
    """One client for the whole analysis, for the same reason as one address.

    The probe resolves the formats the fetch then asks for by id. Probing on one
    client and downloading on another asks for a format the second client was
    never offered, and yt-dlp reports "requested format is not available" — which
    `_UNAVAILABLE_MARKERS` turns into a calm, permanent "can't have this video",
    i.e. a fact about the recording rather than about our own arguments.
    """
    monkeypatch.delenv("CHORDS_YTDLP_PLAYER_CLIENT", raising=False)
    source = ytdlp_source.YtDlpSource()

    probe_args = _extractor_arg(source._common_args())
    fetch_args = _extractor_arg(source._common_args())

    assert probe_args == "youtube:player_client=android"
    assert fetch_args == probe_args


def test_the_player_client_is_overridable(monkeypatch):
    """The one certainty about this setting is that it moves: YouTube retires
    clients and yt-dlp adds them, and the next change must not need a deploy of
    this repo to answer."""
    monkeypatch.setenv("CHORDS_YTDLP_PLAYER_CLIENT", "ios")

    assert _extractor_arg(ytdlp_source.YtDlpSource()._common_args()) == \
        "youtube:player_client=ios"


def test_an_empty_player_client_restores_yt_dlps_own_default(monkeypatch):
    """The escape hatch, and deliberately spelled as *absent* rather than as the
    literal string "default" — `--extractor-args youtube:player_client=default`
    means something specific to yt-dlp, and it is not "leave this alone"."""
    monkeypatch.setenv("CHORDS_YTDLP_PLAYER_CLIENT", "")

    assert "--extractor-args" not in ytdlp_source.YtDlpSource()._common_args()
