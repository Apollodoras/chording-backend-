"""The HTTP surface — §16's contract, and §3's takedown surface.

The error assertions matter more than they look: the iOS client already has copy
written for `401` → not signed in, `403` → email unverified, `429` → quota, and
"other non-2xx → the server's own message" (`MoRosettaError`). Mirroring those
semantics is what §16.3 means by "the new client gets its whole failure UI for
free", so a test that let a bare string or a missing `message` key through would
be letting a blank error dialog through.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.analysis import engines
from app.jobs import JobRunner
from app.main import create_app
from app.store import BLOCK_VIDEO
from tests.conftest import FakeBeatTracker, FakeChordEngine, FakeOnsetDetector, FakeSource

AUTH = {"Authorization": "Bearer dev-token"}
ADMIN = {"X-Admin-Token": "admin-secret", "X-Admin-Actor": "agent@example.com"}
VIDEO = "dQw4w9WgXcQ"


@pytest.fixture(autouse=True)
def registered_engines():
    """Register the fakes under the names `settings` selects, so the routes take
    the same "engines are ready" path production will."""
    engines.register_chord_engine("fake-chords", FakeChordEngine)
    engines.register_beat_tracker("fake-beats", FakeBeatTracker)
    engines.register_onset_detector("fake-beats", FakeOnsetDetector)
    yield
    engines._CHORD_ENGINES.clear()
    engines._BEAT_TRACKERS.clear()
    engines._ONSET_DETECTORS.clear()


@pytest.fixture
def client(settings, store):
    source = FakeSource()
    # The inline runner: a job has finished by the time `submit` returns, which
    # makes the polling assertions readable without sleeping.
    app = create_app(settings, store=store, source=source,
                     runner=JobRunner(settings, store, source))
    with TestClient(app) as client:
        client.source = source
        yield client


def analyze(client, **body):
    return client.post("/v1/analyze", json={"videoId": VIDEO, **body}, headers=AUTH)


# --- health -----------------------------------------------------------------

def test_healthz_reports_what_actually_built(client):
    """Not what config asked for. A green healthz hiding a dead authenticator is
    the failure Mo learned from."""
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["auth"] == "dev-token"
    assert body["analysis"] == "enabled"
    assert body["fetch"] == "configured"
    assert body["admin"] == "configured"
    assert body["enginesReady"] is True


def test_healthz_says_when_the_kill_switch_is_off(settings, store):
    """A protection that is silently off is worse than one that is visibly off,
    because only the second gets turned back on."""
    app = create_app(replace(settings, analysis_enabled=False), store=store, source=FakeSource())
    with TestClient(app) as client:
        assert client.get("/healthz").json()["analysis"] == "disabled"


def test_healthz_is_exempt_from_rate_limiting(settings, store):
    """A liveness probe that can rate-limit itself into red is worse than
    useless."""
    app = create_app(replace(settings, rate_limit_ip_per_min=1), store=store, source=FakeSource())
    with TestClient(app) as client:
        for _ in range(5):
            assert client.get("/healthz").status_code == 200


# --- auth -------------------------------------------------------------------

def test_an_unauthenticated_call_gets_the_shape_the_client_renders(client):
    response = client.post("/v1/analyze", json={"videoId": VIDEO})
    assert response.status_code == 401
    assert response.json()["message"]


def test_me_mirrors_mos_shape(client):
    """So the client's `MoMeService` is reused rather than rewritten."""
    body = client.get("/v1/me", headers=AUTH).json()
    assert set(body) >= {"userID", "displayName", "quota", "emailVerified", "signInProvider"}
    assert set(body["quota"]) == {"used", "limit", "resetsAtUTC"}


# --- analyze ----------------------------------------------------------------

def test_a_cold_request_returns_a_job_id(client):
    response = analyze(client)
    assert response.status_code == 202
    assert response.json()["jobId"]


def test_polling_a_finished_job_returns_the_song_and_its_sidecar(client):
    job_id = analyze(client).json()["jobId"]
    body = client.get(f"/v1/analyze/{job_id}", headers=AUTH).json()
    assert body["status"] == "ready"
    assert body["song"]["id"] == f"yt:{VIDEO}"
    assert body["videoSync"]["videoId"] == VIDEO
    assert body["videoSync"]["beatAnchors"]


def test_polling_a_job_does_not_spend_the_budget_that_starts_one(settings, store):
    """The defect that made long analyses report failure while they succeeded.

    `POST /v1/analyze` returns a job id and the client polls until it is ready —
    so a single analysis is one expensive request followed by however many cheap
    ones the job's duration demands. Sharing one per-uid budget between the two
    meant the cheap ones ate the expensive one's allowance and then started
    429ing themselves: at ten a minute, any job over ~40 s rate-limited its own
    status checks, and the client turned that into "that song couldn't be made".

    Asserted as **more polls than the spend budget allows**, not as a specific
    number, because the number is a deployment's business and the relationship is
    the invariant.
    """
    app = create_app(replace(settings, rate_limit_per_min=2, rate_limit_poll_per_min=30),
                     store=store, source=FakeSource(),
                     runner=JobRunner(settings, store, FakeSource()))
    with TestClient(app) as client:
        job_id = client.post("/v1/analyze", json={"videoId": VIDEO}, headers=AUTH).json()["jobId"]
        for _ in range(20):
            assert client.get(f"/v1/analyze/{job_id}", headers=AUTH).status_code == 200


def test_the_poll_budget_is_a_ceiling_not_an_exemption(settings, store):
    """Cheap is not free. A client whose backoff has broken must still be stopped
    — the fix is a bigger bucket for polls, not the absence of one."""
    app = create_app(replace(settings, rate_limit_per_min=10, rate_limit_poll_per_min=3),
                     store=store, source=FakeSource(),
                     runner=JobRunner(settings, store, FakeSource()))
    with TestClient(app) as client:
        job_id = client.post("/v1/analyze", json={"videoId": VIDEO}, headers=AUTH).json()["jobId"]
        codes = [client.get(f"/v1/analyze/{job_id}", headers=AUTH).status_code for _ in range(5)]
    assert 429 in codes


def test_a_poll_429_tells_the_client_how_long_to_wait(settings, store):
    """`Retry-After` is what makes this recoverable rather than fatal: the client
    treats a poll 429 as backpressure and sleeps, so a header that went missing
    would put it straight back to guessing."""
    app = create_app(replace(settings, rate_limit_per_min=10, rate_limit_poll_per_min=1),
                     store=store, source=FakeSource(),
                     runner=JobRunner(settings, store, FakeSource()))
    with TestClient(app) as client:
        job_id = client.post("/v1/analyze", json={"videoId": VIDEO}, headers=AUTH).json()["jobId"]
        client.get(f"/v1/analyze/{job_id}", headers=AUTH)
        response = client.get(f"/v1/analyze/{job_id}", headers=AUTH)
    assert response.status_code == 429
    assert response.json()["code"] == "rate_limited"
    assert int(response.headers["Retry-After"]) >= 1


def test_starting_a_job_is_still_limited_when_polling_is_generous(settings, store):
    """The separation must not have widened the budget it was meant to protect —
    the spending route is the one the limiter exists for."""
    app = create_app(replace(settings, rate_limit_per_min=1, rate_limit_poll_per_min=60),
                     store=store, source=FakeSource(),
                     runner=JobRunner(settings, store, FakeSource()))
    with TestClient(app) as client:
        client.post("/v1/analyze", json={"videoId": "aaaaaaaaaaa"}, headers=AUTH)
        response = client.post("/v1/analyze", json={"videoId": "bbbbbbbbbbb"}, headers=AUTH)
    assert response.status_code == 429
    assert response.json()["code"] == "rate_limited"


def test_a_warm_request_returns_the_result_inline(client):
    """The common case once warm (§16.1)."""
    analyze(client)
    response = analyze(client)
    assert response.status_code == 200
    assert response.json()["song"]["id"] == f"yt:{VIDEO}"


def test_a_cache_hit_costs_the_player_nothing(client):
    """§16.4: a cached map costs us nothing, so it must not cost the player a
    daily analysis either."""
    analyze(client)
    used = client.get("/v1/me", headers=AUTH).json()["quota"]["used"]
    for _ in range(5):
        assert analyze(client).status_code == 200
    assert client.get("/v1/me", headers=AUTH).json()["quota"]["used"] == used


def test_a_cache_hit_does_not_touch_the_recording_again(client):
    analyze(client)
    decoded = len(client.source.decoded)
    analyze(client)
    assert len(client.source.decoded) == decoded


def test_one_analysis_stores_one_song_under_the_videos_own_id(client):
    """Asking again is a cache hit. The id used to carry a difficulty suffix so
    one video could hold three rows; it is the video's id now."""
    analyze(client)
    response = analyze(client)
    assert response.status_code == 200
    assert response.json()["song"]["id"] == f"yt:{VIDEO}"


def test_a_client_still_sending_difficulty_is_not_rejected(client):
    """The §5.5 tiers are gone from the wire, and a phone in someone's pocket
    does not redeploy with the backend. An already-shipped client still sends
    `{videoId, difficulty}`, and an extra field it no longer means anything to
    read must be ignored rather than answered with a 422."""
    response = client.post("/v1/analyze",
                           json={"videoId": VIDEO, "difficulty": "easy"}, headers=AUTH)
    assert response.status_code in (200, 202), response.text


def test_the_old_difficulty_query_param_is_ignored(client):
    """Same argument on the read path: `GET /v1/maps/{id}?difficulty=hard` is a
    URL an old build constructs, and it addresses the video's one chart."""
    analyze(client)
    response = client.get(f"/v1/maps/{VIDEO}?difficulty=hard", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["song"]["id"] == f"yt:{VIDEO}"


def test_the_catalog_card_no_longer_advertises_a_difficulty(client):
    analyze(client)
    card = client.get("/v1/catalog", headers=AUTH).json()["results"][0]
    assert "difficulty" not in card
    assert card["songId"] == f"yt:{VIDEO}"


def test_a_pasted_url_works_as_well_as_an_id(client):
    """The app's entry point is "paste or pick a video", and a player pastes what
    the share sheet gave them."""
    for url in (f"https://www.youtube.com/watch?v={VIDEO}",
                f"https://youtu.be/{VIDEO}",
                f"https://www.youtube.com/shorts/{VIDEO}"):
        response = client.post("/v1/analyze", json={"url": url}, headers=AUTH)
        assert response.status_code in (200, 202), url


def test_something_that_is_not_a_video_is_rejected_clearly(client):
    # Note "not-a-video" would NOT do here: it is 11 characters of [A-Za-z0-9_-],
    # i.e. a structurally valid YouTube id. The validator checks shape, and only
    # the fetch can tell you a well-formed id doesn't exist.
    response = client.post("/v1/analyze", json={"videoId": "not a video!"}, headers=AUTH)
    assert response.status_code == 400
    assert response.json()["code"] == "bad_request"


def test_the_daily_quota_is_enforced_with_the_code_the_client_reads(settings, store):
    # Inline explicitly: the default is now a background thread (a job must not
    # run on the request thread), and this test asserts on the *second* request's
    # status, so a worker still writing job rows during teardown is only noise.
    app = create_app(replace(settings, daily_quota=1), store=store, source=FakeSource(),
                     runner=JobRunner(settings, store, FakeSource()))
    with TestClient(app) as client:
        client.post("/v1/analyze", json={"videoId": "aaaaaaaaaaa"}, headers=AUTH)
        response = client.post("/v1/analyze", json={"videoId": "bbbbbbbbbbb"}, headers=AUTH)
    assert response.status_code == 429
    assert response.json()["code"] == "quota_exhausted"


def test_the_kill_switch_returns_a_clean_unavailable(settings, store):
    """§3: "returns a clean 'feature unavailable' to clients. Must not require a
    deploy"."""
    app = create_app(replace(settings, analysis_enabled=False), store=store, source=FakeSource())
    with TestClient(app) as client:
        response = client.post("/v1/analyze", json={"videoId": VIDEO}, headers=AUTH)
    assert response.status_code == 503
    assert response.json()["code"] == "feature_disabled"


def test_a_deployment_with_no_audio_stack_serves_cache_but_refuses_new_work(settings, store):
    """The API container's normal state (§4). Cached maps still return; only a
    *new* analysis is refused."""
    app = create_app(settings, store=store, source=None)
    with TestClient(app) as client:
        response = client.post("/v1/analyze", json={"videoId": VIDEO}, headers=AUTH)
    assert response.status_code == 503
    assert response.json()["code"] == "feature_disabled"


def test_two_players_asking_at_once_join_one_job(settings, store):
    """Rather than decoding the same recording twice for an identical result."""
    source = FakeSource()
    app = create_app(settings, store=store, source=source, runner=JobRunner(settings, store, source))
    with TestClient(app) as client:
        store.create_job(job_id="inflight", uid="dev:local", video_id=VIDEO)
        response = client.post("/v1/analyze", json={"videoId": VIDEO}, headers=AUTH)
    assert response.status_code == 202
    assert response.json()["jobId"] == "inflight"
    assert source.decoded == []


def test_another_players_job_is_not_readable(client):
    """404 rather than 403: whether someone else's job exists is not this
    caller's business."""
    client.app.state.store.create_job(job_id="theirs", uid="someone-else",
                                      video_id=VIDEO)
    assert client.get("/v1/analyze/theirs", headers=AUTH).status_code == 404


# --- blocking (§3) ----------------------------------------------------------

def test_blocking_purges_the_cached_map_in_the_same_request(client):
    """A takedown must be satisfiable in minutes: block the ID, purge its cached
    map, done."""
    analyze(client)
    response = client.post("/v1/admin/block",
                           json={"videoId": VIDEO, "reason": "DMCA"}, headers=ADMIN)
    assert response.status_code == 200
    assert response.json()["purged"]["maps"] == 1

    assert client.get(f"/v1/maps/{VIDEO}", headers=AUTH).status_code == 404
    assert analyze(client).status_code == 403


def test_a_video_blocked_after_analysis_stops_being_served_immediately(client):
    """Even before its map is purged — the block is checked on every serve, which
    is what makes it the real guarantee."""
    analyze(client)
    client.app.state.store.block(BLOCK_VIDEO, VIDEO, reason="DMCA", actor="agent")
    assert analyze(client).status_code == 403
    assert client.get(f"/v1/maps/{VIDEO}", headers=AUTH).status_code == 403


def test_admin_routes_refuse_an_unauthenticated_caller(client):
    assert client.post("/v1/admin/block", json={"videoId": VIDEO}).status_code == 401
    assert client.delete(f"/v1/admin/maps/{VIDEO}").status_code == 401


def test_admin_routes_are_closed_when_unconfigured(settings, store):
    """Unconfigured ⇒ 503, never open."""
    app = create_app(replace(settings, admin_token=None), store=store, source=FakeSource())
    with TestClient(app) as client:
        assert client.post("/v1/admin/block", json={"videoId": VIDEO},
                           headers=ADMIN).status_code == 503


def test_the_audit_log_records_who_blocked_what(client):
    client.post("/v1/admin/block", json={"videoId": VIDEO, "reason": "DMCA"}, headers=ADMIN)
    entries = client.get("/v1/admin/audit", headers=ADMIN).json()["entries"]
    blocked = next(e for e in entries if e["action"] == "block")
    assert blocked["key"] == VIDEO
    assert blocked["actor"] == "agent@example.com"
    assert blocked["reason"] == "DMCA"


def test_a_purge_reports_what_it_actually_removed(client):
    analyze(client)
    body = client.delete(f"/v1/admin/maps/{VIDEO}", headers=ADMIN).json()
    assert body["purged"]["maps"] == 1


# --- offset (§6) ------------------------------------------------------------

def test_an_admin_offset_reaches_the_client_without_re_analysis(client):
    """The app has no latency calibration — it was deleted with the scoring it
    corrected — so this and the player's own nudge are the only two correction
    paths that exist."""
    analyze(client)
    client.post(f"/v1/admin/maps/{VIDEO}/offset", json={"offsetMs": -250}, headers=ADMIN)
    assert analyze(client).json()["videoSync"]["offsetMs"] == -250


# --- errors -----------------------------------------------------------------

def test_every_error_carries_a_human_readable_message(client):
    """The exact shape `RemoteMoRosettaService.errorMessage(from:)` parses."""
    responses = [
        client.post("/v1/analyze", json={"videoId": VIDEO}),                       # 401
        client.post("/v1/analyze", json={"videoId": "nope"}, headers=AUTH),        # 400
        client.get("/v1/analyze/missing", headers=AUTH),                           # 404
        client.post("/v1/admin/block", json={}, headers=ADMIN),                    # 400
    ]
    for response in responses:
        assert response.status_code >= 400
        body = response.json()
        assert isinstance(body.get("message"), str) and body["message"]


def test_a_malformed_body_still_gets_the_message_shape(client):
    response = client.post("/v1/analyze", json={"videoId": 12345}, headers=AUTH)
    assert response.status_code == 400
    assert response.json()["message"]


# --- catalog ----------------------------------------------------------------

def test_the_catalog_lists_what_has_been_analyzed(client):
    """Home's whole reason to exist: a player who has analyzed nothing still has
    something to play, because everyone else's analyses are cache hits."""
    analyze(client)

    response = client.get("/v1/catalog", headers=AUTH)
    assert response.status_code == 200
    body = response.json()

    assert [row["videoId"] for row in body["results"]] == [VIDEO]
    row = body["results"][0]
    # The row carries what a card needs without a second round trip: the chords
    # are the whole reason to pick one song over another.
    assert row["chords"]
    assert row["title"]
    assert row["songId"]
    assert body["version"]


def test_an_empty_catalog_is_an_empty_list_not_an_error(client):
    """Before anyone has analyzed anything. The shelf is absent, not broken."""
    response = client.get("/v1/catalog", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_a_blocked_video_never_appears_in_the_catalog(client):
    """§3's takedown has to hold on the listing as firmly as on the detail route
    — a blocked video still sitting on the home screen is a takedown that didn't
    happen."""
    analyze(client)
    client.app.state.store.block(BLOCK_VIDEO, VIDEO, reason="DMCA", actor="agent")

    assert client.get("/v1/catalog", headers=AUTH).json()["results"] == []


def test_a_video_analyzed_twice_is_one_song_in_the_catalog(client):
    """The catalog lists songs, not analyses: re-analyzing a video replaces its
    row."""
    analyze(client)
    analyze(client)

    results = client.get("/v1/catalog", headers=AUTH).json()["results"]
    assert [row["videoId"] for row in results] == [VIDEO]


def test_the_version_moves_when_a_song_is_added(client):
    """What the client polls so a song analyzed by anyone shows up for everyone
    without a relaunch."""
    before = client.get("/v1/catalog/version", headers=AUTH).json()["version"]
    analyze(client)
    after = client.get("/v1/catalog/version", headers=AUTH).json()["version"]

    assert before != after
    # And it agrees with the listing's own token, so one poll answers for both.
    assert after == client.get("/v1/catalog", headers=AUTH).json()["version"]


def test_the_catalog_pages(client):
    analyze(client)
    body = client.get("/v1/catalog?limit=1&offset=1", headers=AUTH).json()
    assert body["results"] == []


def test_the_catalog_is_readable_without_signing_in(client):
    """Home is the landing screen, and it must not be empty for the one person it
    exists for: someone who hasn't signed up and is deciding whether to. Every
    other route needs an identity because it spends quota or starts work; this
    one only reads rows that already exist."""
    analyze(client)

    anonymous = client.get("/v1/catalog")
    assert anonymous.status_code == 200
    assert [row["videoId"] for row in anonymous.json()["results"]] == [VIDEO]
    assert client.get("/v1/catalog/version").status_code == 200


def test_a_stale_token_still_gets_the_catalog(client):
    """An expired session shouldn't turn the home screen into an error about a
    session the player wasn't using."""
    analyze(client)
    response = client.get("/v1/catalog", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 200
    assert response.json()["results"]


# --- rate limits a browser can actually read --------------------------------

class _MultiUser:
    """An authenticator that maps `Bearer <name>` to uid `<name>`.

    The dev token resolves to one fixed uid, which is all any single-player test
    needs — and is exactly why the "two players, one video" defect below could not
    be seen: with one identity in the suite, `job.uid != principal.uid` is never
    true.
    """

    mode = "test-multi-user"

    def __call__(self, authorization, *, check_revoked: bool = False):
        from fastapi import HTTPException

        from app.auth import Principal

        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail={"message": "Sign in."})
        name = authorization[7:].strip()
        return Principal(uid=name, display_name=name, email_verified=True)


ORIGIN = {"Origin": "https://app.example.com"}


@pytest.fixture
def web_client(settings, store):
    """A deployment with CORS configured and a limiter of two per minute."""
    source = FakeSource()
    app = create_app(
        replace(settings, cors_allow_origins="https://app.example.com",
                rate_limit_ip_per_min=2),
        store=store, source=source, runner=JobRunner(settings, store, source))
    with TestClient(app) as client:
        yield client


def test_a_rate_limited_response_still_carries_the_cors_header(web_client):
    """Otherwise being throttled is indistinguishable from the server dying.

    `add_middleware` **prepends**, so the middleware installed last runs first and
    sits outermost. Installing the limiter after CORS therefore put it *outside*
    `CORSMiddleware`: its 429 left the app without ever passing through the CORS
    layer, so it carried no `Access-Control-Allow-Origin` and a browser could not
    read it at all. The `Retry-After` header and the `rate_limited` code were both
    written, and both unreachable from the web client — which saw an opaque CORS
    failure. Native clients ignore CORS entirely, which is why this survived.
    """
    codes = []
    for _ in range(4):
        response = web_client.get("/v1/catalog", headers={**AUTH, **ORIGIN})
        codes.append(response.status_code)
        assert response.headers.get("access-control-allow-origin") == \
            "https://app.example.com", (
            f"a {response.status_code} left the app without the CORS header, so the "
            f"browser sees a network error instead of the message and Retry-After "
            f"this service went to the trouble of setting"
        )
    assert 429 in codes, "the limiter must still actually limit"


def test_a_throttled_browser_can_read_the_retry_after(web_client):
    """The whole point of the header being reachable: backpressure the page can act
    on, rather than a failure it can only report."""
    for _ in range(3):
        response = web_client.get("/v1/catalog", headers={**AUTH, **ORIGIN})
    assert response.status_code == 429
    assert response.json()["code"] == "rate_limited"
    assert int(response.headers["Retry-After"]) >= 1


def test_a_cors_preflight_does_not_spend_the_limiter_budget(web_client):
    """A preflight is the browser's request, not the player's.

    The browser sends one before each non-simple request, so counting them **halves**
    a web client's real budget — and a preflight that 429s fails the request it was
    asking permission for with no status the page can see. It reaches no route, so it
    can neither spend quota nor start work.
    """
    for _ in range(10):
        preflight = web_client.options("/v1/analyze", headers={
            **ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        })
        assert preflight.status_code == 200, preflight.text

    # And the real budget is untouched by all that.
    assert web_client.get("/v1/catalog", headers={**AUTH, **ORIGIN}).status_code == 200


# --- two players, one video -------------------------------------------------

@pytest.fixture
def shared_client(settings, store):
    """Two identities, and a runner that leaves the job in flight."""
    class Queueing(JobRunner):
        def submit(self, **kwargs):
            pass          # accepted, never run — the job stays non-terminal

    source = FakeSource()
    app = create_app(settings, store=store, source=source,
                     authenticator=_MultiUser(), runner=Queueing(settings, store, source))
    with TestClient(app) as client:
        yield client


def test_the_second_player_can_poll_the_job_they_were_given(shared_client):
    """The reproduction.

    `active_job_for` deliberately joins a second caller onto one in-flight analysis
    rather than decoding the same recording twice. But the job row carries a single
    `uid`, and the poll route refused anyone else — so bob was handed alice's job id
    and then told, **forever**, that it had expired. Retrying returned the same dead
    id, so one player asking first made that video permanently un-analyzable for
    everybody else.
    """
    alice = {"Authorization": "Bearer alice"}
    bob = {"Authorization": "Bearer bob"}

    first = shared_client.post("/v1/analyze", json={"videoId": VIDEO}, headers=alice)
    second = shared_client.post("/v1/analyze", json={"videoId": VIDEO}, headers=bob)

    assert first.status_code == second.status_code == 202
    assert second.json()["jobId"] == first.json()["jobId"], \
        "the whole point of joining is that the second caller shares the first's job"

    job_id = second.json()["jobId"]
    poll = shared_client.get(f"/v1/analyze/{job_id}", headers=bob)
    assert poll.status_code == 200, poll.json()
    assert poll.json()["status"] in {"queued", "analyzing"}


def test_joining_a_running_job_costs_the_second_player_nothing(shared_client):
    """Consistent with a cache hit (§16.4): they did not start any work, so there
    is nothing to charge for."""
    shared_client.post("/v1/analyze", json={"videoId": VIDEO},
                       headers={"Authorization": "Bearer alice"})
    shared_client.post("/v1/analyze", json={"videoId": VIDEO},
                       headers={"Authorization": "Bearer bob"})

    assert shared_client.app.state.store.usage_today("bob") == 0
    assert shared_client.app.state.store.usage_today("alice") == 1


def test_a_stranger_still_cannot_poll_someone_elses_job(shared_client):
    """The fix must not have turned the id into the credential. You may poll a job
    because you *joined* it — not because you know its id."""
    job_id = shared_client.post("/v1/analyze", json={"videoId": VIDEO},
                                headers={"Authorization": "Bearer alice"}).json()["jobId"]

    response = shared_client.get(f"/v1/analyze/{job_id}",
                                 headers={"Authorization": "Bearer carol"})
    assert response.status_code == 404


# --- X-Forwarded-For --------------------------------------------------------

def test_the_trusted_hop_count_decides_which_forwarded_address_is_read(settings, store):
    """One hop is today's deployment; two is Cloudflare in front of a load balancer.

    Reading the rightmost entry is right behind exactly one trusted proxy. Behind
    two it is the *load balancer's* address — identical for every caller — so the
    whole internet collapses into one rate-limit bucket and one noisy client
    throttles every player at once.
    """
    from app.main import client_ip

    class _Request:
        def __init__(self, app, header):
            self.app = app
            self.headers = {"x-forwarded-for": header}
            self.client = None

    source = FakeSource()
    one_hop = create_app(settings, store=store, source=source,
                         runner=JobRunner(settings, store, source))
    two_hops = create_app(replace(settings, trusted_proxy_hops=2), store=store,
                          source=source, runner=JobRunner(settings, store, source))

    chain = "203.0.113.9, 198.51.100.7, 10.0.0.1"
    assert client_ip(_Request(one_hop, chain)) == "10.0.0.1"
    assert client_ip(_Request(two_hops, chain)) == "198.51.100.7"

    # Fewer entries than configured hops: attribute the leftmost rather than
    # walking off the front. A short header means the request did not come through
    # the chain we think it did.
    assert client_ip(_Request(two_hops, "203.0.113.9")) == "203.0.113.9"


def test_a_forged_forwarded_header_cannot_pick_its_own_bucket(settings, store):
    """The reason the rightmost entry is read at all: everything left of our own
    proxies is caller-supplied text, and a fresh fake address per request would
    start a fresh rate-limit window every time."""
    from app.main import client_ip

    class _Request:
        def __init__(self, app, header):
            self.app = app
            self.headers = {"x-forwarded-for": header}
            self.client = None

    source = FakeSource()
    app = create_app(settings, store=store, source=source,
                     runner=JobRunner(settings, store, source))
    assert client_ip(_Request(app, "1.1.1.1, 10.0.0.1")) == "10.0.0.1"
    assert client_ip(_Request(app, "2.2.2.2, 10.0.0.1")) == "10.0.0.1"


# --- a ready job whose map went away ----------------------------------------

def _drop_maps_only(store, video_id):
    """Delete a video's maps but leave its job rows.

    `purge` cascades to jobs, so it cannot produce the state under test — which is
    the state this branch is *for*: a job that reached `ready` and whose map is no
    longer there when the client polls. Reached by a takedown landing in that
    window, or by a cache invalidation, and defensive either way.
    """
    with store._cursor(write=True) as cur:
        cur.execute(store._sql("DELETE FROM chord_maps WHERE video_id = ?"), (video_id,))


def test_a_missing_map_is_only_called_blocked_when_it_is_blocked(client):
    """`video_blocked` renders in the app as "this video is banned".

    Answering it for *any* missing map accused the video of something whenever the
    map had merely been cleared. The two cases are distinguishable — the blocklist
    is right there — and they call for different copy.
    """
    job_id = analyze(client).json()["jobId"]
    _drop_maps_only(client.app.state.store, VIDEO)      # cleared, not blocked

    response = client.get(f"/v1/analyze/{job_id}", headers=AUTH)
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_a_blocked_video_still_says_blocked(client):
    """The other half: a takedown landing mid-flight is exactly what
    `video_blocked` is for, and must keep saying so."""
    job_id = analyze(client).json()["jobId"]
    store = client.app.state.store

    store.block(BLOCK_VIDEO, VIDEO, reason="DMCA", actor="agent")
    _drop_maps_only(store, VIDEO)

    response = client.get(f"/v1/analyze/{job_id}", headers=AUTH)
    assert response.status_code == 403
    assert response.json()["code"] == "video_blocked"


# --- healthz and the worker it cannot see -----------------------------------

def test_healthz_names_the_runner_the_capability_came_from(client):
    """`canAnalyze: true` beside `engines: {}` is not a contradiction — it is one
    page describing two containers. Saying which runner answered is what makes that
    readable instead of alarming."""
    body = client.get("/healthz").json()
    assert body["runner"] == "JobRunner"


def test_an_in_process_deployment_has_no_separate_worker_to_report(client):
    """`/v1/admin/worker` exists for the two-container shape. In-process, `/healthz`
    already describes the container that would run the analysis, and a second copy
    could only disagree with it."""
    response = client.get("/v1/admin/worker", headers=ADMIN)
    assert response.status_code == 503
    assert response.json()["code"] == "feature_disabled"


def test_the_worker_report_is_admin_only(client):
    """It costs a worker cold start, so it must not be reachable by anyone who
    fancies starting containers."""
    assert client.get("/v1/admin/worker").status_code == 401


def test_a_remote_runners_worker_report_is_served(settings, store):
    """What an operator actually gets on Modal: the worker image's own answer about
    what it built — the gap that made a worker missing an engine indistinguishable
    from a healthy deployment."""
    from app.jobs import RemoteJobRunner

    class Reporting(RemoteJobRunner):
        def submit(self, **kwargs):
            pass

        def worker_report(self):
            return {"engines": {"chords": ["btc"]}, "enginesReady": True}

    app = create_app(settings, store=store, source=None,
                     runner=Reporting(settings, store, None))
    with TestClient(app) as remote:
        body = remote.get("/v1/admin/worker", headers=ADMIN).json()
    assert body["enginesReady"] is True


# --- the user table is not a signup log -------------------------------------

class _Unverified:
    """A password account that has not proven its address — which anyone can mint
    through Firebase's public Auth REST API."""

    mode = "test-unverified"

    def __call__(self, authorization, *, check_revoked: bool = False):
        from fastapi import HTTPException

        from app.auth import PASSWORD_PROVIDER, Principal

        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail={"message": "Sign in."})
        return Principal(uid=authorization[7:].strip(), display_name=None,
                         email_verified=False, sign_in_provider=PASSWORD_PROVIDER)


def test_an_unverified_account_leaves_no_row_behind(settings, store):
    """`_charge` upserted the user row *before* the verified-email check, so every
    refused request from an unverified account still wrote one — an unbounded table
    any script could grow, populated entirely by accounts that may not spend
    anything. The check is cheap and comes first now."""
    source = FakeSource()
    app = create_app(settings, store=store, source=source,
                     authenticator=_Unverified(), runner=JobRunner(settings, store, source))
    with TestClient(app) as client:
        response = client.post("/v1/analyze", json={"videoId": VIDEO},
                               headers={"Authorization": "Bearer squatter"})

    assert response.status_code == 403
    assert response.json()["code"] == "email_unverified"
    assert store.display_name("squatter") is None
    with store._cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM users")
        assert cur.fetchone()[0] == 0


def test_a_verified_account_still_gets_its_row(settings, store):
    """The reordering must not have dropped the upsert that `/v1/me` and the quota
    depend on."""
    source = FakeSource()
    app = create_app(settings, store=store, source=source,
                     runner=JobRunner(settings, store, source))
    with TestClient(app) as client:
        assert client.post("/v1/analyze", json={"videoId": VIDEO},
                           headers=AUTH).status_code == 202

    with store._cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM users")
        assert cur.fetchone()[0] == 1
