"""Machine-readable error codes, and the one exception the routes raise.

The wire contract is Mo's (handoff §16.3): **every error is
``{"message": "<human-readable>"}``**, so the iOS client's existing
``MoRosettaError`` mapping renders this service's failures with no new copy —
401 → sign-in CTA, 403 → email unverified, 429 → quota, other non-2xx → the
server's own message, ``URLError`` → offline.

``code`` rides *alongside* ``message``, never instead of it. A client that
ignores it behaves exactly as it does against Mo today; a client that reads it
can tell the §16.3 cases apart without matching on prose.
"""

from __future__ import annotations

from fastapi import HTTPException

# --- Inherited from Mo (same strings, so one client mapping covers both) -----
CODE_EMAIL_UNVERIFIED = "email_unverified"
CODE_QUOTA_EXHAUSTED = "quota_exhausted"
CODE_RATE_LIMITED = "rate_limited"
CODE_INSUFFICIENT_CREDITS = "insufficient_credits"

# --- This service's own (handoff §16.3's list, verbatim) ---------------------
# "Give each a stable machine-readable code alongside message."
CODE_VIDEO_BLOCKED = "video_blocked"
CODE_VIDEO_UNAVAILABLE = "video_unavailable"
CODE_VIDEO_TOO_LONG = "video_too_long"
CODE_ANALYSIS_FAILED = "analysis_failed"
CODE_LOW_CONFIDENCE = "low_confidence"
CODE_FEATURE_DISABLED = "feature_disabled"

# Not in §16.3's list but needed by the same UI: the caller asked for a video id
# that isn't one, or a job id that doesn't exist.
CODE_BAD_REQUEST = "bad_request"
CODE_NOT_FOUND = "not_found"

# Also not in §16.3: a narrowing of `analysis_failed`, for the one analysis
# failure the pipeline can actually diagnose. A client that doesn't know it
# renders `message` and behaves exactly as it does today.
CODE_TEMPO_UNREADABLE = "tempo_unreadable"


def fail(status: int, message: str, code: str | None = None, **kwargs) -> HTTPException:
    """Build the {"message", "code"} HTTPException the error handler reshapes.

    A function rather than a bare `raise` at each site so the detail dict can
    never be spelled a second way — the client parses `message` by key.
    """
    detail: dict[str, str] = {"message": message}
    if code:
        detail["code"] = code
    return HTTPException(status_code=status, detail=detail, **kwargs)


# ---------------------------------------------------------------------------
# Analysis failures
# ---------------------------------------------------------------------------

class AnalysisError(Exception):
    """A job failed for a reason worth telling the player about.

    Carries the §16.3 `code` and a message already written in the app's voice,
    so the failure survives the trip through the job row and comes back out of
    `GET /v1/analyze/{jobId}` looking like every other error this service emits.
    """

    def __init__(self, message: str, code: str = CODE_ANALYSIS_FAILED, *, status: int = 422):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


class VideoUnavailable(AnalysisError):
    def __init__(self, message: str = "That video can’t be reached — it may be private, removed, or region-locked."):
        super().__init__(message, CODE_VIDEO_UNAVAILABLE, status=422)


class EgressBlocked(VideoUnavailable):
    """YouTube's bot check refused *this container's IP* — the video is fine.

    A subclass of `VideoUnavailable` on purpose: to the player it is the same
    calm "can't have this one" outcome, with the same code, status, message and
    refund, so nothing on the wire needs to learn that this exists. What it adds
    is a distinction the **worker** can act on, because the two failures want
    opposite responses — a private video fails identically forever, while this
    one is a property of the egress IP, and a fresh container is a fresh draw.

    Measured 2026-08-04, six Modal containers × eight yt-dlp player clients: one
    IP served every client, five refused every client. The check is per-IP, not
    per-client — so retrying *elsewhere* is the fix and retrying *here* is not.
    """

    def __init__(self) -> None:
        super().__init__()


class TempoUnreadable(AnalysisError):
    """The beat tracker's tempo is one the container cannot carry.

    Almost always an octave error — the tracker counted the eighths, or the
    half-notes — and the analysis knows that (`meter.tempo_octave_suspect`). It
    used to arrive as a `LintFailure`, i.e. as "that video didn't produce a song
    we could play", which is both true and useless: it names none of what the
    pipeline had already diagnosed, and it reads like a fault in the video rather
    than in the reading of it. §13.3 is "degrade honestly", so it says what it
    saw.
    """

    def __init__(self, tempo: int, low: int, high: int):
        super().__init__(
            f"That track’s beat came out at {tempo} BPM, which is outside the "
            f"{low}–{high} BPM a song can be played at — usually it means the beat was "
            f"counted at double or half speed. Try a video with a steadier beat.",
            CODE_TEMPO_UNREADABLE,
            status=422,
        )
        self.tempo = tempo


class VideoTooLong(AnalysisError):
    def __init__(self, limit_seconds: int):
        minutes = limit_seconds // 60
        super().__init__(
            f"That video is longer than {minutes} minutes — try a single song.",
            CODE_VIDEO_TOO_LONG,
            status=422,
        )


class VideoBlocked(AnalysisError):
    def __init__(self, message: str = "That video isn’t available for chord analysis."):
        super().__init__(message, CODE_VIDEO_BLOCKED, status=403)


class FeatureDisabled(AnalysisError):
    def __init__(self, message: str = "Chord analysis is unavailable right now."):
        super().__init__(message, CODE_FEATURE_DISABLED, status=503)
