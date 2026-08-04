"""Modal deployment — and where §4's worker isolation actually comes from.

Two functions, **two images**, and that split is the whole security posture:

    fastapi_app   API image     no ffmpeg, no yt-dlp, no numpy, no engines.
                                Cannot fetch or decode audio even if asked to.
                                Serves cached maps, /v1/me, /healthz, admin.
                                  │  .spawn()
                                  ▼
    analysis_worker  worker image  ffmpeg + yt-dlp + the chosen engines.
                                   NO Volume mounted → nothing durable to write to.
                                   Its own secret. Hard timeout. Memory cap.

§4 asks for `--read-only` with an explicit tmpfs mount for scratch. Modal has no
`--read-only` flag, so the equivalent is stated plainly rather than assumed: a
Modal container's filesystem is **ephemeral and dies with the call**, and the only
way to write something that outlives it is to mount a Volume — so the worker
mounts none. Scratch goes to `/tmp`, which is container-local, and
`app/analysis/scratch.py` refuses to run anywhere that isn't an ephemeral mount.
The two together are the "can't write anywhere durable" guarantee; neither alone
is.

**Do not mount `data_volume` on the worker.** If a future change needs the worker
to persist something, that something is a chord map and it belongs in Postgres,
via the API — not on a disk the audio-handling code can reach.

Deploy:
    modal deploy modal_app.py

Secrets (Modal Secrets, edited in the dashboard — `modal secret create --force`
replaces the WHOLE secret and would drop keys):

- `chords-secrets` — the API's. FIREBASE_PROJECT_ID, FIREBASE_SERVICE_ACCOUNT_JSON,
  CHORDS_REQUIRE_AUTH=1, CHORDS_ADMIN_TOKEN, CHORDS_DATABASE_URL.
- `chords-worker-secrets` — the worker's. **Its own Firebase-free set**: the
  worker never authenticates anyone, so it gets no auth credentials at all. Only
  CHORDS_DATABASE_URL and the analysis knobs. §19.2's "do not let the
  chord-analysis service inherit Mo's deployment or Mo's blast radius", applied
  one level further in.

  It deliberately carries **no YouTube cookies**. They were supplied, measured
  against the bot check and removed: across ten containers on ten distinct Modal
  egress IPs, three ordinary uploads resolved on two of them — 20%, and byte-for
  -byte identical with and without cookies (6/30 in both arms). The bot check is
  per **IP reputation**, not per account, so a session credential buys nothing
  and a real one is worth not storing. `ytdlp_source` still *accepts*
  CHORDS_YTDLP_COOKIES_CONTENT for age-restricted or members-only video, and it
  would be worth revisiting if egress ever stops being a datacentre IP.

  Label-owned music is a second, separate wall: every official Beatles upload in
  `bench/corpus.json` is refused in every player client, and on the one occasion
  cookies did clear the bot check YouTube answered with storyboard images and no
  audio format at all — the PO-token/SABR path, which cookies also do not solve.

CHORDS_DEV_TOKEN must never be set here (CHORDS_REQUIRE_AUTH refuses to start
with it).

Check what actually built:  curl .../healthz
"""

from __future__ import annotations

import os

import modal

# Its OWN Modal app, not a function bolted onto Mo's `main` (§19.2). A deploy, a
# rollback, or a bad build here must not be able to take Mo down with it — the
# blast-radius argument that splits the two containers below applies just as much
# one level up. Modal app names are `[a-zA-Z0-9-_.]+`, so "Rosetta Dechorder"
# lands as this; the dashboard shows it at /apps/<workspace>/rosetta-dechorder.
app = modal.App("rosetta-dechorder")

# --- the single-writer pin --------------------------------------------------
#
# SQLite on a network volume tolerates exactly ONE writer. Postgres removes that
# constraint, but this file runs on *your machine* at `modal deploy` time, where
# it cannot see the secret the container will get. So the pin is lifted by an
# explicit signal in the deploy shell:
#
#     CHORDS_DATABASE_URL="$(the DSN in chords-secrets)" modal deploy modal_app.py
#
# Unset ⇒ the pin stays. Fail-safe direction: a deployment still on SQLite keeps
# its single writer, and the cost of forgetting the variable is "correct but not
# scaled out" rather than "silently losing writes". (Same reasoning, and the same
# shape, as Mo's `modal_app.py`.)
_ON_POSTGRES = bool(os.environ.get("CHORDS_DATABASE_URL"))
MAX_CONTAINERS = None if _ON_POSTGRES else 1

# MUST stay in step with `[project].dependencies` in pyproject.toml. This list is
# a duplicate of it — Modal images are built from an explicit package list, not
# from the local project — and `tests/test_deployment.py` asserts the two agree,
# because the one time they drifted it took the whole API down.
#
# The failure mode is worth stating: a dependency added to pyproject and not here
# is present in every test run and absent from the deployed image, so the suite
# is green, the deploy *succeeds*, and the container then dies at import. That is
# not a degraded service — `create_app()` runs at module scope, so the ASGI app
# never builds and every route 503s, including `/healthz`.
BASE_PACKAGES = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.8",
    # Required at *import* time by `POST /v1/analyze/upload`: FastAPI raises
    # while registering any route that declares File()/Form(), not when one is
    # first called. Omitting it does not disable the upload path, it unbuilds
    # the application.
    "python-multipart>=0.0.9",
    "httpx>=0.27",
    "firebase-admin>=6.5",
    "psycopg[binary,pool]>=3.2",
]

# The API image. Deliberately does NOT contain ffmpeg, yt-dlp, numpy or any
# engine: the container that faces the internet has no ability to fetch or decode
# a recording, so a bug or a compromise there cannot become an audio problem.
api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*BASE_PACKAGES)
    .add_local_python_source("app")
)

# The worker image. Everything that touches audio lives here and nowhere else.
#
# The engines are the §8-step-2 benchmark's picks (README has the table): BTC for
# chords, Beat This! for beats, librosa for onsets. Anything added here must NOT
# be added to the API image (§4).
BTC_ROOT = "/opt/BTC-ISMIR19"

# Both engines are pinned to a commit rather than to a branch. Neither is a
# packaged release — one is a zip of a git ref, the other a clone — so "latest"
# would mean the image quietly changes between two deploys of identical code.
# For BTC it is stronger than housekeeping: the pretrained weights live in the
# repo and have to agree with the model code that loads them.
BTC_COMMIT = "2682317be668032e6e4b269ded36adaa2ad57df0"
BEAT_THIS_COMMIT = "b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c"

worker_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        *BASE_PACKAGES,
        "yt-dlp>=2024.8",
        "numpy>=1.26,<2.1",
        "soundfile>=0.12",
        "librosa>=0.10",
        "pyyaml>=6",
        # BTC's own `utils/mir_eval_modules` imports this, and that module is
        # where `idx2voca_chord` — the 170-label vocabulary — comes from. It is
        # a dependency of the *checkout*, so nothing in this repo's own imports
        # reveals it, and a machine that happens to have it installed (a bench
        # run will) analyzes perfectly while the image fails every job.
        "mir_eval>=0.7",
    )
    # CPU torch explicitly. The default wheel carries the whole CUDA runtime —
    # some two gigabytes of image, per container pull, for hardware this
    # deployment does not have. Both models are small enough that a GPU would
    # mostly buy cold-start latency (§18), and the pipeline is already async.
    #
    # **torchaudio belongs in THIS call, not the next one**, and that is what the
    # first build of this image got wrong. `beat_this` requires torchaudio and
    # does not pin it, so leaving it to be resolved below pulled the *default*
    # PyPI wheel — the CUDA build — which links `libcudart.so.13` and cannot be
    # imported on a CPU image at all. The failure surfaces at the checkpoint bake
    # further down (`OSError: libcudart.so.13`), several layers away from the
    # line that caused it. Naming it here resolves torch and torchaudio together,
    # from one index, as the matched CPU pair they have to be — and because pip
    # then finds the requirement already satisfied, the install below leaves it
    # alone. A local run never sees this: macOS wheels have no CUDA variant.
    .pip_install("torch>=2.0,<3", "torchaudio",
                 index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "einops", "soxr", "rotary-embedding-torch",
        f"https://github.com/CPJKU/beat_this/archive/{BEAT_THIS_COMMIT}.zip",
    )
    # BTC is research code with no package and no PyPI release, so it is cloned
    # rather than installed. Pinned to a commit: the weights and the model code
    # have to agree, and "whatever main was on deploy day" is not a version.
    .run_commands(
        f"git clone https://github.com/jayg996/BTC-ISMIR19 {BTC_ROOT}",
        f"git -C {BTC_ROOT} checkout --quiet {BTC_COMMIT}",
        # The weights ship in the repo, so a checkout that silently lacks them
        # would produce an image whose first job fails. Fail the build instead.
        f"test -f {BTC_ROOT}/test/btc_model_large_voca.pt",
        # And the stronger version of the same idea: `test -f` proves the files
        # arrived, not that they *import*. `utils.mir_eval_modules` pulls in
        # `mir_eval`, and `btc_model` needs the NumPy aliases 1.24 removed —
        # neither is visible from a file check, and both fail identically at run
        # time: every job dying in the chord stage behind a green health check,
        # which is the one shape of failure this deployment cannot see, because
        # the fetch stage's bot check stops jobs before they ever get here.
        # Assert the vocabulary size too — a checkout whose model code and
        # weights disagree is exactly what pinning BTC_COMMIT exists to prevent.
        # The alias restoration must be **guarded by hasattr**, exactly as
        # `adapters/btc.py::_restore_numpy_aliases` does it. NumPy 2.0 brought
        # `np.bool` back as a real scalar type, so assigning it unconditionally
        # does not restore a removed alias — it overwrites a live one, and numpy
        # then fails deep inside itself (`'bool' object has no attribute
        # 'view'`). Only `np.float` is genuinely absent on 2.x.
        'python -c "'
        "import sys; sys.path.insert(0, '" + BTC_ROOT + "'); "
        "import numpy as np; "
        "[setattr(np, a, b) for a, b in "
        "(('float', float), ('int', int), ('bool', bool), ('object', object)) "
        "if not hasattr(np, a)]; "
        "from utils.mir_eval_modules import idx2voca_chord; "
        "from btc_model import BTC_model; "
        "assert len(idx2voca_chord()) == 170, len(idx2voca_chord()); "
        "print('BTC imports OK')"
        '"',
    )
    # Bake Beat This!'s checkpoint into the image. Left to run time it is
    # downloaded on the first request of every cold container — which turns a
    # cold start into a dependency on someone else's file server, inside a job
    # that already has a 300 s timeout.
    .run_commands(
        "python -c \"from beat_this.inference import load_model; load_model('final0', device='cpu')\""
    )
    .env({
        "CHORDS_SCRATCH_ROOT": "/tmp/chords-scratch",
        "CHORDS_BTC_ROOT": BTC_ROOT,
    })
    .add_local_python_source("app")
)

# Quota/blocklist/map state. Mounted on the API only — see the module docstring.
data_volume = modal.Volume.from_name("chords-data", create_if_missing=True)

api_secrets = [modal.Secret.from_name("chords-secrets")]
worker_secrets = [modal.Secret.from_name("chords-worker-secrets")]


@app.function(
    image=worker_image,
    secrets=worker_secrets,
    # NO volumes. This is the isolation, not an omission — see the docstring.
    timeout=300,
    memory=4096,
    # A DSP job is CPU-bound and holds a decoded track in memory; one at a time
    # per container keeps the memory cap meaningful.
    max_containers=8,
    retries=0,          # a failed analysis is reported, never silently retried
)
def analysis_worker(job_id: str, video_id: str, difficulty: str, uid: str,
                    audio: bytes | None = None, filename: str | None = None) -> None:
    """One analysis, in its own container, with its own image and secret.

    Imports happen inside the function so the API image never has to satisfy
    them: `app.analysis.fetch` pulls in yt-dlp, which does not exist over there.

    `audio` carries an upload's bytes (`POST /v1/analyze/upload`). They arrive as
    a function argument rather than through storage because §2.1 leaves nowhere
    to put them — the worker mounts no Volume, and a recording parked anywhere
    durable is the invariant broken. They exist in this container's memory, get
    written only into the scratch directory, and die with the container.
    """
    from app.config import load_settings
    from app.jobs import run_job
    from app.analysis.fetch import build_source
    from app.analysis.file_source import FileSource
    from app.analysis.scratch import assert_clean
    from app.store import build_store

    settings = load_settings()
    store = build_store(settings)
    source = (FileSource(audio, filename=filename, settings=settings)
              if audio is not None else build_source(settings))

    try:
        run_job(job_id=job_id, video_id=video_id, difficulty=difficulty, uid=uid,
                settings=settings, store=store, source=source)
    finally:
        # The check a per-job cleanup can't do: catches a crash between mkdir and
        # the `with`. The container is about to die and take its filesystem with
        # it, so this is belt-and-braces — but it is how you find out that the
        # cleanup path stopped working, rather than assuming it still does.
        assert_clean(settings.scratch_root)


@app.function(
    image=api_image,
    secrets=api_secrets,
    volumes={"/data": data_volume},
    timeout=120,
    max_containers=MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app():
    from app.config import load_settings
    from app.jobs import RemoteJobRunner
    from app.main import create_app
    from app.store import build_store

    class ModalJobRunner(RemoteJobRunner):
        """Hand the job to the isolated worker and return immediately.

        `.spawn()` rather than `.remote()`: the client is polling `GET
        /v1/analyze/{jobId}`, so the request that started the job must not wait
        for it. The worker writes every status transition to the job row, which
        is the only channel between the two containers — deliberately, since it
        means the API never has to hold a handle to a running worker.

        `RemoteJobRunner` rather than `JobRunner` is what makes this container
        answer "yes, an analysis can run" despite having no audio stack of its
        own. Subclassing the inline runner instead published the API image's own
        (correctly empty) capabilities as the service's, and every uncached
        `POST /v1/analyze` answered 503 on a deployment that was working.
        """

        def submit(self, *, job_id: str, video_id: str, difficulty: str, uid: str,
                   audio: bytes | None = None, filename: str | None = None) -> None:
            analysis_worker.spawn(job_id=job_id, video_id=video_id,
                                  difficulty=difficulty, uid=uid,
                                  audio=audio, filename=filename)

        def can_accept_uploads(self) -> bool:
            """True for the same reason `can_analyze` is: the *worker* image has
            ffmpeg and the engines, and this container's job is to delegate, not
            to decode."""
            return True

    settings = load_settings()
    store = build_store(settings)
    # `source=None` is correct and load-bearing here: this container cannot fetch
    # or decode, and `create_app` would otherwise try to build one. The runner is
    # passed in rather than swapped in afterwards so `create_app` never builds
    # the default `ThreadJobRunner` — which would start a thread pool this
    # container must never use, since running an analysis in the API process is
    # precisely what §4's isolation exists to prevent.
    return create_app(settings, store=store, runner=ModalJobRunner(settings, store, None),
                      source=None)
