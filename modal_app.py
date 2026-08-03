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

CHORDS_DEV_TOKEN must never be set here (CHORDS_REQUIRE_AUTH refuses to start
with it).

Check what actually built:  curl .../healthz
"""

from __future__ import annotations

import os

import modal

app = modal.App("chords-rosetta-gp")

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

BASE_PACKAGES = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.8",
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
    )
    # CPU torch explicitly. The default wheel carries the whole CUDA runtime —
    # some two gigabytes of image, per container pull, for hardware this
    # deployment does not have. Both models are small enough that a GPU would
    # mostly buy cold-start latency (§18), and the pipeline is already async.
    .pip_install("torch>=2.0,<3", index_url="https://download.pytorch.org/whl/cpu")
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
def analysis_worker(job_id: str, video_id: str, difficulty: str, uid: str) -> None:
    """One analysis, in its own container, with its own image and secret.

    Imports happen inside the function so the API image never has to satisfy
    them: `app.analysis.fetch` pulls in yt-dlp, which does not exist over there.
    """
    from app.config import load_settings
    from app.jobs import run_job
    from app.analysis.fetch import build_source
    from app.analysis.scratch import assert_clean
    from app.store import build_store

    settings = load_settings()
    store = build_store(settings)
    source = build_source(settings)

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
    from app.jobs import JobRunner
    from app.main import create_app

    class ModalJobRunner(JobRunner):
        """Hand the job to the isolated worker and return immediately.

        `.spawn()` rather than `.remote()`: the client is polling `GET
        /v1/analyze/{jobId}`, so the request that started the job must not wait
        for it. The worker writes every status transition to the job row, which
        is the only channel between the two containers — deliberately, since it
        means the API never has to hold a handle to a running worker.
        """

        def submit(self, *, job_id: str, video_id: str, difficulty: str, uid: str) -> None:
            analysis_worker.spawn(job_id=job_id, video_id=video_id,
                                  difficulty=difficulty, uid=uid)

    settings = load_settings()
    # `source=None` is correct and load-bearing here: this container cannot fetch
    # or decode, and `create_app` would otherwise try to build one.
    web_app = create_app(settings, source=None)
    web_app.state.runner = ModalJobRunner(settings, web_app.state.store, None)
    return web_app
