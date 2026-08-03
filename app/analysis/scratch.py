"""§2.1 in code: **audio is never persisted.**

    "Fetch → decode → analyze → destroy, inside one worker process. No disk
    writes outside tmpfs, no object storage, no cache of decoded PCM, no
    'temporary' debug dumps that survive the request. **Enforce it in code, not
    in a comment.**"

So this module is the enforcement, and it has three parts, because a `finally:
rmtree` alone is the version of this that quietly stops being true:

1. **`scratch` refuses to run somewhere durable.** A scratch root inside the
   repo, under `$HOME`, or on any path that isn't a recognised ephemeral mount
   raises at the moment of use — before anything is fetched. Misconfiguring the
   worker should fail loudly and immediately, not produce a container that works
   perfectly while accumulating other people's music on a volume.
2. **Cleanup runs on every exit path**, including a raised exception, a timeout
   killing the job, and the success path.
3. **`assert_clean` verifies it worked** rather than assuming. The handoff asks
   for the test that asserts the scratch dir is empty after every job "including
   failure paths"; `verify_on_exit` makes the *runtime* do the same check the
   test does, so a cleanup that silently fails in production is loud.

None of this substitutes for the container-level guarantee — §4's `--read-only`
root filesystem with an explicit tmpfs mount for scratch, which is what makes
"can't write anywhere durable" true even for code that never calls this module.
This is the belt; that is the braces.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("chords.scratch")


class ScratchError(RuntimeError):
    """The scratch root is unusable, or refused to empty. Always fatal to a job."""


# Prefixes we accept as ephemeral. `/dev/shm` and `/tmp` are tmpfs on a normal
# Linux container; the rest are macOS's, for local runs — note `/tmp` and
# `/var/folders` there **resolve** to `/private/...`, and this check runs against
# the resolved path, so both spellings have to be listed.
# Anything else has to be opted into explicitly via CHORDS_SCRATCH_ALLOW_PREFIX,
# so a new deployment target is a deliberate decision rather than a surprise.
_EPHEMERAL_PREFIXES = (
    "/tmp", "/dev/shm", "/var/tmp", "/var/folders",
    "/private/tmp", "/private/var/tmp", "/private/var/folders",
)

_ALLOW_ENV = "CHORDS_SCRATCH_ALLOW_PREFIX"


def _allowed_prefixes() -> tuple[str, ...]:
    extra = os.environ.get(_ALLOW_ENV, "")
    return _EPHEMERAL_PREFIXES + tuple(p for p in (x.strip() for x in extra.split(":")) if p)


def check_root(root: str | Path) -> Path:
    """Validate a scratch root, or raise. Called before every job, not at import.

    Deliberately re-checked per job rather than cached: the check is
    microseconds, and a cached answer would survive a config change that made it
    wrong.
    """
    path = Path(root)
    if not path.is_absolute():
        raise ScratchError(f"scratch root must be an absolute path (got {root!r})")

    resolved = str(path.resolve()) if path.exists() else str(path)
    home = str(Path.home().resolve())
    cwd = str(Path.cwd().resolve())

    if resolved == home or resolved.startswith(home + os.sep):
        raise ScratchError(
            f"scratch root {resolved} is under $HOME — decoded audio must never land "
            f"on a durable filesystem (§2.1). Point CHORDS_SCRATCH_ROOT at a tmpfs mount."
        )
    if resolved == cwd or resolved.startswith(cwd + os.sep):
        raise ScratchError(
            f"scratch root {resolved} is inside the working tree — decoded audio must "
            f"never land in the repo (§2.1)."
        )
    if not any(resolved == p or resolved.startswith(p + os.sep) for p in _allowed_prefixes()):
        raise ScratchError(
            f"scratch root {resolved} is not a recognised ephemeral mount. Mount a tmpfs "
            f"there, or add its prefix to {_ALLOW_ENV} if you are certain it is ephemeral."
        )
    return path


@contextmanager
def scratch(root: str | Path, *, label: str = "job", verify_on_exit: bool = True):
    """A unique scratch directory that **cannot** outlive this block.

    Usage is the whole contract::

        with scratch(settings.scratch_root, label=video_id) as workdir:
            audio = decode(fetch(video_id), into=workdir)
            result = analyze(audio)
        # workdir no longer exists — checked, not assumed

    The directory is removed on every exit path. With `verify_on_exit` (the
    default) its absence is then asserted, and a failure to remove it raises —
    because a scratch directory that survives a job is not a hygiene problem to
    log and move past, it is the invariant being broken.
    """
    base = check_root(root)
    base.mkdir(parents=True, exist_ok=True)
    workdir = base / f"{label}-{uuid.uuid4().hex[:12]}"
    workdir.mkdir(parents=True, exist_ok=False)
    try:
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if workdir.exists():
            # Last resort before raising: something is holding a file open, or
            # the mount went read-only mid-job.
            shutil.rmtree(workdir, ignore_errors=False)
        if verify_on_exit and workdir.exists():
            raise ScratchError(f"scratch directory survived the job: {workdir}")
        log.debug("scratch cleaned: %s", workdir)


def assert_clean(root: str | Path) -> None:
    """Raise unless the scratch root holds nothing.

    The runtime hook for §8 step 4's test ("assert scratch is empty after every
    job, including failure paths") — and usable as a worker-shutdown check, which
    is where it catches the case a per-job cleanup can't: a crash between mkdir
    and the `with`.
    """
    path = Path(root)
    if not path.exists():
        return
    leftovers = sorted(p.name for p in path.iterdir())
    if leftovers:
        raise ScratchError(f"scratch root {path} is not empty: {leftovers}")
