"""§2.1 — audio is never persisted. "Enforce it in code, not in a comment."

The handoff asks for one test specifically (§8 step 4): *"Write the test asserting
scratch is empty after every job, including failure paths."* That is
`test_the_scratch_directory_never_survives_a_failure` below, and the rest of this
file guards the ways that assertion could be made vacuously true — a scratch root
that was durable all along, a cleanup that silently no-ops, a job that crashes
before the `with` block.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.scratch import ScratchError, assert_clean, check_root, scratch


def test_a_scratch_root_under_home_is_refused():
    """Misconfiguring the worker must fail loudly and immediately, not produce a
    container that works perfectly while accumulating other people's music."""
    with pytest.raises(ScratchError, match="under \\$HOME"):
        check_root(Path.home() / "chords-scratch")


def test_a_scratch_root_inside_the_working_tree_is_refused():
    """Refused whichever guard catches it first — on a checkout that lives under
    `$HOME` (the usual case) that is the `$HOME` rule, and the working-tree rule
    covers a checkout that doesn't."""
    with pytest.raises(ScratchError):
        check_root(Path.cwd() / "scratch")


def test_an_unrecognised_mount_is_refused():
    with pytest.raises(ScratchError, match="not a recognised ephemeral mount"):
        check_root("/srv/data/chords")


def test_a_relative_path_is_refused():
    with pytest.raises(ScratchError, match="absolute"):
        check_root("scratch")


def test_a_tmpfs_style_root_is_accepted(scratch_root):
    assert check_root(scratch_root)


def test_the_scratch_directory_exists_during_the_job_and_not_after(scratch_root):
    with scratch(scratch_root, label="video") as workdir:
        assert workdir.exists()
        (workdir / "audio.raw").write_bytes(b"\x00" * 4096)
        held = workdir
    assert not held.exists()
    assert_clean(scratch_root)


def test_the_scratch_directory_never_survives_a_failure(scratch_root):
    """§8 step 4's test. The failure path is the one that matters: a job that
    raises must not leave a decoded recording behind."""
    held = None
    with pytest.raises(RuntimeError, match="engine exploded"):
        with scratch(scratch_root, label="video") as workdir:
            held = workdir
            (workdir / "audio.raw").write_bytes(b"\x00" * 4096)
            raise RuntimeError("engine exploded")
    assert held is not None and not held.exists()
    assert_clean(scratch_root)


def test_concurrent_jobs_do_not_share_a_directory(scratch_root):
    with scratch(scratch_root, label="a") as first, scratch(scratch_root, label="b") as second:
        assert first != second
    assert_clean(scratch_root)


def test_assert_clean_notices_a_stray_directory(scratch_root):
    """The case a per-job cleanup cannot catch: a crash between `mkdir` and the
    `with`. Usable as a worker-shutdown check."""
    (Path(scratch_root) / "leftover").mkdir()
    with pytest.raises(ScratchError, match="not empty"):
        assert_clean(scratch_root)


def test_nothing_in_the_analysis_types_can_carry_audio():
    """§2.2 as a structural property rather than a promise: there is no field on
    any type crossing a stage boundary that could hold PCM, a spectrogram, a
    chroma matrix, or a path to a decoded file. Persisting audio would take
    *adding* one first."""
    import dataclasses

    from app.analysis import types

    forbidden = {"pcm", "audio", "samples", "path", "file", "chroma", "spectrogram", "waveform"}
    for name in ("RawChordSpan", "BeatGrid", "Onset", "GridSpan", "VideoMeta", "EngineInfo"):
        for field in dataclasses.fields(getattr(types, name)):
            assert field.name.lower() not in forbidden, f"{name}.{field.name}"
