"""Continuing an interrupted sampling run.

**No teacher, no GPU.** The calculator here is Lennard-Jones and the base frames are
small synthetic cells, so the whole file runs on a login node in seconds. That is not
only about speed. The real teacher is a compiled model on a GPU, and two runs of it
over identical geometry disagree by a few times 1e-6 eV -- floating-point reductions
are not associative and the order is not fixed. So with the real teacher, "the resumed
dataset is the dataset an uninterrupted run would have produced" can only be checked
to a tolerance. Lennard-Jones on the CPU gives the same number every time, which
turns that claim into a comparison of file hashes: strictly stronger, and it fails
loudly if a single structure is dropped, duplicated, or put in the wrong file.

What is under test is bookkeeping -- what is on disk, what settings produced it, what
the config says now -- and none of that depends on where the labels came from.
"""

import hashlib

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.io import read, write

from nequip_extension_template.sample import MDSampler, RattleSampler
from nequip_extension_template.sample.sampler import STATE_FILE

SPLITS = ("train", "val", "test")

RATTLE_SETTINGS = dict(
    strain_magnitudes=[-0.05, 0.0, 0.05],
    n_random_strain_samples=2,
    anisotropic_strain_magnitude=0.05,
    max_displacement_ang=0.25,
    seed=1,
    split={"train": 0.8, "val": 0.1, "test": 0.1},
    split_policy="scattered",
    split_seed=0,
)

MD_SETTINGS = dict(
    temperature_K=300.0,
    timestep_fs=1.0,
    friction_per_fs=0.01,
    n_equilibrate_steps=5,
    sample_interval=5,
    n_samples=20,
    seed=1,
    split={"train": 0.8, "val": 0.1, "test": 0.1},
    split_policy="blocked",
    split_seed=0,
)


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    """Twelve distinct 8-atom periodic cells, written out at three file lengths.

    Distinct on purpose. A base frame is identified by its contents, so two identical
    frames are one frame as far as the sampler is concerned, and a test built on a
    duplicate would pass without proving anything.
    """
    directory = tmp_path_factory.mktemp("frames")
    generator = np.random.default_rng(0)
    cells = [
        Atoms(
            "Ar8",
            positions=generator.uniform(0, 5, (8, 3)),
            cell=np.eye(3) * 5.0,
            pbc=True,
        )
        for _ in range(12)
    ]
    paths = {}
    for count in (10, 11, 12):
        paths[count] = directory / f"base{count}.xyz"
        write(str(paths[count]), cells[:count])
    return paths


def build(path, base_frames, sampler_class=RattleSampler, defaults=None, **overrides):
    """A sampler plus the config record it would have been built from.

    Mirrors what `nequip-distill` does: the same settings both construct the sampler
    and are handed to it as the config, which it stores and later compares against.
    Tests that change a setting change it in one place and get both.
    """
    settings = dict(RATTLE_SETTINGS if defaults is None else defaults)
    settings.update(overrides)
    sampler = sampler_class(
        calculator=LennardJones(),
        base_frames=str(base_frames),
        sample_path=str(path),
        **settings,
    )
    sampler.sampler_config = {
        "_target_": f"{sampler_class.__module__}.{sampler_class.__qualname__}",
        "calculator": {"_target_": "ase.calculators.lj.LennardJones"},
        "base_frames": str(base_frames),
        **settings,
    }
    return sampler


def md(path, base_frames, **overrides):
    return build(
        path,
        base_frames,
        sampler_class=MDSampler,
        defaults=MD_SETTINGS,
        **overrides,
    )


def digests(path):
    """A hash per split file. The whole dataset, in three numbers."""
    return {
        s: (
            hashlib.sha256((path / f"{s}.extxyz").read_bytes()).hexdigest()
            if (path / f"{s}.extxyz").exists()
            else None
        )
        for s in SPLITS
    }


def counts(path):
    return {
        s: len(read(str(path / f"{s}.extxyz"), index=":"))
        if (path / f"{s}.extxyz").exists()
        else 0
        for s in SPLITS
    }


def kill_after(sampler, n_structures):
    """Run until ``n_structures`` have been appended, then die, as a killed job would.

    A precise structure count rather than a wall-clock timeout, so a test kills at a
    known point instead of wherever the clock happened to fall.
    """
    real_step, appended = sampler.step, [0]

    def step():
        if appended[0] >= n_structures:
            raise KeyboardInterrupt("simulated kill")
        appended[0] += 1
        real_step()

    sampler.step = step
    with pytest.raises(KeyboardInterrupt):
        sampler.generate()


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def reference(tmp_path, frames):
    """A complete dataset -- 10 base frames x 5 variants -- produced in one run."""
    path = tmp_path / "reference"
    assert build(path, frames[10]).generate() == 50
    return path


@pytest.fixture
def md_reference(tmp_path, frames):
    path = tmp_path / "md_reference"
    assert md(path, frames[10]).generate() == 20
    return path


# ----------------------------------------------------------------- the record itself


def test_the_record_says_what_is_actually_on_disk(reference):
    import torch

    state = torch.load(reference / STATE_FILE, weights_only=False)
    progress = state["progress"]
    assert progress["n_written"] == 50
    assert progress["split_counts"] == {"train": 40, "val": 5, "test": 5}
    assert sum(progress["split_counts"].values()) == progress["n_written"]
    # the point of the offsets: each is that file's length, so a resume can cut back
    for split, offset in progress["offsets"].items():
        assert offset == (reference / f"{split}.extxyz").stat().st_size
    assert not (reference / "sampler_state.pt.tmp").exists()


def test_the_record_stores_the_config_and_the_base_frame_contents(reference, frames):
    import torch

    goal = torch.load(reference / STATE_FILE, weights_only=False)["goal"]
    assert goal["config"]["seed"] == 1
    assert goal["config"]["base_frames"] == str(frames[10])
    assert isinstance(goal["base_frames"], str) and goal["base_frames"]


# ------------------------------------------------------------------------- resuming


def test_a_fresh_run_is_complete_and_split_as_asked(reference):
    assert counts(reference) == {"train": 40, "val": 5, "test": 5}


def test_resuming_reproduces_an_uninterrupted_run(tmp_path, frames, reference):
    path = tmp_path / "killed"
    kill_after(build(path, frames[10]), 17)
    assert sum(counts(path).values()) == 17

    sampler = build(path, frames[10])
    assert sampler.generate() == 50
    assert sampler.n_resumed == 17
    assert digests(path) == digests(reference)


def test_a_structure_appended_after_the_last_record_write_is_truncated(
    tmp_path, frames, reference
):
    """An append and a record write are two operations, so a kill can land between.

    With the record written every fifth structure and the kill on the thirteenth, the
    record knows about ten. The three beyond it must be cut off and produced again,
    not appended to -- otherwise the dataset gains duplicates.
    """
    path = tmp_path / "torn"
    kill_after(build(path, frames[10], state_interval=5), 13)
    assert sum(counts(path).values()) == 13

    import torch

    offsets = torch.load(path / STATE_FILE, weights_only=False)["progress"]["offsets"]
    excess = sum(
        (path / f"{s}.extxyz").stat().st_size - offsets[s]
        for s in SPLITS
        if (path / f"{s}.extxyz").exists()
    )
    assert excess > 0, "nothing to truncate -- this test is not exercising the path"

    assert build(path, frames[10], state_interval=5).generate() == 50
    assert digests(path) == digests(reference)


def test_rerunning_a_finished_dataset_appends_nothing(reference, frames):
    before = digests(reference)
    sampler = build(reference, frames[10])
    assert sampler.generate() == 50
    assert sampler.n_resumed == 50
    assert digests(reference) == before


# ------------------------------------------------------------------------- refusals


def test_a_changed_setting_is_refused_and_named(reference, frames):
    with pytest.raises(ValueError, match="produced under different settings"):
        build(reference, frames[10], seed=99).generate()
    with pytest.raises(ValueError, match=r"seed: 1 -> 99"):
        build(reference, frames[10], seed=99).generate()


def test_even_a_setting_that_cannot_change_the_structures_is_refused(reference, frames):
    """Current behaviour, deliberately: no setting is exempt yet.

    ``state_interval`` only governs how often the record is written. Refusing it is
    over-strict, and the classification step will exempt it -- but until that
    classification exists, exempting things one at a time is how a real difference
    gets waved through by accident.
    """
    with pytest.raises(ValueError, match="produced under different settings"):
        build(reference, frames[10], state_interval=5).generate()


def test_editing_the_base_frames_file_is_refused_even_though_its_path_is_the_same(
    tmp_path, frames
):
    """The config names the base frames by path. A path is not its contents."""
    path = tmp_path / "edited"
    copied = tmp_path / "frames.xyz"
    write(str(copied), read(str(frames[10]), index=":"))
    assert build(path, copied).generate() == 50

    write(str(copied), read(str(frames[11]), index=":"))
    with pytest.raises(ValueError, match="base frame contents"):
        build(path, copied).generate()


def test_split_files_with_no_record_are_refused(tmp_path, frames):
    """No record means no way to know what those structures are, or what made them."""
    path = tmp_path / "orphaned"
    path.mkdir()
    (path / "train.extxyz").write_text("")
    with pytest.raises(FileExistsError, match=STATE_FILE):
        build(path, frames[10]).generate()


def test_a_dataset_belongs_to_one_procedure(reference, frames):
    with pytest.raises(ValueError, match="one procedure"):
        md(reference, frames[10]).generate()


def test_a_record_from_an_unknown_format_is_refused(reference, frames):
    import torch

    state = torch.load(reference / STATE_FILE, weights_only=False)
    state["version"] = 99
    torch.save(state, reference / STATE_FILE)
    with pytest.raises(ValueError, match="record format"):
        build(reference, frames[10]).generate()


def test_a_split_file_shorter_than_the_record_is_refused(reference, frames):
    """Truncation cannot fix this: the record describes data that is not there."""
    with open(reference / "train.extxyz", "r+b") as f:
        f.truncate(100)
    with pytest.raises(ValueError, match="not from the same run"):
        build(reference, frames[10]).generate()


def test_resuming_without_the_config_is_refused(reference, frames):
    """A sampler built outside the CLI has nothing to compare against."""
    sampler = build(reference, frames[10])
    sampler.sampler_config = None
    with pytest.raises(ValueError, match="not given the config"):
        sampler.generate()


# ------------------------------------------------------------------------------- md


def test_md_starts_fresh(md_reference):
    assert counts(md_reference) == {"train": 16, "val": 2, "test": 2}


def test_md_cannot_resume_yet_and_says_so(md_reference, frames):
    with pytest.raises(NotImplementedError, match="MDSampler cannot resume yet"):
        md(md_reference, frames[10]).generate()
