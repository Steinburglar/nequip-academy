import torch
import pytest
from ase import Atoms
from ase.io import read, write

from nequip_academy.data.paths import SPLITS
from nequip_academy.data.state import STATE_FILE
from nequip_academy.sample.generator import Generator, frames_digest


class ToyGenerator(Generator):
    """Small deterministic generator for base-class state tests."""

    def __init__(self, *, target=4, splits=None, **kwargs):
        super().__init__(**kwargs)
        self.target = int(target)
        self.splits = list(splits or SPLITS)
        self.i = 0

    @property
    def finished(self):
        return self.i >= self.target

    def procedure_state(self):
        return {"i": self.i}

    def restore_progress(self, procedure_state):
        self.i = int(procedure_state["i"])

    def step(self):
        atoms = self.base_frames[0].copy()
        atoms.info["toy_index"] = self.i
        self.append(atoms, self.splits[self.i % len(self.splits)])
        self.i += 1


def write_base_frame(path, x=0.0):
    write(str(path), Atoms("Ar", positions=[[x, 0.0, 0.0]], cell=[5, 5, 5], pbc=True))


def build_generator(path, base_frames, **overrides):
    settings = {
        "target": 4,
        "splits": list(SPLITS),
        "state_interval": 1,
    }
    settings.update(overrides)
    generator = ToyGenerator(
        # ToyGenerator fabricates structures instead of labeling them, so the object
        # here is never used. It is supplied because the base class's contract is
        # that stepping needs a teacher, and generate() now says so up front.
        calculator=object(),
        base_frames=str(base_frames),
        dataset_path=str(path),
        target=settings["target"],
        splits=settings["splits"],
        state_interval=settings["state_interval"],
    )
    generator.generation_config = {
        "_target_": f"{ToyGenerator.__module__}.{ToyGenerator.__qualname__}",
        "calculator": {"device": "cpu"},
        "base_frames": str(base_frames),
        **settings,
    }
    return generator


def split_count(path, split):
    split_path = path / f"{split}.extxyz"
    if not split_path.exists():
        return 0
    return len(read(str(split_path), index=":"))


def test_generate_writes_state_that_matches_disk(tmp_path):
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "sampled"
    write_base_frame(base)

    generator = build_generator(dataset_path, base)

    assert generator.generate() == 4
    assert generator.n_written == 4
    assert generator.split_counts == {"train": 2, "val": 1, "test": 1}
    assert {split: split_count(dataset_path, split) for split in SPLITS} == {
        "train": 2,
        "val": 1,
        "test": 1,
    }

    state = torch.load(dataset_path / STATE_FILE, weights_only=False)
    contents = state["contents"]
    assert contents["n_written"] == 4
    assert contents["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert state["progress"] == {"i": 4}
    assert contents["offsets"] == {
        split: (dataset_path / f"{split}.extxyz").stat().st_size for split in SPLITS
    }
    assert all(contents["digests"][split] is not None for split in SPLITS)
    assert not (dataset_path / "generation_state.pt.tmp").exists()


def test_resume_truncates_bytes_after_last_record(tmp_path):
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "sampled"
    write_base_frame(base)

    interrupted = build_generator(dataset_path, base, state_interval=2)
    real_step = interrupted.step

    def die_after_third_append():
        real_step()
        if interrupted.n_written == 3:
            raise KeyboardInterrupt("simulated interruption")

    interrupted.step = die_after_third_append
    with pytest.raises(KeyboardInterrupt):
        interrupted.generate()

    state = torch.load(dataset_path / STATE_FILE, weights_only=False)
    recorded_offsets = state["contents"]["offsets"]
    assert state["contents"]["n_written"] == 2
    assert sum(split_count(dataset_path, split) for split in SPLITS) == 3
    assert any(
        (dataset_path / f"{split}.extxyz").stat().st_size > recorded_offsets[split]
        for split in SPLITS
        if (dataset_path / f"{split}.extxyz").exists()
    )

    resumed = build_generator(dataset_path, base, state_interval=2)

    assert resumed.generate() == 4
    assert resumed.n_resumed == 2
    assert {split: split_count(dataset_path, split) for split in SPLITS} == {
        "train": 2,
        "val": 1,
        "test": 1,
    }
    assert resumed.i == 4


def test_resume_refuses_goal_differences_and_names_them(tmp_path):
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "sampled"
    write_base_frame(base)
    assert build_generator(dataset_path, base).generate() == 4

    changed = build_generator(dataset_path, base)
    changed.generation_config["calculator"]["device"] = "cuda"

    with pytest.raises(ValueError) as error:
        changed.generate()

    message = str(error.value)
    assert "produced under different settings" in message
    assert "calculator.device: 'cpu' -> 'cuda'" in message


def test_resume_refuses_changed_base_frame_contents(tmp_path):
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "sampled"
    write_base_frame(base, x=0.0)
    assert build_generator(dataset_path, base).generate() == 4

    write_base_frame(base, x=1.0)

    with pytest.raises(ValueError, match="base frame contents"):
        build_generator(dataset_path, base).generate()


def test_resume_refuses_unrecorded_split_files(tmp_path):
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "sampled"
    write_base_frame(base)
    dataset_path.mkdir()
    write(str(dataset_path / "train.extxyz"), Atoms("Ar", positions=[[0.0, 0.0, 0.0]]))

    with pytest.raises(FileExistsError, match=STATE_FILE):
        build_generator(dataset_path, base).generate()


# ------------------------------------------- provenance the generator owns


def test_frames_digest_tracks_loaded_structure_contents():
    """Hashed from the parsed structures, so reformatting the file is not a change."""
    a = Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    b = Atoms("Ar", positions=[[0.1, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    assert frames_digest([a]) == frames_digest([a.copy()])
    assert frames_digest([a]) != frames_digest([b])
    assert frames_digest([a]) != frames_digest([a, b])


def test_check_compatible_refuses_a_changed_setting(tmp_path):
    base = tmp_path / "base.xyz"
    write_base_frame(base)
    generator = build_generator(tmp_path / "sampled", base)
    stored = generator.provenance()
    generator.generation_config = dict(generator.generation_config, target=99)

    with pytest.raises(ValueError, match=r"target: 4 -> 99"):
        generator.check_compatible(stored, n_written=4)


def test_check_compatible_refuses_changed_base_frame_contents(tmp_path):
    base = tmp_path / "base.xyz"
    write_base_frame(base)
    generator = build_generator(tmp_path / "sampled", base)
    stored = dict(generator.provenance(), base_frames="not-the-same-digest")

    with pytest.raises(ValueError, match="base frame contents"):
        generator.check_compatible(stored, n_written=4)


def test_check_compatible_refuses_another_procedure(tmp_path):
    base = tmp_path / "base.xyz"
    write_base_frame(base)
    generator = build_generator(tmp_path / "sampled", base)
    stored = dict(generator.provenance(), generator_class="pkg.SomethingElse")

    with pytest.raises(ValueError, match="one procedure"):
        generator.check_compatible(stored, n_written=4)


def test_check_compatible_refuses_a_generator_with_no_config(tmp_path):
    base = tmp_path / "base.xyz"
    write_base_frame(base)
    generator = build_generator(tmp_path / "sampled", base)
    stored = generator.provenance()
    generator.generation_config = None

    with pytest.raises(ValueError, match="was not given the config"):
        generator.check_compatible(stored, n_written=4)


def test_check_compatible_accepts_unchanged_settings(tmp_path):
    base = tmp_path / "base.xyz"
    write_base_frame(base)
    generator = build_generator(tmp_path / "sampled", base)
    generator.check_compatible(generator.provenance(), n_written=4)


# ------------------------------------------------- lazy teacher acquisition


def build_teacherless(path, base_frames, **overrides):
    """Like `build_generator`, but with no calculator attached yet."""
    generator = build_generator(path, base_frames, **overrides)
    generator.release_calculator()
    return generator


def test_generate_builds_the_teacher_only_once_and_only_when_stepping(tmp_path):
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "sampled"
    write_base_frame(base)

    built = []

    def factory():
        built.append(1)
        return object()

    assert build_teacherless(dataset_path, base).generate(teacher_factory=factory) == 4
    assert built == [1], "the teacher was not built, or was built more than once"


def test_generate_does_not_build_the_teacher_for_a_finished_dataset(tmp_path):
    """The whole point of deferring it: a second student pays nothing for the model."""
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "sampled"
    write_base_frame(base)
    build_teacherless(dataset_path, base).generate(teacher_factory=lambda: object())

    built = []
    resumed = build_teacherless(dataset_path, base)
    assert resumed.generate(teacher_factory=lambda: built.append(1)) == 4
    assert built == [], "a complete dataset still loaded the teacher"
    assert resumed.n_resumed == 4


def test_generate_refuses_to_step_with_no_teacher_at_all(tmp_path):
    base = tmp_path / "base.xyz"
    write_base_frame(base)
    with pytest.raises(ValueError, match="has no teacher"):
        build_teacherless(tmp_path / "sampled", base).generate()


def test_an_attached_calculator_needs_no_factory(tmp_path):
    base = tmp_path / "base.xyz"
    write_base_frame(base)
    generator = build_teacherless(tmp_path / "sampled", base)
    generator.attach_calculator(object())
    assert generator.generate() == 4
