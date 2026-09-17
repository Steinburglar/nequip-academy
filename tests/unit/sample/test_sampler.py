import torch
import pytest
from ase import Atoms
from ase.io import read, write

from nequip_extension_template.sample.sampler import STATE_FILE, SPLITS, Sampler


class ToySampler(Sampler):
    """Small deterministic sampler for base-class state tests."""

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


def build_sampler(path, base_frames, **overrides):
    settings = {
        "target": 4,
        "splits": list(SPLITS),
        "state_interval": 1,
    }
    settings.update(overrides)
    sampler = ToySampler(
        calculator=None,
        base_frames=str(base_frames),
        sample_path=str(path),
        target=settings["target"],
        splits=settings["splits"],
        state_interval=settings["state_interval"],
    )
    sampler.sampler_config = {
        "_target_": f"{ToySampler.__module__}.{ToySampler.__qualname__}",
        "calculator": {"device": "cpu"},
        "base_frames": str(base_frames),
        **settings,
    }
    return sampler


def split_count(path, split):
    split_path = path / f"{split}.extxyz"
    if not split_path.exists():
        return 0
    return len(read(str(split_path), index=":"))


def test_generate_writes_state_that_matches_disk(tmp_path):
    base = tmp_path / "base.xyz"
    sample_path = tmp_path / "sampled"
    write_base_frame(base)

    sampler = build_sampler(sample_path, base)

    assert sampler.generate() == 4
    assert sampler.n_written == 4
    assert sampler.split_counts == {"train": 2, "val": 1, "test": 1}
    assert {split: split_count(sample_path, split) for split in SPLITS} == {
        "train": 2,
        "val": 1,
        "test": 1,
    }

    state = torch.load(sample_path / STATE_FILE, weights_only=False)
    progress = state["progress"]
    assert progress["n_written"] == 4
    assert progress["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert progress["procedure"] == {"i": 4}
    assert progress["offsets"] == {
        split: (sample_path / f"{split}.extxyz").stat().st_size for split in SPLITS
    }
    assert not (sample_path / "sampler_state.pt.tmp").exists()


def test_resume_truncates_bytes_after_last_record(tmp_path):
    base = tmp_path / "base.xyz"
    sample_path = tmp_path / "sampled"
    write_base_frame(base)

    interrupted = build_sampler(sample_path, base, state_interval=2)
    real_step = interrupted.step

    def die_after_third_append():
        real_step()
        if interrupted.n_written == 3:
            raise KeyboardInterrupt("simulated interruption")

    interrupted.step = die_after_third_append
    with pytest.raises(KeyboardInterrupt):
        interrupted.generate()

    state = torch.load(sample_path / STATE_FILE, weights_only=False)
    recorded_offsets = state["progress"]["offsets"]
    assert state["progress"]["n_written"] == 2
    assert sum(split_count(sample_path, split) for split in SPLITS) == 3
    assert any(
        (sample_path / f"{split}.extxyz").stat().st_size > recorded_offsets[split]
        for split in SPLITS
        if (sample_path / f"{split}.extxyz").exists()
    )

    resumed = build_sampler(sample_path, base, state_interval=2)

    assert resumed.generate() == 4
    assert resumed.n_resumed == 2
    assert {split: split_count(sample_path, split) for split in SPLITS} == {
        "train": 2,
        "val": 1,
        "test": 1,
    }
    assert resumed.i == 4


def test_resume_refuses_goal_differences_and_names_them(tmp_path):
    base = tmp_path / "base.xyz"
    sample_path = tmp_path / "sampled"
    write_base_frame(base)
    assert build_sampler(sample_path, base).generate() == 4

    changed = build_sampler(sample_path, base)
    changed.sampler_config["calculator"]["device"] = "cuda"

    with pytest.raises(ValueError) as error:
        changed.generate()

    message = str(error.value)
    assert "produced under different settings" in message
    assert "calculator.device: 'cpu' -> 'cuda'" in message


def test_resume_refuses_changed_base_frame_contents(tmp_path):
    base = tmp_path / "base.xyz"
    sample_path = tmp_path / "sampled"
    write_base_frame(base, x=0.0)
    assert build_sampler(sample_path, base).generate() == 4

    write_base_frame(base, x=1.0)

    with pytest.raises(ValueError, match="base frame contents"):
        build_sampler(sample_path, base).generate()


def test_resume_refuses_unrecorded_split_files(tmp_path):
    base = tmp_path / "base.xyz"
    sample_path = tmp_path / "sampled"
    write_base_frame(base)
    sample_path.mkdir()
    write(str(sample_path / "train.extxyz"), Atoms("Ar", positions=[[0.0, 0.0, 0.0]]))

    with pytest.raises(FileExistsError, match=STATE_FILE):
        build_sampler(sample_path, base).generate()
