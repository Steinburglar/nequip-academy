import torch
import pytest
from ase import Atoms
from ase.io import write

from nequip_extension_template.data.paths import SPLITS, split_file
from nequip_extension_template.data.state import (
    STATE_FILE,
    STATE_VERSION,
    check_goal,
    check_state_header,
    flatten,
    frames_digest,
    read_state,
    refuse_existing_split_files_without_state,
    split_offsets,
    state_file,
    truncate_to,
    write_state,
)


def frame(x: float = 0.0) -> Atoms:
    return Atoms("Ar", positions=[[x, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)


def test_split_file_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="unknown split"):
        split_file(tmp_path, "holdout")


def test_state_file_uses_current_compatibility_filename(tmp_path):
    assert state_file(tmp_path) == tmp_path / STATE_FILE


def test_flatten_reports_nested_config_keys():
    assert flatten({"calculator": {"device": "cpu"}, "seed": 1}) == {
        "calculator.device": "cpu",
        "seed": 1,
    }


def test_frames_digest_tracks_loaded_structure_contents():
    assert frames_digest([frame(0.0)]) == frames_digest([frame(0.0)])
    assert frames_digest([frame(0.0)]) != frames_digest([frame(1.0)])


def test_write_state_is_atomic_and_read_state_round_trips(tmp_path):
    path = state_file(tmp_path)
    payload = {
        "version": STATE_VERSION,
        "sampler_class": "example.Sampler",
        "goal": {"config": {}, "base_frames": frames_digest([frame()])},
        "progress": {"n_written": 0},
    }

    write_state(path, payload)

    assert read_state(path) == payload
    assert not path.with_suffix(".pt.tmp").exists()


def test_read_state_returns_none_when_absent(tmp_path):
    assert read_state(state_file(tmp_path)) is None


def test_check_state_header_refuses_version_mismatch(tmp_path):
    state = {"version": 999, "sampler_class": "example.Sampler"}

    with pytest.raises(ValueError, match="record format 999"):
        check_state_header(
            state,
            expected_version=STATE_VERSION,
            expected_class="example.Sampler",
            sample_path=tmp_path,
        )


def test_check_state_header_refuses_class_mismatch(tmp_path):
    state = {"version": STATE_VERSION, "sampler_class": "old.Sampler"}

    with pytest.raises(ValueError, match="one procedure"):
        check_state_header(
            state,
            expected_version=STATE_VERSION,
            expected_class="new.Sampler",
            sample_path=tmp_path,
        )


def test_check_goal_refuses_config_difference_and_names_it(tmp_path):
    stored_goal = {
        "config": {"calculator": {"device": "cpu"}, "seed": 1},
        "base_frames": frames_digest([frame()]),
    }

    with pytest.raises(ValueError) as error:
        check_goal(
            stored_goal,
            live_config={"calculator": {"device": "cuda"}, "seed": 1},
            base_frames=[frame()],
            sample_path=tmp_path,
            n_written=4,
        )

    message = str(error.value)
    assert "produced under different settings" in message
    assert "calculator.device: 'cpu' -> 'cuda'" in message


def test_check_goal_refuses_changed_base_frame_contents(tmp_path):
    stored_goal = {
        "config": {"seed": 1},
        "base_frames": frames_digest([frame(0.0)]),
    }

    with pytest.raises(ValueError, match="base frame contents"):
        check_goal(
            stored_goal,
            live_config={"seed": 1},
            base_frames=[frame(1.0)],
            sample_path=tmp_path,
            n_written=4,
        )


def test_check_goal_refuses_missing_live_config(tmp_path):
    stored_goal = {
        "config": {"seed": 1},
        "base_frames": frames_digest([frame()]),
    }

    with pytest.raises(ValueError, match="was not given the config"):
        check_goal(
            stored_goal,
            live_config=None,
            base_frames=[frame()],
            sample_path=tmp_path,
            n_written=4,
        )


def test_split_offsets_and_truncate_to_recorded_lengths(tmp_path):
    sample_path = tmp_path / "samples"
    sample_path.mkdir()
    train = split_file(sample_path, "train")
    write(str(train), frame(0.0))
    offsets = split_offsets(sample_path)
    assert offsets["train"] == train.stat().st_size
    assert offsets["val"] == 0
    assert offsets["test"] == 0

    with open(train, "ab") as f:
        f.write(b"extra")
    assert train.stat().st_size > offsets["train"]

    truncate_to(sample_path, offsets)

    assert train.stat().st_size == offsets["train"]


def test_truncate_to_refuses_shorter_file_than_recorded(tmp_path):
    sample_path = tmp_path / "samples"
    sample_path.mkdir()
    train = split_file(sample_path, "train")
    train.write_bytes(b"short")
    offsets = {"train": 10, "val": 0, "test": 0}

    with pytest.raises(ValueError, match="shorter than"):
        truncate_to(sample_path, offsets)


def test_truncate_to_refuses_missing_file_with_recorded_bytes(tmp_path):
    sample_path = tmp_path / "samples"
    sample_path.mkdir()
    offsets = {"train": 10, "val": 0, "test": 0}

    with pytest.raises(FileNotFoundError, match="does not exist"):
        truncate_to(sample_path, offsets)


def test_refuse_existing_split_files_without_state(tmp_path):
    sample_path = tmp_path / "samples"
    sample_path.mkdir()
    write(str(split_file(sample_path, "train")), frame())

    with pytest.raises(FileExistsError, match=STATE_FILE):
        refuse_existing_split_files_without_state(sample_path)


def test_refuse_existing_split_files_allows_empty_directory(tmp_path):
    refuse_existing_split_files_without_state(tmp_path)


def test_state_file_payload_is_torch_loadable(tmp_path):
    path = state_file(tmp_path)
    payload = {"version": STATE_VERSION, "sampler_class": "example.Sampler"}

    write_state(path, payload)

    assert torch.load(path, weights_only=False) == payload
