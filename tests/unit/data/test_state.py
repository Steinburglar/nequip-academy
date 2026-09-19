import torch
import pytest
from ase import Atoms
from ase.io import write

from nequip_extension_template.data.paths import SPLITS, split_file
from nequip_extension_template.data.state import (
    STATE_FILE,
    STATE_VERSION,
    flatten,
    read_state,
    refuse_changed_settings,
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


def test_write_state_is_atomic_and_read_state_round_trips(tmp_path):
    path = state_file(tmp_path)
    payload = {
        "version": STATE_VERSION,
        "generator_class": "example.Generator",
        "goal": {"config": {}, "base_frames": "a1b2c3d4"},
        "progress": {"n_written": 0},
    }

    write_state(path, payload)

    assert read_state(path) == payload
    assert not path.with_suffix(".pt.tmp").exists()


def test_read_state_returns_none_when_absent(tmp_path):
    assert read_state(state_file(tmp_path)) is None


def test_refuse_changed_settings_names_what_changed(tmp_path):
    with pytest.raises(ValueError) as error:
        refuse_changed_settings(
            {"calculator.device": "cpu", "seed": 1},
            {"calculator.device": "cuda", "seed": 1},
            dataset_path=tmp_path,
            n_written=4,
        )
    message = str(error.value)
    assert "produced under different settings" in message
    assert "calculator.device: 'cpu' -> 'cuda'" in message
    assert "seed" not in message.split("settings:")[1].split("A resumed")[0]


def test_refuse_changed_settings_accepts_identical_settings(tmp_path):
    refuse_changed_settings(
        {"seed": 1}, {"seed": 1}, dataset_path=tmp_path, n_written=4
    )


def test_refuse_changed_settings_reports_a_key_only_one_side_has(tmp_path):
    with pytest.raises(ValueError, match=r"extra: '<not set>' -> 7"):
        refuse_changed_settings(
            {}, {"extra": 7}, dataset_path=tmp_path, n_written=1
        )


def test_refuse_changed_settings_appends_an_explanation(tmp_path):
    with pytest.raises(ValueError, match="path has not"):
        refuse_changed_settings(
            {"base frame contents": "aaaa"},
            {"base frame contents": "bbbb"},
            dataset_path=tmp_path,
            n_written=4,
            explanations={
                "base frame contents": "(the file behind `base_frames` has changed, "
                "even if its path has not)"
            },
        )


def test_split_offsets_and_truncate_to_recorded_lengths(tmp_path):
    dataset_path = tmp_path / "samples"
    dataset_path.mkdir()
    train = split_file(dataset_path, "train")
    write(str(train), frame(0.0))
    offsets = split_offsets(dataset_path)
    assert offsets["train"] == train.stat().st_size
    assert offsets["val"] == 0
    assert offsets["test"] == 0

    with open(train, "ab") as f:
        f.write(b"extra")
    assert train.stat().st_size > offsets["train"]

    truncate_to(dataset_path, offsets)

    assert train.stat().st_size == offsets["train"]


def test_truncate_to_refuses_shorter_file_than_recorded(tmp_path):
    dataset_path = tmp_path / "samples"
    dataset_path.mkdir()
    train = split_file(dataset_path, "train")
    train.write_bytes(b"short")
    offsets = {"train": 10, "val": 0, "test": 0}

    with pytest.raises(ValueError, match="shorter than"):
        truncate_to(dataset_path, offsets)


def test_truncate_to_refuses_missing_file_with_recorded_bytes(tmp_path):
    dataset_path = tmp_path / "samples"
    dataset_path.mkdir()
    offsets = {"train": 10, "val": 0, "test": 0}

    with pytest.raises(FileNotFoundError, match="does not exist"):
        truncate_to(dataset_path, offsets)


def test_refuse_existing_split_files_without_state(tmp_path):
    dataset_path = tmp_path / "samples"
    dataset_path.mkdir()
    write(str(split_file(dataset_path, "train")), frame())

    with pytest.raises(FileExistsError, match=STATE_FILE):
        refuse_existing_split_files_without_state(dataset_path)


def test_refuse_existing_split_files_allows_empty_directory(tmp_path):
    refuse_existing_split_files_without_state(tmp_path)


def test_state_file_payload_is_torch_loadable(tmp_path):
    path = state_file(tmp_path)
    payload = {"version": STATE_VERSION, "generator_class": "example.Generator"}

    write_state(path, payload)

    assert torch.load(path, weights_only=False) == payload
