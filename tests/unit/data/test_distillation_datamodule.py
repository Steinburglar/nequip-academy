import logging
from pathlib import Path

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from hydra.utils import instantiate
from omegaconf import OmegaConf

import nequip_academy.data.datamodule as datamodule_module
from nequip_academy.data import DistillationDataModule
from nequip_academy.data.paths import SPLITS
from nequip_academy.data.state import STATE_FILE

pytestmark = pytest.mark.filterwarnings("ignore:Length of split at index .*:UserWarning")


def base_frame(offset: float) -> Atoms:
    return Atoms(
        "Ar2",
        positions=[[0.0 + offset, 0.0, 0.0], [1.8 + offset, 0.0, 0.0]],
        cell=[6.0, 6.0, 6.0],
        pbc=True,
    )


def write_base_frames(path: Path, n: int = 4) -> None:
    write(str(path), [base_frame(float(i)) for i in range(n)])


def generation_config(base_frames: Path, split: dict | None = None) -> dict:
    return {
        "_target_": "nequip_academy.sample.RattleGenerator",
        "base_frames": str(base_frames),
        "split": split or {"train": 0.5, "val": 0.5},
        "split_policy": "blocked",
        "split_seed": 0,
        "strain_magnitudes": [0.0],
        "n_random_strain_samples": 0,
        "anisotropic_strain_magnitude": 0.02,
        "max_displacement_ang": 0.0,
        "seed": 1,
    }


def teacher_config() -> dict:
    return {
        "_target_": "ase.calculators.lj.LennardJones",
        "sigma": 3.4,
        "epsilon": 0.0104,
        "rc": 5.0,
    }


def transforms() -> list[dict]:
    return [
        {
            "_target_": "nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper",
            "model_type_names": ["Ar"],
        },
        {
            "_target_": "nequip.data.transforms.NeighborListTransform",
            "r_max": 4.0,
        },
    ]


def datamodule(
    dataset_path: Path, base_frames: Path, **overrides
) -> DistillationDataModule:
    split = overrides.pop("split", None)
    return DistillationDataModule(
        seed=1,
        dataset_path=dataset_path,
        generation=generation_config(base_frames, split),
        teacher=teacher_config(),
        transforms=transforms(),
        **overrides,
        train_dataloader={
            "_target_": "torch.utils.data.DataLoader",
            "batch_size": 2,
            "num_workers": 0,
            "shuffle": True,
        },
        val_dataloader={
            "_target_": "torch.utils.data.DataLoader",
            "batch_size": 2,
            "num_workers": 0,
        },
        test_dataloader={
            "_target_": "torch.utils.data.DataLoader",
            "batch_size": 2,
            "num_workers": 0,
        },
    )


def split_counts(dataset_path: Path) -> dict[str, int]:
    """Structures per split, zero for a split whose file was never written."""
    counts = {}
    for split in SPLITS:
        path = dataset_path / f"{split}.extxyz"
        counts[split] = len(read(str(path), index=":")) if path.exists() else 0
    return counts


def test_prepare_data_generates_pre_split_dataset(tmp_path: Path) -> None:
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "samples"
    write_base_frames(base)

    dm = datamodule(dataset_path, base)
    assert dm.train_dataset_config[0]["file_path"] == str(dataset_path / "train.extxyz")
    assert dm.val_dataset_config[0]["file_path"] == str(dataset_path / "val.extxyz")
    assert dm.test_dataset_config == []

    dm.prepare_data()

    assert (dataset_path / STATE_FILE).exists()
    assert split_counts(dataset_path) == {"train": 2, "val": 2, "test": 0}
    assert not (dataset_path / "test.extxyz").exists()


def test_generated_files_load_through_nequip_setup(tmp_path: Path) -> None:
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "samples"
    write_base_frames(base)

    dm = datamodule(dataset_path, base)
    dm.prepare_data()
    dm.setup("fit")

    assert len(dm.train_dataset[0]) == 2
    assert len(dm.val_dataset[0]) == 2

    batch = next(iter(dm.train_dataloader()))
    assert batch["pos"].shape[-1] == 3


def test_hydra_can_instantiate_datamodule_without_recursive_teacher_load(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "samples"
    write_base_frames(base)

    cfg = OmegaConf.create(
        {
            "_target_": "nequip_academy.data.DistillationDataModule",
            "_recursive_": False,
            "seed": 1,
            "dataset_path": str(dataset_path),
            "teacher": teacher_config(),
            "generation": generation_config(base),
            "transforms": transforms(),
            "train_dataloader": {
                "_target_": "torch.utils.data.DataLoader",
                "batch_size": 2,
                "num_workers": 0,
                "shuffle": True,
            },
            "val_dataloader": {
                "_target_": "torch.utils.data.DataLoader",
                "batch_size": 2,
                "num_workers": 0,
            },
            "test_dataloader": {
                "_target_": "torch.utils.data.DataLoader",
                "batch_size": 2,
                "num_workers": 0,
            },
        }
    )

    dm = instantiate(cfg)
    assert isinstance(dm, DistillationDataModule)
    assert isinstance(dm.teacher_config, dict)
    assert not dataset_path.exists()

    dm.prepare_data()

    assert split_counts(dataset_path) == {"train": 2, "val": 2, "test": 0}


def test_completed_dataset_does_not_reinstantiate_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "samples"
    write_base_frames(base)
    datamodule(dataset_path, base).prepare_data()

    real_instantiate = datamodule_module.instantiate

    def fail_on_teacher(config, *args, **kwargs):
        if (
            isinstance(config, dict)
            and config.get("_target_") == teacher_config()["_target_"]
        ):
            raise AssertionError("teacher should not be instantiated for completed data")
        return real_instantiate(config, *args, **kwargs)

    monkeypatch.setattr(datamodule_module, "instantiate", fail_on_teacher)

    datamodule(dataset_path, base).prepare_data()


def write_ground_truth_test_set(path: Path, n: int = 2) -> Path:
    """A held-out file standing in for an externally labeled (e.g. DFT) test set."""
    frames = []
    for i in range(n):
        atoms = base_frame(float(10 + i))
        atoms.calc = SinglePointCalculator(
            atoms, energy=-1.0 * (i + 1), forces=[[0.0] * 3] * len(atoms)
        )
        frames.append(atoms)
    write(str(path), frames)
    return path


def test_external_test_set_is_used_and_not_generated(tmp_path: Path) -> None:
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "samples"
    write_base_frames(base)
    held_out = write_ground_truth_test_set(tmp_path / "dft_test.xyz")

    dm = datamodule(dataset_path, base, test_file_path=str(held_out))
    assert dm.test_dataset_config[0]["file_path"] == str(held_out)

    dm.prepare_data()
    dm.setup("test")

    # the generated dataset holds train+val only; the test set is the untouched file
    assert split_counts(dataset_path) == {"train": 2, "val": 2, "test": 0}
    assert len(dm.test_dataset[0]) == 2


def test_external_test_set_may_live_anywhere(tmp_path: Path) -> None:
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "samples"
    write_base_frames(base)
    elsewhere = tmp_path / "somewhere" / "else"
    elsewhere.mkdir(parents=True)
    held_out = write_ground_truth_test_set(elsewhere / "dft_test.xyz")

    dm = datamodule(dataset_path, base, test_file_path=held_out)

    assert dm.test_dataset_config[0]["file_path"] == str(held_out)
    assert dataset_path not in held_out.parents


def test_external_test_set_and_generated_test_share_refused(tmp_path: Path) -> None:
    base = tmp_path / "base.xyz"
    write_base_frames(base)
    held_out = write_ground_truth_test_set(tmp_path / "dft_test.xyz")

    with pytest.raises(ValueError, match="Pick one"):
        datamodule(
            tmp_path / "samples",
            base,
            test_file_path=str(held_out),
            split={"train": 0.5, "val": 0.25, "test": 0.25},
        )


def test_generated_test_share_refused_without_opt_in(tmp_path: Path) -> None:
    base = tmp_path / "base.xyz"
    write_base_frames(base)

    with pytest.raises(ValueError, match="TEACHER-LABELED"):
        datamodule(
            tmp_path / "samples",
            base,
            split={"train": 0.5, "val": 0.25, "test": 0.25},
        )


def test_generated_test_share_allowed_with_loud_opt_in(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "samples"
    write_base_frames(base)

    with caplog.at_level(logging.WARNING, logger=datamodule_module.logger.name):
        dm = datamodule(
            dataset_path,
            base,
            split={"train": 0.5, "val": 0.25, "test": 0.25},
            teacher_labeled_test=True,
        )
    assert "TEACHER-LABELED" in caplog.text

    assert dm.test_dataset_config[0]["file_path"] == str(dataset_path / "test.extxyz")
    dm.prepare_data()
    assert split_counts(dataset_path) == {"train": 2, "val": 1, "test": 1}


def test_opt_in_without_a_test_share_refused(tmp_path: Path) -> None:
    base = tmp_path / "base.xyz"
    write_base_frames(base)

    with pytest.raises(ValueError, match="no test share"):
        datamodule(tmp_path / "samples", base, teacher_labeled_test=True)


def test_missing_external_test_set_refused_before_generating(tmp_path: Path) -> None:
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "samples"
    write_base_frames(base)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        datamodule(dataset_path, base, test_file_path=str(tmp_path / "nope.xyz"))
    assert not dataset_path.exists()


@pytest.mark.parametrize("key", ["train_file_path", "val_file_path"])
def test_generated_split_paths_still_refused(tmp_path: Path, key: str) -> None:
    base = tmp_path / "base.xyz"
    write_base_frames(base)

    with pytest.raises(ValueError, match="owns their paths"):
        datamodule(tmp_path / "samples", base, **{key: ["nonexistent.xyz"]})
