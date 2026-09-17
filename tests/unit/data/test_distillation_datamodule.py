from pathlib import Path

import pytest
from ase import Atoms
from ase.io import read, write
from hydra.utils import instantiate
from omegaconf import OmegaConf

import nequip_extension_template.data.datamodule as datamodule_module
from nequip_extension_template.data import DistillationDataModule
from nequip_extension_template.sample import SPLITS
from nequip_extension_template.sample.sampler import STATE_FILE

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


def generation_config(base_frames: Path) -> dict:
    return {
        "_target_": "nequip_extension_template.sample.RattleSampler",
        "base_frames": str(base_frames),
        "split": {"train": 0.5, "val": 0.25, "test": 0.25},
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


def datamodule(sample_path: Path, base_frames: Path) -> DistillationDataModule:
    return DistillationDataModule(
        seed=1,
        sample_path=sample_path,
        generation=generation_config(base_frames),
        teacher=teacher_config(),
        transforms=transforms(),
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


def split_counts(sample_path: Path) -> dict[str, int]:
    return {
        split: len(read(str(sample_path / f"{split}.extxyz"), index=":"))
        for split in SPLITS
    }


def test_prepare_data_generates_pre_split_dataset(tmp_path: Path) -> None:
    base = tmp_path / "base.xyz"
    sample_path = tmp_path / "samples"
    write_base_frames(base)

    dm = datamodule(sample_path, base)
    assert dm.train_dataset_config[0]["file_path"] == str(sample_path / "train.extxyz")
    assert dm.val_dataset_config[0]["file_path"] == str(sample_path / "val.extxyz")
    assert dm.test_dataset_config[0]["file_path"] == str(sample_path / "test.extxyz")

    dm.prepare_data()

    assert (sample_path / STATE_FILE).exists()
    assert split_counts(sample_path) == {"train": 2, "val": 1, "test": 1}


def test_generated_files_load_through_nequip_setup(tmp_path: Path) -> None:
    base = tmp_path / "base.xyz"
    sample_path = tmp_path / "samples"
    write_base_frames(base)

    dm = datamodule(sample_path, base)
    dm.prepare_data()
    dm.setup("fit")

    assert len(dm.train_dataset[0]) == 2
    assert len(dm.val_dataset[0]) == 1

    batch = next(iter(dm.train_dataloader()))
    assert batch["pos"].shape[-1] == 3


def test_hydra_can_instantiate_datamodule_without_recursive_teacher_load(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.xyz"
    sample_path = tmp_path / "samples"
    write_base_frames(base)

    cfg = OmegaConf.create(
        {
            "_target_": "nequip_extension_template.data.DistillationDataModule",
            "_recursive_": False,
            "seed": 1,
            "sample_path": str(sample_path),
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
    assert not sample_path.exists()

    dm.prepare_data()

    assert split_counts(sample_path) == {"train": 2, "val": 1, "test": 1}


def test_completed_dataset_does_not_reinstantiate_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.xyz"
    sample_path = tmp_path / "samples"
    write_base_frames(base)
    datamodule(sample_path, base).prepare_data()

    real_instantiate = datamodule_module.instantiate

    def fail_on_teacher(config, *args, **kwargs):
        if (
            isinstance(config, dict)
            and config.get("_target_") == teacher_config()["_target_"]
        ):
            raise AssertionError("teacher should not be instantiated for completed data")
        return real_instantiate(config, *args, **kwargs)

    monkeypatch.setattr(datamodule_module, "instantiate", fail_on_teacher)

    datamodule(sample_path, base).prepare_data()
