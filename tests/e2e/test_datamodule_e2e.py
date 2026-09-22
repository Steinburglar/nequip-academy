"""End-to-end exercise of `nequip-train` driving `DistillationDataModule`.

This runs the real command as a subprocess, once per test, and most tests train a
student, so it takes minutes rather than seconds. Run it with pytest:

    pytest -m e2e
    pytest tests/e2e/test_datamodule_e2e.py -k <substring>

**No teacher, no GPU.** The calculator is Lennard-Jones on 32-atom argon cells and
the student is a one-layer model trained for two epochs on the CPU. Nothing here
checks that a student learns anything; what is under test is the wiring -- what
reaches nequip, which mistakes are refused, and when the teacher is and is not
loaded.

Subprocesses rather than in-process calls because `@hydra.main` owns global state
and does not survive being invoked twice in one interpreter, and because the point
is to check the command as a user meets it.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

N_BASE_FRAMES = 10
# 3 isotropic strains + 1 random anisotropic one per base frame
N_VARIANTS = 4
N_STRUCTURES = N_BASE_FRAMES * N_VARIANTS
N_TEST_FRAMES = 3
SPLITS = ("train", "val", "test")
GENERATED_SPLITS = ("train", "val")

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


# ------------------------------------------------------------------ fixtures

def make_base_frames(path: Path) -> None:
    """Ten distinct 32-atom argon cells.

    Distinct on purpose: a base frame is identified by its contents, so duplicates
    would collapse into one as far as the generator is concerned. Built from an fcc
    lattice rather than random positions so Lennard-Jones returns sane energies --
    random points in a box put atoms on top of each other.
    """
    generator = np.random.default_rng(0)
    frames = []
    for _ in range(N_BASE_FRAMES):
        atoms = bulk("Ar", "fcc", a=5.26, cubic=True) * (2, 2, 2)
        atoms.positions += generator.normal(0.0, 0.05, atoms.positions.shape)
        frames.append(atoms)
    write(str(path), frames)


def make_test_set(path: Path) -> None:
    """A held-out, already-labeled test set, standing in for a DFT one.

    Labeled here with the same Lennard-Jones calculator that plays the teacher, which
    makes it a poor scientific test set and a perfectly good plumbing one: what is
    under test is that an externally supplied file drives the test stage, not what the
    numbers are. Structures are distinct from the base frames.
    """
    generator = np.random.default_rng(1)
    frames = []
    for _ in range(N_TEST_FRAMES):
        atoms = bulk("Ar", "fcc", a=5.26, cubic=True) * (2, 2, 2)
        atoms.positions += generator.normal(0.0, 0.05, atoms.positions.shape)
        atoms.calc = LennardJones(sigma=3.4, epsilon=0.0104, rc=5.0)
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=atoms.get_potential_energy(),
            forces=atoms.get_forces(),
        )
        frames.append(atoms)
    write(str(path), frames)


@pytest.fixture
def work(tmp_path: Path) -> Path:
    """A scratch directory with the base frames and the held-out test set in it."""
    make_base_frames(tmp_path / "base_frames.xyz")
    make_test_set(tmp_path / "ground_truth_test.xyz")
    return tmp_path


def teacher_section() -> dict:
    return {
        "_target_": "ase.calculators.lj.LennardJones",
        "sigma": 3.4,
        "epsilon": 0.0104,
        "rc": 5.0,
    }


def generation_section() -> dict:
    return {
        "_target_": "nequip_academy.sample.RattleGenerator",
        "base_frames": "base_frames.xyz",
        "split": {"train": 0.9, "val": 0.1},
        "split_policy": "scattered",
        "split_seed": 0,
        "strain_magnitudes": [-0.02, 0.0, 0.02],
        "n_random_strain_samples": 1,
        "anisotropic_strain_magnitude": 0.02,
        "max_displacement_ang": 0.1,
        "seed": 1,
    }


def student_sections() -> dict:
    """A student small enough to train on a login node in a few seconds."""
    return {
        "seed": 1,
        "cutoff_radius": 4.0,
        "model_type_names": ["Ar"],
        "monitored_metric": "val0_epoch/weighted_sum",
        "data": {
            "seed": "${seed}",
            "transforms": [
                {
                    "_target_": "nequip.data.transforms"
                    ".ChemicalSpeciesToAtomTypeMapper",
                    "model_type_names": "${model_type_names}",
                },
                {
                    "_target_": "nequip.data.transforms.NeighborListTransform",
                    "r_max": "${cutoff_radius}",
                },
            ],
            "train_dataloader": {
                "_target_": "torch.utils.data.DataLoader",
                "batch_size": 4,
                "num_workers": 0,
                "shuffle": True,
            },
            "val_dataloader": {
                "_target_": "torch.utils.data.DataLoader",
                "batch_size": 4,
                "num_workers": 0,
            },
            "test_dataloader": "${data.val_dataloader}",
            "stats_manager": {
                "_target_": "nequip.data.CommonDataStatisticsManager",
                "dataloader_kwargs": {"batch_size": 4},
                "type_names": "${model_type_names}",
            },
        },
        "trainer": {
            "_target_": "lightning.Trainer",
            "accelerator": "cpu",
            "enable_checkpointing": True,
            "max_epochs": 2,
            "log_every_n_steps": 4,
            "enable_progress_bar": False,
            "logger": {
                "_target_": "lightning.pytorch.loggers.CSVLogger",
                "save_dir": "${hydra:runtime.output_dir}",
            },
            "callbacks": [
                {
                    "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "monitor": "${monitored_metric}",
                    "dirpath": "${hydra:runtime.output_dir}",
                    "filename": "best",
                    "save_last": True,
                    "enable_version_counter": False,
                }
            ],
        },
        "training_module": {
            "_target_": "nequip.train.NequIPLightningModule",
            "loss": {
                "_target_": "nequip.train.EnergyForceLoss",
                "per_atom_energy": True,
                "coeffs": {"total_energy": 1.0, "forces": 1.0},
            },
            "val_metrics": {
                "_target_": "nequip.train.EnergyForceMetrics",
                "coeffs": {"total_energy_mae": 1.0, "forces_mae": 1.0},
            },
            "train_metrics": "${training_module.val_metrics}",
            "test_metrics": "${training_module.val_metrics}",
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.01},
            "model": {
                "_target_": "nequip.model.NequIPGNNModel",
                "seed": "${seed}",
                "model_dtype": "float64",
                "type_names": "${model_type_names}",
                "r_max": "${cutoff_radius}",
                "num_bessels": 4,
                "num_layers": 1,
                "l_max": 0,
                "parity": True,
                "num_features": 8,
                "radial_mlp_depth": 1,
                "radial_mlp_width": 16,
                "avg_num_neighbors": "${training_data_stats:num_neighbors_mean}",
                "per_type_energy_shifts": "${training_data_stats:per_atom_energy_mean}",
            },
        },
    }


def config(dataset_path: str, **extra) -> dict:
    """A full `nequip-train` config whose `data:` generates its own dataset.

    `ckpt_path` is deliberately absent rather than null: nequip tests for the key's
    presence, not its value, so `ckpt_path: null` sends the run down the restart
    path with nothing to load.
    """
    cfg = student_sections()
    cfg["run"] = ["train", "val", "test"]
    cfg["data"].update(
        {
            "_target_": "nequip_academy.data.DistillationDataModule",
            "_recursive_": False,
            "dataset_path": dataset_path,
            "teacher": teacher_section(),
            "generation": generation_section(),
            # generation makes train+val; the test set is supplied, as it would be
            # with real ground-truth labels
            "test_file_path": "ground_truth_test.xyz",
        }
    )
    cfg.update(extra)
    return cfg


# -------------------------------------------------------------------- driver

class TrainRun:
    """One `nequip-train` invocation and everything it left behind."""

    def __init__(self, workdir: Path, name: str, cfg: dict):
        self.workdir = workdir
        self.name = name
        config_dir = workdir / "configs"
        config_dir.mkdir(exist_ok=True)
        (config_dir / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        # pinned rather than left to the timestamp, so the test can look inside it
        self.run_dir = workdir / "runs" / name
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "nequip.scripts.train",
                "-cp",
                str(config_dir),
                "-cn",
                name,
                f"hydra.run.dir={self.run_dir}",
            ],
            cwd=str(workdir),
            capture_output=True,
            text=True,
        )
        self.returncode = completed.returncode
        self.output = completed.stdout + completed.stderr

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_digests(dataset_path: Path) -> dict:
    return {s: digest(dataset_path / f"{s}.extxyz") for s in GENERATED_SPLITS}


def count_structures(dataset_path: Path) -> int:
    from ase.io import read

    return sum(
        len(read(str(dataset_path / f"{s}.extxyz"), index=":"))
        for s in SPLITS
        if (dataset_path / f"{s}.extxyz").exists()
    )


def expect_failure(run: TrainRun, *fragments: str) -> None:
    assert not run.ok, f"expected {run.name} to fail, it exited 0:\n{run.output}"
    for fragment in fragments:
        assert fragment in run.output, (
            f"{run.name}: expected {fragment!r} in the output:\n{run.output}"
        )


def expect_success(run: TrainRun, *fragments: str) -> None:
    assert run.ok, f"expected {run.name} to succeed:\n{run.output}"
    for fragment in fragments:
        assert fragment in run.output, (
            f"{run.name}: expected {fragment!r} in the output:\n{run.output}"
        )


# --------------------------------------------------------------------- tests

def test_full_pipeline(work: Path) -> None:
    """The primary path: nequip's trainer owns the run, `data:` owns generation."""
    run = TrainRun(work, "full", config("out/full"))
    expect_success(run, "TRAIN RUN END", "VAL RUN END", "TEST RUN END")
    dataset_path = work / "out/full"
    assert count_structures(dataset_path) == N_STRUCTURES
    assert (dataset_path / "generation_state.pt").exists()
    for name in ("best.ckpt", "last.ckpt"):
        assert (run.run_dir / name).exists(), f"{name} missing from {run.run_dir}"


def test_second_student_reuses_the_dataset_untouched(work: Path) -> None:
    """The dataset outlives the command; each command gets its own student.

    Also the idempotence check: a finished dataset is finished, and a second run
    must not rewrite a byte of it.
    """
    cfg = config("out/two_students")
    first = TrainRun(work, "two_students_a", cfg)
    expect_success(first, "TRAIN RUN END")
    before = dataset_digests(work / "out/two_students")

    second = TrainRun(work, "two_students_b", cfg)
    expect_success(second, f"0 from this run, {N_STRUCTURES} already present")
    assert dataset_digests(work / "out/two_students") == before, (
        "the second run rewrote the dataset"
    )
    assert second.run_dir != first.run_dir
    assert (second.run_dir / "best.ckpt").exists()
    assert (first.run_dir / "best.ckpt").exists(), "the earlier student was clobbered"


TEACHER_PROBE = '''\
"""A Lennard-Jones teacher that records the fact that it was constructed."""

from pathlib import Path

from ase.calculators.lj import LennardJones


class ProbeCalculator(LennardJones):
    def __init__(self, *args, marker: str = "teacher_loaded.marker", **kwargs):
        Path(marker).touch()
        super().__init__(*args, **kwargs)
'''


def test_complete_dataset_never_loads_the_teacher(work: Path) -> None:
    """A finished dataset must be trainable without instantiating the teacher.

    This is what the no-teacher fast path exists for: a second student should not
    have to load a multi-gigabyte model onto a GPU to read files already on disk.

    Proving it needs a teacher whose *construction* is observable while its config
    stays byte-identical across both runs -- the config is provenance, so changing
    it would trip the settings-changed refusal instead of testing anything. Hence a
    LennardJones subclass that touches a marker file when built. `python -m` puts
    the working directory on `sys.path`, so the subprocess can import it.
    """
    (work / "teacher_probe.py").write_text(TEACHER_PROBE)
    marker = work / "teacher_loaded.marker"
    cfg = config("out/no_teacher")
    cfg["data"]["teacher"] = {"_target_": "teacher_probe.ProbeCalculator"}

    first = TrainRun(work, "probe_a", cfg)
    expect_success(first, "TRAIN RUN END")
    assert marker.exists(), "the probe never fired, so it proves nothing below"
    marker.unlink()

    second = TrainRun(work, "probe_b", cfg)
    expect_success(
        second, "TRAIN RUN END", f"0 from this run, {N_STRUCTURES} already present"
    )
    assert not marker.exists(), "the teacher was constructed for a finished dataset"


def test_split_dataset_refused(work: Path) -> None:
    """Re-splitting at train time is what a frozen split exists to prevent."""
    cfg = config("out/split_dataset")
    cfg["data"]["split_dataset"] = {"file_path": "x.xyz", "train": 0.8, "val": 0.2}
    run = TrainRun(work, "split_dataset", cfg)
    expect_failure(run, "do not also set")
    assert not (work / "out/split_dataset").exists(), (
        "refused config still created dataset_path"
    )


def test_generated_split_paths_refused(work: Path) -> None:
    """Train and val are generated, so their paths are not the user's to set.

    `test_file_path` is the deliberate exception, covered by
    `test_external_test_set_drives_the_test_stage`.
    """
    cfg = config("out/file_paths")
    cfg["data"]["train_file_path"] = ["nonexistent.xyz"]
    run = TrainRun(work, "file_paths", cfg)
    expect_failure(run, "owns their paths through `dataset_path`")
    assert not (work / "out/file_paths").exists()


def test_broken_student_config_fails_before_generating(work: Path) -> None:
    """A config that cannot train must not cost a full generation run first.

    Under the old CLI this was an explicit pre-flight check. On the `nequip-train`
    path it falls to nequip's own config validation, so this test records whether
    that validation really does run before Lightning calls `prepare_data()`.
    """
    cfg = config("out/no_trainer")
    del cfg["trainer"]
    run = TrainRun(work, "no_trainer", cfg)
    assert not run.ok, f"expected no_trainer to fail:\n{run.output}"
    assert not (work / "out/no_trainer").exists(), (
        "generation started before the config was found to be untrainable"
    )


def test_restart_from_checkpoint(work: Path) -> None:
    """Hand back a checkpoint with the dataset unchanged: nequip's own restart."""
    cfg = config("out/restart")
    first = TrainRun(work, "restart_a", cfg)
    expect_success(first, "TRAIN RUN END")

    resumed = config("out/restart", ckpt_path=str(first.run_dir / "last.ckpt"))
    second = TrainRun(work, "restart_b", resumed)
    expect_success(second, "Continuing training with checkpoint file")


def test_external_test_set_drives_the_test_stage(work: Path) -> None:
    """The reported test error is measured against the supplied labels.

    The generated dataset holds train and val only -- no `test.extxyz` is ever
    written -- and the test stage reads the held-out file instead, so the number
    `nequip-train` prints means "error against the labels the user brought", not
    "agreement with the teacher".
    """
    run = TrainRun(work, "external_test", config("out/external_test"))
    expect_success(run, "TEST RUN END")

    dataset_path = work / "out/external_test"
    assert not (dataset_path / "test.extxyz").exists()
    assert count_structures(dataset_path) == N_STRUCTURES

    metrics = next((run.run_dir).rglob("metrics.csv"))
    header = metrics.read_text().splitlines()[0]
    assert "test0_epoch/forces_mae" in header

    # the held-out file is read, never written to
    assert len(read(str(work / "ground_truth_test.xyz"), index=":")) == N_TEST_FRAMES


def test_generated_test_split_refused_without_opt_in(work: Path) -> None:
    """Asking for a teacher-labeled test split is possible, but never by accident."""
    cfg = config("out/teacher_test")
    cfg["data"].pop("test_file_path")
    cfg["data"]["generation"]["split"] = {"train": 0.8, "val": 0.1, "test": 0.1}
    run = TrainRun(work, "teacher_test", cfg)
    expect_failure(run, "TEACHER-LABELED", "teacher_labeled_test")
    assert not (work / "out/teacher_test").exists(), (
        "nothing should be generated when the config is refused"
    )


def test_generated_test_split_allowed_when_opted_into(work: Path) -> None:
    """With the opt-in the run proceeds, and says loudly what the metrics mean."""
    cfg = config("out/opted_in")
    cfg["data"].pop("test_file_path")
    cfg["data"]["generation"]["split"] = {"train": 0.8, "val": 0.1, "test": 0.1}
    cfg["data"]["teacher_labeled_test"] = True
    run = TrainRun(work, "opted_in", cfg)
    expect_success(run, "TEST RUN END", "TEACHER-LABELED")
    assert (work / "out/opted_in/test.extxyz").exists()
