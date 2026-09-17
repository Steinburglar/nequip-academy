"""End-to-end exercise of the `nequip-distill` command.

This runs the real command as a subprocess, once per case, and several cases
train a student, so it takes minutes rather than seconds. Run it with pytest:

    pytest -m e2e
    pytest tests/e2e/test_distill_cli.py -k <substring>

**No teacher, no GPU.** The calculator is Lennard-Jones on 32-atom argon cells and
the student is a one-layer model trained for two epochs on the CPU. Nothing here
checks that a student learns anything; what is under test is the wiring -- which
stages run, what reaches nequip, which mistakes are refused and how early.

Subprocesses rather than in-process calls because `@hydra.main` owns global state
and does not survive being invoked twice in one interpreter, and because the point
is to check the command as a user meets it.
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.build import bulk
from ase.io import write

REPO_ROOT = Path(__file__).resolve().parent.parent

N_BASE_FRAMES = 10
# 3 isotropic strains + 1 random anisotropic one per base frame
N_VARIANTS = 4
N_STRUCTURES = N_BASE_FRAMES * N_VARIANTS
SPLITS = ("train", "val", "test")


# ------------------------------------------------------------------ fixtures

def make_base_frames(path: Path) -> None:
    """Ten distinct 32-atom argon cells.

    Distinct on purpose: a base frame is identified by its contents, so duplicates
    would collapse into one as far as the sampler is concerned. Built from an fcc
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


def sampler_section() -> dict:
    return {
        "_target_": "nequip_extension_template.sample.RattleSampler",
        "calculator": {
            "_target_": "ase.calculators.lj.LennardJones",
            "sigma": 3.4,
            "epsilon": 0.0104,
            "rc": 5.0,
        },
        "base_frames": "base_frames.xyz",
        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
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
            "_target_": "nequip.data.datamodule.ASEDataModule",
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


def config(run, sample_path, train=True, **extra) -> dict:
    cfg = {"run": list(run), "sample_path": sample_path, "sampler": sampler_section()}
    if train:
        cfg.update(student_sections())
    cfg.update(extra)
    return cfg


# --------------------------------------------------------------------- driver

class Run:
    """One `nequip-distill` invocation and everything it left behind."""

    def __init__(self, workdir: Path, name: str, cfg: dict):
        self.workdir = workdir
        self.name = name
        config_dir = workdir / "configs"
        config_dir.mkdir(exist_ok=True)
        (config_dir / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        # pinned rather than left to the timestamp, so the case can look inside it
        self.run_dir = workdir / "runs" / name
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "nequip_extension_template.scripts.distill",
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


def dataset_digests(sample_path: Path) -> dict:
    return {s: digest(sample_path / f"{s}.extxyz") for s in SPLITS}


def count_structures(sample_path: Path) -> int:
    from ase.io import read

    return sum(len(read(str(sample_path / f"{s}.extxyz"), index=":")) for s in SPLITS)


# ---------------------------------------------------------------------- cases

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def expect_failure(run: Run, *fragments: str) -> None:
    assert not run.ok, f"expected {run.name} to fail, it exited 0:\n{run.output}"
    for fragment in fragments:
        assert fragment in run.output, (
            f"{run.name}: expected {fragment!r} in the output:\n{run.output}"
        )


def expect_success(run: Run, *fragments: str) -> None:
    assert run.ok, f"expected {run.name} to succeed:\n{run.output}"
    for fragment in fragments:
        assert fragment in run.output, (
            f"{run.name}: expected {fragment!r} in the output:\n{run.output}"
        )


@case
def unknown_run_type(work: Path) -> None:
    """`predict` is nequip's, not ours, and is refused by name."""
    run = Run(work, "unknown_run_type", config(["sample", "predict"], "out/unused"))
    expect_failure(run, "unknown run type 'predict'")
    assert not (work / "out/unused").exists(), "refused run still created sample_path"


@case
def sample_must_come_first(work: Path) -> None:
    """Training before sampling would train on the previous dataset."""
    run = Run(work, "sample_last", config(["train", "sample"], "out/unused"))
    expect_failure(run, "must come first")


@case
def empty_run_list(work: Path) -> None:
    run = Run(work, "empty_run", config([], "out/unused"))
    expect_failure(run, "`run` is empty")


@case
def missing_dataset(work: Path) -> None:
    """No `sample` stage means the dataset has to be there already."""
    run = Run(work, "missing_dataset", config(["train", "val", "test"], "out/absent"))
    expect_failure(run, "missing or empty", "train.extxyz")


@case
def missing_trainer_section_fails_before_sampling(work: Path) -> None:
    """The point of the check: it costs teacher calls to find this out late."""
    cfg = config(["sample", "train"], "out/no_trainer")
    del cfg["trainer"]
    run = Run(work, "no_trainer", cfg)
    expect_failure(run, "`trainer` must be provided")
    assert not (work / "out/no_trainer").exists(), (
        "sampling started before the config was found to be untrainable"
    )


@case
def wrong_datamodule(work: Path) -> None:
    """The dataset is wired in through `ASEDataModule`'s file-path arguments."""
    cfg = config(["sample", "train"], "out/wrong_dm")
    cfg["data"]["_target_"] = "nequip.data.datamodule.NequIPDataModule"
    run = Run(work, "wrong_datamodule", cfg)
    expect_failure(run, "ASEDataModule")
    assert not (work / "out/wrong_dm").exists()


@case
def split_dataset_refused(work: Path) -> None:
    """Re-splitting at train time is what a frozen split exists to prevent."""
    cfg = config(["sample", "train"], "out/split_dataset")
    cfg["data"]["split_dataset"] = {"file_path": "x.xyz", "train": 0.8, "val": 0.2}
    run = Run(work, "split_dataset", cfg)
    expect_failure(run, "never re-split")


@case
def sample_only(work: Path) -> None:
    """The original behaviour: build the dataset, stop."""
    run = Run(work, "sample_only", config(["sample"], "out/sample_only", train=False))
    expect_success(run, f"{N_STRUCTURES} labeled structures", "train=32, val=4, test=4")
    sample_path = work / "out/sample_only"
    assert (sample_path / "sampler_state.pt").exists()
    assert count_structures(sample_path) == N_STRUCTURES


@case
def sample_only_rerun_adds_nothing(work: Path) -> None:
    """A finished dataset is finished, and the files are not touched again."""
    cfg = config(["sample"], "out/rerun", train=False)
    first = Run(work, "rerun_a", cfg)
    expect_success(first)
    before = dataset_digests(work / "out/rerun")
    second = Run(work, "rerun_b", cfg)
    expect_success(second, f"0 from this run, {N_STRUCTURES} already present")
    assert dataset_digests(work / "out/rerun") == before, "re-run rewrote the dataset"


@case
def full_pipeline(work: Path) -> None:
    """Sample, then hand the result to nequip: train, val, test."""
    run = Run(
        work,
        "full",
        config(["sample", "train", "val", "test"], "out/full"),
    )
    expect_success(run, "TRAIN RUN END", "VAL RUN END", "TEST RUN END")
    assert count_structures(work / "out/full") == N_STRUCTURES
    for name in ("best.ckpt", "last.ckpt"):
        assert (run.run_dir / name).exists(), f"{name} missing from {run.run_dir}"


@case
def second_student_on_the_same_dataset(work: Path) -> None:
    """The dataset outlives the command; each command gets its own student."""
    cfg = config(["sample", "train", "val", "test"], "out/two_students")
    first = Run(work, "two_students_a", cfg)
    expect_success(first, "TRAIN RUN END")
    before = dataset_digests(work / "out/two_students")
    second = Run(work, "two_students_b", cfg)
    expect_success(second, f"0 from this run, {N_STRUCTURES} already present")
    assert dataset_digests(work / "out/two_students") == before
    assert second.run_dir != first.run_dir
    assert (second.run_dir / "best.ckpt").exists()
    assert (first.run_dir / "best.ckpt").exists(), "the earlier student was clobbered"


@case
def train_only_never_builds_the_sampler(work: Path) -> None:
    """A run with no `sample` stage must not load the teacher.

    Checked by pointing the calculator at something that does not exist: if the
    sampler were instantiated, the run would die. The bogus `train_file_path` in the
    same config checks the other half -- that whatever the user wrote there is
    replaced by the sampled files rather than read.
    """
    prepared = Run(
        work, "prep_train_only", config(["sample"], "out/train_only", train=False)
    )
    expect_success(prepared)
    cfg = config(["train", "val", "test"], "out/train_only")
    cfg["sampler"]["calculator"] = {"_target_": "does.not.Exist"}
    cfg["data"]["train_file_path"] = ["nonexistent.xyz"]
    run = Run(work, "train_only", cfg)
    expect_success(run, "TRAIN RUN END", "ignoring `data.train_file_path")


@case
def explicit_null_ckpt_path(work: Path) -> None:
    """`ckpt_path: null` is a fresh start, not a restart.

    nequip tests for the key's presence rather than its value, so left in place this
    sends it down the restart path with nothing to load.
    """
    cfg = config(["sample", "train"], "out/null_ckpt", ckpt_path=None)
    run = Run(work, "null_ckpt", cfg)
    expect_success(run, "TRAIN RUN END")
    assert "Starting fresh training" in run.output, run.output


@case
def restart_from_checkpoint(work: Path) -> None:
    """Hand back a checkpoint with the dataset unchanged: nequip's own restart."""
    cfg = config(["sample", "train", "val", "test"], "out/restart")
    first = Run(work, "restart_a", cfg)
    expect_success(first, "TRAIN RUN END")
    resumed = config(["train", "val", "test"], "out/restart")
    resumed["ckpt_path"] = str(first.run_dir / "last.ckpt")
    second = Run(work, "restart_b", resumed)
    expect_success(second, "Continuing training with checkpoint file")


@case
def new_structures_plus_checkpoint_refused(work: Path) -> None:
    """The trap D11 exists for.

    A restart resumes at the checkpoint's epoch and stops at the same `max_epochs`,
    so structures produced after that checkpoint get only the leftover epochs --
    none at all if the run had finished. Confirmed by deleting the guard: the source
    run logged epochs ['0', '1'], the restarted run logged only ['1'] and exited 0.
    """
    cfg = config(["sample", "train", "val", "test"], "out/trap_source")
    source = Run(work, "trap_source", cfg)
    expect_success(source, "TRAIN RUN END")
    grown = config(["sample", "train", "val", "test"], "out/trap_fresh")
    grown["ckpt_path"] = str(source.run_dir / "last.ckpt")
    run = Run(work, "trap", grown)
    expect_failure(run, f"added {N_STRUCTURES} structure(s)", "Drop `ckpt_path`")


# ---------------------------------------------------------------------- pytest


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("case_fn", CASES, ids=lambda fn: fn.__name__)
def test_distill_cli_case(case_fn, tmp_path):
    make_base_frames(tmp_path / "base_frames.xyz")
    case_fn(tmp_path)


# ----------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", default="", help="only run cases matching this substring")
    parser.add_argument(
        "--keep", action="store_true", help="keep the working directory"
    )
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="distill-smoke-"))
    make_base_frames(work / "base_frames.xyz")
    print(f"working directory: {work}\n")

    selected = [c for c in CASES if args.k in c.__name__]
    failures = []
    for index, case_fn in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {case_fn.__name__} ... ", end="", flush=True)
        try:
            case_fn(work)
        except AssertionError as error:
            failures.append((case_fn.__name__, error))
            print("FAIL")
            print(f"    {error}")
        else:
            print("ok")

    print()
    if failures:
        print(f"{len(failures)} of {len(selected)} cases failed:")
        for name, _ in failures:
            print(f"  - {name}")
    else:
        print(f"all {len(selected)} cases passed")
    if args.keep or failures:
        print(f"working directory kept: {work}")
    else:
        shutil.rmtree(work)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
