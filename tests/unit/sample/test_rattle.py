from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from nequip_academy.sample.rattle import (
    RattleGenerator,
    derive_seed,
    frame_key,
    random_anisotropic_strain_matrix,
)
from nequip_academy.data.paths import SPLITS

pytestmark = pytest.mark.filterwarnings("ignore:Length of split at index .*:UserWarning")

Geometry = tuple[np.ndarray, np.ndarray, np.ndarray]
GeometryByIdentity = dict[tuple[str, str], Geometry]


def base_frame(offset: float) -> Atoms:
    """Distinct Ar dimer cell used as one base structure."""
    return Atoms(
        "Ar2",
        positions=[[0.0 + offset, 0.0, 0.0], [1.8 + offset, 0.0, 0.0]],
        cell=[6.0, 6.0, 6.0],
        pbc=True,
    )


def build_rattler(path: Path, base_frames: Path, **overrides: Any) -> RattleGenerator:
    """Build a CLI-like generator with matching stored config."""
    settings = {
        "strain_magnitudes": [0.0],
        "n_random_strain_samples": 2,
        "anisotropic_strain_magnitude": 0.03,
        "max_displacement_ang": 0.05,
        "seed": 7,
        "split": {"train": 1.0, "val": 0.0, "test": 0.0},
        "split_seed": 0,
        "split_policy": "blocked",
    }
    settings.update(overrides)
    generator = RattleGenerator(
        calculator=LennardJones(),
        base_frames=str(base_frames),
        dataset_path=str(path),
        **settings,
    )
    generator.generation_config = {
        "_target_": f"{RattleGenerator.__module__}.{RattleGenerator.__qualname__}",
        "calculator": {"_target_": "ase.calculators.lj.LennardJones"},
        "base_frames": str(base_frames),
        **settings,
    }
    return generator


def generated_frames(path: Path) -> list[Atoms]:
    """Read all split files in stable train/val/test order."""
    frames: list[Atoms] = []
    for split in SPLITS:
        split_path = path / f"{split}.extxyz"
        if split_path.exists():
            frames.extend(read(str(split_path), index=":"))
    return frames


def geometries_by_identity(path: Path) -> GeometryByIdentity:
    """Map each generated structure identity to its geometry."""
    return {
        (atoms.info["base_frame_key"], atoms.info["variant"]): (
            atoms.get_atomic_numbers().copy(),
            atoms.get_positions().copy(),
            np.asarray(atoms.cell).copy(),
        )
        for atoms in generated_frames(path)
    }


def assert_same_geometries(
    left: GeometryByIdentity, right: GeometryByIdentity
) -> None:
    assert left.keys() == right.keys()
    for key in left:
        left_numbers, left_positions, left_cell = left[key]
        right_numbers, right_positions, right_cell = right[key]
        np.testing.assert_array_equal(left_numbers, right_numbers)
        np.testing.assert_allclose(left_positions, right_positions, atol=0.0)
        np.testing.assert_allclose(left_cell, right_cell, atol=0.0)


def test_rattle_splits_by_base_frame_not_variant(tmp_path: Path) -> None:
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "sampled"
    write(str(base), [base_frame(i) for i in range(4)])

    build_rattler(
        dataset_path,
        base,
        split={"train": 0.5, "val": 0.25, "test": 0.25},
        split_policy="blocked",
    ).generate()

    splits_by_base_frame = {}
    for split in SPLITS:
        split_path = dataset_path / f"{split}.extxyz"
        for atoms in read(str(split_path), index=":"):
            splits_by_base_frame.setdefault(atoms.info["base_frame_key"], set()).add(
                split
            )

    assert len(splits_by_base_frame) == 4
    assert all(len(splits) == 1 for splits in splits_by_base_frame.values())


def test_rattle_generation_is_identity_seeded_not_order_seeded(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.xyz"
    reversed_ = tmp_path / "reversed.xyz"
    write(str(first), [base_frame(0), base_frame(1)])
    write(str(reversed_), [base_frame(1), base_frame(0)])

    first_path = tmp_path / "first_out"
    reversed_path = tmp_path / "reversed_out"
    build_rattler(first_path, first).generate()
    build_rattler(reversed_path, reversed_).generate()

    assert_same_geometries(
        geometries_by_identity(first_path),
        geometries_by_identity(reversed_path),
    )


def test_rattle_variant_labels_do_not_renumber_anisotropic_variants(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.xyz"
    write(str(base), [base_frame(0)])

    original_path = tmp_path / "original"
    extended_path = tmp_path / "extended"
    build_rattler(original_path, base, strain_magnitudes=[0.0]).generate()
    build_rattler(extended_path, base, strain_magnitudes=[-0.05, 0.0]).generate()

    # Adding "iso:-0.05" must not redraw the existing "iso:0.0" or "aniso:*" cases.
    original = geometries_by_identity(original_path)
    extended = geometries_by_identity(extended_path)
    common = {key: extended[key] for key in original}

    assert_same_geometries(original, common)


def test_rattle_labels_output_with_teacher_and_drops_input_labels(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.xyz"
    dataset_path = tmp_path / "sampled"
    atoms = base_frame(0)
    atoms.calc = SinglePointCalculator(
        atoms, energy=123.0, forces=np.full((len(atoms), 3), 456.0)
    )
    write(str(base), [atoms])

    build_rattler(
        dataset_path,
        base,
        strain_magnitudes=[0.0],
        n_random_strain_samples=0,
        max_displacement_ang=0.0,
    ).generate()

    [labeled] = generated_frames(dataset_path)
    assert labeled.info["base_frame"] == 0
    assert labeled.info["base_frame_key"]
    assert labeled.info["variant"] == "iso:0.0"
    assert labeled.get_potential_energy() != 123.0
    np.testing.assert_allclose(
        labeled.get_potential_energy(), LennardJones().get_potential_energy(labeled)
    )
    np.testing.assert_allclose(
        labeled.get_forces(), LennardJones().get_forces(labeled)
    )


def triclinic_frame() -> Atoms:
    """Base frame with a cell that is not a multiple of the identity.

    The row-vs-column strain convention is invisible on a cubic cell, so a
    regression test for it has to use a genuinely triclinic one. These are the
    lattice vectors of the CsH2PO4 frames in `sandbox/inputs/`, rounded.
    """
    return Atoms(
        "Ar2",
        positions=[[0.5, 0.5, 0.5], [2.3, 0.5, 0.5]],
        cell=[[9.377, 0.0, 0.0], [1.673, 13.025, 0.0], [-0.976, -1.179, 9.053]],
        pbc=True,
    )


def test_anisotropic_strain_uses_ase_row_vector_convention(tmp_path: Path) -> None:
    """ASE keeps lattice vectors as rows, so deforming by F is `cell @ F.T`.

    Doing `F @ cell` instead deforms the transpose. It is identical for an
    isotropic strain and for a cubic cell, which is why this went unnoticed; on a
    triclinic cell it silently produces a differently shaped box of the same volume.
    """
    base = tmp_path / "triclinic.xyz"
    write(str(base), [triclinic_frame()])

    generator = build_rattler(
        tmp_path / "out",
        base,
        strain_magnitudes=[0.0],
        n_random_strain_samples=1,
        max_displacement_ang=0.0,
    )
    generator.generate()

    strained = next(
        f for f in generated_frames(tmp_path / "out") if f.info["variant"] == "aniso:0"
    )

    cell = np.asarray(triclinic_frame().cell)
    rng = np.random.default_rng(
        derive_seed(generator.seed, frame_key(triclinic_frame()), "aniso:0")
    )
    strain_matrix = random_anisotropic_strain_matrix(
        rng, generator.anisotropic_strain_magnitude
    )

    assert np.allclose(np.asarray(strained.cell), cell @ strain_matrix.T)
    assert not np.allclose(np.asarray(strained.cell), strain_matrix @ cell)
