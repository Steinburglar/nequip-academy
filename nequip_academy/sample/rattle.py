"""Rattle/deform generator.

Ported from ``../distillation/scripts/gen_synthetic_geoms.py``. Each base frame is
deformed and then rattled, once per *variant*:

* one isotropic volume scan point per entry in ``strain_magnitudes`` -- the cell is
  scaled by ``(1 + strain) ** (1/3)`` with the atoms scaled along with it
* ``n_random_strain_samples`` structures with a random symmetric anisotropic strain,
  bounded by ``anisotropic_strain_magnitude``

followed in both cases by a per-atom random displacement of up to
``max_displacement_ang``.

Keep the magnitudes conservative. In the reference work, +/-10%/5% strain with 0.5 Ang
displacement pushed structures outside the teacher's domain and produced students
*worse* than using no synthetic data at all; halving to +/-5%/2.5% with 0.25 Ang
fixed it.
"""

import hashlib
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from .generator import Generator
from .split import assign_splits

def frame_key(atoms: Atoms) -> str:
    """Content hash of a single structure, used as its stable identity.

    Stable under reordering the base-frame file, so adding or reordering base frames
    never renames the existing ones.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(atoms.get_atomic_numbers(), dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(np.round(atoms.get_positions(), 8)).tobytes())
    h.update(np.ascontiguousarray(np.round(np.asarray(atoms.cell), 8)).tobytes())
    h.update(np.ascontiguousarray(atoms.get_pbc()).tobytes())
    return h.hexdigest()[:16]


def derive_seed(seed: int, key: str, variant: str) -> int:
    """Seed for one structure, derived from its identity rather than its position."""
    digest = hashlib.sha256(f"{seed}|{key}|{variant}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def isotropic_strain_matrix(volumetric_strain: float) -> np.ndarray:
    return np.eye(3) * (1.0 + volumetric_strain) ** (1.0 / 3.0)


def random_anisotropic_strain_matrix(rng, max_magnitude: float) -> np.ndarray:
    strain = rng.uniform(-max_magnitude, max_magnitude, size=(3, 3))
    return np.eye(3) + 0.5 * (strain + strain.T)


def rattle_positions(rng, atoms: Atoms, max_displacement_ang: float) -> Atoms:
    directions = rng.normal(size=(len(atoms), 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    magnitudes = rng.uniform(0.0, max_displacement_ang, size=(len(atoms), 1))
    atoms.positions += directions * magnitudes
    return atoms


class RattleGenerator(Generator):
    """Deform and rattle each base frame, once per variant.

    Parameters
    ----------
    strain_magnitudes
        Volumetric strains for the isotropic scan, one variant each.
    n_random_strain_samples
        Number of random anisotropic-strain variants per base frame.
    anisotropic_strain_magnitude
        Bound on each component of the random anisotropic strain.

        Deliberately independent of ``strain_magnitudes``. The reference
        implementation derived it as ``max(abs(strain_magnitudes))``, which couples
        the two knobs: adding one point to the isotropic scan silently widens the
        anisotropic distribution, so every ``aniso`` structure already on disk was
        drawn from a different distribution than the config now describes -- with no
        change to its name, and so nothing to detect it by. Changing this value on
        purpose has the same effect, which is why it belongs in the goal diff that
        resume compares against.
    max_displacement_ang
        Upper bound on the per-atom displacement, in Angstrom.
    seed
        Base RNG seed. Each structure gets its own stream derived from ``seed``, the
        content hash of its base frame, and its variant label -- not from one shared
        stream. So a structure depends only on its own identity, never on how many
        structures were generated before it. That is what lets a resumed or extended
        run reproduce exactly the structures a fresh run would have produced, with no
        RNG state to checkpoint.
    split
        Target train/val/test fractions.
    split_seed
        Seed for the shuffle that decides which base frame lands in which split.
    split_policy
        ``"scattered"`` (default) or ``"blocked"``. Base frames have no meaningful
        order here, so scattered is the sensible default; ``"blocked"`` hands out
        contiguous ranges of the input file instead.

    The split is assigned per base frame: the structures rattled from one base frame
    are near-duplicates of each other, so splitting them apart would put effectively
    the same structure in both train and validation.
    """

    def __init__(
        self,
        strain_magnitudes: Sequence[float] = (-0.05, -0.025, 0.0, 0.025, 0.05),
        n_random_strain_samples: int = 3,
        anisotropic_strain_magnitude: float = 0.05,
        max_displacement_ang: float = 0.25,
        seed: int = 0,
        split: dict = None,
        split_seed: int = 0,
        split_policy: str = "scattered",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.strain_magnitudes = [float(s) for s in strain_magnitudes]
        self.n_random_strain_samples = int(n_random_strain_samples)
        self.max_displacement_ang = float(max_displacement_ang)
        self.anisotropic_strain_magnitude = float(anisotropic_strain_magnitude)
        self.seed = int(seed)
        self.n_steps = 0
        self.base_frame_keys = [frame_key(a) for a in self.base_frames]

        # Variant labels are part of the scientific identity of a synthetic structure.
        # They enter the RNG seed so extending the strain grid later does not redraw
        # structures that already had a label. For example, adding a new isotropic
        # strain point before the anisotropic variants must not change the random
        # strain/displacement used for "aniso:0"; otherwise a resumed or extended
        # dataset would contain structures whose names match old ones but whose
        # sampling distribution changed.
        self.variants = [f"iso:{s}" for s in self.strain_magnitudes]
        self.variants += [f"aniso:{i}" for i in range(self.n_random_strain_samples)]
        if not self.variants:
            raise ValueError(
                "this procedure has no variants -- `strain_magnitudes` is empty and "
                "`n_random_strain_samples` is 0"
            )

        self.base_frame_split = assign_splits(
            len(self.base_frames), split, split_seed, split_policy
        )

    def procedure_state(self) -> dict:
        """How many structures have been produced. That is the whole of it.

        The order is fixed -- variant-major over the base frames -- and a resume is
        only allowed when the settings and the base frames are unchanged, so the count
        is also the position: structure ``n_steps`` is the next one either way.
        """
        return {"n_steps": self.n_steps}

    def restore_progress(self, procedure_state: dict) -> None:
        self.n_steps = int(procedure_state["n_steps"])

    @property
    def n_total(self) -> int:
        """One structure per (base frame, variant) pair. This is the whole procedure."""
        return len(self.base_frames) * len(self.variants)

    @property
    def finished(self) -> bool:
        return self.n_steps >= self.n_total

    def label(self, atoms: Atoms) -> Atoms:
        """Evaluate the teacher and return a detached, labeled copy.

        ``atoms.copy()`` drops ``atoms.calc``, and with it the labels, so the results
        are read out first and reattached as a ``SinglePointCalculator``. Skipping
        this writes geometry with no energy or forces, which is only noticed at
        student-training time. Dropping the calculator is also what discards any
        label the base frame arrived with.
        """
        atoms.calc = self.calculator
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        labeled = atoms.copy()
        labeled.calc = SinglePointCalculator(labeled, energy=energy, forces=forces)
        return labeled

    def step(self) -> None:
        # Variant-major: every base frame gets variant 0 before any gets variant 1, so
        # a run that stops early has covered all the base frames rather than
        # exhausting the first few.
        base_index = self.n_steps % len(self.base_frames)
        variant_index = self.n_steps // len(self.base_frames)
        self.n_steps += 1

        variant = self.variants[variant_index]
        key = self.base_frame_keys[base_index]
        rng = np.random.default_rng(derive_seed(self.seed, key, variant))

        kind, value = variant.split(":", 1)
        if kind == "iso":
            strain_matrix = isotropic_strain_matrix(float(value))
        else:
            strain_matrix = random_anisotropic_strain_matrix(
                rng, self.anisotropic_strain_magnitude
            )

        atoms = self.base_frames[base_index].copy()
        # ASE stores lattice vectors as ROWS, so deforming by F is `cell @ F.T`, not
        # `F @ cell`. The two agree for an isotropic strain (a multiple of the
        # identity) and for a cubic cell, and differ for anything else -- on the
        # triclinic CsH2PO4 sandbox frames by ~0.13 Ang per lattice vector.
        atoms.set_cell(atoms.cell[:] @ strain_matrix.T, scale_atoms=True)
        rattle_positions(rng, atoms, self.max_displacement_ang)

        labeled = self.label(atoms)
        labeled.info["base_frame"] = base_index
        labeled.info["base_frame_key"] = key
        labeled.info["variant"] = variant
        self.append(labeled, self.base_frame_split[base_index])
