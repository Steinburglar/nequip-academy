"""MD generator: one trajectory, snapshots taken along it at a fixed interval.

Scaffolding. The dynamics are the simplest thing that samples a thermal ensemble --
Langevin NVT from Maxwell-Boltzmann initial velocities -- and the parameters below
are placeholders, not recommendations.

This exists mainly as the second consumer of :class:`~.generator.Generator`, so the base
class is checked against a procedure that is nothing like rattling:

* it runs *one* trajectory, from the first base frame, and ignores the rest, where
  rattling visits every base frame
* its extent is a snapshot count, not an enumeration over inputs
* it splits per *snapshot*, where rattling splits per base frame
* its labels come from the dynamics, which already evaluated the teacher to take the
  step, so no extra teacher call is made to label a snapshot

**This generator cannot resume yet.** A count is enough for rattling, which walks a
fixed list, but a snapshot only exists by integrating the trajectory up to it, so
continuing one means restoring positions, velocities and the state of the random
number generator. Until that is built, pointing a second run at a ``dataset_path``
that already holds MD output is refused by
:meth:`~.generator.Generator.restore_progress`. Fresh runs are unaffected.

**Assumption.** Splitting per snapshot is only sound if ``sample_interval`` is long
enough that consecutive snapshots are decorrelated. Nothing here checks that, and
nothing can: if the interval is too short, neighbouring snapshots are near-duplicates
and a validation snapshot is effectively in the training set. Choose the interval
from the system's correlation time, not from how many structures you want.
"""

import logging
from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from .generator import Generator
from .split import assign_splits

logger = logging.getLogger(__name__)


class MDGenerator(Generator):
    """Run a single Langevin trajectory, keeping a snapshot every ``sample_interval``.

    Parameters
    ----------
    temperature_K
        Thermostat temperature, also used for the initial velocities.
    timestep_fs
        MD timestep, in fs.
    friction_per_fs
        Langevin friction coefficient, in 1/fs.
    n_equilibrate_steps
        Steps run before any snapshot is kept, to reach temperature.
    sample_interval
        Steps between kept snapshots. Must exceed the system's correlation time --
        see the module docstring.
    n_samples
        How many snapshots to keep. This is the whole extent of the procedure.
    seed
        RNG seed for the initial velocities and the thermostat noise.

        Unlike :class:`~.rattle.RattleGenerator`, this is one *streaming* generator, and
        it has to be. A trajectory is sequential by nature: snapshot n is reached by
        integrating through snapshots 1..n-1, so a snapshot cannot be derived from its
        own identity alone. Resuming therefore means checkpointing the generator state
        (``self.rng.bit_generator.state``) along with the positions and velocities.
    split, split_seed, split_policy
        Target fractions, assigned per snapshot. Defaults to ``"blocked"``: snapshots
        are ordered in time, so a contiguous holdout at the end of the trajectory is
        honest even if ``sample_interval`` under-decorrelates, whereas a scattered one
        is not. Set ``"scattered"`` only if the interval is known to decorrelate.
    """

    def __init__(
        self,
        temperature_K: float = 300.0,
        timestep_fs: float = 1.0,
        friction_per_fs: float = 0.01,
        n_equilibrate_steps: int = 100,
        sample_interval: int = 50,
        n_samples: int = 100,
        seed: int = 0,
        split: Optional[dict] = None,
        split_seed: int = 0,
        split_policy: str = "blocked",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temperature_K = float(temperature_K)
        self.timestep_fs = float(timestep_fs)
        self.friction_per_fs = float(friction_per_fs)
        self.n_equilibrate_steps = int(n_equilibrate_steps)
        self.sample_interval = int(sample_interval)
        self.n_samples = int(n_samples)
        self.rng = np.random.default_rng(seed)

        if len(self.base_frames) > 1:
            logger.warning(
                f"{len(self.base_frames)} base frames provided; MD starts one "
                "trajectory from the first and ignores the rest."
            )

        # one split label per snapshot, fixed up front
        self.snapshot_split = assign_splits(
            self.n_samples, split, split_seed, split_policy
        )

        # procedure state
        self.n_taken = 0
        self.md_step = 0
        self.dynamics = None

    @property
    def finished(self) -> bool:
        return self.n_taken >= self.n_samples

    def _start(self) -> None:
        atoms = self.base_frames[0].copy()
        atoms.calc = self.calculator
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=self.temperature_K, rng=self.rng
        )
        self.dynamics = Langevin(
            atoms,
            timestep=self.timestep_fs * units.fs,
            temperature_K=self.temperature_K,
            # ASE wants the friction per ASE time unit, not per fs
            friction=self.friction_per_fs / units.fs,
            rng=self.rng,
        )
        if self.n_equilibrate_steps:
            self.dynamics.run(self.n_equilibrate_steps)
            self.md_step += self.n_equilibrate_steps

    def _snapshot(self) -> Atoms:
        """Detach the current configuration, keeping the labels the dynamics computed.

        The energy and forces are read from the calculator's cached results at the
        current positions -- the dynamics needed them to take its last step -- so this
        costs no additional teacher evaluation. ``atoms.copy()`` drops the calculator,
        and with it those results, hence the ``SinglePointCalculator``.
        """
        atoms = self.dynamics.atoms
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        snapshot = atoms.copy()
        snapshot.calc = SinglePointCalculator(snapshot, energy=energy, forces=forces)
        return snapshot

    def step(self) -> None:
        if self.dynamics is None:
            self._start()
        self.dynamics.run(self.sample_interval)
        self.md_step += self.sample_interval

        snapshot = self._snapshot()
        snapshot.info["md_step"] = self.md_step
        self.append(snapshot, self.snapshot_split[self.n_taken])
        self.n_taken += 1
