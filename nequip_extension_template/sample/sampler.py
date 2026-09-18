"""Base sampler.

Deliberately minimal. All any sampler shares is the teacher calculator, the frames
it starts from, and where the dataset is written. Everything else -- how many
structures there are, what a step consists of, when it is finished, and which split
a structure belongs to -- belongs to the procedure, so it belongs to the subclass.

The base class knows only that the dataset is split three ways and where each part
is written. It does not decide what goes where; concrete samplers do.

:meth:`Sampler.step` does the whole job for one step: produce the next structure or
structures, label them, append them to the dataset, and advance the procedure's own
state. Producing and labeling are not separated, because for MD-type procedures they
are not separate events -- the dynamics needs the energy and forces to take the step,
so the label is already in hand. Splitting them would either pay the teacher twice or
require a cache across the seam.

A run writes down what it has done, to ``sampler_state.pt`` beside the dataset, and
reads it back if it finds one: pointing ``sample_path`` at a directory that
already holds a dataset continues that dataset rather than refusing it.

The record holds two separate things, and keeping them separate is the point:

* **progress** -- what is on disk. The counters, the byte offset of the end of each
  split file, and whatever the procedure needs to pick up where it stopped.
* **the goal** -- a copy of the ``sampler`` config the dataset was built under, plus a
  hash of the base frames that config named.

A resumed run compares the goal in the record against the live config. At present any
difference at all is refused. Deciding which differences are legal -- adding a strain
magnitude is fine, changing the seed is not -- comes next; until then nothing can go
quietly wrong, because nothing is allowed to change.
"""

import logging
from pathlib import Path
from typing import Optional, Union

from ase import Atoms
from ase.io import read, write

from nequip_extension_template.data.state import (
    check_goal as check_stored_goal,
    frames_digest,
)
from nequip_extension_template.data.store import SampleStore

logger = logging.getLogger(__name__)


class Sampler:
    """Generates teacher-labeled structures into ``sample_path``.

    Parameters
    ----------
    calculator
        The teacher, as an ASE calculator.
    base_frames
        Path to a file ASE can read, holding the structure(s) the procedure starts
        from. These may carry labels of their own; those labels are not used and do not reach the output.
    sample_path
        Output directory. Passed in by ``DistillationDataModule`` from the
        top-level ``sample_path`` config key, not from the ``sampler`` section.
    state_interval
        How many structures to append between writes of the progress record. The
        default writes after every structure.
    """

    def __init__(
        self,
        calculator,
        base_frames: Union[str, Path],
        sample_path: Union[str, Path],
        state_interval: int = 1,
    ):
        self.calculator = calculator
        self.base_frames = read(str(base_frames), index=":")
        self.store = SampleStore(sample_path)
        self.state_interval = int(state_interval)
        if self.state_interval < 1:
            raise ValueError(
                f"`state_interval` must be at least 1, got {state_interval!r}"
            )
        self.n_resumed = 0
        # Set by `DistillationDataModule`, which is what has the config. Not a
        # constructor argument: hydra's `instantiate` recurses into the arguments it
        # is handed looking for things to build, and this dict contains the
        # calculator's own config, so passing it that way would load the teacher
        # twice.
        self.sampler_config: Optional[dict] = None

    # ------------------------------------------------------------------ subclass API

    @property
    def finished(self) -> bool:
        """Whether the procedure has nothing left to do. Defined by the subclass.
        """
        raise NotImplementedError

    def step(self) -> None:
        """Produce, label, and append the next structure(s), and advance state. Defined by the subclass.
        """
        raise NotImplementedError

    def procedure_state(self) -> dict:
        """Whatever this procedure needs to pick up where it stopped.

        Plain numbers, strings, dicts and arrays only -- never the sampler itself and
        never the calculator. 
        
        Empty by default, overwritten by subclass. 
        """
        return {}

    def restore_progress(self, procedure_state: dict) -> None:
        """Pick up where the record says this procedure stopped.

        Called after the base class has restored the counters and cut the split files
        back to their recorded lengths, so the dataset is already in the state the
        record describes by the time this runs.

        Refusing by default is deliberate: a sampler that has not been taught to
        resume must say so rather than start from the beginning and append a second
        copy of everything.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot resume yet. {self.sample_path} already "
            "holds a dataset from an earlier run -- point `sample_path` somewhere "
            "else, or delete it."
        )

    # ----------------------------------------------------------------- dataset files

    @property
    def sample_path(self) -> Path:
        return self.store.sample_path

    @property
    def n_written(self) -> int:
        return self.store.n_written

    @property
    def split_counts(self) -> dict:
        return self.store.split_counts

    def split_file(self, split: str) -> Path:
        return self.store.split_file(split)

    def append(self, atoms: Atoms, split: str) -> None:
        """Append one labeled structure to the named split."""
        self.store.append(atoms, split)

    # ------------------------------------------------------------- progress record

    def provenance(self) -> dict:
        """What defines which run is allowed to continue this dataset.

        Written and checked by the procedure, not by the store: only a procedure
        knows which of its settings are load-bearing. The class name rides along
        here rather than in the record header, so that a dataset produced by a
        different procedure is refused by the same comparison as any other
        incompatible setting.
        """
        return {
            "generator_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "config": self.sampler_config,
            "base_frames": frames_digest(self.base_frames),
        }

    def check_goal(self, stored_goal: dict, n_written: int) -> None:
        """Refuse to continue a dataset whose settings have since changed.

        Future versions should be able to carefully identify settings that ARE
        allowed to change, but that is not yet implemented.

        `n_written` comes from the record rather than from the store, because this
        runs before the store has been reconciled -- the count is only there to say
        how much is at stake.
        """
        live = f"{type(self).__module__}.{type(self).__qualname__}"
        stored_class = stored_goal.get("generator_class")
        if stored_class != live:
            raise ValueError(
                f"{self.sample_path} was sampled by {stored_class}, the config asks "
                f"for {live}. One dataset is the output of one procedure -- point "
                "`sample_path` somewhere else."
            )
        check_stored_goal(
            stored_goal,
            live_config=self.sampler_config,
            base_frames=self.base_frames,
            sample_path=self.sample_path,
            n_written=n_written,
        )

    def write_state(self) -> None:
        """Record the current durable state atomically."""
        self.store.save_record(self.provenance(), self.procedure_state())

    def resume_from(self, record: dict) -> None:
        """Take up the position a record describes.

        Settings are checked before a byte is touched, so a refused run truncates
        nothing and leaves the store's counters alone. Only then does the store cut
        the split files back to their recorded lengths and adopt the counts, and only
        then is the procedure asked to restore itself -- by which point the dataset is
        already in the state the record describes.
        """
        self.check_goal(record["provenance"], record["contents"]["n_written"])
        self.store.reconcile(record["contents"])
        self.n_resumed = self.n_written
        self.restore_progress(record["progress"])

    # --------------------------------------------------------------------- main loop

    def generate(self) -> int:
        """Step until finished, continuing an existing dataset if there is one.

        Returns the number of structures in the dataset, including any that were
        already there before this call. ``n_resumed`` is how many of those predate it.

        The directory is not created up front: the store makes it when the first
        structure or record is written, so a run refused below leaves nothing behind.
        """
        record = self.store.load_record()

        if record is None:
            self.store.refuse_orphan_files()
        else:
            self.resume_from(record)
            counts = ", ".join(f"{s}={n}" for s, n in self.split_counts.items())
            logger.info(
                f"continuing {self.sample_path}: {self.n_written} structure(s) "
                f"already written ({counts})"
            )

        since_write = 0
        while not self.finished:
            self.step()
            since_write += 1
            if since_write >= self.state_interval:
                self.write_state()
                since_write = 0
        self.write_state()
        return self.n_written
