"""Base generator.

Deliberately minimal. All any generator shares is the teacher calculator, the frames
it starts from, and where the dataset is written. Everything else -- how many
structures there are, what a step consists of, when it is finished, and which split
a structure belongs to -- belongs to the procedure, so it belongs to the subclass.

The base class knows only that the dataset is split three ways and where each part
is written. It does not decide what goes where; concrete generators do.

:meth:`Generator.step` does the whole job for one step: produce the next structure or
structures, label them, append them to the dataset, and advance the procedure's own
state. Producing and labeling are not separated, because for MD-type procedures they
are not separate events -- the dynamics needs the energy and forces to take the step,
so the label is already in hand. Splitting them would either pay the teacher twice or
require a cache across the seam.

A run writes down what it has done, to ``generation_state.pt`` beside the dataset, and
reads it back if it finds one: pointing ``dataset_path`` at a directory that
already holds a dataset continues that dataset rather than refusing it.

The record holds two separate things, and keeping them separate is the point:

* **progress** -- what is on disk. The counters, the byte offset of the end of each
  split file, and whatever the procedure needs to pick up where it stopped.
* **the goal** -- a copy of the ``generator`` config the dataset was built under, plus a
  hash of the base frames that config named.

A resumed run compares the goal in the record against the live config. At present any
difference at all is refused. Deciding which differences are legal -- adding a strain
magnitude is fine, changing the seed is not -- comes next; until then nothing can go
quietly wrong, because nothing is allowed to change.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from ase import Atoms
from ase.io import read

from nequip_extension_template.data.state import (
    flatten,
    refuse_changed_settings,
)
from nequip_extension_template.data.store import SampleStore

logger = logging.getLogger(__name__)


def frames_digest(frames: Sequence[Atoms]) -> str:
    """One content hash for a sequence of loaded structures.

    Lives here rather than in the data layer because it answers a question only a
    procedure asks: "am I being told to continue from the same starting material?"
    Note what it hashes -- the parsed numbers, positions, cell and pbc, not the file's
    bytes. Reformatting the input file is therefore fine; moving an atom is not. That
    is the opposite of how the store hashes its own output, where the question really
    is whether the bytes are the ones it wrote.
    """
    h = hashlib.sha256()
    for atoms in frames:
        numbers = atoms.get_atomic_numbers()
        h.update(np.ascontiguousarray(numbers, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(np.round(atoms.get_positions(), 8)).tobytes())
        h.update(np.ascontiguousarray(np.round(np.asarray(atoms.cell), 8)).tobytes())
        h.update(np.ascontiguousarray(atoms.get_pbc()).tobytes())
    return h.hexdigest()[:16]


class Generator:
    """Generates teacher-labeled structures into ``dataset_path``.

    Parameters
    ----------
    calculator
        The teacher, as an ASE calculator.
    base_frames
        Path to a file ASE can read, holding the structure(s) the procedure starts
        from. These may carry labels of their own; those labels are not used and do not reach the output.
    dataset_path
        Output directory. Passed in by ``DistillationDataModule`` from the
        top-level ``dataset_path`` config key, not from the ``generator`` section.
    state_interval
        How many structures to append between writes of the progress record. The
        default writes after every structure.
    """

    def __init__(
        self,
        base_frames: Union[str, Path],
        dataset_path: Union[str, Path],
        calculator=None,
        state_interval: int = 1,
    ):
        self.calculator = calculator
        self.base_frames = read(str(base_frames), index=":")
        self.store = SampleStore(dataset_path)
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
        self.generation_config: Optional[dict] = None

    # ------------------------------------------------------------- teacher lifecycle

    def attach_calculator(self, calculator) -> None:
        """Give this generator the teacher it needs in order to :meth:`step`.

        A generator without a calculator is a legitimate state, not a half-built one:
        it can read a record, check settings, restore its position and answer
        :attr:`finished`. It just cannot produce a new structure. That is what lets a
        caller find out whether a dataset is already complete before paying to load a
        multi-gigabyte model onto a GPU.
        """
        self.calculator = calculator

    def release_calculator(self) -> None:
        """Drop the teacher once no more structures will be produced."""
        self.calculator = None

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

        Plain numbers, strings, dicts and arrays only -- never the generator itself and
        never the calculator. 
        
        Empty by default, overwritten by subclass. 
        """
        return {}

    def restore_progress(self, procedure_state: dict) -> None:
        """Pick up where the record says this procedure stopped.

        Called after the base class has restored the counters and cut the split files
        back to their recorded lengths, so the dataset is already in the state the
        record describes by the time this runs.

        Refusing by default is deliberate: a generator that has not been taught to
        resume must say so rather than start from the beginning and append a second
        copy of everything.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot resume yet. {self.dataset_path} already "
            "holds a dataset from an earlier run -- point `dataset_path` somewhere "
            "else, or delete it."
        )

    # ----------------------------------------------------------------- dataset files

    @property
    def dataset_path(self) -> Path:
        return self.store.dataset_path

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
            "config": self.generation_config,
            "base_frames": frames_digest(self.base_frames),
        }

    def _comparable(self, provenance: dict) -> dict:
        """Flatten one provenance into the named settings a refusal talks about.

        The names are chosen here, not in the data layer, because they are what the
        user reads. A subclass with settings that need different wording, or that are
        not plain config values, overrides this.
        """
        flat = flatten(provenance["config"])
        flat["base frame contents"] = provenance["base_frames"]
        return flat

    def check_compatible(self, stored_provenance: dict, n_written: int) -> None:
        """Refuse to continue a dataset whose settings have since changed.

        The procedure owns this rather than the datamodule because the comparison is
        not always a key diff: a future rattle may accept an added strain magnitude so
        long as the existing structures are untouched, which is a judgement only
        rattle can make. Subclasses that need that override this method; the default
        below is the ordinary case and uses the shared differ.

        No setting is exempt yet -- classification into immutable and mutable comes
        later (`planning.md` 11.10).

        `n_written` comes from the record rather than from the store, because this
        runs before the store has been reconciled; the count is only there to say how
        much is at stake.
        """
        live = f"{type(self).__module__}.{type(self).__qualname__}"
        stored_class = stored_provenance.get("generator_class")
        if stored_class != live:
            raise ValueError(
                f"{self.dataset_path} was sampled by {stored_class}, the config asks "
                f"for {live}. One dataset is the output of one procedure -- point "
                "`dataset_path` somewhere else."
            )
        if self.generation_config is None:
            raise ValueError(
                f"{self.dataset_path} holds a dataset from an earlier run, but this "
                "generator was not given the config it is being asked to continue, so "
                "there is nothing to compare against. `DistillationDataModule` "
                "supplies it; a generator built directly in a script must set "
                "`generation_config` itself."
            )
        refuse_changed_settings(
            self._comparable(stored_provenance),
            self._comparable(self.provenance()),
            dataset_path=self.dataset_path,
            n_written=n_written,
            explanations={
                "base frame contents": (
                    "(the file behind `base_frames` has changed, even if its path "
                    "has not)"
                )
            },
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
        self.check_compatible(record["provenance"], record["contents"]["n_written"])
        self.store.reconcile(record["contents"])
        self.n_resumed = self.n_written
        self.restore_progress(record["progress"])

    # --------------------------------------------------------------------- main loop

    def generate(self, teacher_factory=None) -> int:
        """Step until finished, continuing an existing dataset if there is one.

        Returns the number of structures in the dataset, including any that were
        already there before this call. ``n_resumed`` is how many of those predate it.

        The directory is not created up front: the store makes it when the first
        structure or record is written, so a run refused below leaves nothing behind.

        `teacher_factory` is a zero-argument callable that builds the teacher, and it
        is called only after this method has established that there is something left
        to produce. That ordering is the point: an already-complete dataset can be
        verified -- record read, settings checked, files reconciled -- without loading
        a model onto a GPU for nothing. A caller that already holds a calculator
        should :meth:`attach_calculator` instead and pass nothing here.
        """
        record = self.store.load_record()

        if record is None:
            self.store.refuse_orphan_files()
        else:
            self.resume_from(record)
            counts = ", ".join(f"{s}={n}" for s, n in self.split_counts.items())
            logger.info(
                f"continuing {self.dataset_path}: {self.n_written} structure(s) "
                f"already written ({counts})"
            )

        # The only thing skipped for an already-complete dataset is building the
        # teacher. Everything else below still runs, so a finished run rewrites its
        # record exactly as it always did.
        if not self.finished and self.calculator is None:
            if teacher_factory is None:
                raise ValueError(
                    f"{self.dataset_path} needs more structures, but this "
                    f"{type(self).__name__} has no teacher. Pass `teacher_factory` "
                    "to `generate()`, or call `attach_calculator()` first."
                )
            self.attach_calculator(teacher_factory())

        since_write = 0
        while not self.finished:
            self.step()
            since_write += 1
            if since_write >= self.state_interval:
                self.write_state()
                since_write = 0
        self.write_state()
        return self.n_written
