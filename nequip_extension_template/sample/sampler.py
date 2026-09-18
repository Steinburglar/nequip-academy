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
reads it back if it finds one: pointing ``nequip-distill`` at a ``sample_path`` that
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

from nequip_extension_template.data.paths import SPLITS, split_file
from nequip_extension_template.data.state import (
    STATE_FILE,
    STATE_VERSION,
    check_goal as check_stored_goal,
    check_state_header,
    frames_digest,
    flatten,
    read_state as read_state_file,
    refuse_existing_split_files_without_state,
    split_offsets,
    state_file,
    truncate_to as truncate_split_files_to,
    write_state as write_state_file,
)

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
        Output directory. Passed in by the ``nequip-distill`` script from the
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
        self.sample_path = Path(sample_path)
        self.state_interval = int(state_interval)
        if self.state_interval < 1:
            raise ValueError(
                f"`state_interval` must be at least 1, got {state_interval!r}"
            )
        self.n_written = 0
        self.n_resumed = 0
        self.split_counts = {s: 0 for s in SPLITS}
        # Set by the `nequip-distill` script, which is what has the config. Not a
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

    def split_file(self, split: str) -> Path:
        return split_file(self.sample_path, split)

    def append(self, atoms: Atoms, split: str) -> None:
        """Append one labeled structure to the named split."""
        path = self.split_file(split)
        with open(path, "a") as f:
            write(f, atoms, format="extxyz")
        self.n_written += 1
        self.split_counts[split] += 1

    # ------------------------------------------------------------- progress record

    @property
    def state_file(self) -> Path:
        return state_file(self.sample_path)

    def goal_state(self) -> dict:
        """Metadata that defines which run is allowed to continue this dataset."""
        return {
            "version": STATE_VERSION,
            "sampler_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "goal": {
                "config": self.sampler_config,
                "base_frames": frames_digest(self.base_frames),
            },
        }

    def split_offsets(self) -> dict:
        """Byte length of each split file at the moment progress is recorded."""
        return split_offsets(self.sample_path)

    def progress_state(self) -> dict:
        """Mutable progress needed to continue after interruption."""
        return {
            "n_written": self.n_written,
            "split_counts": dict(self.split_counts),
            "offsets": self.split_offsets(),
            "procedure": self.procedure_state(),
        }

    def state_payload(self) -> dict:
        """Full durable state.

        Goal metadata and progress are built separately because they change on
        different timelines, but they are written to one file so each checkpoint is a
        single atomic snapshot.
        """
        payload = self.goal_state()
        payload["progress"] = self.progress_state()
        return payload

    def write_state(self) -> None:
        """Record the current durable state atomically."""
        write_state_file(self.state_file, self.state_payload())

    def read_state(self) -> Optional[dict]:
        """Load the record, or ``None`` if there is none. Refuse one we cannot trust.

        ``weights_only=False`` because this is a record of plain python values, not a
        tensor checkpoint, and nothing writes it but :meth:`write_state`.
        """
        state = read_state_file(self.state_file)
        if state is None:
            return None
        live = f"{type(self).__module__}.{type(self).__qualname__}"
        check_state_header(
            state,
            expected_version=STATE_VERSION,
            expected_class=live,
            sample_path=self.sample_path,
        )
        return state

    def check_goal(self, stored_goal: dict) -> None:
        """Refuse to continue a dataset whose settings have since changed.
            Future versions should be able to carefully identify settings that ARE allowed to change, but that is not yet implemented.
        """
        check_stored_goal(
            stored_goal,
            live_config=self.sampler_config,
            base_frames=self.base_frames,
            sample_path=self.sample_path,
            n_written=self.n_written,
        )

    def truncate_to(self, offsets: dict) -> None:
        """Cut each split file back to the length the record gives for it.

        A file *longer* than its recorded length holds structures appended after the
        record was last written -- the run died between the append and the next
        record write. They are dropped and produced again.

        """
        truncate_split_files_to(self.sample_path, offsets)

    # --------------------------------------------------------------------- main loop

    def generate(self) -> int:
        """Step until finished, continuing an existing dataset if there is one.

        Returns the number of structures in the dataset, including any that were
        already there before this call. ``n_resumed`` is how many of those predate it.
        """
        self.sample_path.mkdir(parents=True, exist_ok=True)
        state = self.read_state()

        if state is None:
            refuse_existing_split_files_without_state(self.sample_path)
        else:
            progress = state["progress"]
            # counters first, so the refusal below can say how much is at stake
            self.n_written = int(progress["n_written"])
            self.n_resumed = self.n_written
            self.split_counts = {s: int(progress["split_counts"][s]) for s in SPLITS}
            self.check_goal(state["goal"])
            # before anything trusts the file lengths
            self.truncate_to(progress["offsets"])
            self.restore_progress(progress["procedure"])
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
