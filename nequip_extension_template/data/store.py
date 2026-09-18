"""The physical half of a generated dataset: bytes on disk, and the record of them.

`SampleStore` is the only thing in the package that touches the generated files. It
knows about paths, byte offsets, content digests, atomic record writes and torn-write
repair, and it knows nothing at all about sampling. A generator never calls into the
filesystem; it asks the store to append a structure and hands the store an opaque
progress dict to persist.

The record it owns has one writer per section, and that writer is also the section's
checker:

    version       the format of this file, written and checked HERE
    provenance    written and checked by the generator, INCLUDING its own class name
    contents      offsets, digests, counts -- written and checked HERE
    progress      procedure position, written and restored by the generator

The store does not check which procedure produced a dataset. That is provenance, so
the generator carries its own class name there and refuses a mismatch itself; the
store would only be duplicating a check it is worse at explaining. Order makes this
safe: the generator agrees the provenance matches before anything asks it to restore
another procedure's progress.

Offsets and digests are deliberately not part of ``progress``: if they were, a
generator's ``state()`` would have to produce them, which would force procedure code
to know about byte offsets. They are equally not provenance -- provenance is fixed for
the dataset's whole life, while contents change at every checkpoint.

See ``planning.md`` section 11 for the full picture. This module is Phase A: the API
below is the final shape, but it still serializes to the version-1 ``sampler_state.pt``
layout so that datasets generated before the refactor keep working. The three-section
split already exists there, just nested differently and without digests; Phase C bumps
the format rather than changing this interface.
"""

from pathlib import Path
from typing import Optional, Union

from ase import Atoms

from nequip_extension_template.data.paths import SPLITS, split_file


class SampleStore:
    """Owns the split files and the durable record beside them.

    The store's only state is its counters. Offsets come free from ``stat()``, but
    ``n_written`` and ``split_counts`` cannot be recovered from disk without parsing
    every structure, so they are carried in memory, incremented by :meth:`append`, and
    seeded from a record by :meth:`reconcile`.

    Parameters
    ----------
    sample_path
        Directory holding ``train.extxyz`` / ``val.extxyz`` / ``test.extxyz`` and the
        record. Created on demand rather than at construction, so building a store
        costs nothing and refusing a run leaves no directory behind.
    """

    def __init__(self, sample_path: Union[str, Path]) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------- paths

    @property
    def sample_path(self) -> Path:
        """The dataset directory."""
        raise NotImplementedError

    def split_file(self, split: str) -> Path:
        """Where the structures of one split live."""
        raise NotImplementedError

    @property
    def record_file(self) -> Path:
        """Where the durable record lives."""
        raise NotImplementedError

    # ---------------------------------------------------------------- counters

    @property
    def n_written(self) -> int:
        """Structures this store has appended, or been seeded with by a record."""
        raise NotImplementedError

    @property
    def split_counts(self) -> dict:
        """Per-split structure counts, same basis as :attr:`n_written`."""
        raise NotImplementedError

    def offsets(self) -> dict:
        """Current byte length of each split file, zero where the file is absent."""
        raise NotImplementedError

    def digests(self) -> dict:
        """Content hash of each split file, ``None`` where the file is absent.

        Not yet compared against anything: the version-1 record does not carry
        digests. Phase C stores these and makes :meth:`reconcile` check them.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ writing

    def append(self, atoms: Atoms, split: str) -> None:
        """Append one labeled structure to a split and update the counters.

        The only way a structure reaches disk. Generators call this; they never open
        a file themselves.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------- record

    def load_record(self) -> Optional[dict]:
        """Read the record, or ``None`` if there is none.

        Checks ``version`` before returning, because a record in an unknown format
        cannot be interpreted at all. Everything else is left to the generator.

        Does not touch the counters: seeding them is :meth:`reconcile`'s job, and it
        must not happen before the generator has agreed the provenance matches --
        otherwise a refused run has already mutated this store.
        """
        raise NotImplementedError

    def save_record(self, provenance: dict, progress: dict) -> None:
        """Write the record atomically, injecting the ``contents`` section.

        `provenance` and `progress` are the generator's; the store adds the header and
        the byte-level contents, because only it knows them. One file and one
        ``os.replace`` so the sections cannot tear apart from each other.
        """
        raise NotImplementedError

    # -------------------------------------------------------------- consistency

    def reconcile(self, contents: dict) -> None:
        """Bring the files on disk and my own counters into agreement with a record.

        Three things have to agree: the bytes on disk, the record, and this store's
        counters. On a fresh run all three start empty and stay in step. On a resume
        the process is new, so the counters are zero while disk and record may be well
        along; this is what puts them back in step.

        Disk first, in this order, because the order is forced:

        1. a split file shorter than its recorded offset, or missing while the record
           says it holds bytes, means the record and the dataset are not from the same
           run -- refuse, there is nothing safe to do
        2. a split file longer than its recorded offset is a torn write: structures
           appended after the last checkpoint. Truncate; they get produced again
        3. compare digests, once the tail is gone and there is something to compare
           against (Phase C)

        Then seed the counters from the record, since disk and record now agree.

        `contents` is passed in rather than read from a cached record so that the
        datamodule's sequencing stays visible in the datamodule, and so this method
        has no hidden dependency on :meth:`load_record` having been called.
        """
        raise NotImplementedError

    def refuse_orphan_files(self) -> None:
        """Refuse split files that exist with no record beside them.

        Without a record there is no account of what those structures are or what
        produced them, so they can neither be continued nor safely appended to.
        """
        raise NotImplementedError
