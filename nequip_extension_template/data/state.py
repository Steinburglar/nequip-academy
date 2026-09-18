"""Durable state helpers for generated distillation datasets.

Free functions behind :class:`~nequip_extension_template.data.store.SampleStore`.
Nothing here knows what a structure is: this layer deals in paths, bytes and plain
dicts, and anything that must be loaded or hashed to be compared is reduced to a value
by the generator that owns it before it gets here.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import torch

from nequip_extension_template.data.paths import SPLITS, split_file

logger = logging.getLogger(__name__)

STATE_FILE = "sampler_state.pt"
STATE_VERSION = 1


def state_file(sample_path: Union[str, Path]) -> Path:
    """Return the durable generation state path for a sample directory."""
    return Path(sample_path) / STATE_FILE


def flatten(value, prefix: str = "") -> dict:
    """Convert a nested config dictionary to dotted-key flat form."""
    if isinstance(value, dict) and value:
        flat = {}
        for key, item in value.items():
            flat.update(flatten(item, f"{prefix}{key}."))
        return flat
    return {prefix.rstrip("."): value}


def split_offsets(sample_path: Union[str, Path]) -> dict:
    """Return byte lengths for all split files in a generated dataset."""
    return {
        split: (
            split_file(sample_path, split).stat().st_size
            if split_file(sample_path, split).exists()
            else 0
        )
        for split in SPLITS
    }


def write_state(path: Union[str, Path], payload: dict) -> None:
    """Write a generation state record atomically."""
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def read_state(path: Union[str, Path]) -> Optional[dict]:
    """Load a generation state record, or ``None`` if it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, weights_only=False)


def refuse_changed_settings(
    stored: dict,
    live: dict,
    *,
    sample_path: Union[str, Path],
    n_written: int,
    explanations: Optional[dict] = None,
) -> None:
    """Refuse to continue a dataset whose settings have since changed.

    Pure mechanism: `stored` and `live` are FLAT ``{name: value}`` dicts and the names
    are the caller's own wording, because only the caller knows what its settings are
    called or which of them matter. `explanations` adds a trailing note to named keys,
    for the cases where the values alone do not say what went wrong.

    This layer deliberately cannot compute either side. Anything that needs to be
    loaded, parsed or hashed to be compared -- base frames, teacher artifacts -- is
    reduced to a value by whoever owns it before it arrives here.
    """
    explanations = explanations or {}
    differences = []
    for key in sorted(set(stored) | set(live)):
        before = stored.get(key, "<not set>")
        after = live.get(key, "<not set>")
        if before != after:
            line = f"  {key}: {before!r} -> {after!r}"
            if key in explanations:
                line = f"{line} {explanations[key]}"
            differences.append(line)
    if differences:
        joined = "\n".join(differences)
        raise ValueError(
            f"{sample_path} holds {n_written} structure(s) produced under different "
            f"settings:\n{joined}\n"
            "A resumed run can only continue a dataset it would have produced "
            "itself. Put these back, or point `sample_path` somewhere else."
        )


def truncate_to(sample_path: Union[str, Path], offsets: dict) -> None:
    """Cut split files back to the byte lengths recorded in durable state."""
    sample_path = Path(sample_path)
    for split in SPLITS:
        path = split_file(sample_path, split)
        offset = int(offsets[split])
        if not path.exists():
            if offset:
                raise FileNotFoundError(
                    f"{state_file(sample_path)} says {split} holds {offset} bytes, "
                    f"but {path} does not exist."
                )
            continue
        size = path.stat().st_size
        if size < offset:
            raise ValueError(
                f"{path} is {size} bytes, shorter than the {offset} bytes "
                f"{state_file(sample_path)} records for it. The record and the "
                f"dataset in {sample_path} are not from the same run -- point "
                "`sample_path` somewhere else."
            )
        if size > offset:
            logger.warning(
                f"{path}: dropping the last {size - offset} byte(s), appended after "
                "the record was last written; they will be produced again."
            )
            with open(path, "r+b") as f:
                f.truncate(offset)


def existing_split_files(sample_path: Union[str, Path]) -> list[str]:
    """Return generated split files that already exist."""
    sample_path = Path(sample_path)
    return [
        str(split_file(sample_path, s))
        for s in SPLITS
        if split_file(sample_path, s).exists()
    ]


def refuse_existing_split_files_without_state(sample_path: Union[str, Path]) -> None:
    """Raise if split files exist without their durable state record."""
    existing = existing_split_files(sample_path)
    if existing:
        raise FileExistsError(
            f"{existing} already exist, but {STATE_FILE} does not. Without it there "
            "is no record of what those structures are or what settings produced "
            "them, so they can neither be continued nor safely appended to -- delete "
            "them, or point `sample_path` somewhere else."
        )
