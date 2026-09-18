"""Path helpers for generated distillation datasets."""

from pathlib import Path
from typing import Union

SPLITS = ("train", "val", "test")


def split_file(sample_path: Union[str, Path], split: str) -> Path:
    """Return the extxyz path for one generated dataset split."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {list(SPLITS)}")
    return Path(sample_path) / f"{split}.extxyz"
