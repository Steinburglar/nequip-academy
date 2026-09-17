"""Assigning things to train/val/test.

Not part of the base :class:`~.sampler.Sampler`. The base class only creates the
three empty boxes and knows where each is written; each procedure decides what fills
them, and this labels ``n`` of whatever that procedure considers independent.

What ``n`` counts differs by procedure, and that is the whole point. Rattling splits
per *base frame* -- the structures rattled from one base frame are near-duplicates,
so they must not be separated. MD splits per *snapshot*, on the assumption that the
sampling interval is long enough to decorrelate consecutive snapshots.

The apportionment itself is ``torch.utils.data.random_split`` -- floor each fraction,
distribute the remainder round-robin. That is also what nequip's own splitting uses
(``nequip/data/dataset/utils.py``), so a ``nequip-distill`` split and a
``nequip-train`` split produce the same sizes. Only *which* items land where differs,
by policy:

``scattered``
    Shuffle, then hand out blocks -- ``random_split``'s own behaviour, and what
    ``nequip-train`` does. Right when the items are exchangeable, e.g. base frames
    that have no meaningful order.

``blocked``
    Hand out contiguous ranges in order: train first, then val, then test. Right when
    the items are ordered and neighbours are correlated -- snapshots along one MD
    trajectory. A scattered validation snapshot sits between two training snapshots,
    so if the sampling interval does not fully decorrelate, scattered flatters the
    validation metric and blocked does not. This is the usual time-series holdout.
"""

from typing import List, Optional

import torch

from .sampler import SPLITS

DEFAULT_SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}


def assign_splits(
    n: int,
    split: Optional[dict],
    seed: int,
    policy: str = "scattered",
) -> List[str]:
    """Assign ``n`` items to splits, returning one split name per item index.

    ``policy`` is ``"scattered"`` or ``"blocked"``; see the module docstring. Sizes are
    identical either way, only the membership differs. ``seed`` affects ``"scattered"``
    only -- ``"blocked"`` is fully determined by the item order.
    """
    if policy not in ("scattered", "blocked"):
        raise ValueError(
            f"unknown split policy {policy!r}, expected 'scattered' or 'blocked'"
        )
    fractions = dict(DEFAULT_SPLIT if split is None else split)
    unknown = set(fractions) - set(SPLITS)
    if unknown:
        raise ValueError(
            f"unknown split names {sorted(unknown)}, expected {list(SPLITS)}"
        )
    fractions = {s: float(fractions.get(s, 0.0)) for s in SPLITS}

    generator = torch.Generator().manual_seed(seed)
    subsets = torch.utils.data.random_split(
        range(n), [fractions[s] for s in SPLITS], generator=generator
    )

    # torch only warns when a split comes out empty, and the symptom would surface far
    # away -- as a training run with no validation set. The cause is always the same
    # and is worth naming here.
    empty = [
        s for s, subset in zip(SPLITS, subsets) if fractions[s] > 0 and not len(subset)
    ]
    if empty:
        raise ValueError(
            f"splitting {n} item(s) as {fractions} leaves {empty} empty. Increase the "
            "number of items the procedure splits on, or zero out a split fraction."
        )

    assignment: List[Optional[str]] = [None] * n
    if policy == "scattered":
        for name, subset in zip(SPLITS, subsets):
            for index in subset.indices:
                assignment[index] = name
    else:
        cursor = 0
        for name, subset in zip(SPLITS, subsets):
            for index in range(cursor, cursor + len(subset)):
                assignment[index] = name
            cursor += len(subset)
    return assignment
