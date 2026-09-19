"""Contract tests for `SampleStore`.

The round-trip tests go through the real `Sampler` rather than a hand-written record,
so that they cannot pass by sharing a misunderstanding with the store.
"""

import torch
import pytest
from ase import Atoms
from ase.io import write

from nequip_extension_template.data.paths import SPLITS
from nequip_extension_template.data.state import STATE_FILE, STATE_VERSION
from nequip_extension_template.data.store import SampleStore
from nequip_extension_template.sample.sampler import Sampler


def frame(x: float = 0.0) -> Atoms:
    return Atoms("Ar", positions=[[x, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)


def bare_sampler(tmp_path, sample_path):
    """A `Sampler` built far enough to use its record builders, and nothing more."""
    base = tmp_path / "base.xyz"
    write(str(base), [frame()])
    sampler = Sampler(base_frames=base, sample_path=sample_path)
    sampler.sampler_config = {"seed": 1, "calculator": {"device": "cpu"}}
    return sampler


# ------------------------------------------------------------------ paths

def test_paths_are_derived_from_sample_path(tmp_path):
    store = SampleStore(tmp_path / "ds")
    assert store.sample_path == tmp_path / "ds"
    assert store.split_file("train") == tmp_path / "ds" / "train.extxyz"
    assert store.record_file == tmp_path / "ds" / STATE_FILE


def test_construction_creates_nothing(tmp_path):
    SampleStore(tmp_path / "ds")
    assert not (tmp_path / "ds").exists()


# --------------------------------------------------------------- appending

def test_append_writes_and_counts(tmp_path):
    store = SampleStore(tmp_path / "ds")
    store.append(frame(0.0), "train")
    store.append(frame(1.0), "train")
    store.append(frame(2.0), "val")

    assert store.n_written == 3
    assert store.split_counts == {"train": 2, "val": 1, "test": 0}
    assert store.offsets()["train"] > 0
    assert store.offsets()["test"] == 0


def test_digests_are_none_for_absent_files_and_track_content(tmp_path):
    store = SampleStore(tmp_path / "ds")
    assert store.digests() == {s: None for s in SPLITS}

    store.append(frame(0.0), "train")
    before = store.digests()
    assert before["train"] is not None and before["val"] is None

    store.append(frame(1.0), "train")
    assert store.digests()["train"] != before["train"]


# ------------------------------------------------------------------ record

def test_load_record_returns_none_when_absent(tmp_path):
    assert SampleStore(tmp_path / "ds").load_record() is None


def test_record_round_trips_through_the_three_section_shape(tmp_path):
    store = SampleStore(tmp_path / "ds")
    store.append(frame(0.0), "train")
    store.append(frame(1.0), "val")
    provenance = {"generator_class": "pkg.Rattle", "config": {"seed": 1}}
    store.save_record(provenance, {"n_steps": 7})

    record = SampleStore(tmp_path / "ds").load_record()
    assert record["version"] == STATE_VERSION
    assert record["provenance"] == provenance
    assert record["progress"] == {"n_steps": 7}
    assert record["contents"]["n_written"] == 2
    assert record["contents"]["split_counts"] == {"train": 1, "val": 1, "test": 0}
    assert record["contents"]["offsets"] == store.offsets()


def test_saved_record_has_the_three_section_layout_on_disk(tmp_path):
    """Checked on the raw dict, because the round-trip test cannot see the layout.

    One section per writer: `version` and `contents` are the store's, `provenance` and
    `progress` are the generator's and pass through untouched.
    """
    store = SampleStore(tmp_path / "ds")
    store.append(frame(0.0), "train")
    store.save_record({"generator_class": "pkg.Rattle", "config": {"seed": 1}}, {"n": 1})

    raw = torch.load(store.record_file, weights_only=False)
    assert set(raw) == {"version", "provenance", "contents", "progress"}
    assert raw["version"] == STATE_VERSION
    assert raw["provenance"] == {"generator_class": "pkg.Rattle", "config": {"seed": 1}}
    assert raw["progress"] == {"n": 1}
    assert set(raw["contents"]) == {"n_written", "split_counts", "offsets", "digests"}


def test_load_record_refuses_an_unknown_format(tmp_path):
    path = tmp_path / "ds"
    path.mkdir()
    torch.save({"version": 99}, path / STATE_FILE)
    with pytest.raises(ValueError, match="record format"):
        SampleStore(path).load_record()


# ---------------------------------------------- round trip via the sampler

def test_store_reads_a_record_the_sampler_wrote(tmp_path):
    """What the generator put in provenance comes back out unchanged."""
    sample_path = tmp_path / "ds"
    sampler = bare_sampler(tmp_path, sample_path)
    sampler.append(frame(0.0), "train")
    sampler.append(frame(1.0), "test")
    sampler.write_state()

    record = SampleStore(sample_path).load_record()
    assert record["provenance"]["generator_class"] == (
        "nequip_extension_template.sample.sampler.Sampler"
    )
    assert record["provenance"]["config"] == sampler.sampler_config
    assert "base_frames" in record["provenance"]
    assert record["contents"]["n_written"] == 2
    assert record["contents"]["split_counts"] == {"train": 1, "val": 0, "test": 1}
    assert record["progress"] == {}


def test_sampler_accepts_a_record_the_store_wrote(tmp_path):
    """The direction that lets a later run pick up what an earlier one recorded."""
    sample_path = tmp_path / "ds"
    written = bare_sampler(tmp_path, sample_path)
    written.store.append(frame(0.0), "train")
    written.store.save_record(written.provenance(), {})

    fresh = bare_sampler(tmp_path, sample_path)
    record = fresh.store.load_record()
    assert record["provenance"]["generator_class"] == (
        "nequip_extension_template.sample.sampler.Sampler"
    )
    assert record["contents"]["n_written"] == 1
    # the settings are unchanged, so this is the case that must NOT refuse
    fresh.check_compatible(record["provenance"], record["contents"]["n_written"])


# ------------------------------------------------------------- reconciling

def test_reconcile_truncates_a_torn_tail_and_seeds_the_counters(tmp_path):
    store = SampleStore(tmp_path / "ds")
    store.append(frame(0.0), "train")
    store.save_record({"generator_class": "pkg.X"}, {})
    recorded = store.load_record()["contents"]

    # a structure appended after the record was last written
    store.append(frame(1.0), "train")
    grown = store.offsets()["train"]

    fresh = SampleStore(tmp_path / "ds")
    fresh.reconcile(recorded)
    assert fresh.offsets()["train"] < grown
    assert fresh.offsets()["train"] == recorded["offsets"]["train"]
    assert fresh.n_written == 1
    assert fresh.split_counts == {"train": 1, "val": 0, "test": 0}


def test_reconcile_refuses_a_file_shorter_than_recorded(tmp_path):
    store = SampleStore(tmp_path / "ds")
    store.append(frame(0.0), "train")
    contents = {
        "n_written": 1,
        "split_counts": {"train": 1, "val": 0, "test": 0},
        "offsets": dict(store.offsets(), train=store.offsets()["train"] + 500),
        "digests": store.digests(),
    }
    with pytest.raises(ValueError, match="shorter than"):
        SampleStore(tmp_path / "ds").reconcile(contents)


def test_reconcile_leaves_the_counters_alone_when_it_refuses(tmp_path):
    store = SampleStore(tmp_path / "ds")
    store.append(frame(0.0), "train")
    contents = {
        "n_written": 99,
        "split_counts": {"train": 99, "val": 0, "test": 0},
        "offsets": dict(store.offsets(), train=store.offsets()["train"] + 500),
        "digests": store.digests(),
    }
    fresh = SampleStore(tmp_path / "ds")
    with pytest.raises(ValueError):
        fresh.reconcile(contents)
    assert fresh.n_written == 0


def test_reconcile_refuses_content_that_is_the_right_length_and_wrong_bytes(tmp_path):
    """Offsets cannot catch this; it is the whole reason digests are recorded."""
    store = SampleStore(tmp_path / "ds")
    store.append(frame(0.0), "train")
    store.save_record({"generator_class": "pkg.X"}, {})
    recorded = store.load_record()["contents"]

    # same byte count, different structure
    path = store.split_file("train")
    swapped = path.read_bytes()
    path.write_bytes(swapped[:-2] + b"9\n")
    assert path.stat().st_size == recorded["offsets"]["train"]

    with pytest.raises(ValueError, match="not the recorded content"):
        SampleStore(tmp_path / "ds").reconcile(recorded)


def test_reconcile_accepts_an_empty_file_recorded_as_absent(tmp_path):
    """Truncating back to offset zero leaves a file that exists and is empty.

    The record, written before that split was ever touched, has `None` for it. Those
    are the same zero structures and must not be read as a content mismatch.
    """
    store = SampleStore(tmp_path / "ds")
    store.append(frame(0.0), "train")
    store.save_record({"generator_class": "pkg.X"}, {})
    recorded = store.load_record()["contents"]
    assert recorded["digests"]["test"] is None

    store.append(frame(1.0), "test")          # a split the record knows nothing about
    fresh = SampleStore(tmp_path / "ds")
    fresh.reconcile(recorded)                 # truncates test.extxyz to empty
    assert fresh.split_file("test").exists()
    assert fresh.n_written == 1


# ------------------------------------------------- refusing older formats

def test_load_record_refuses_a_format_1_record(tmp_path):
    """The pre-refactor layout. No migration: contents and digests are not in it."""
    path = tmp_path / "ds"
    path.mkdir()
    torch.save(
        {
            "version": 1,
            "sampler_class": "pkg.Rattle",
            "goal": {"config": {"seed": 1}, "base_frames": "a1b2"},
            "progress": {
                "n_written": 1,
                "split_counts": {"train": 1, "val": 0, "test": 0},
                "offsets": {"train": 10, "val": 0, "test": 0},
                "procedure": {},
            },
        },
        path / STATE_FILE,
    )
    with pytest.raises(ValueError, match="format-1 record"):
        SampleStore(path).load_record()


# ------------------------------------------------------------------ orphans

def test_refuse_orphan_files(tmp_path):
    store = SampleStore(tmp_path / "ds")
    store.append(frame(0.0), "train")
    with pytest.raises(FileExistsError, match=STATE_FILE):
        store.refuse_orphan_files()


def test_refuse_orphan_files_allows_an_empty_directory(tmp_path):
    SampleStore(tmp_path / "ds").refuse_orphan_files()
