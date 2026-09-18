"""Contract tests for `SampleStore`.

Two of these matter more than the rest: `SampleStore` must read a record that the
existing `Sampler` wrote, and `Sampler` must read a record that `SampleStore` wrote.
Phase A only earns its keep if the on-disk format is untouched, so both directions are
checked against the real `Sampler` payload builders rather than against a hand-written
dict, which would only prove the test and the store share a misunderstanding.
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
    """A `Sampler` built far enough to use its record builders, and nothing more.

    The directory is created here because `Sampler.append` expects `generate()` to
    have done it already; `SampleStore.append` creates it itself, so that a refused
    run leaves nothing behind.
    """
    base = tmp_path / "base.xyz"
    write(str(base), [frame()])
    sample_path.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(calculator=None, base_frames=base, sample_path=sample_path)
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


def test_saved_record_has_the_version_1_layout_on_disk(tmp_path):
    """Checked on the raw dict, because the round-trip test cannot see this.

    The class name belongs in the header under version 1 and in provenance under the
    new shape. Writing it to both places round-trips fine and is still wrong: it
    changes the bytes of every record, which Phase C would then have to migrate.
    """
    store = SampleStore(tmp_path / "ds")
    store.append(frame(0.0), "train")
    store.save_record({"generator_class": "pkg.Rattle", "config": {"seed": 1}}, {})

    raw = torch.load(store.record_file, weights_only=False)
    assert set(raw) == {"version", "sampler_class", "goal", "progress"}
    assert raw["sampler_class"] == "pkg.Rattle"
    assert raw["goal"] == {"config": {"seed": 1}}
    assert set(raw["progress"]) == {"n_written", "split_counts", "offsets", "procedure"}


def test_load_record_refuses_an_unknown_format(tmp_path):
    path = tmp_path / "ds"
    path.mkdir()
    torch.save({"version": 99}, path / STATE_FILE)
    with pytest.raises(ValueError, match="record format"):
        SampleStore(path).load_record()


# ------------------------------------------- version-1 format compatibility

def test_store_reads_a_record_the_sampler_wrote(tmp_path):
    """The direction that keeps datasets generated before the refactor usable."""
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


def test_sampler_reads_a_record_the_store_wrote(tmp_path):
    """The direction that lets the two coexist while the call sites are switched."""
    sample_path = tmp_path / "ds"
    sampler = bare_sampler(tmp_path, sample_path)

    store = SampleStore(sample_path)
    store.append(frame(0.0), "train")
    store.save_record(sampler.goal_state()["goal"] | {
        "generator_class": "nequip_extension_template.sample.sampler.Sampler"
    }, {})

    state = sampler.read_state()
    assert state["version"] == STATE_VERSION
    assert state["sampler_class"] == "nequip_extension_template.sample.sampler.Sampler"
    sampler.check_goal(state["goal"])
    assert state["progress"]["n_written"] == 1
    assert state["progress"]["procedure"] == {}


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
    }
    fresh = SampleStore(tmp_path / "ds")
    with pytest.raises(ValueError):
        fresh.reconcile(contents)
    assert fresh.n_written == 0


# ------------------------------------------------------------------ orphans

def test_refuse_orphan_files(tmp_path):
    store = SampleStore(tmp_path / "ds")
    store.append(frame(0.0), "train")
    with pytest.raises(FileExistsError, match=STATE_FILE):
        store.refuse_orphan_files()


def test_refuse_orphan_files_allows_an_empty_directory(tmp_path):
    SampleStore(tmp_path / "ds").refuse_orphan_files()
