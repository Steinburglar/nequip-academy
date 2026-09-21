# NequipAcademy — Agent Context

Durable state only. **Design rationale, measured facts about nequip/lightning/ASE, rejected
designs, open problems and the ordered next steps all live in `planning.md`** — read it on
demand, don't duplicate it here.

## Purpose

`nequip` extension package. Ships NO CLI of its own — the user runs plain **`nequip-train`**, and
the package contributes a `data:` datamodule. Given a teacher artifact loadable as an ASE
calculator + base frame(s): **sample** frames (rattle or ASE MD) → **label** with the teacher →
hand the generated splits to nequip's ordinary training.

Finetuning is OUT OF SCOPE (nequip core does it). Scope is a hackathon one-off being cleaned up
for shipping, NOT a broad extensible framework — reject abstraction that only pays off
hypothetically.

**Mid-refactor.** The boundary is being redrawn: physical (bytes on disk) vs semantic (science
and resume position). `planning.md` §11 is the spec and the phase plan — read it before touching
`sample/` or `data/`.

## Working paradigm (READ FIRST, every step)

**SMALL STEPS, ONE AT A TIME. Plan before code, wait for approval between steps.** Before any
code: short plain-language plan — exactly what the step adds AND what it deliberately leaves out.
One step per plan, no bundling, no "while I'm here." No unrequested extras (no smoke tests, no
extra verification, no GPU jobs, no refactors) unless asked. Drop caveman mode for
plans/explanations/design talk, keep it for status. User challenges design premises hard and is
usually right — take the challenge seriously, don't defend.

User explicit, twice: an agent once wrote the whole resume feature in one go (~800 lines + a
26-test suite + an unrequested GPU Slurm job). User: *"i feel like you wrote too much at once for
me to follow it... i need to trust your code and i dont right now."* All reverted, rebuilt from
scratch in small approved steps.

## Code map

`nequip_academy/sample/`
- `generator.py` (~330) — base `Generator`. Owns a `SampleStore` (`self.store`) and delegates ALL
  file/record mechanics to it; holds only calculator, `base_frames`, `state_interval`,
  `n_resumed`, `generation_config`. `dataset_path`/`n_written`/`split_counts` are read-only
  properties off the store. Abstract surface = `finished` (property) + `step()`. Overridable:
  `procedure_state()` (default `{}`), `restore_progress(blob)` (default RAISES
  `NotImplementedError` → a generator not taught to resume refuses instead of duplicating
  everything). Concrete: `append`, `split_file`, `provenance()`, `_comparable(provenance)`,
  `check_compatible(stored, n_written)`, `write_state()`, `resume_from(record)`,
  `attach_calculator`/`release_calculator`, `generate(teacher_factory=None)`.
  **`calculator` is optional.** A generator with none is a valid state: it can read a record,
  check settings, reconcile and answer `finished` — it just cannot `step()`. `generate()` calls
  `teacher_factory` ONLY after deciding it has something left to produce, so a complete dataset
  never loads the model. No teacher and nothing to produce ⇒ clear refusal.
  Module fn `frames_digest(frames)` — hashes the PARSED structures (numbers, rounded
  positions, cell, pbc), NOT file bytes, so reformatting the input file is not a change but
  moving an atom is. Opposite of `store.digests()`, which hashes output bytes.
- `rattle.py` (215) — `RattleGenerator`. Owns `label()` (teacher call + `SinglePointCalculator`).
  Resumes: `procedure_state()` = `{"n_steps": ...}`, nothing else — order is variant-major and
  fixed, and the goal must be unchanged, so the count IS the position. Module fns: `frame_key`,
  `derive_seed`, `isotropic_strain_matrix`, `random_anisotropic_strain_matrix`, `rattle_positions`.
- `md.py` (164) — `MDGenerator`. Langevin NVT, ONE trajectory from `base_frames[0]`, rest ignored
  (warns). CANNOT resume — inherits the base refusal. SCAFFOLDING, not production.
- `split.py` (92) — free fns, NOT base-class methods. `assign_splits(n, split, seed, policy)`,
  `policy` = `"scattered"` (torch shuffle) or `"blocked"` (contiguous train→val→test ranges).
- `__init__.py` — exports `Generator`, `RattleGenerator`, `MDGenerator`.

`nequip_academy/data/`
- `paths.py` — `SPLITS`, `split_file(dataset_path, split)`. **The only home for these** —
  `sample/` no longer re-exports them.
- `state.py` — free fns behind the store: `STATE_FILE`/`STATE_VERSION`, `flatten`,
  `split_offsets`, atomic `read_state`/`write_state`, `refuse_changed_settings`, `truncate_to`,
  `refuse_existing_split_files_without_state`. **Imports no ASE** — `refuse_changed_settings`
  diffs two FLAT `{name: value}` dicts the caller built, so the data layer never computes
  either side and never learns what a base frame is.
- `store.py` — `SampleStore`, the physical layer as an object and **the only thing that touches
  the generated files**. `append` / `offsets` / `digests` / `load_record` / `save_record` /
  `reconcile(contents)` / `refuse_orphan_files`. Record is format **2**, flat:
  `version`+`contents` are the store's, `provenance`+`progress` pass through from the generator.
  `contents` = `n_written`/`split_counts`/`offsets`/`digests`; `reconcile` truncates a torn tail
  then verifies digests. A format-1 record is REFUSED with "regenerate", no migration. `digests()`
  returns `None` for a zero-byte file whether it exists or not — an emptied file and an absent one
  are the same zero structures. `save_record` re-reads all three files, so a huge dataset should
  raise `state_interval`. Creates `dataset_path` lazily, so a refused run leaves nothing behind.
  Its only state is `n_written`/`split_counts`.
- `datamodule.py` — `DistillationDataModule(ASEDataModule)`. Computes the 3 split paths in
  `__init__` and passes them to `ASEDataModule` before the files exist; `prepare_data()`
  generates. Requires `_recursive_: false` so hydra does not build the teacher eagerly. Sets
  `generator.generation_config` AFTER `instantiate` — NOT as a ctor arg, `instantiate`
  recurses into args hunting `_target_` and would build the teacher twice. `prepare_data()` is split into a
  labelled **adapter** half (everything touching `self`; hydra config → objects) and a
  **pipeline** half (local variables only, paste-able into a script — `planning.md` §11.1a).
  Builds the generator ONCE, with no calculator, and hands `generate()` a `teacher_factory`.

Leftovers from the upstream template, delete or replace when touched: `_keys.py` registers two
unused placeholder fields; `model/`, `nn/`, `train/` are README-only stub dirs.

## What works / what refuses

| | State |
|---|---|
| rattle sampling | works, GPU-validated, deterministic per structure |
| 3-file split output | works; splits frozen at generation |
| student training via `nequip-train` | works, e2e-covered; GPU-validated on the old CLI path |
| rattle resume | works (truncate to recorded offset, continue) |
| MD sampling | runs, but scaffolding — see `planning.md` §8 |
| MD resume | REFUSES (base-class `NotImplementedError`) |
| resume onto edited-but-same-length split files | REFUSES (digest check, format 2) |
| resume from a pre-2026-09-18 dataset | REFUSES — format-1 record, regenerate |
| resume w/ ANY generator-config difference | REFUSES, incl. `calculator.device` — deliberate; `immutable_keys()` classification deferred to §11.10 |
| growing a dataset + `ckpt_path` | **NO LONGER GUARDED** — guard died with the CLI, replacement deferred to §11.10 |
| warm start on grown data | not implemented (D11 Path B) |
| MD under the datamodule | untested, no example |
| `student_path` / stable checkpoint dir | not implemented (D15) |
| parallel sweeps into one `dataset_path` | unsafe, no lock |
| compiled `.pt2` teacher | segfaults; use packaged `.nequip.zip` |

## Run commands

```bash
CONDA=/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311
export LD_LIBRARY_PATH=$CONDA/lib:${LD_LIBRARY_PATH:-}
# from repo root; hydra does NOT chdir, so relative paths resolve to launch cwd
$CONDA/bin/nequip-train -cp $PWD/examples -cn rattle_train_datamodule
```

- **`-cp` MUST be absolute.** Hydra resolves a relative `--config-path` against the DECORATED
  FUNCTION'S MODULE, so `-cp testartifacts` fails with `Primary config module
  'nequip.scripts.configs' not found`. Auto-absolutizing it in a wrapper was proposed and
  REJECTED (user).
- **`ckpt_path: null` must NOT appear.** nequip tests for the KEY's presence, not its value, so a
  null value sends the run down the restart path with nothing to load. Omit it entirely.
- Installed `pip install -e . --no-deps --no-build-isolation` into nequip311. Editable, so edits
  are live. `pyproject.toml` declares the license PEP 639 style (`license = "MIT"` +
  `license-files = ["LICENSE"]`), which is why `build-system.requires` is `setuptools>=77` — the
  old `license = {file = ...}` table is deprecated on setuptools 81.
- **All real runs go through Slurm (user explicit, never the login node.)** Smoke-scale generator
  tests are fine on login.
- `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` fixes `GLIBCXX_3.4.31 not found`.

## Test suite

```bash
pytest                                      # default suite, excludes e2e
pytest tests/integration                   # resume/integration tests
pytest -m e2e                              # nequip-train subprocess suite, 7 tests
```

Both CPU only, NO teacher, NO GPU — a correctness decision, not just speed (user explicit).
Calculator is `ase.calculators.lj.LennardJones`, so "resumed == uninterrupted" is a comparison of
file hashes, which also catches a dropped, duplicated or mis-filed structure. **Byte-identity is
NOT a valid criterion whenever the teacher runs on GPU** — see `planning.md` §6 for the measured
nondeterminism and the tolerance-based check to use instead.

- `tests/integration/sample/test_resume.py` — resume machinery. Helpers: `build(...)` mirrors what
  `DistillationDataModule` does (the same settings both construct the generator AND become
  `generation_config`); `kill_after(generator, n)` wraps `step` to raise after exactly n appends, so
  the kill lands at a known point. Mutation-checked: `truncate_to` → no-op ⇒ 2 failures; rattle
  `restore_progress` forgets `n_steps` ⇒ 3; `check_goal` early-return ⇒ 3.
- Portability invariant (before touching `sample/` or `data/`) — `sample/`, `data/store.py`,
  `data/state.py`, `data/paths.py` must import NO nequip/hydra/lightning/omegaconf, must print
  nothing:
```bash
grep -rn "import nequip\|from nequip\b\|import hydra\|from hydra\|import lightning\|from lightning\|omegaconf" \
  nequip_academy/sample/ nequip_academy/data/store.py \
  nequip_academy/data/state.py nequip_academy/data/paths.py
```

- `tests/e2e/test_datamodule_e2e.py` — ordinary pytest tests (no case registry), marked `e2e` and
  `slow`. Each writes a config and runs `python -m nequip.scripts.train` as a SUBPROCESS —
  `@hydra.main` owns global state and does not survive two calls in one interpreter.
  `hydra.run.dir` is pinned per test so assertions can look inside it. 10 synthetic 32-atom fcc
  argon cells (fcc, NOT random positions — random points in a box overlap atoms and LJ energies
  explode); student = 1 layer, `l_max: 0`, `num_features: 8`, 2 epochs, `accelerator: cpu`.
  Covers: full pipeline, two students on one dataset byte-identical, **the no-teacher fast path**
  (a LennardJones subclass that touches a marker file when constructed — the teacher config is
  provenance, so breaking its `_target_` would trip the settings-changed refusal instead of
  proving anything), `split_dataset` refusal, explicit `*_file_path` refusal, broken student
  config refused before generation, checkpoint restart.

## Examples And Local Artifacts

ONE tracked example config: `examples/rattle_train_datamodule.yaml` — `run: [train, val, test]`,
`data:` = `DistillationDataModule` with `_recursive_: false`, `teacher:` and `generation:`; student
= `EMALightningModule` + `NequIPGNNModel` (2 layers, `l_max: 1`, `num_features: [32, 16]`,
`r_max: 6.0` — the CDP student arch from `../distillation/config/base.yaml`), CSVLogger,
`ModelCheckpoint(dirpath=${hydra:runtime.output_dir}, filename=best, save_last=true)`, and NO
`ckpt_path` key at all. Local-only inputs and generated outputs live under gitignored `sandbox/`
paths. `testartifacts/` and `configs/` have been retired. **No MD example any more** — the CLI-era
`md_only.yaml` went with the CLI and MD has not been ported to the datamodule.
- `sandbox/inputs/teacher.nequip.zip` — THE teacher, packaged (not compiled). Copied from
  `../distillation/results/CDP/student_direct/S1b/n200_seed1/model.nequip.zip`.
- `sandbox/inputs/teacher.nequip.pt2` — the segfaulting compiled one, kept as evidence only.
- `sandbox/inputs/base_frames.xyz` — 10 CsH2PO4 frames (64 atoms, periodic), first 10 of a 200-frame REAL
  DFT subset. DFT energy/forces DISCARDED, the teacher relabels. 10 chosen so 0.8/0.1/0.1
  apportions to 8/1/1. Named "base frames" not "seed frames" — collided with `n200_seed1` naming
  AND with the RNG seed.

## Known debt (the shippable-repo cleanup list)

- `pyproject.toml` placeholder: `description = "TODO"`. MIT `LICENSE` and `authors` are done.
- `README.md` is a scaffold with empty sections — **user writes the prose, do not fill it in.**
- `README.md` and `docs/tutorial/` still describe the deleted `nequip-distill` CLI. The tutorial
  is SHIPPED (public Colab link) and is therefore BROKEN until its config and section are ported
  to the `nequip-train` path — user's call, not yet made. Never run the tutorial locally.
- The D11 guard (growing a dataset + `ckpt_path` silently under-trains) went with the CLI. Its
  replacement is the datamodule checkpoint fingerprint, deferred to `planning.md` §11.10. User
  accepted this window knowingly.
- `.pre-commit-config.yaml` exists (ruff line-length 88 double quotes, yamllint, whitespace,
  `fail_fast: true`) but no hook is installed in `.git/hooks/`, and ruff is not in nequip311
  (`No module named ruff`). Lint by hand before committing.
- Latent bug: `rattle.py` strains with `F @ cell`; ASE stores lattice vectors as ROWS so it should
  be `cell @ F.T`. Silent on cubic cells (everything used so far). `planning.md` §7.
- `generation_config` must be set post-hoc (not a ctor arg) so hydra `instantiate` doesn't double-build
  the teacher — means a standalone (non-hydra) script has to set it by hand or resume refuses
  "was not given the config". Clean fix: generator derives its own provenance from ctor args
  instead. Not scheduled. `planning.md` §11.1a.

## Where things are documented

- `planning.md` — design record. §1 original intent, §2–3 decisions D1–D15, §4 on-disk layout,
  §5 rejected designs (**check before re-proposing anything**), §6 measured nequip/lightning/hydra
  facts, §7 rattle + ASE gotchas, §8 open problems, §9 ordered next steps, §10 the tutorial.
- `README.md` — human-facing, scaffold only.
- `docs/tutorial/` — vendored upstream notebook + our distillation section. **NEVER RUN LOCALLY**;
  the user tests it in Colab. `planning.md` §10 for the file map and the build step.
