# NequIP Academy — Agent Context

Durable state only. This is the ONLY context file — `planning.md` was the refactor design record
and was deleted once that refactor finished (2026-09-21). Its history is in git
(`git show 9ec7030:planning.md`) if an argument ever needs reconstructing; the verdicts
that still bind are recorded below.

## Purpose

`nequip` extension package. Ships NO CLI of its own — the user runs plain **`nequip-train`**, and
the package contributes a `data:` datamodule. Given a teacher artifact loadable as an ASE
calculator + base frame(s): **sample** frames (rattle or ASE MD) → **label** with the teacher →
hand the generated splits to nequip's ordinary training.

Finetuning is OUT OF SCOPE (nequip core does it). Scope is a hackathon one-off being cleaned up
for shipping, NOT a broad extensible framework — reject abstraction that only pays off
hypothetically.

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
  recurses into args hunting `_target_` and would build the teacher twice. `prepare_data()` is
  split into a labelled **adapter** half (everything touching `self`; hydra config → objects) and
  a **pipeline** half (local variables only, paste-able into a script).
  Builds the generator ONCE, with no calculator, and hands `generate()` a `teacher_factory`.

Leftovers from the upstream template, delete or replace when touched: `_keys.py` registers two
unused placeholder fields; `model/`, `nn/`, `train/` are README-only stub dirs.

**Portability invariant.** `sample/`, `data/store.py`, `data/state.py`, `data/paths.py` must
import NO nequip/hydra/lightning/omegaconf and must print nothing — generating data must not
require nequip. Check before touching them:
```bash
grep -rn "import nequip\|from nequip\b\|import hydra\|from hydra\|import lightning\|from lightning\|omegaconf" \
  nequip_academy/sample/ nequip_academy/data/store.py \
  nequip_academy/data/state.py nequip_academy/data/paths.py
```

## What works / what refuses

| | State |
|---|---|
| rattle sampling | works, GPU-validated, deterministic per structure |
| 3-file split output | works; splits frozen at generation |
| student training via `nequip-train` | works, e2e-covered; GPU-validated on the old CLI path |
| rattle resume | works (truncate to recorded offset, continue) |
| MD sampling | runs, but scaffolding |
| MD resume | REFUSES (base-class `NotImplementedError`) |
| resume onto edited-but-same-length split files | REFUSES (digest check, format 2) |
| resume from a pre-2026-09-18 dataset | REFUSES — format-1 record, regenerate |
| resume w/ ANY generator-config difference | REFUSES, incl. `calculator.device` — deliberate |
| growing a dataset + `ckpt_path` | **NO LONGER GUARDED** — guard died with the CLI |
| warm start on grown data | not implemented |
| MD under the datamodule | untested, no example |
| `student_path` / stable checkpoint dir | not implemented |
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
- Hydra `_target_` values are import paths, independent of cwd. Ordinary file paths in YAML
  (`model_path`, `base_frames`, `dataset_path`) resolve against process cwd, not config location.
- Installed `pip install -e . --no-deps --no-build-isolation` into nequip311. Editable, so edits
  are live. `pyproject.toml` declares the license PEP 639 style (`license = "MIT"` +
  `license-files = ["LICENSE"]`), which is why `build-system.requires` is `setuptools>=77`.
- **All real runs go through Slurm (user explicit, never the login node.)** Smoke-scale generator
  tests are fine on login. Interactive GPU: `salloc -p gpu_test --gres=gpu:1`, then run the plain
  command inside the allocation (no nested `srun` — it mangles interactive output for a
  single-process job).
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
NOT a valid criterion whenever the teacher runs on GPU** — see the determinism note below.

- `tests/integration/sample/test_resume.py` — resume machinery. Helpers: `build(...)` mirrors what
  `DistillationDataModule` does (the same settings both construct the generator AND become
  `generation_config`); `kill_after(generator, n)` wraps `step` to raise after exactly n appends, so
  the kill lands at a known point. Mutation-checked: `truncate_to` → no-op ⇒ 2 failures; rattle
  `restore_progress` forgets `n_steps` ⇒ 3; `check_goal` early-return ⇒ 3.
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
`ckpt_path` key at all. **No MD example any more** — the CLI-era `md_only.yaml` went with the CLI
and MD has not been ported to the datamodule.

Local-only inputs and generated outputs live under gitignored `sandbox/`. `testartifacts/` and
`configs/` are retired; do not recreate them. Promote sandbox material only when its role is
clear: runnable example configs → `examples/`, test inputs → `tests/fixtures/`, tutorial assets →
`docs/tutorial/`.
- `sandbox/inputs/teacher.nequip.zip` — THE teacher, packaged (not compiled). Copied from
  `../distillation/results/CDP/student_direct/S1b/n200_seed1/model.nequip.zip`. It is really a
  *student* model — fine for plumbing, not a real teacher, so no quoted number off it means
  anything. Sanity check: teacher E vs DFT E on unrattled frame 0 = −374.61 vs −374.51 eV
  (64 atoms).
- `sandbox/inputs/teacher.nequip.pt2` — the segfaulting compiled one, kept as evidence only.
- `sandbox/inputs/base_frames.xyz` — 10 CsH2PO4 frames (64 atoms, periodic), first 10 of a
  200-frame REAL DFT subset. DFT energy/forces DISCARDED, the teacher relabels. 10 chosen so
  0.8/0.1/0.1 apportions to 8/1/1. Named "base frames" not "seed frames" — collided with
  `n200_seed1` naming AND with the RNG seed.
- `sandbox/configs/rattle_train_datamodule_local.yaml` (untracked) — same semantics as the tracked
  example, paths relative to `sandbox/out`. Run from there:
  `cd sandbox/out && nequip-train -cp $PWD/../configs -cn rattle_train_datamodule_local`.

## Design decisions that still bind

Everything else is readable from the code. These are the ones whose *reason* is not.

- **Splits frozen at generation, 3 files, never recomputed at train time.** This is what stops a
  grown dataset from silently moving val/test frames into train.
- **Split unit is per-procedure.** Rattle splits per BASE FRAME (rattles of one frame are
  near-duplicates, must stay together); MD per snapshot. Defaults differ on purpose: MD
  `"blocked"` (time-ordered, a contiguous tail stays honest if `sample_interval`
  under-decorrelates), rattle `"scattered"` (base frames have no order). Apportionment reuses
  `torch.utils.data.random_split` — don't reimplement; `split.py` adds only index→label inversion
  and a hard `ValueError` on an empty split where torch merely warns.
- **Provenance is a COPY of the generator config, not a hand-picked param list.** A hand-written
  list silently omits any newly added kwarg — exactly how the `anisotropic_strain_magnitude` bug
  happened. Two things the config can't express are stored beside it: `base_frames` is a path
  (hence `frames_digest` over loaded structures), and MD uses only `base_frames[0]` so its digest
  is over-strict (unaddressed).
- **Live config wins on resume; the stored goal is a diff baseline and legality guard, never
  behavior-driving.**
- **Never pickle the generator or calculator.** A pickle carries a stale config (live config
  wins), CUDA-bound models aren't portable across nodes/GPU archs, and a pickle couples to class
  layout so a renamed attribute breaks every existing `dataset_path` silently — a plain dict
  breaks loudly on the version check.
- **RNG asymmetry is load-bearing.** Rattle has NO streaming RNG: each structure is seeded from
  `derive_seed(seed, frame_key(frame), variant_label)`, so a structure depends only on its own
  identity, never on generation order — nothing to checkpoint. Variant identity is a LABEL
  (`"iso:-0.05"`, `"aniso:0"`) not a position, so adding a strain magnitude doesn't renumber
  existing variants. MD must keep one streaming RNG (snapshot n only exists by integrating
  1..n-1), so MD resume needs `rng.bit_generator.state` checkpointed with positions/velocities.
- **`state_interval` isn't in the yaml**, so it needs `+generator.state_interval=N` (hydra ADD,
  not override) — and a run started that way must be resumed with the same flag, or the live
  config is missing the key and the diff refuses.
- **Repo layout**: `nequip_academy/` code, `examples/` shipped configs, `tests/` verification,
  `docs/tutorial/` shipped tutorial, `sandbox/` gitignored playground.
- **`dataset_path` lives beside the hydra run folder, never nested either direction** — different
  lifetimes. Never point `hydra.run.dir` at or inside `dataset_path`: hydra creates its run dir at
  command start, so the generator then sees a non-empty dir with no state file and hard-errors on
  every fresh run (tested). The reverse nesting is safe but pointless.

## Rejected — do not re-propose

- Separate `labeler` slot (user).
- A shared `label()` on the `Generator` base class (user, 2026-09-19). Rattle calls the teacher
  itself; MD reads energy/forces the dynamics already computed. Producing and labeling are not
  separable for MD-type procedures, so hoisting would either pay the teacher twice or force a
  fake label step.
- `warm_start_from` as a second checkpoint key (user). One `ckpt_path`.
- A single `runs.jsonl` provenance file — split-brain. Each stage's record lives with its own
  artifact.
- Two-command workflow (`[sample]` then `[train,val,test]`) as the sweep-race fix — defeats the
  one-command product.
- Two hydra processes in one command — tested, both resolve to the SAME output folder when run in
  the same second.
- A second unconditional `ModelCheckpoint(monitor: null, filename: latest)` to fix the stale
  `last.ckpt` (user, 2026-08-28): *"this is nequip behavior, which we dont want to change."*
- Auto-absolutizing `-cp` in a wrapper (user, 2026-08-28) — match nequip, don't diverge.
- GPU tests in the automated suite (user) — correctness decision, see the determinism note.
- An incremental hash carried across appends to make `save_record` linear — optimising for a size
  nobody runs.
- Aliases for the old `Sampler`/`sample_path` names — unreleased package, no external importers.

## Measured facts — nequip / lightning / hydra

Env: nequip **0.19.1**, lightning 2.6.1, hydra 1.3.2, torch 2.11.0+cu128. `pyproject.toml` says
`nequip>=0.17.1`, no upper bound (user wants 0.19 usable).

- **Teacher → ASE calculator.** `nequip.integrations.ase.NequIPCalculator` (`nequip/ase/` is a
  deprecated shim). `_from_saved_model(...)` is the ONLY path for `.nequip.zip` or raw `.ckpt`;
  `from_compiled_model(...)` is `nequip-compile` output only.
- **`run` accepts only** `train`/`val`/`test`/`predict`, at most one `train`.
- **`ASEDataModule` takes `test_file_path`/`val_file_path` as `str` OR `List[str]`** and the base
  datamodule supports multiple test datasets natively — hence the `val0`/`test0` metric naming.
  In-tree precedent: `samd23_datamodule.py` appends an OOD test set, `_3bpa_datamodule.py` passes
  a list.
- **On a `ckpt_path` restart:** `training_module` comes from the checkpoint (live config ignored,
  warned); `data`, `trainer` and `run` come from the live config; the checkpoint's `run_stage`
  only sets the START INDEX. Warm start is instead `ModelFromCheckpoint`/`ModelFromPackage` as the
  `model` builder — no conflict guard, only a version-string warning.
- **`run_stage` carries almost no information.** It advances after each stage returns, but only
  Lightning writes checkpoints and only during `fit`, so every checkpoint from a `train`-first run
  holds `run_stage == 0` however far it got (verified on a finished 20-epoch GPU run:
  `epoch=19`, `run_stage=0`). A restart therefore always replays train/val/test; `train` returns
  immediately when the restored epoch is already `max_epochs`.
- **`last.ckpt` IS NOT THE NEWEST EPOCH under a monitored callback.** Measured in pure lightning
  2.6.1: with `monitor=<metric>, save_top_k=1` (our config AND nequip's tutorial), `last.ckpt` is
  the BEST epoch, not the last. `save_last` is relative to SAVES, not epochs, and with a `monitor`
  a save only fires on improvement. `monitor=None` or `save_top_k=-1` give the newest.
  Consequence: `ckpt_path=.../last.ckpt` resumes from the last IMPROVING epoch and silently drops
  everything after — harmless while improving, expensive on a plateau. Fix rejected above.
- **Lightning checkpoint versioning must be OFF for a shared student dir.** With
  `enable_version_counter=True` (default) a second run into the same `dirpath` writes
  `best-v1.ckpt`, and unsuffixed `last.ckpt` is then the OLDEST. Precondition for any future
  `student_path`.
- **`ModelCheckpoint.state_dict()` persists `best_model_score`/`best_model_path`**, restored on a
  `ckpt_path` resume — but a warm start has no `ckpt_path`, so best-tracking restarts from zero
  and can overwrite `best.ckpt` with something worse, irreversibly if versioning is off.
- **`exclude_keys` is a NON-ISSUE for our provenance keys — do not "fix" it.** `nequip/data/ase.py`
  reads only ASE calculator properties plus the user's `include_keys` out of
  `atoms.info`/`atoms.arrays`, so `base_frame`, `base_frame_key`, `variant`, `md_step` are never
  looked at. The `../distillation` crash was `dipole`/`free_energy` — real ASE properties present
  on only some frames.
- **Student test metrics land on disk** — CSVLogger writes `test0_epoch/forces_mae`,
  `test0_epoch/per_atom_energy_rmse` etc. to `metrics.csv` under the hydra output dir.
- **Compiled artifacts are BLOCKED.** TorchScript compile is dead (`not supported in PyTorch >=
  2.10`), so AOTInductor is the only mode and its output is GPU-arch-locked. `nequip-compile
  --target ase` `.pt2` SEGFAULTS — at first forward on the CDP model, inside the compile itself on
  OAM-S. Suspect the `aten._linalg_det.default is missing a c-shim implementation` warning
  (`linalg_det` is in no c-shim header in torch 2.11 and is unavoidable — every periodic model
  hits `torch.linalg.det(cell)` for the stress volume), but that is correlation, not proof. The
  packaged `.nequip.zip` of the same model works. Compiling from a `.ckpt` also fails
  (`FileNotFoundError`, rebuilds the datamodule); from a package it doesn't.
- **Teacher determinism on GPU.** Two runs over BIT-IDENTICAL geometry disagree by ~3e-6 eV /
  ~4e-6 eV/Å (float reductions aren't associative). For a manual GPU check use: same structure
  set, same split per structure, positions/cell exactly equal, |dE| and |dF| < 1e-5.
- `nequip-package` output must never be relocated after creation — path baked in.

## Rattle + ASE gotchas

- **Algorithm** (from `../distillation/scripts/gen_synthetic_geoms.py`, don't reinvent): per base
  frame, one structure per `strain_magnitudes` entry (isotropic volume scaling `(1+strain)^(1/3)`)
  plus `n_random_strain_samples` random anisotropic strains, each followed by a per-atom rattle
  bounded by `max_displacement_ang`. Ordering is variant-major: every base frame gets variant 0
  before any gets variant 1.
- **LATENT BUG, unconfirmed impact — ASE stores lattice vectors as ROWS.** Straining should be
  `cell @ F.T`; `rattle.py` does `F @ cell`. Identical for a CUBIC cell (everything used so far is
  cubic), diverges for hexagonal/triclinic. Fix before any non-cubic base frame.
- **`anisotropic_strain_magnitude` is decoupled from `strain_magnitudes` on purpose.** The derived
  form (`max(abs(strain_magnitudes))`) coupled two knobs: adding one isotropic scan point silently
  widened the anisotropic distribution with nothing to detect it. Default 0.05 preserves the old
  value.
- **Magnitude lesson, hard-won**: ±10%/5% strain + 0.5 Å displacement pushed frames OOD for the
  teacher and students came out WORSE than with no synthetic data at all. Halved defaults
  (±5%/2.5%, 0.25 Å — current) fixed it. Displacement is measured against the STRAINED parent,
  not the unstrained one.
- **Snapshot every kept frame with `atoms.copy()`** — ASE MD mutates one `Atoms` in place.
- **`atoms.copy()` drops `atoms.calc`** — reattach `SinglePointCalculator(snap, energy=e,
  forces=f)`, or extxyz gets geometry with no labels and you find out at student-training time.
- ASE `Langevin(fixcm=True)` (current `md.py` default) does NOT strictly sample NVT, deprecated
  since ASE 3.28; fix is `fixcm=False` + `ase.constraints.FixCom`. Left alone while MD is
  scaffolding; a real correctness issue the moment MD output is trusted.
- **Determinism-test trap:** `frames[0].copy()` has the same content hash as `frames[0]` by
  construction, so testing "adding a base frame leaves existing structures identical" with a
  copied frame passes trivially. Use a genuinely unused frame.
- numpy 2: `ndarray.ptp()` is gone, use `np.ptp(arr)`. Don't name a scratch script `inspect.py` —
  shadows stdlib, breaks numpy/ase import.

## Backlog

Unordered. Each line is the verdict; the argument is gone on purpose.

**Science**
- **Test error means error vs the TEACHER — wrong default.** SETTLED (user, 2026-09-21): one
  DFT-labelled test set supplied by the user; generator makes train+val only (`test: 0.0`);
  implement by relaxing the datamodule's explicit-`*_file_path` refusal for `test` only. No
  teacher inference at test time (the teacher still labels train/val in `prepare_data()`). All
  teacher comparison is post-hoc — student numbers are already in `metrics.csv`, so the tool only
  evaluates the teacher. Report teacher-vs-DFT as the floor; student-vs-DFT alone can't separate
  "distillation failed" from "teacher was already bad here". *Rejected:* two test dataloaders
  (works natively, but positional `test0`/`test1` naming is ambiguous to a reader); teacher
  inference in the test stage (puts a student-independent constant into a per-epoch metrics
  stream, and needs custom field registration + ASE reader changes + a metric that ignores the
  model output); a "pre" script writing teacher labels (duplicates the post-hoc tool).
  *Deferred:* carrying teacher labels through the pipeline for side-by-side test output — revisit
  only if reading two places annoys in practice; teacher columns would need clearly non-standard
  names so such a file wandering into `train_file_path` fails loudly.
- **Post-hoc grading tool.** Teacher + one or more students + a labelled test file → table and a
  Pareto plot of accuracy vs inference time. NOT `nequip-distill` returning: no run stages, no
  hydra schema, no resume semantics. Timing must be honest — fixed batch size, warmup discarded,
  median over repeats with spread, one GPU, compiled/uncompiled stated. Parameter count is not a
  substitute x-axis; it misses what `l_max` costs.
- **Energy zero.** Teacher and DFT labels may sit on different absolute references while the
  student's per-type shifts are fit from teacher-labelled data, so total-energy MAE against an
  external DFT set can be dominated by a constant offset. Forces are immune. Decide raw vs
  per-atom vs shift-corrected. Doesn't bite the sandbox teacher (same DFT); bites a fine-tuned
  foundation model, which is the real use case.
- **Strain bug** — `F @ cell` should be `cell @ F.T`. Silent on cubic cells.
- **MD is unmeasured.** `sample_interval` decorrelation is an assumption (if false, scattered
  splitting leaks and nothing detects it); `friction_per_fs=0.01` and the 5-step equilibration are
  placeholders, not recommendations.
- *Optional, not required (user):* a leakage check that the supplied DFT test set doesn't overlap
  the base frames. Datamodule is the natural home (it holds the frames, sees the test path, and
  `frames_digest` exists); post-hoc would be too late. Skipped because most implementations make
  no such guarantee and train/test hygiene is the user's job. Related caveat worth documenting:
  a near-equilibrium test set says little about the rattled, strained regime the student trained
  on.

**Restart / compatibility**
- **`immutable_keys()` classification.** Today ANY config difference is fatal. *Immutable:*
  generator class, base-frame contents, teacher identity, any knob affecting written structures,
  split policy/seed/fractions. *Mutable:* teacher device, `state_interval`, dataloader settings,
  transforms/stats, a coherently moved output path. *Conditionally mutable:* sample budget, added
  rattle variants, added base frames, teacher path when a hash proves it's the same teacher.
  Unclassified defaults to fatal. Messages must name which immutable settings changed and which
  mutable ones were ignored.
- **Teacher identity separate from plumbing.** `strict_hash` (hash the artifact) / `metadata`
  (path+size+mtime) / explicit id for non-file calculators. **Device is never identity** —
  `cuda` → `cpu` must not invalidate a finished dataset.
- **Training checkpoint fingerprint** — replaces the dead CLI guard, as native datamodule state:
  `state_dict()` stores a dataset fingerprint, `load_state_dict()` refuses a mismatch. Semantics:
  training crash + unchanged dataset resumes; sampling crash resumes generation then trains;
  extended dataset + `ckpt_path` refuses; extended + warm start is an explicit
  model-from-checkpoint path, never `ckpt_path`.
- **MD resume** — positions, cell, velocities/momenta, MD step count, snapshot count, RNG
  bit-generator state, thermostat state, split assignment. Refuses until then.
- **Growing a dataset moves split assignment.** Re-running `assign_splits` at a larger total moves
  already-written items between files. Items on disk must keep their split; new items close the
  gap to the new target sizes. Needs the enumerable-generator manifest (`item_id = base_frame_key
  | variant_label`) so rattle progress becomes "how far through the manifest" and new variants
  inherit their base frame's split. MD stays sequential with an opaque blob — that asymmetry is a
  property of the procedures, not an accident. Adding BASE FRAMES stays fatal until membership is
  stored per key.
- **Lock** — `dataset_path/.generation.lock` around datamodule generation; acquire, re-read the
  record, generate or no-op, release. Lightning's rank-zero `prepare_data()` does NOT protect two
  separate jobs pointing at one `dataset_path`.
- **Warm start on grown data** — `ModelFromCheckpoint` rewrite, unimplemented.
- **`student_path`** — stable checkpoint dir + versioning off + a `student_state.json`
  (`{dataset_path, sample_n_frames, hydra_run_dir, epochs, max_epochs, status, best_metric}`), so
  training resume becomes automatic. Cost accepted: we'd set `ModelCheckpoint.dirpath` ourselves,
  the first real intervention in nequip's territory. Also needs the warm-start archive hook so a
  warm start can't clobber `best.ckpt` with something worse.
- **No generate-without-training path exists.** Revisit if sampling on a GPU node and sweeping
  students elsewhere becomes real.

**Tests**
- Hand-inspect the suite (user wants to read it, not just trust it).
- Gaps: store unit tests (atomic write, offsets, truncation, short/missing file, digest mismatch,
  orphan files); datamodule unit tests (notably "second `prepare_data()` no-ops WITHOUT
  instantiating the teacher"); a v1-record migration refusal test.

**Shipping hygiene**
- `pyproject.toml` still says `description = "TODO"`.
- `README.md` is a scaffold with empty sections — **user writes the prose, do not fill it in.**
- `README.md` and `docs/tutorial/` still describe the deleted `nequip-distill` CLI. The tutorial
  is SHIPPED (public Colab link) and therefore BROKEN until ported — user's call, not yet made.
  **Never run the tutorial locally**; the user tests it in Colab.
- `.pre-commit-config.yaml` exists (ruff line-length 88 double quotes, yamllint, whitespace,
  `fail_fast: true`) but no hook is installed in `.git/hooks/`, and ruff is not in nequip311
  (`No module named ruff`). Lint by hand before committing.
- `generation_config` must be set post-hoc (not a ctor arg) so hydra `instantiate` doesn't
  double-build the teacher — so a standalone non-hydra script must set it by hand or resume
  refuses "was not given the config". Clean fix: the generator derives its own provenance from
  ctor args.
- Template leftovers: `_keys.py` placeholder fields, empty `model/`, `nn/`, `train/` stub dirs.
- Teacher compiling: plan is to DEMAND a ready calculator-compatible artifact and compile nothing
  here. Confirm with the user before adding — and the `.pt2` segfault makes "bring a fast compiled
  teacher" unshippable advice right now.
