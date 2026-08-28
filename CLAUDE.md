# TuneandDistill — Agent Context

## Purpose

`nequip` extension pkg. Ships ONE new CLI: **`nequip-distill`**.

Given teacher model artifact loadable as ASE calc + base frame(s):
1. **sample** frames (ASE MD, or rattle/deform of base frames)
2. **label** w/ teacher calc
3. **train** student via nequip's `train()`

Finetuning OUT OF SCOPE (nequip core does it). This repo = distillation only.

**Scope: hackathon one-off, warm-up for possible later extensibility work. NOT a broad
extensible framework.** Reject abstraction/hooks/plugin seams that only pay off hypothetically.
Product pitch = ONE command; any design needing 2 commands or 2 configs is rejected.

Design source: `planning.md` (this repo). Broader framework vision: `../Zero2Tuned/DESIGN.md`.
`../distillation/` is **NOT precedent/source of truth** here (different repo, Snakemake-based).
Cite it ONLY for the rattle algorithm + ASE artifact gotchas below.

## Working paradigm (READ FIRST, every step)

**SMALL STEPS, ONE AT A TIME. Plan before code, wait for approval between steps.** Before any
code: short plain-language plan — exactly what the step adds AND what it deliberately leaves
out. One step per plan, no bundling, no "while I'm here." No unrequested extras (no smoke
tests, no extra verification, no GPU jobs, no refactors) unless asked. Drop caveman mode for
plans/explanations/design talk, keep it for status. User challenges design premises hard and is
usually right — take the challenge seriously, don't defend.
User explicit, twice: agent once wrote the whole resume feature in one go (~800 lines: base
class + rattle + md + split logic + 26-test suite + an unrequested GPU Slurm job). User: *"i
feel like you wrote too much at once for me to follow it... i need to trust your code and i
dont right now."* All reverted, GPU job cancelled, rebuilt from scratch in small approved steps.

## State — 2026-08-28

**Sampling half DONE + verified on GPU (`d61ee43`, RNG fix `96d54c8`). Sampler resume: HALF DONE
(`f9689f3`) — record written + read back, rattle resumes, MD does NOT, every config change
refused. Student side WIRED (D8/D9/D11 minus warm start): `run: [sample, train, val, test]`
RAN END TO END ON GPU (2026-08-28, user): `nequip-distill -cp <abs>/testartifacts -cn
rattle_train` sampled 50 structures then trained + val + test a student. `nequip-distill` CLI
installed editable into nequip311.**

`nequip_extension_template/sample/`
- `sampler.py` — base `Sampler`. Holds calculator, `base_frames`, `sample_path`, `n_written`,
  `n_resumed`, `split_counts`, `state_interval`, `sampler_config`. Abstract surface = `finished`
  (property) + `step()`. Overridable: `procedure_state()` (default `{}`),
  `restore_progress(blob)` (default RAISES `NotImplementedError` → a sampler not taught to
  resume refuses instead of duplicating everything). Concrete: `split_file(s)` /
  `train_file` / `val_file` / `test_file`, `append(atoms, split)`, `state_file`, `write_state()`,
  `read_state()`, `check_goal(stored)`, `truncate_to(offsets)`, `generate()`.
  Module fns: `frames_digest(frames)`, `flatten(dict)`, `split_file(sample_path, split)` (also
  exported from `sample/__init__` w/ `SPLITS` — `distill.py` needs the 3 paths in runs that build
  no sampler).
- **resume, as implemented (`f9689f3`)** — `sampler_state.pt` beside the dataset holds
  `{version, sampler_class, goal: {config, base_frames}, progress: {n_written, split_counts,
  offsets, procedure}}`. `offsets` = byte length of each split file. Written every
  `state_interval` structures (default 1) + at end, atomically (`.pt.tmp` + `os.replace`).
  `generate()` order on resume: read → restore counters → `check_goal` → `truncate_to` →
  `restore_progress`. Refuses: unknown `version`, different `sampler_class`, split files with NO
  record, file shorter than recorded offset, `sampler_config is None`, ANY config difference.
  File LONGER than offset → truncated + warned (this fires even at `state_interval=1`, measured).
- `rattle.py` — `RattleSampler`. Owns `label()` (teacher call + `SinglePointCalculator`).
  Resumes: `procedure_state()` = `{"n_steps": ...}`, nothing else. Order is fixed (variant-major)
  and the goal must be unchanged, so the count IS the position.
- `md.py` — `MDSampler`. Langevin NVT, ONE trajectory from `base_frames[0]`, rest ignored (warns).
  CANNOT resume — inherits the base `restore_progress` refusal, no md.py code needed. Needs
  positions + velocities + `rng.bit_generator.state` (Next steps #2).
- `split.py` — free fns, NOT a base-class method. `assign_splits(n, split, seed, policy)`,
  `policy` = `"scattered"` (torch shuffle) or `"blocked"` (contiguous train→val→test ranges).
- `__init__.py` — exports `Sampler`, `RattleSampler`, `MDSampler`.

`nequip_extension_template/scripts/distill.py` — hydra entry, `@hydra.main(version_base=None,
config_path=os.getcwd(), config_name="config")`, same shape as `nequip.scripts.train.main`.
`run` = optional `sample` (first, at most once) + any of `train`/`val`/`test`; the tail goes to
`nequip.scripts.train.main(train_config)` in-process. Helpers: `_split_run_list`,
`_check_train_config` (runs BEFORE sampling — `data`/`trainer`/`training_module` present,
`data._target_` is an `ASEDataModule` subclass, `data.split_dataset` empty), `_dataset_files`,
`_require_dataset` (no-`sample` runs need 3 non-empty files, sampler NOT built — would load the
teacher onto a GPU for nothing), `_train_config`, `_release_teacher` (drops `sampler.calculator`
+ `torch.cuda.empty_cache()` before the student starts). Sets
`sampler.sampler_config = OmegaConf.to_container(config.sampler, resolve=True)` AFTER
`instantiate` — NOT as a ctor arg, `instantiate` recurses into args hunting `_target_` and would
build the teacher twice. Logs new-vs-already-present via `sampler.n_resumed`.
`nequip_extension_template/scripts/__init__.py` — added (was missing, needed for import).

**Two live traps handled in `_train_config`/`main`, both from reading nequip 0.17.1's source:**
- nequip does `if "ckpt_path" in config:` — PRESENCE, not value. A config spelling out
  `ckpt_path: null` would enter the restart branch and `torch.load(None)`. So the key is POPPED
  when it is None.
- D11 guard: `main` counts `n_written` before/after sampling, RAISES if it grew AND `ckpt_path`
  is given. Mechanism/trap detail: D11.

**`exclude_keys` is a NON-ISSUE for our provenance keys** (checked, do not "fix" it):
`nequip/data/ase.py:55` builds `include_keys` from `ase_all_properties` + the user's
`include_keys`, and reads ONLY those out of `atoms.info`/`atoms.arrays`. `base_frame`,
`base_frame_key`, `variant`, `md_step` are not ASE calculator properties, so nequip never looks
at them. The `../distillation` crash was `dipole`/`free_energy` — real ASE properties, present on
some frames only. Our frames all carry exactly energy+forces from one `SinglePointCalculator`.

`tests/test_resume.py` — 16 tests, ~26 s, CPU only, NO teacher and NO GPU. `pytest` from repo
root (config in `pyproject.toml`). `tests/smoke_tests.py` — 15 `nequip-distill` end-to-end cases,
~4 min, run by hand, NOT collected by pytest. See "Test suite" below.

`pyproject.toml` — `[project.scripts] nequip-distill = "nequip_extension_template.scripts.distill:main"`,
`[project.optional-dependencies] test = ["pytest"]`, `[tool.pytest.ini_options] testpaths`.
`nequip>=0.17.1`, NO upper bound (user, 2026-08-28 — wants it usable with nequip 0.19; env still
has 0.17.1, which is what every integration fact here was validated against, so 0.19 is UNTESTED.
If it misbehaves, re-check the two things the student side leans on: `nequip.scripts.train.main`
taking a config as its first positional arg, and `run_stage` restart semantics). Still
template-shaped otherwise: `description = "TODO"`, `authors = [{name = "your name here"}]`
placeholders (no `license` key — removed this session, build refused without a LICENSE file).
Package name stays `nequip_extension_template` — DELIBERATE, no final name chosen; rename later
touches `pyproject.toml` (`name`, `packages.find.include`, entry-points, `version.attr`) + every
intra-pkg import.

`_keys.py` still registers upstream template placeholder fields (`user_facing_graph_field_name`,
`user_facing_node_field_name`) — unused, delete or replace whenever touched.
`configs/distill_template.yaml` — stale UI spec from before this session's `split_policy`/3-file
design; don't trust its `sampler:` shape, `testartifacts/*.yaml` are the current source of truth.

## Design decisions — sampling half (SETTLED, implemented)

**D1. Sampler owns calculator AND labeling, exactly one calculator, no separate labeler.**
Separate `labeler` slot REJECTED (user). Labeling itself is procedure-specific, not on the base
class: `label()` lives on `RattleSampler` only (static-structure eval); `MDSampler` gets labels
free from the dynamics step (Langevin already evaluates energy/forces every step — MEASURED
0 extra teacher calls, by wrapping `calc.calculate`).

**D2. `step()` does everything for one step**: produce + label + append + advance own state, in
one call. `generate()` just loops `while not self.finished: self.step()`. No separate
produce/label seam (D1). **No guard against a subclass whose `step()` doesn't advance
`finished`** — would loop forever. D2's original "no base-class `checkpoint_every`" is SUPERSEDED
(`f9689f3`): the base class owns the state file, so it owns the cadence — `state_interval`, default
1 (a teacher call costs far more than one `torch.save`).

**D3. Base class owns ONLY three empty boxes**: the three split file paths + `append(atoms,
split)`. It does NOT decide sample count, procedure params, or what goes in them — those are
each subclass's job (no `sample_size` on the base class; rattle's extent =
`len(base_frames) * len(variants)`, MD's = `n_samples`, no shared meaning to force into one key).

**D4. Split assignment is per-procedure-chosen unit, computed once at generation time.**
`assign_splits(n, ...)` labels `n` *items*; each sampler decides what `n` counts. Rattle → n =
base frames (rattles of one base frame are near-duplicates, must stay together). MD → n =
snapshots (assumes `sample_interval` decorrelates — UNVERIFIED, see Open risks).
Apportionment itself = `torch.utils.data.random_split` (nequip uses this too,
`nequip/data/dataset/utils.py:55` — don't reimplement). `split.py` only adds: (a) index→label
inversion (need labels before a dataset object exists, since frames are written straight to 3
files as generated), (b) empty-split → hard `ValueError` where torch only warns.
`split_policy` defaults differ ON PURPOSE: `MDSampler` → `"blocked"` (time-ordered, contiguous
tail holdout stays honest if interval under-decorrelates); `RattleSampler` → `"scattered"` (base
frames have no meaningful order). Both `testartifacts/*.yaml` configs set it explicitly anyway.

**D5. Output = three files** `train.extxyz` / `val.extxyz` / `test.extxyz` in `sample_path`, NOT
one `samples.extxyz`. Splits are frozen at generation, never recomputed at train time — this is
what keeps growing the dataset from silently moving val/test frames into train (the fraction-
based leakage risk that a single-file + `data.split_dataset`-fractions design would have). Once
wired to the trainer (Next steps #2), point `train_file_path`/`val_file_path`/`test_file_path`
at these three files directly; `data.split_dataset` fractions become irrelevant to sampled data.

**D6. Restart precedence for sampling: REVERSED from the original plan.** Live config wins;
sampler diffs stored goal vs. live goal, does not silently ignore the live config. Three-way
distinction: progress state (n_written, offsets, procedure blob) / stored goal / live goal —
stored goal is a diff baseline + legality guard, not behavior-driving.

**PARTLY IMPLEMENTED (`f9689f3`)**: the diff exists (`Sampler.check_goal`) but ANY difference is
FATAL, nothing is a warning yet and nothing is legal yet. Classification (which changes are legal,
which fatal) = Next steps #1.

**D6a. Stored goal = a COPY OF THE `sampler` CONFIG, not a hand-picked list of params (user,
settled).** A hand-written `goal()` per sampler duplicates every ctor kwarg and silently omits any
new one — the exact shape of the `anisotropic_strain_magnitude` bug. `distill.py` hands over
`OmegaConf.to_container(config.sampler, resolve=True)`; the sampler stores it verbatim.
Two things the config alone CANNOT express, so they are stored beside it:
- `base_frames` is a PATH. Edit the file, keep the path, and a config-vs-config diff sees nothing.
  Fixed by `frames_digest(self.base_frames)` (hashes the LOADED structures, not file bytes, so
  re-exporting the same frames in a different text layout is not a false alarm).
- MD uses only `base_frames[0]`, so its digest over all frames is over-strict. Not yet addressed.
NO exclusion list yet (user explicit): `state_interval` and `calculator.device` are compared too,
so a resume on a different device currently REFUSES. Deliberate — exempting keys one at a time is
how a real difference gets waved through. Exclusions land with the classification step.
**Consequence to know:** `state_interval` isn't in the yaml, so it needs `+sampler.state_interval=N`
(hydra ADD not override), and a run started that way must be resumed with the same flag or the
live config is missing the key → `state_interval: 5 -> <not set>`.

**D7. Never pickle sampler/calculator. IMPLEMENTED (`f9689f3`, mechanism = the `sampler.py`
"resume, as implemented" bullet above).** Three reasons, in order: (a) a pickled sampler carries
its OLD config, and D6 says the live config wins — so the pickle buys nothing and makes it easy
to miss an attribute; (b) compiled/CUDA-bound torch models are non-portable across nodes/GPU
archs; (c) a pickle is coupled to class layout, so renaming an attribute silently breaks every
existing `sample_path` — a plain dict breaks loudly via the `version` check instead.

**D7a. RNG derivation is ASYMMETRIC between samplers (settled, `96d54c8`) — load-bearing for
resume.**
- `RattleSampler`: no streaming RNG. Each structure seeded by `derive_seed(seed,
  frame_key(base_frame), variant_label)`, both module-level fns in `rattle.py`. `frame_key` =
  sha256 of numbers + positions (rounded 8) + cell (rounded 8) + pbc, truncated 16 hex chars.
  Structure depends ONLY on its own identity, never generation order — **nothing to checkpoint
  for rattle's RNG.** Variant identity is a LABEL not a position (`"iso:-0.05"`, `"aniso:0"`) so
  adding a strain magnitude doesn't renumber existing variants. `atoms.info` now carries
  `base_frame`, `base_frame_key`, `variant` (MD still carries `md_step`).
- `MDSampler`: KEEPS one streaming `np.random.default_rng(seed)` — must, trajectory is
  sequential, snapshot n only exists via integrating 1..n-1. **Resuming MD requires
  checkpointing `self.rng.bit_generator.state` alongside positions/velocities** (documented in
  `md.py` docstring, not implemented — see Next steps #2).
- **Open trap: CLOSED (`f9689f3`).** Changing `anisotropic_strain_magnitude` on purpose had the
  same silent effect as the bug the decoupling fixed. Now caught, because D6a stores the whole
  config — as are `seed`, `strain_magnitudes`, `max_displacement_ang`, and base-frame content.
  Tested (`tests/test_resume.py`).

## Design decisions — student side (SETTLED, NOT yet implemented)

**D8. `sample` stays in `run`, earns place by being omissible.**
| `run` | `sample_path` | Meaning |
|---|---|---|
| `[sample, train, val, test]` | fresh | full distillation |
| `[sample]` | fresh | build dataset only |
| `[train, val, test]` | existing, complete | retrain diff student on existing dataset (sweep) |
Assert: `sample` first if present, at most one.

**D9. Call nequip's train in-process, do NOT reimplement.** Mechanism:
`nequip.scripts.train.main(config)` — hydra's `@hydra.main` decorator takes `cfg_passthrough` as
first-class first param (`hydra/main.py` `decorated_main`, verified in installed hydra): no new
hydra folder, no sys.argv parsing, inner fn sees outer's `HydraConfig.get().runtime.output_dir`.
Must ALWAYS pass the config (`train.main()` with no arg parses sys.argv, mints a second hydra
folder). Consequence: `${hydra:runtime.output_dir}` in the config resolves to OUR folder —
template's `ModelCheckpoint.dirpath`/`logger.save_dir` land in the one folder, no rewriting by
us. Version pin still wanted for `run_stage`/restart behavior (`pyproject.toml` →
`>=0.17.1`, no upper bound). Config handed over: strip `sample` from `run`, drop `sampler`/`sample_path`,
point `train_file_path`/`val_file_path`/`test_file_path` at the 3 sample files (supersedes D5's
older single-file plan).

**D10. Progress DERIVED from artifacts, no distill-level program counter.** Sample stage →
sample-side state file (D7). Train/val/test → nequip's own `run_stage` (registered buffer in
checkpoint, `nequip/train/lightning.py:161`), untouched. **Caveat, measured: `run_stage` carries
almost no information** — see D11's corrected trap note. It is 0 in every checkpoint a
`train`-first run produces, so it does not distinguish "finished" from "died during epoch 1".

**D11. ONE ckpt key `ckpt_path`.** `warm_start_from` proposed + REJECTED (user). Script picks
Path A vs B from whether dataset grew (only thing that knows):
- unchanged → Path A, `ckpt_path` passthrough = crash resume (via `last.ckpt`, D13).
- grew → Path B, rewrite `training_module.model` to `ModelFromCheckpoint` from `best.ckpt`
  (D12), warm start on extended data. Log loudly (expensive, easy accident either way).
- escape hatch: user-written `ModelFromCheckpoint`/`ModelFromPackage` builder respected, untouched.
- **TRAP guarded against, IMPLEMENTED (`distill.py` raises).** Bump sample count + keep
  `ckpt_path` → sampler appends frames, nequip restarts the OLD run: it resumes at the restored
  epoch and stops at the same `max_epochs`, so the new structures get only the leftover epochs,
  or none if that run finished. Only the distill script can catch this: record `n_written` before
  sampling, compare after. Measured with the guard removed (`tests/smoke_tests.py`): source run
  logged epochs `['0','1']`, restarted run logged only `['1']`, exit 0.
  **The original premise was WRONG and is corrected here: a completed run does NOT store
  `run_stage==3`.** nequip advances `run_stage` after each stage RETURNS, but only Lightning
  writes checkpoints and only during `fit`; `val`/`test` write none. So every checkpoint from a
  `train`-first run holds `run_stage == 0` no matter how far the run got — verified on a finished
  20-epoch GPU run (`epoch=19`, `run_stage=0`, `runs=['train','val','test']`). A `ckpt_path`
  restart therefore ALWAYS replays train, val and test; `train` just returns immediately when the
  restored epoch is already `max_epochs`. `run_stage` would only be nonzero if `train` were not
  the first stage.

**D12. `best.ckpt` vs `last.ckpt` are different jobs — never say "ckpt_path" generically.**
`last.ckpt` = resume point (Path A). `best.ckpt` = best on monitored metric, warm-start (Path B)
must start from THIS not last-epoch weights (`nequip/scripts/train.py` dispatch loop sets
`ckpt_path="best"` after a `train` stage, confirmed).

**D12a. `last.ckpt` IS NOT THE NEWEST EPOCH under a monitored callback (MEASURED, lightning
2.6.1). Documented behaviour, not a bug — `save_last` is relative to SAVES, not to epochs.**
`model_checkpoint.py:117`: "saves a `last.ckpt` copy whenever a checkpoint file gets saved".
Mechanism at `model_checkpoint.py:514-517` — `_save_last_checkpoint` runs only `if
self._last_global_step_saved == trainer.global_step`, i.e. only when the top-k save fired at that
same step; with a `monitor` that is only on an improvement. `on_train_end` (line 536) saves last
only `if not self._last_checkpoint_saved`, so it does not fix it either.

Measured in PURE lightning (no nequip), 5 epochs, metric worsening every epoch so best = epoch 0,
newest = epoch 4:

| `ModelCheckpoint` args | `last.ckpt` |
|---|---|
| `monitor=<metric>, save_top_k=1` (our config, AND nequip's tutorial) | epoch 0 — the best |
| `monitor=<metric>, save_last="link"` | epoch 0 — the best |
| `monitor=<metric>, save_top_k=-1` | epoch 4 — newest |
| `monitor=None` | epoch 4 — newest |

Not a short-run artifact (checked at 5 and 6 epochs and against a 20-epoch GPU run, where best
happened to BE the last epoch so it hid the effect).

**Consequence: `ckpt_path=.../last.ckpt` resumes from the last IMPROVING epoch and silently drops
everything after it.** Harmless on a still-improving run, expensive on a plateau. A second,
unconditional `ModelCheckpoint(monitor: null, filename: latest)` callback PROPOSED then REJECTED
by user (2026-08-28): "this is nequip behavior, which we dont want to change. lets not worry
about it." **Do not re-propose** — left alone on purpose as nequip/lightning's own checkpoint
semantics. `testartifacts/rattle_train.yaml` keeps the single monitored callback.

**D13. Lightning checkpoint versioning must be OFF for a shared student dir.** Confirmed in
installed lightning `model_checkpoint.py`: version counter (`enable_version_counter=True`
default) makes a second run into the same `dirpath` write `best-v1.ckpt`/`-v2`, etc. **Trap:
unsuffixed `last.ckpt` is then the OLDEST, not newest.** `ModelCheckpoint
(enable_version_counter=False)` gives one always-current `best.ckpt`/`last.ckpt` — PRECONDITION
for automatic training resume via `student_path` below.

**D14. Warm-start best-clobbering → ARCHIVE HOOK, NOT yet designed/implemented.**
`ModelCheckpoint.state_dict()` persists `best_model_score`/`best_model_path` (confirmed,
installed lightning); Lightning restores this on a `ckpt_path` resume but a WARM START has no
`ckpt_path` → best tracking starts from zero → new run can overwrite `best.ckpt` even if worse,
irreversible with versioning off (D13). Plan: hook that MOVES the outgoing `best` checkpoint into
an archive folder before a warm start, keeping main slot for true best on current val set.

**D15. `student_path` — SETTLED, adopt (was open, user resolved this session).** Symmetric to
`sample_path`: stable home for checkpoints + D13 versioning off → training resume becomes
automatic (default = resume from last/best checkpoint in `student_path`; new start = hand a new
`student_path`). Cost accepted: we set `ModelCheckpoint.dirpath` ourselves — first real
intervention in nequip's territory (everything else in D9 passes the trainer config through
untouched). Record next to checkpoint: `student_state.json` w/ `{sample_path, sample_n_frames,
sample_updated, hydra_run_dir, epochs, max_epochs, status, best_metric}`. Convergence readable
via `trainer.early_stopping_callback.stopped_epoch` (0 if never fired), `trainer.current_epoch`,
`trainer.max_epochs`, `trainer.checkpoint_callback.best_model_score` (confirmed present,
installed lightning).

## On-disk layout (settled)

- One hydra timestamped folder per `nequip-distill` command, siblings accumulate. Holds resolved
  config + log (sampling log lines land there too), checkpoints once training begins — nequip's
  own convention (`configs/tutorial.yaml` v0.17.1 sets both `ModelCheckpoint.dirpath` and
  `logger.save_dir` to `${hydra:runtime.output_dir}`).
- `sample_path` lives ELSEWHERE, side by side with the hydra folder, not nested either direction
  — different lifetimes (dataset outlives any command, run folder is per-command). Nesting hydra
  folder INSIDE `sample_path` is REJECTED (tested): hydra creates its run dir at command START,
  so `hydra.run.dir: ${sample_path}/...` makes hydra create `sample_path` itself → sampler sees a
  dir with contents but no state file → hard error every fresh run. Reverse (sample_path as
  subdir of hydra folder) is SAFE, tested fine — just never point `hydra.run.dir` at/inside
  `sample_path`.
- Interruption during SAMPLING: no checkpoint to hand back, `ckpt_path` stays null, resume driven
  entirely by `sample_path`. Interruption during TRAINING: new hydra folder on resume unless
  `student_path` (D15) makes `dirpath` stable — this was the "resume asymmetry", now resolved by
  D15's adoption.
- Hydra does NOT chdir (`hydra.job.chdir` unset → False, confirmed installed 1.3.2) → every
  relative path in a nequip config resolves against LAUNCH CWD, not the hydra run dir. Why
  `${hydra:runtime.output_dir}` is used explicitly rather than relative paths.
- `--multirun` groups siblings under `multirun/<date>/<time>/<job_num>/` instead of `outputs/`.

## OPEN — not yet settled

1. **Provenance.** Single `runs.jsonl` REJECTED (split-brain). Settled instead: each stage's
   record lives with its own artifact — sampling's is `sampler_config.yaml`/state file in
   `sample_path` (once D7 lands); training's is the hydra folder's `.hydra/config.yaml` +
   `student_state.json` (D15) next to the checkpoint.
2. Teacher compiling: plan is DEMAND ready calculator-compatible artifact, no compiling in this
   repo. Confirm w/ user before adding — worth revisiting given the `.pt2` segfault below.
3. Sweep race: parallel sweep runs would all sample into the same 3 files simultaneously. Fix =
   lock file in `sample_path`, undesigned.
4. Two-command workflow (`run:[sample]` then `run:[train,val,test]`) REJECTED as the sweep-race
   fix — defeats one-command product. Still legal per D8, just not the answer.
5. Two separate hydra processes in one command REJECTED — tested, both resolve to the SAME
   output folder when run same second (`${now:...}` identical), timing-dependent failure.

**Scope call:** sampling / student-side are separable in TIME. Sampling built first (held the
only real unknowns: ASE MD state capture, extxyz append+truncate, `step()` shape per sampler).

## nequip 0.17.1 integration facts (verified in installed env)

Env: `/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311` (conda prefix, no activate).

**Teacher → ASE calculator.** `nequip.integrations.ase.NequIPCalculator`
(`nequip/ase/nequip_calculator.py` is a deprecated shim, don't use).
- `from_compiled_model(...)` — `nequip-compile` output only (`.nequip.pth`/`.nequip.pt2`).
- `_from_saved_model(...)` — only path for `.nequip.zip` (nequip-package) or raw `.ckpt`.

**nequip's train entry** (`nequip/scripts/train.py`): required top-level sections `run`, `data`,
`trainer`, `training_module`. `run` accepts ONLY `train`/`val`/`test`/`predict` (or a dict w/
`function` key, UNIMPLEMENTED upstream); at most one `train`. `sample` must be stripped before
delegating.

**Path A vs B precedence** (drives D9/D11):
| Section | On `ckpt_path` restart (Path A) | Guard |
|---|---|---|
| `training_module` | checkpoint wins, live config ignored | warns config ignored |
| `data` | live config, always re-instantiated | none |
| `trainer` | live config, always | none |
| `run` | live config decides stages+order; ckpt's `run_stage` decides only START INDEX | assert live `run` matches ckpt `run` as prefix |
| `global_options` | live, folded into `info_dict` only | none |
Path B = `ModelFromCheckpoint`/`ModelFromPackage` as `model` builder — no conflict guard, only a
version-string warning (`nequip/model/saved_models/checkpoint.py:64-71`).

**Workflow state.** `nequip/scripts/_workflow_utils.py::set_workflow_state(state)` asserts
`state in ["train","package","compile",None]` — cannot register `"distill"`/`"sample"`, private
module, don't try.

**Datamodules.** Base `nequip.data.datamodule.NequIPDataModule`. On-disk ASE files:
`nequip.data.datamodule.ASEDataModule`, kwargs incl. `train_file_path=[]`, `val_file_path=[]`,
`test_file_path=[]`, `split_dataset=[]`, `exclude_keys=[]` (this is what D5/D9's 3-file output
plugs into).

**No prior art in nequip core.** No sampler, rattle, active learning, MD driver.
`nequip/data/_sampler.py::PartialSampler` is an unrelated torch DataLoader sampler. Only MD code
is `nequip/ase/nosehoover.py::NoseHoover` — an NVT thermostat class, not a driver.

**Compiled-artifact findings (this session, matters for OPEN #2):**
- TorchScript compile is DEAD: `nequip-compile --mode torchscript` →
  `ValueError: TorchScript compilation is deprecated and not supported in PyTorch >= 2.10`
  (env has torch `2.11.0+cu128`). AOTInductor is the only mode → compiled artifacts always
  GPU-arch-locked.
- AOTInductor compile needs `module load cuda/12.9.1-fasrc01` (env's pip nvidia headers
  incomplete — `fatal error: crt/host_defines.h`; setting `CPATH` alone is NOT enough).
- **`nequip-compile --target ase` AOTInductor `.pt2` SEGFAULTS** — compiles fine, calculator
  builds fine. THIS session (CDP model): crashes at first forward (`get_potential_energy()`).
  Tutorial session (OAM-S model, 2 attempts): crashes IN THE COMPILE ITSELF instead, right after
  the same warning. Crash POINT is inconsistent across model/attempt — UNRESOLVED, root cause
  unconfirmed. Suspect export warning `aten._linalg_det.default is missing a c-shim
  implementation` both times — confirmed `linalg_det` is in no c-shim header in installed torch
  2.11, and the op is unavoidable (`nequip/nn/grad_output.py:249`, stress volume, every periodic
  model hits it). Packaged `.nequip.zip` on the SAME model works fine — isolated to the `.pt2`
  path.
- `nequip-compile` from a `.ckpt` needs the checkpoint's training data (rebuilds datamodule →
  `FileNotFoundError`); from a `.nequip.zip` package does not.
- gpu_test node used = A100-SXM4-40GB, compute_cap 8.0.

## Rattle prior art — reused, don't reinvent

`../distillation/scripts/gen_synthetic_geoms.py` algorithm now implemented in `rattle.py`: per
base frame, one structure per entry in `strain_magnitudes` (isotropic volume scaling
`(1+strain)^(1/3)`) + `n_random_strain_samples` random anisotropic strains, each followed by
per-atom rattle bounded by `max_displacement_ang`. Variant-major ordering (confirmed): every base
frame gets variant 0 before any gets variant 1.

**Anisotropic strain bound decoupled from `strain_magnitudes` (`96d54c8`).** `RattleSampler` kwarg
`anisotropic_strain_magnitude: float = 0.05` replaces the derived
`max_strain_magnitude = max(abs(strain_magnitudes))` inherited from `gen_synthetic_geoms.py`. The
derived form coupled two knobs: adding one isotropic scan point silently widened the anisotropic
distribution with no change to any structure's name, so nothing could detect it (measured before
fix: adding a strain magnitude left 20/20 `iso` structures bit-identical but changed 10/10
`aniso`; after fix, all 30 bit-identical). Default 0.05 matches old expression's value for the
default `strain_magnitudes`, so default behaviour unchanged. `testartifacts/rattle_only.yaml`
sets it explicitly.

**Latent bug, impact unconfirmed — ASE stores lattice vectors as ROWS.** Straining should be
`cell @ F.T`; `rattle.py` does `F @ cell`. Identical for a CUBIC cell (every base-frame set used
so far — CsH2PO4 testartifacts, Si `sitraj.xyz` — is cubic), so nothing is observably wrong
today; diverges for hexagonal/triclinic. Fix if a non-cubic base frame is ever used.

**Hard-won magnitude lesson**: ±10%/5% strain + 0.5 Å displacement pushed frames OOD for the
teacher, students WORSE than no synthetic data. Halved defaults (±5%/2.5% strain, 0.25 Å
displacement, matching `rattle.py`'s `strain_magnitudes`/`max_displacement_ang` defaults) fixed
it. Rattle displacement must be measured against the STRAINED parent, not the unstrained one
(comparing to unstrained overstates spread) — confirmed correct in `rattle.py`.

## ASE gotchas that bit MDSampler / RattleSampler

- **Snapshot every kept frame w/ `atoms.copy()`.** ASE MD mutates one `Atoms` in place;
  appending the live object gives N copies of the final step.
- **`atoms.copy()` drops `atoms.calc`.** Reattach: `snap.calc = SinglePointCalculator(snap,
  energy=e, forces=f)` (`ase.calculators.singlepoint`). Without it, extxyz gets geometry w/ no
  labels, found out only at student-training time.
- ASE `Langevin(fixcm=True)` (current `md.py` default) does NOT strictly sample the correct NVT
  distribution — deprecated since ASE 3.28, fix is `fixcm=False` + `ase.constraints.FixCom`. Left
  alone because MD is scaffolding; becomes a real correctness issue if MD output is trusted.
- numpy 2: `ndarray.ptp()` is gone, use `np.ptp(arr)`.
- Don't name a scratch script `inspect.py` — shadows stdlib, breaks numpy/ase import.
- **Determinism-test trap:** `frames[0].copy()` has the SAME content hash as `frames[0]` by
  construction (D7a's `frame_key`) — testing "adding a base frame leaves existing structures
  identical" with a copied frame passes trivially, proves nothing. Use a genuinely unused frame
  (source has 200, `base_frames.xyz` uses first 10 — frames 10+ are free for this).

## Artifact gotchas inherited from `../distillation/`

- `nequip-package` output must never be relocated after creation — path baked in (one
  `.nequip.zip` did survive a copy this session, don't rely on that holding in general).
- `ASEDataset` auto-includes `energy`/`forces` regardless of `include_keys`; inconsistent
  per-frame extra ASE properties (`dipole`, `free_energy`) crash batching. Set `exclude_keys`
  broadly — rattle/MD tag DIFFERENT provenance keys (`base_frame` vs `md_step`), must cover both.
- `nequip-compile --mode aotinductor` bakes GPU-arch-specific code — A100-compiled artifact
  invalid on H200.
- `nequip-train -cp <path>` needs ABSOLUTE path. Same applies to `nequip-distill`.

## Sandbox — `testartifacts/`

Tracked configs: `test_distill.yaml` (user's, pre-existing, water/`WaterDataModule`, NOT used by
our configs), `rattle_only.yaml`, `md_only.yaml`, `rattle_train.yaml`. Gitignored:
`testartifacts/inputs/`, `testartifacts/out/`.
- `rattle_train.yaml` — the FULL distillation config: `run: [sample, train, val, test]`, rattle
  half copied from `rattle_only.yaml`, student half = `ASEDataModule` + `EMALightningModule` +
  `NequIPGNNModel` (2 layers, `l_max: 1`, `num_features: [32, 16]`, `r_max: 6.0` — the CDP
  student arch from `../distillation/config/base.yaml`), CSVLogger, `max_epochs: 20`,
  `ModelCheckpoint(dirpath=${hydra:runtime.output_dir}, filename=best, save_last=true)`. Its
  `data:` block deliberately has NO `*_file_path` and NO `split_dataset` — `distill.py` fills
  those in. 50 structures → 40/5/5.
- `inputs/teacher.nequip.zip` — copied from
  `../distillation/results/CDP/student_direct/S1b/n200_seed1/model.nequip.zip`. THE teacher,
  loaded eagerly (packaged, not compiled — works, just slower).
- `inputs/teacher.nequip.pt2` — the segfaulting compiled one, kept as evidence only.
- `inputs/base_frames.xyz` — 10 CsH2PO4 frames (64 atoms, periodic), first 10 of
  `../distillation/results/CDP/dft_subset/n200_seed1.xyz` (200-frame REAL DFT subset). DFT
  energy/forces DISCARDED, teacher relabels. 10 chosen so 0.8/0.1/0.1 apportions to 8/1/1.
- Named "base frames" not "seed frames" — collided with `n200_seed1` naming AND with RNG seed.

## Tutorial — `docs/tutorial/`

Upstream `mir-group/nequip-tutorial` VENDORED VERBATIM @ `8f90935` (2026-03-09, MIT):
`NequIP_Tutorial.ipynb` (32 cells), `config.yaml`, `config_finetuning.yaml`,
`sitraj.xyz` (110 frames, 64-atom Si, PBC), `README.md`, `LICENSE`. Never hand-edit —
`SOURCE.md` holds the refresh recipe. Tutorial is NOT in the nequip repo itself.

Ours:
- `distill_section.md` — our section, one cell per `## [markdown]` / `## [code]` marker.
  Preamble + `<!-- TODO -->` comments stripped at build. Written to match upstream's
  terse 1–3-paragraph cells (user rejected a 3x longer first draft).
- `distill.yaml` — Si distillation config. Teacher = fine-tuned OAM-S, packaged from
  `results_ft/best.ckpt`. 110 base frames x 5 variants = 550 structures, 88/11/11 base
  frames → 440/55/55. Student: 2 layers, `l_max: 1`, `num_features: 32`, `[Si]` only,
  r_max 5.0, ZBL on. Heavily commented — detail lives HERE, not in the prose.
- `build_notebook.py` → `NequIP_Distill_Tutorial.ipynb` (38 cells = upstream verbatim +
  our 6). GENERATED. Edit the md/yaml and re-run; never edit the notebook.

**NEVER RUN. `distill.yaml` has not been executed once**, and Colab installs current
nequip, not the 0.17.1 everything here was validated against. User tests in Colab.

Colab link needs repo PUBLIC + pushed `main`:
`colab.research.google.com/github/Steinburglar/TuneandDistill/blob/main/docs/tutorial/NequIP_Distill_Tutorial.ipynb`.
Two URLs hardcode `Steinburglar/TuneandDistill@main` (the `pip install git+` cell and
the `wget` of `distill.yaml`) — both live in `distill_section.md`, change there +
regenerate.

**Validated on GPU (job 42612714, `gpu_test`, 8m06s, 2026-08-28)**: packaged-teacher
config runs end to end. 550 structures sampled in 18 s → 440/55/55 exactly as the config
predicts, student trained, `TEST RUN END`, `best.ckpt` 997 KB vs teacher 9.9 MB. Student vs
teacher on held-out: per-atom energy MAE 6.0 meV/atom, forces MAE 0.425 eV/Ang (early-stopped
epoch 97 of 200, so converged by its own criterion, not truncated). Colab-specific cells
(install-from-git, `/content` paths) still UNTESTED.

**Compiled teacher REVERTED, do not re-add until a working `.pt2` exists.** Switched to
`from_compiled_model` + `./finetuned_ase.nequip.pt2`, then reverted on user's call: `nequip-compile
--mode aotinductor --target ase` SEGFAULTED (core dumped) on BOTH attempts here, in the compile
itself, right after `aten._linalg_det.default is missing a c-shim implementation, using proxy
executor as fallback`. Confirmed `linalg_det` is in NO c-shim header in installed torch 2.11;
the op is real and unavoidable — `nequip/nn/grad_output.py:249` does `torch.linalg.det(cell)` for
the stress volume, so every periodic model hits it. Both attempts were MIG slices
(`nvidia_a100_3g.20gb`); the full-A100 control job was cancelled while still PENDING, so
**MIG-vs-full is still UNTESTED** and the c-shim story remains correlation, not proof.
`max_epochs: 200` / `max_time: 00:00:30:00` deviate from upstream's 1000 / 3 days — user's call
to keep, do not "fix" to match upstream.

**Upstream bug, do not inherit:** notebook cell 29 compiles `/content/results/best.ckpt`
— the FROM-SCRATCH model — and cell 30 plots it as "Fine-tuned model". Real one is
`/content/results_ft/best.ckpt` (`results_dir: ./results_ft`). Our section uses the
correct path. Worth reporting upstream.

## Tutorial — open questions (user, 2026-08-28)

1. **Test split measures FIDELITY TO THE TEACHER, not accuracy.** Splitting is per base frame,
   so val/test structures do come from base frames that fed nothing into train — a real geometric
   holdout. But every label is the teacher's, so `test0_epoch/*` answers "how well does the
   student imitate the teacher", and CANNOT be compared against the teacher's own DFT test error.
   Fix = a final eval on held-out DFT frames. Note both contaminations: all 110 `sitraj.xyz`
   frames are base frames, AND `config_finetuning.yaml` feeds all 110 to the teacher via
   `split_dataset` 0.8/0.1/0.1. So a clean benchmark must be carved out BEFORE both the finetune
   and the sampling. Mentioned in the tutorial text, NOT implemented.
2. **The Si strain/rattle magnitudes are GUESSES.** +/-5% volumetric, 5% anisotropic, 0.25 Ang
   were tuned for CsH2PO4 in `../distillation`, not for silicon, and were carried over unchanged.
   Open question worth answering IN the tutorial, since every user faces it: how do you know the
   right magnitudes? Candidate diagnostics, none implemented — compare generated min interatomic
   distance against the source trajectory's; compare rattle displacement against thermal
   displacement at the trajectory's temperature; compare strain range against the volume
   fluctuations MD actually visits; teacher-ensemble disagreement as an OOD flag. The one hard
   datum stays the CDP result: too wide made students WORSE than no synthetic data at all.
3. Rattle recipe for the tutorial is exactly `rattle.py`'s, not Si-tuned — see "Rattle prior
   art" section above for the algorithm and the cubic-only `F @ cell` latent bug (applies here
   too, `sitraj.xyz` is cubic so it's silent).
4. **Missing section: compile + compare FM vs toy vs student, on accuracy AND speed.** This is
   what makes the point land, and it is the payoff the section currently only asserts. Needs (1)'s
   DFT holdout for the accuracy half, and a working compile for the speed half — see the segfault
   above.

## Run commands

```bash
CONDA=/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311
export LD_LIBRARY_PATH=$CONDA/lib:${LD_LIBRARY_PATH:-}
# from repo root; hydra does NOT chdir so relative paths resolve to launch cwd; -cp must be ABSOLUTE
$CONDA/bin/nequip-distill -cp $PWD/testartifacts -cn rattle_train   # sample + train + val + test
$CONDA/bin/nequip-distill -cp $PWD/testartifacts -cn rattle_only    # sample only
$CONDA/bin/nequip-distill -cp $PWD/testartifacts -cn md_only
```
`nequip-distill` is on PATH because the repo is `pip install -e . --no-deps --no-build-isolation`
into nequip311. Editable, so edits are live; re-run only if entry points change. Needed
`license = {file = "LICENSE"}` dropped from `pyproject.toml` first — no LICENSE file exists.
`nequip-distill --help` FAILS with no `config.yaml` in cwd (hydra composes before help); use
`-cp ... -cn ... --cfg job` to inspect a resolved config instead.
**`-cp` MUST be absolute** — hydra resolves a relative `--config-path` against the DECORATED
FUNCTION'S MODULE, so `-cp testartifacts` fails with `Primary config module
'nequip_extension_template.scripts.testartifacts' not found` (user hit this). `nequip-train` has
the identical trap. Auto-absolutizing `-cp` in a wrapper was PROPOSED and REJECTED (user,
2026-08-28): match nequip's behavior, do not diverge from it for convenience.
All runs go through Slurm (user explicit, never login node).

**`testartifacts/out/rattle_smoke` and `md_smoke` predate `f9689f3`** — no `sampler_state.pt`, so
pointing a run at them now hard-errors (split files, no record). Same for any leftover
`testartifacts/out/s2_*`; safe to delete, all gitignored scratch.

No pre-commit git hook installed in `.git/hooks/` (config present, hook never run); ruff not in
the nequip311 env (`No module named ruff`) — lint by hand before committing.

## Test suite

`tests/test_resume.py`, 16 tests, ~26 s, run with plain `pytest` from the repo root.

**CPU only, NO teacher, NO GPU — and that is a correctness decision, not just speed (user
explicit: no GPU test).** Calculator = `ase.calculators.lj.LennardJones`, base frames = 12
synthetic 8-atom Ar cells built in a module fixture at three file lengths (10/11/12). Reason: the
real teacher is a compiled model on GPU and two runs over BIT-IDENTICAL geometry disagree by
~3e-6 eV / ~4e-6 eV/Ang (float reductions not associative, order not fixed — MEASURED between two
uninterrupted runs). So with the real teacher, "resumed == uninterrupted" is only checkable to a
tolerance; with LJ it is a comparison of file hashes, which also catches a dropped, duplicated or
mis-filed structure. **So: byte-identity of extxyz is NOT a valid criterion whenever the teacher
runs on GPU.** For manual GPU checks use instead: same structure set, same split per structure,
positions/cell EXACTLY equal, |dE| and |dF| < 1e-5.

Test helpers: `build(...)` mirrors what `distill.py` does — same settings both construct the
sampler AND become `sampler_config`. `kill_after(sampler, n)` wraps `step` to raise after exactly
n appends (a structure count, not a wall-clock timeout, so the kill lands at a known point).

**Suite was mutation-checked**, each mutation caught only by the right tests:
`truncate_to` → no-op ⇒ 2 failures; rattle `restore_progress` forgets `n_steps` ⇒ 3;
`check_goal` early-return ⇒ 3.

### `tests/smoke_tests.py` — the `nequip-distill` command end to end

15 cases, ~4 min, CPU only, NO teacher and NO GPU. **NOT collected by pytest** (name is
deliberate — `pytest` only picks up `test_*.py`). Run by hand:

```bash
OMP_NUM_THREADS=2 $CONDA/bin/python tests/smoke_tests.py [-k <substring>] [--keep]
```

Each case writes a config and runs `python -m nequip_extension_template.scripts.distill`
as a SUBPROCESS — `@hydra.main` owns global state and does not survive two calls in one
interpreter, and the point is to exercise the command as a user meets it. `hydra.run.dir`
is pinned per case so assertions can look inside it. Calculator = `LennardJones` (Ar
params) over 10 synthetic 32-atom fcc argon cells (fcc, NOT random positions — random
points in a box put atoms on top of each other and LJ energies explode); student = 1
layer, `l_max: 0`, `num_features: 8`, 2 epochs, `accelerator: cpu`. 40 structures →
32/4/4. A full sample+train+val+test case costs ~16 s.

Covers: 7 refusals (unknown run type, `sample` not first, empty `run`, missing dataset,
missing `trainer`, non-`ASEDataModule`, `split_dataset` set), sample-only + its re-run
(byte-identical files), full pipeline, two students on one dataset (earlier `best.ckpt`
not clobbered — different hydra dirs), train-only with a DELIBERATELY BROKEN calculator
target (proves no sampler is built) plus a bogus `train_file_path` (proves the
overwrite), `ckpt_path: null`, checkpoint restart, and the D11 refusal.

**Mutation-checked**, 3 mutations, each caught by exactly its case: don't pop a null
`ckpt_path` ⇒ nequip logs "Building `training_module` from checkpoint file None";
delete the D11 guard ⇒ run exits 0 after `TRAIN RUN START` + `max_epochs=2 reached`,
having trained on none of the 40 new structures (this is the trap, MEASURED); don't
overwrite a user-set `*_file_path` ⇒ nequip reads `nonexistent.xyz`.

## Conventions

- Env: prepend `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` on `GLIBCXX_3.4.31 not found`.
- Real compute → compute node. Smoke-scale sampler tests OK on login.
- `pre-commit` configured: ruff lint+format (line-length 88, double quotes), yamllint,
  whitespace hooks, `fail_fast: true` (see hook-not-installed note above).

## Open risks / unverified

- MD `sample_interval` decorrelation (D4) is an ASSUMPTION, unmeasured — if false, per-snapshot
  scattered splitting leaks; `blocked` default mitigates, nothing detects it.
- MD params in `md.py` are placeholders not recommendations (`friction_per_fs=0.01`, 5-step
  equilibration in earlier smoke run).
- `.pt2` AOTInductor segfault (compiled-artifact findings above) unresolved — blocks OPEN #2
  ("demand a ready compiled artifact") until fixed or worked around.
- Teacher used is a *student* model from `../distillation/`, a stand-in — fine for plumbing, not
  a real teacher. Sanity check available for free: teacher E on an unrattled base frame vs its
  DFT E — −374.61 vs −374.51 eV on frame 0 (64 atoms, ~0.1 eV total).

## Next steps (ordered)

**Follow "Working paradigm" at top of file for every step below — one at a time, plan first.**

1. **Classify goal changes** (finishes D6). Today ANY config difference is fatal. Split into:
   fatal (`seed`, `max_displacement_ang`, `anisotropic_strain_magnitude`, split params — change
   these and structures on disk were drawn under settings the config no longer describes);
   legal-and-extending (more `strain_magnitudes`, higher `n_random_strain_samples`, more base
   frames, higher `n_samples`); irrelevant (`state_interval`, `calculator.device`). Prefer
   declaring the SMALL legal/irrelevant sets and defaulting everything else to fatal, so a knob
   added later is safe until someone thinks about it. Removing anything already on disk stays
   fatal — a written structure cannot be un-written.
2. **MD trajectory state** — positions + velocities + `rng.bit_generator.state` (D7a), replacing
   the inherited `restore_progress` refusal.
3. **Growing a dataset changes SPLIT ASSIGNMENT, and that is its own problem.** Re-running
   `assign_splits` at a larger total MOVES already-written items between files. Items on disk must
   keep their split; new items must be apportioned to close the gap to the target sizes at the new
   total. Rattle splits per base frame (so the map must be keyed by frame hash, not list index);
   MD splits per snapshot, and with `blocked` each extension gets its OWN contiguous val/test tail
   rather than one tail for the whole trajectory — a real cost, flag it before building.
4. `student_path` (D15) + `student_state.json` + D14 archive hook. Deferred ON PURPOSE (user,
   2026-08-28): student uses nequip's normal checkpoints-in-the-hydra-dir semantics for now, so a
   training restart needs an explicit `ckpt_path`.
5. Warm start on a grown dataset (D11 Path B, `ModelFromCheckpoint` rewrite). Until then
   `distill.py` REFUSES that combination rather than silently training on nothing.

**DONE (2026-08-28)**: old #4, the train handoff — see State. Order above reflects the user's
call to get a runnable `nequip-distill` out first and improve resume behavior after.
