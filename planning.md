# NequIP Academy — design record

The **why** behind the code, plus every fact measured about someone else's code.
On-demand reading, not per-turn context — `CLAUDE.md` holds the state an agent needs every
turn and points here. `README.md` is the human-facing front door and owes this file nothing.

Everything below marked "measured" / "verified" was checked against the environment named in
§6. A dependency bump invalidates §6 and §7 first; §2–§4 are our own decisions and survive it.

---

## 1. Original intent (pre-implementation brainstorm)

Kept verbatim as the starting point. Superseded in places by §2 — where they disagree, §2 wins.

> User interface should center on specifying the "run[train, val, test, sample, distill]" line
> Student can be similar to model, but by default be from scratch, and match the foundation model unless otherwise specified.
> Generator will require the most design, most open ended. How should it know which model to use in a script
> Separate labelling from sampling?
>
> Steps:
> finetune/train FM
> Sample frames (with or without FM)
> Label frames synthetically (with FM)
> distill/train student model
>
> Problems:
> How to handle restarts? Restarts are meaningful if there is just one model. Becomes unclear if there are multiple.
>
> In order to generate frames/labels, do we have to compile the teacher? Should we even be doing compiling in the training script.
>
> Possible solutions:
> Separate into two separate configs, a teacher and student config, provide new command line script that manages things on top. "nequip-PFD". Then need config for sampling procedure as well
> All in one config, make checkpoint refer to teacher, need "student_checkpoint" for student. All train, val, test, of student are just stu_train, stu_val, stu_test. Student config is inherited from teacher config except for checkpoint path and where otherwise specified?
> Distinction to draw: two slightly different things
> What is specified in what config file
> Which processes are called in the train() script, or run in a higher level script that runs train() or a wrapped version of train on a config, then does other things (like sampling/labeling frames)
>
> So, we can have 1 config with the specifications, but have a script that calls out the train, package, and compile scripts separately, and does not call one from the other.
>
> **Final Plan:**
>
> Separate Finetuning from Distillation. Finetuning is largely written already, just need to make sure the modifiers make sense.
>
> Instead of the primary goal being a full finetune then distill script, we just want to offer a new distillation script, nequip-distill.
>
> What you need for nequip-distill
> A teacher model artifact and a compatible ase calculator; (ideally compiled/fast).
> Frame(s) that you would like to either run MD from, or jiggle or something to sample new ones.
> A normal config for the student model you want (same requirements as nequip train())
> Plus: a new sampling config section that clarifies sampling procedure.
> Path and calculator type of the teacher model
>
> Given a compiled, packaged, or otherwise ase calculator compatible artifact, you simply run nequip-distill –cn distil.yaml, and it will do and save the sampling, then do a normal training for you.
>
> Restart handling:
>
> Should be handled similar to nequip-train. If you have an existing half or fully finished dataset, you can hand samples_path, and it will resume from there.
>
> Similarly, once you enter training, if it sees a checkpoint path in the config, it will do the normal nequip train() behavior with that sample.

**Scope, as settled:** hackathon one-off, warm-up for possible later extensibility work. NOT a
broad extensible framework. Reject abstraction/hooks/plugin seams that only pay off
hypothetically. Product pitch = ONE command; any design needing 2 commands or 2 configs is
rejected. Finetuning stays out of scope — nequip core does it.

Broader framework vision lives in `../Zero2Tuned/DESIGN.md`. `../distillation/` is **NOT**
precedent or source of truth here (different repo, Snakemake-based) — cite it only for the
rattle algorithm (§7) and the ASE artifact gotchas (§7).

---

## 2. Settled design decisions — sampling half

All implemented unless marked otherwise.

**D1. Generator owns calculator AND labeling, exactly one calculator, no separate labeler.**
Separate `labeler` slot REJECTED (user). Labeling itself is procedure-specific, not on the base
class: `label()` lives on `RattleGenerator` only (static-structure eval); `MDGenerator` gets labels
free from the dynamics step (Langevin already evaluates energy/forces every step — MEASURED
0 extra teacher calls, by wrapping `calc.calculate`).

**D2. `step()` does everything for one step**: produce + label + append + advance own state, in
one call. `generate()` just loops `while not self.finished: self.step()`. No separate
produce/label seam (D1). **No guard against a subclass whose `step()` doesn't advance
`finished`** — would loop forever. D2's original "no base-class `checkpoint_every`" is SUPERSEDED:
the base class owns the state file, so it owns the cadence — `state_interval`, default 1 (a
teacher call costs far more than one `torch.save`).

**D3. Base class owns ONLY three empty boxes**: the three split file paths + `append(atoms,
split)`. It does NOT decide sample count, procedure params, or what goes in them — those are
each subclass's job (no `sample_size` on the base class; rattle's extent =
`len(base_frames) * len(variants)`, MD's = `n_samples`, no shared meaning to force into one key).

**D4. Split assignment is per-procedure-chosen unit, computed once at generation time.**
`assign_splits(n, ...)` labels `n` *items*; each generator decides what `n` counts. Rattle → n =
base frames (rattles of one base frame are near-duplicates, must stay together). MD → n =
snapshots (assumes `sample_interval` decorrelates — UNVERIFIED, see §8).
Apportionment itself = `torch.utils.data.random_split` (nequip uses this too,
`nequip/data/dataset/utils.py:55` — don't reimplement). `split.py` only adds: (a) index→label
inversion (need labels before a dataset object exists, since frames are written straight to 3
files as generated), (b) empty-split → hard `ValueError` where torch only warns.
`split_policy` defaults differ ON PURPOSE: `MDGenerator` → `"blocked"` (time-ordered, contiguous
tail holdout stays honest if interval under-decorrelates); `RattleGenerator` → `"scattered"` (base
frames have no meaningful order). The `examples/*.yaml` configs set it explicitly anyway.

**D5. Output = three files** `train.extxyz` / `val.extxyz` / `test.extxyz` in `dataset_path`, NOT
one `samples.extxyz`. Splits are frozen at generation, never recomputed at train time — this is
what keeps growing the dataset from silently moving val/test frames into train (the fraction-
based leakage risk that a single-file + `data.split_dataset`-fractions design would have).

**D6. Restart precedence for sampling: REVERSED from §1's plan.** Live config wins; generator
diffs stored goal vs. live goal, does not silently ignore the live config. Three-way distinction:
progress state (n_written, offsets, procedure blob) / stored goal / live goal — stored goal is a
diff baseline + legality guard, not behavior-driving.

**PARTLY IMPLEMENTED**: the diff exists (`Generator.check_goal`) but ANY difference is FATAL,
nothing is a warning yet and nothing is legal yet. Classification = §9 next-step #1.

**D6a. Stored goal = a COPY OF THE `generator` CONFIG, not a hand-picked list of params (user,
settled).** A hand-written `goal()` per generator duplicates every ctor kwarg and silently omits any
new one — the exact shape of the `anisotropic_strain_magnitude` bug (§7). `distill.py` hands over
`OmegaConf.to_container(config.generator, resolve=True)`; the generator stores it verbatim.
Two things the config alone CANNOT express, so they are stored beside it:
- `base_frames` is a PATH. Edit the file, keep the path, and a config-vs-config diff sees nothing.
  Fixed by `frames_digest(self.base_frames)` (hashes the LOADED structures, not file bytes, so
  re-exporting the same frames in a different text layout is not a false alarm).
- MD uses only `base_frames[0]`, so its digest over all frames is over-strict. Not yet addressed.

NO exclusion list yet (user explicit): `state_interval` and `calculator.device` are compared too,
so a resume on a different device currently REFUSES. Deliberate — exempting keys one at a time is
how a real difference gets waved through. Exclusions land with the classification step.
**Consequence to know:** `state_interval` isn't in the yaml, so it needs `+generator.state_interval=N`
(hydra ADD not override), and a run started that way must be resumed with the same flag or the
live config is missing the key → `state_interval: 5 -> <not set>`.

**D7. Never pickle generator/calculator.** Three reasons, in order: (a) a pickled generator carries
its OLD config, and D6 says the live config wins — so the pickle buys nothing and makes it easy
to miss an attribute; (b) compiled/CUDA-bound torch models are non-portable across nodes/GPU
archs; (c) a pickle is coupled to class layout, so renaming an attribute silently breaks every
existing `dataset_path` — a plain dict breaks loudly via the `version` check instead.

**Resume, as implemented.** `generation_state.pt` beside the dataset holds `{version,
generator_class, goal: {config, base_frames}, progress: {n_written, split_counts, offsets,
procedure}}`. `offsets` = byte length of each split file. Written every `state_interval`
structures (default 1) + at end, atomically (`.pt.tmp` + `os.replace`). `generate()` order on
resume: read → restore counters → `check_goal` → `truncate_to` → `restore_progress`.
Refuses: unknown `version`, different `generator_class`, split files with NO record, file shorter
than recorded offset, `generation_config is None`, ANY config difference. File LONGER than offset →
truncated + warned (fires even at `state_interval=1`, measured).

**D7a. RNG derivation is ASYMMETRIC between generators — load-bearing for resume.**
- `RattleGenerator`: no streaming RNG. Each structure seeded by `derive_seed(seed,
  frame_key(base_frame), variant_label)`, both module-level fns in `rattle.py`. `frame_key` =
  sha256 of numbers + positions (rounded 8) + cell (rounded 8) + pbc, truncated 16 hex chars.
  Structure depends ONLY on its own identity, never generation order — **nothing to checkpoint
  for rattle's RNG.** Variant identity is a LABEL not a position (`"iso:-0.05"`, `"aniso:0"`) so
  adding a strain magnitude doesn't renumber existing variants. `atoms.info` carries
  `base_frame`, `base_frame_key`, `variant` (MD carries `md_step` instead).
- `MDGenerator`: KEEPS one streaming `np.random.default_rng(seed)` — must, trajectory is
  sequential, snapshot n only exists via integrating 1..n-1. **Resuming MD requires
  checkpointing `self.rng.bit_generator.state` alongside positions/velocities** (documented in
  `md.py` docstring, not implemented — §9 next-step #2).

---

## 3. Settled design decisions — student side

**D8. `sample` stays in `run`, earns its place by being omissible.**

| `run` | `dataset_path` | Meaning |
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
template's `ModelCheckpoint.dirpath`/`logger.save_dir` land in the one folder, no rewriting by us.
Config handed over: strip `sample` from `run`, drop `generator`/`dataset_path`, point
`train_file_path`/`val_file_path`/`test_file_path` at the 3 sample files.

**D10. Progress DERIVED from artifacts, no distill-level program counter.** Sample stage →
sample-side state file (D7). Train/val/test → nequip's own `run_stage` (registered buffer in
checkpoint, `nequip/train/lightning.py:161`), untouched. **Caveat, measured: `run_stage` carries
almost no information** — see D11. It is 0 in every checkpoint a `train`-first run produces, so it
does not distinguish "finished" from "died during epoch 1".

**D11. ONE ckpt key `ckpt_path`.** `warm_start_from` proposed + REJECTED (user). Script picks
Path A vs B from whether dataset grew (only thing that knows):
- unchanged → Path A, `ckpt_path` passthrough = crash resume (via `last.ckpt`, D13).
- grew → Path B, rewrite `training_module.model` to `ModelFromCheckpoint` from `best.ckpt` (D12),
  warm start on extended data. Log loudly (expensive, easy accident either way). **NOT
  IMPLEMENTED** — `distill.py` refuses this combination instead (§9 next-step #5).
- escape hatch: user-written `ModelFromCheckpoint`/`ModelFromPackage` builder respected, untouched.
- **TRAP guarded against, IMPLEMENTED (`distill.py` raises).** Bump sample count + keep
  `ckpt_path` → generator appends frames, nequip restarts the OLD run: it resumes at the restored
  epoch and stops at the same `max_epochs`, so the new structures get only the leftover epochs, or
  none if that run finished. Only the distill script can catch this: record `n_written` before
  sampling, compare after. Measured with the guard removed (`tests/e2e/test_distill_cli.py`): source run
  logged epochs `['0','1']`, restarted run logged only `['1']`, exit 0.
- **The original premise was WRONG and is corrected here: a completed run does NOT store
  `run_stage==3`.** nequip advances `run_stage` after each stage RETURNS, but only Lightning
  writes checkpoints and only during `fit`; `val`/`test` write none. So every checkpoint from a
  `train`-first run holds `run_stage == 0` no matter how far the run got — verified on a finished
  20-epoch GPU run (`epoch=19`, `run_stage=0`, `runs=['train','val','test']`). A `ckpt_path`
  restart therefore ALWAYS replays train, val and test; `train` just returns immediately when the
  restored epoch is already `max_epochs`. `run_stage` would only be nonzero if `train` were not
  the first stage.

**D12. `best.ckpt` vs `last.ckpt` are different jobs — never say "ckpt_path" generically.**
`last.ckpt` = resume point (Path A). `best.ckpt` = best on monitored metric; a warm start (Path B)
must start from THIS not last-epoch weights (`nequip/scripts/train.py` dispatch loop sets
`ckpt_path="best"` after a `train` stage, confirmed).

**D13. Lightning checkpoint versioning must be OFF for a shared student dir.** Confirmed in
installed lightning `model_checkpoint.py`: version counter (`enable_version_counter=True` default)
makes a second run into the same `dirpath` write `best-v1.ckpt`/`-v2`, etc. **Trap: unsuffixed
`last.ckpt` is then the OLDEST, not newest.** `ModelCheckpoint(enable_version_counter=False)`
gives one always-current `best.ckpt`/`last.ckpt` — PRECONDITION for automatic training resume via
`student_path` (D15).

**D14. Warm-start best-clobbering → ARCHIVE HOOK, NOT designed or implemented.**
`ModelCheckpoint.state_dict()` persists `best_model_score`/`best_model_path` (confirmed, installed
lightning); Lightning restores this on a `ckpt_path` resume but a WARM START has no `ckpt_path` →
best tracking starts from zero → new run can overwrite `best.ckpt` even if worse, irreversible
with versioning off (D13). Sketch: hook that MOVES the outgoing `best` checkpoint into an archive
folder before a warm start, keeping the main slot for true best on the current val set.

**D15. `student_path` — SETTLED, adopt. NOT implemented (§9 next-step #4).** Symmetric to
`dataset_path`: stable home for checkpoints + D13 versioning off → training resume becomes
automatic (default = resume from last/best checkpoint in `student_path`; new start = hand a new
`student_path`). Cost accepted: we set `ModelCheckpoint.dirpath` ourselves — first real
intervention in nequip's territory (everything else in D9 passes the trainer config through
untouched). Record next to checkpoint: `student_state.json` w/ `{dataset_path, sample_n_frames,
sample_updated, hydra_run_dir, epochs, max_epochs, status, best_metric}`. Convergence readable via
`trainer.early_stopping_callback.stopped_epoch` (0 if never fired), `trainer.current_epoch`,
`trainer.max_epochs`, `trainer.checkpoint_callback.best_model_score` (confirmed present,
installed lightning).

---

## 4. On-disk layout (settled)

- One hydra timestamped folder per `nequip-distill` command, siblings accumulate. Holds resolved
  config + log (sampling log lines land there too), checkpoints once training begins — nequip's
  own convention (`configs/tutorial.yaml` v0.17.1 sets both `ModelCheckpoint.dirpath` and
  `logger.save_dir` to `${hydra:runtime.output_dir}`).
- `dataset_path` lives ELSEWHERE, side by side with the hydra folder, not nested either direction —
  different lifetimes (dataset outlives any command, run folder is per-command). Nesting the hydra
  folder INSIDE `dataset_path` is REJECTED (tested): hydra creates its run dir at command START, so
  `hydra.run.dir: ${dataset_path}/...` makes hydra create `dataset_path` itself → generator sees a dir
  with contents but no state file → hard error every fresh run. Reverse (dataset_path as subdir of
  hydra folder) is SAFE, tested fine — just never point `hydra.run.dir` at/inside `dataset_path`.
- Interruption during SAMPLING: no checkpoint to hand back, `ckpt_path` stays null, resume driven
  entirely by `dataset_path`. Interruption during TRAINING: new hydra folder on resume unless
  `student_path` (D15) makes `dirpath` stable — this was the "resume asymmetry", resolved on paper
  by D15, not yet in code.
- Hydra does NOT chdir (`hydra.job.chdir` unset → False, confirmed installed 1.3.2) → every
  relative path in a nequip config resolves against LAUNCH CWD, not the hydra run dir. Why
  `${hydra:runtime.output_dir}` is used explicitly rather than relative paths.
- `--multirun` groups siblings under `multirun/<date>/<time>/<job_num>/` instead of `outputs/`.

### Repository layout policy
Keep four different kinds of files separate:

- `nequip_academy/` is package code.
- `examples/` is shipped runnable example configuration.
- `tests/` is automated verification, ideally split by the source module or CLI surface under test.
- `docs/tutorial/` is shipped tutorial material.
- `sandbox/` is the local playground for config experiments, datasets, teacher artifacts, generated
  samples, Slurm scripts, and run outputs. It is fully gitignored.

Do not recreate `testartifacts/`; it was retired in favor of explicit locations. Promote sandbox
material only when its role is clear: user-facing runnable example configs belong in `examples/`,
automated-test inputs belong in `tests/fixtures/`, and tutorial-specific assets belong in
`docs/tutorial/`.

---

## 5. Rejected designs — do not re-propose

- **Separate `labeler` slot** (user). D1.
- **A shared `label()` on the `Generator` base class** (user, 2026-09-19). Phase D listed pulling
  `RattleGenerator.label()` up into the base. Rejected for the reason already written into the
  base class's own module docstring: rattle calls the teacher itself, while MD reads energy and
  forces the dynamics already computed to take the step. Producing and labeling are not separable
  for MD-type procedures, so hoisting `label()` would either pay the teacher twice or force a fake
  label step. Each generator keeps its own. The only shared thing is the 5-line "read results,
  `copy()`, reattach `SinglePointCalculator`" detach — a free function at most, not an abstraction.
- **`warm_start_from` as a second checkpoint key** (user). D11.
- **Single `runs.jsonl` provenance file** — split-brain. Settled instead: each stage's record
  lives with its own artifact — sampling's is the state file in `dataset_path`; training's is the
  hydra folder's `.hydra/config.yaml` + `student_state.json` (D15) next to the checkpoint.
- **Two-command workflow (`run:[sample]` then `run:[train,val,test]`) as the sweep-race fix** —
  defeats the one-command product. Still legal per D8, just not the answer.
- **Two hydra processes in one command** — tested, both resolve to the SAME output folder when run
  in the same second (`${now:...}` identical); timing-dependent failure.
- **A second unconditional `ModelCheckpoint(monitor: null, filename: latest)` callback** to fix
  D12a's stale `last.ckpt` (user, 2026-08-28): *"this is nequip behavior, which we dont want to
  change. lets not worry about it."* Left alone on purpose as nequip/lightning's own semantics.
- **Auto-absolutizing `-cp` in a wrapper** (user, 2026-08-28): match nequip's behavior, do not
  diverge from it for convenience.
- **GPU tests in the automated suite** (user). See §6's byte-identity finding for why this is a
  correctness decision, not just a speed one.

---

## 6. Measured facts — other people's code

Env: `/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311` (conda prefix, no
activate). nequip **0.17.1**, lightning 2.6.1, hydra 1.3.2, torch 2.11.0+cu128.
`pyproject.toml` says `nequip>=0.17.1` with NO upper bound (user, 2026-08-28 — wants it usable
with nequip 0.19), **so everything in this section is UNTESTED above 0.17.1.** If it misbehaves,
re-check the two things the student side leans on: `nequip.scripts.train.main` taking a config as
its first positional arg, and `run_stage` restart semantics.

**Current package state (2026-09-08, smoke-validated):** `nequip311` now has nequip
**0.19.1** installed, which PyPI reports as the latest nequip release. The first integration
assumption still holds at the signature level: `nequip.scripts.train.main(config:
omegaconf.dictconfig.DictConfig) -> None`. `tests/e2e/test_distill_cli.py` passed all 15 cases with
this nequip version. `nequip-allegro` is still one patch behind (`0.8.2` installed, `0.8.3`
latest), and `pip check` reports unrelated environment issues (`aimsgb` missing
`mp-api`/`pymatgen`; `sympy 1.14.0` vs `mpmath 1.4.0`).

### Teacher → ASE calculator
`nequip.integrations.ase.NequIPCalculator` (`nequip/ase/nequip_calculator.py` is a deprecated
shim, don't use).
- `from_compiled_model(...)` — `nequip-compile` output only (`.nequip.pth`/`.nequip.pt2`).
- `_from_saved_model(...)` — only path for `.nequip.zip` (nequip-package) or raw `.ckpt`.

### nequip's train entry
`nequip/scripts/train.py`: required top-level sections `run`, `data`, `trainer`,
`training_module`. `run` accepts ONLY `train`/`val`/`test`/`predict` (or a dict w/ `function` key,
UNIMPLEMENTED upstream); at most one `train`. `sample` must be stripped before delegating.

**Two live traps, both from reading 0.17.1's source, both handled in `distill.py`:**
- nequip does `if "ckpt_path" in config:` — PRESENCE, not value. A config spelling out
  `ckpt_path: null` would enter the restart branch and `torch.load(None)`. So the key is POPPED
  when it is None.
- D11's grown-dataset trap. See D11.

### Path A vs B precedence (drives D9/D11)

| Section | On `ckpt_path` restart (Path A) | Guard |
|---|---|---|
| `training_module` | checkpoint wins, live config ignored | warns config ignored |
| `data` | live config, always re-instantiated | none |
| `trainer` | live config, always | none |
| `run` | live config decides stages+order; ckpt's `run_stage` decides only START INDEX | assert live `run` matches ckpt `run` as prefix |
| `global_options` | live, folded into `info_dict` only | none |

Path B = `ModelFromCheckpoint`/`ModelFromPackage` as `model` builder — no conflict guard, only a
version-string warning (`nequip/model/saved_models/checkpoint.py:64-71`).

### D12a. `last.ckpt` IS NOT THE NEWEST EPOCH under a monitored callback
MEASURED, lightning 2.6.1. Documented behaviour, not a bug — `save_last` is relative to SAVES,
not to epochs. `model_checkpoint.py:117`: "saves a `last.ckpt` copy whenever a checkpoint file
gets saved". Mechanism at `model_checkpoint.py:514-517` — `_save_last_checkpoint` runs only `if
self._last_global_step_saved == trainer.global_step`, i.e. only when the top-k save fired at that
same step; with a `monitor`, only on an improvement. `on_train_end` (line 536) saves last only
`if not self._last_checkpoint_saved`, so it does not fix it either.

Measured in PURE lightning (no nequip), 5 epochs, metric worsening every epoch so best = epoch 0,
newest = epoch 4:

| `ModelCheckpoint` args | `last.ckpt` |
|---|---|
| `monitor=<metric>, save_top_k=1` (our config, AND nequip's tutorial) | epoch 0 — the best |
| `monitor=<metric>, save_last="link"` | epoch 0 — the best |
| `monitor=<metric>, save_top_k=-1` | epoch 4 — newest |
| `monitor=None` | epoch 4 — newest |

Not a short-run artifact (checked at 5 and 6 epochs, and against a 20-epoch GPU run where best
happened to BE the last epoch, which hid the effect).

**Consequence: `ckpt_path=.../last.ckpt` resumes from the last IMPROVING epoch and silently drops
everything after it.** Harmless on a still-improving run, expensive on a plateau. Fix REJECTED,
§5.

### Workflow state
`nequip/scripts/_workflow_utils.py::set_workflow_state(state)` asserts `state in
["train","package","compile",None]` — cannot register `"distill"`/`"sample"`, private module,
don't try.

### Datamodules
Base `nequip.data.datamodule.NequIPDataModule`. On-disk ASE files:
`nequip.data.datamodule.ASEDataModule`, kwargs incl. `train_file_path=[]`, `val_file_path=[]`,
`test_file_path=[]`, `split_dataset=[]`, `exclude_keys=[]` — what D5/D9's 3-file output plugs into.

**`exclude_keys` is a NON-ISSUE for our provenance keys** (checked, do not "fix" it):
`nequip/data/ase.py:55` builds `include_keys` from `ase_all_properties` + the user's
`include_keys`, and reads ONLY those out of `atoms.info`/`atoms.arrays`. `base_frame`,
`base_frame_key`, `variant`, `md_step` are not ASE calculator properties, so nequip never looks at
them. The `../distillation` crash was `dipole`/`free_energy` — real ASE properties, present on
some frames only. Our frames all carry exactly energy+forces from one `SinglePointCalculator`.

### No prior art in nequip core
No generator, rattle, active learning, MD driver. `nequip/data/_sampler.py::PartialSampler` is an
unrelated torch DataLoader generator. Only MD code is `nequip/ase/nosehoover.py::NoseHoover` — an
NVT thermostat class, not a driver.

### Compiled artifacts — currently BLOCKED
- TorchScript compile is DEAD: `nequip-compile --mode torchscript` → `ValueError: TorchScript
  compilation is deprecated and not supported in PyTorch >= 2.10`. AOTInductor is the only mode →
  compiled artifacts always GPU-arch-locked (an A100-compiled artifact is invalid on H200).
- AOTInductor compile needs `module load cuda/12.9.1-fasrc01` (env's pip nvidia headers
  incomplete — `fatal error: crt/host_defines.h`; setting `CPATH` alone is NOT enough).
- **`nequip-compile --target ase` AOTInductor `.pt2` SEGFAULTS.** Compiles fine, calculator builds
  fine. On the CDP model: crashes at first forward (`get_potential_energy()`). On the OAM-S model
  (2 attempts): crashes IN THE COMPILE ITSELF instead, right after the same warning. Crash POINT
  is inconsistent across model/attempt — UNRESOLVED, root cause unconfirmed. Suspect the export
  warning `aten._linalg_det.default is missing a c-shim implementation, using proxy executor as
  fallback` — confirmed `linalg_det` is in NO c-shim header in installed torch 2.11, and the op is
  unavoidable (`nequip/nn/grad_output.py:249` does `torch.linalg.det(cell)` for the stress volume,
  so every periodic model hits it). Packaged `.nequip.zip` on the SAME model works fine — isolated
  to the `.pt2` path. Both OAM-S attempts were MIG slices (`nvidia_a100_3g.20gb`); the full-A100
  control job was cancelled while still PENDING, so **MIG-vs-full is UNTESTED** and the c-shim
  story remains correlation, not proof.
- `nequip-compile` from a `.ckpt` needs the checkpoint's training data (rebuilds datamodule →
  `FileNotFoundError`); from a `.nequip.zip` package it does not.
- gpu_test node used = A100-SXM4-40GB, compute_cap 8.0.

### Teacher determinism on GPU — why tests are CPU-only
Two runs of the real teacher over BIT-IDENTICAL geometry disagree by ~3e-6 eV / ~4e-6 eV/Ang
(float reductions not associative, order not fixed — MEASURED between two uninterrupted runs).
**So byte-identity of extxyz is NOT a valid criterion whenever the teacher runs on GPU.** For a
manual GPU check use instead: same structure set, same split per structure, positions/cell EXACTLY
equal, |dE| and |dF| < 1e-5.

---

## 7. Rattle prior art + ASE gotchas

### Rattle algorithm — reused, don't reinvent
`../distillation/scripts/gen_synthetic_geoms.py` algorithm, now implemented in `rattle.py`: per
base frame, one structure per entry in `strain_magnitudes` (isotropic volume scaling
`(1+strain)^(1/3)`) + `n_random_strain_samples` random anisotropic strains, each followed by
per-atom rattle bounded by `max_displacement_ang`. Variant-major ordering (confirmed): every base
frame gets variant 0 before any gets variant 1.

**Anisotropic strain bound decoupled from `strain_magnitudes`.** `RattleGenerator` kwarg
`anisotropic_strain_magnitude: float = 0.05` replaces the derived `max_strain_magnitude =
max(abs(strain_magnitudes))` inherited from `gen_synthetic_geoms.py`. The derived form coupled two
knobs: adding one isotropic scan point silently widened the anisotropic distribution with no
change to any structure's name, so nothing could detect it (measured before fix: adding a strain
magnitude left 20/20 `iso` structures bit-identical but changed 10/10 `aniso`; after fix, all 30
bit-identical). Default 0.05 matches the old expression's value for the default
`strain_magnitudes`, so default behaviour is unchanged.

**LATENT BUG, impact unconfirmed — ASE stores lattice vectors as ROWS.** Straining should be
`cell @ F.T`; `rattle.py` does `F @ cell`. Identical for a CUBIC cell (every base-frame set used
so far — CsH2PO4 sandbox inputs, Si `sitraj.xyz` — is cubic), so nothing is observably wrong today;
diverges for hexagonal/triclinic. Fix if a non-cubic base frame is ever used.

**Hard-won magnitude lesson**: ±10%/5% strain + 0.5 Å displacement pushed frames OOD for the
teacher, and students came out WORSE than with no synthetic data at all. Halved defaults (±5%/2.5%
strain, 0.25 Å displacement — `rattle.py`'s current defaults) fixed it. Rattle displacement must be
measured against the STRAINED parent, not the unstrained one (comparing to unstrained overstates
spread) — confirmed correct in `rattle.py`.

### ASE gotchas that bit the generators
- **Snapshot every kept frame w/ `atoms.copy()`.** ASE MD mutates one `Atoms` in place; appending
  the live object gives N copies of the final step.
- **`atoms.copy()` drops `atoms.calc`.** Reattach: `snap.calc = SinglePointCalculator(snap,
  energy=e, forces=f)` (`ase.calculators.singlepoint`). Without it, extxyz gets geometry with no
  labels — discovered only at student-training time.
- ASE `Langevin(fixcm=True)` (current `md.py` default) does NOT strictly sample the correct NVT
  distribution — deprecated since ASE 3.28, fix is `fixcm=False` + `ase.constraints.FixCom`. Left
  alone because MD is scaffolding; becomes a real correctness issue if MD output is trusted.
- numpy 2: `ndarray.ptp()` is gone, use `np.ptp(arr)`.
- Don't name a scratch script `inspect.py` — shadows stdlib, breaks numpy/ase import.
- **Determinism-test trap:** `frames[0].copy()` has the SAME content hash as `frames[0]` by
  construction (D7a's `frame_key`) — testing "adding a base frame leaves existing structures
  identical" with a copied frame passes trivially and proves nothing. Use a genuinely unused frame.

### Artifact gotchas inherited from `../distillation/`
- `nequip-package` output must never be relocated after creation — path baked in (one
  `.nequip.zip` did survive a copy, don't rely on that holding in general).
- `ASEDataset` auto-includes `energy`/`forces` regardless of `include_keys`; inconsistent
  per-frame extra ASE properties (`dipole`, `free_energy`) crash batching.
- `nequip-train -cp <path>` needs an ABSOLUTE path. Same applies to `nequip-distill`.

---

## 8. Open problems

1. **Sweep race.** Parallel sweep runs would all sample into the same 3 files simultaneously. Fix =
   lock file in `dataset_path`, undesigned. (Two-command workaround rejected, §5.)
2. **Growing a dataset changes SPLIT ASSIGNMENT.** Re-running `assign_splits` at a larger total
   MOVES already-written items between files. Items on disk must keep their split; new items must
   be apportioned to close the gap to the target sizes at the new total. Rattle splits per base
   frame (so the map must be keyed by frame hash, not list index); MD splits per snapshot, and with
   `blocked` each extension gets its OWN contiguous val/test tail rather than one tail for the whole
   trajectory — a real cost, flag it before building.
3. **MD `sample_interval` decorrelation (D4) is an ASSUMPTION, unmeasured.** If false,
   per-snapshot scattered splitting leaks; the `blocked` default mitigates, nothing detects it.
4. **MD params in `md.py` are placeholders, not recommendations** (`friction_per_fs=0.01`, 5-step
   equilibration in an earlier smoke run).
5. **Teacher compiling.** Plan is to DEMAND a ready calculator-compatible artifact and do no
   compiling in this repo. Confirm with user before adding — and note §6's `.pt2` segfault makes
   "bring a fast compiled teacher" unshippable advice right now.
6. **The teacher in `sandbox/inputs/` is a *student* model from `../distillation/`** — a
   stand-in, fine for plumbing, not a real teacher. Free sanity check: teacher E on an unrattled
   base frame vs its DFT E — −374.61 vs −374.51 eV on frame 0 (64 atoms, ~0.1 eV total).

---

## 9. Next steps (ordered)

Order reflects the user's call to get a runnable `nequip-distill` out first and improve resume
behaviour after. Follow `CLAUDE.md`'s working paradigm for every one — one at a time, plan first.

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
3. **Growing a dataset** — §8 item 2.
4. **`student_path` (D15)** + `student_state.json` + D14 archive hook. Deferred ON PURPOSE (user,
   2026-08-28): the student uses nequip's normal checkpoints-in-the-hydra-dir semantics for now, so
   a training restart needs an explicit `ckpt_path`.
5. **Warm start on a grown dataset** (D11 Path B, `ModelFromCheckpoint` rewrite). Until then
   `distill.py` REFUSES that combination rather than silently training on nothing.

---

## 10. Tutorial — `docs/tutorial/`

Upstream `mir-group/nequip-tutorial` VENDORED VERBATIM @ `8f90935` (2026-03-09, MIT):
`NequIP_Tutorial.ipynb` (32 cells), `config.yaml`, `config_finetuning.yaml`, `sitraj.xyz` (110
frames, 64-atom Si, PBC), `README.md`, `LICENSE`. Never hand-edit — `SOURCE.md` holds the refresh
recipe. The tutorial is NOT in the nequip repo itself.

Ours:
- `distill_section.md` — our section, one cell per `## [markdown]` / `## [code]` marker. Preamble
  and `<!-- TODO -->` comments stripped at build. Written to match upstream's terse 1–3-paragraph
  cells (user rejected a 3x longer first draft).
- `distill.yaml` — Si distillation config. Teacher = fine-tuned OAM-S, packaged from
  `results_ft/best.ckpt`. 110 base frames x 5 variants = 550 structures, 88/11/11 base frames →
  440/55/55. Student: 2 layers, `l_max: 1`, `num_features: 32`, `[Si]` only, r_max 5.0, ZBL on.
  Heavily commented — detail lives THERE, not in the prose.
- `build_notebook.py` → `NequIP_Distill_Tutorial.ipynb` (38 cells = upstream verbatim + our 6).
  GENERATED. Edit the md/yaml and re-run; never edit the notebook.

**NEVER RUN LOCALLY.** Colab installs current nequip, not the 0.17.1 everything here was validated
against. User tests in Colab.

Colab link needs the repo PUBLIC and pushed to `main`:
`colab.research.google.com/github/Steinburglar/nequip-academy/blob/main/docs/tutorial/NequIP_Distill_Tutorial.ipynb`.
Two URLs hardcode `Steinburglar/nequip-academy@main` (the `pip install git+` cell and the `wget` of
`distill.yaml`) — both live in `distill_section.md`; change there and regenerate.

**Validated on GPU (job 42612714, `gpu_test`, 8m06s, 2026-08-28)**: the packaged-teacher config
runs end to end. 550 structures sampled in 18 s → 440/55/55 exactly as the config predicts, student
trained, `TEST RUN END`, `best.ckpt` 997 KB vs teacher 9.9 MB. Student vs teacher on held-out:
per-atom energy MAE 6.0 meV/atom, forces MAE 0.425 eV/Ang (early-stopped at epoch 97 of 200, so
converged by its own criterion, not truncated). Colab-specific cells (install-from-git, `/content`
paths) still UNTESTED.

**Compiled teacher REVERTED, do not re-add until a working `.pt2` exists** — §6's segfault.
`max_epochs: 200` / `max_time: 00:00:30:00` deviate from upstream's 1000 / 3 days — user's call to
keep, do not "fix" to match upstream.

**Upstream bug, do not inherit:** notebook cell 29 compiles `/content/results/best.ckpt` — the
FROM-SCRATCH model — and cell 30 plots it as "Fine-tuned model". The real one is
`/content/results_ft/best.ckpt` (`results_dir: ./results_ft`). Our section uses the correct path.
Worth reporting upstream.

### Tutorial open questions (user, 2026-08-28)

1. **The test split measures FIDELITY TO THE TEACHER, not accuracy.** Splitting is per base frame,
   so val/test structures do come from base frames that fed nothing into train — a real geometric
   holdout. But every label is the teacher's, so `test0_epoch/*` answers "how well does the student
   imitate the teacher", and CANNOT be compared against the teacher's own DFT test error. Fix = a
   final eval on held-out DFT frames. Note both contaminations: all 110 `sitraj.xyz` frames are
   base frames, AND `config_finetuning.yaml` feeds all 110 to the teacher via `split_dataset`
   0.8/0.1/0.1. So a clean benchmark must be carved out BEFORE both the finetune and the sampling.
   Mentioned in the tutorial text, NOT implemented.
2. **The Si strain/rattle magnitudes are GUESSES.** ±5% volumetric, 5% anisotropic, 0.25 Ang were
   tuned for CsH2PO4 in `../distillation`, not for silicon, and were carried over unchanged. Worth
   answering IN the tutorial, since every user faces it: how do you know the right magnitudes?
   Candidate diagnostics, none implemented — compare generated min interatomic distance against the
   source trajectory's; compare rattle displacement against thermal displacement at the
   trajectory's temperature; compare strain range against the volume fluctuations MD actually
   visits; teacher-ensemble disagreement as an OOD flag. The one hard datum stays the CDP result:
   too wide made students WORSE than no synthetic data at all.
3. The rattle recipe in the tutorial is exactly `rattle.py`'s, not Si-tuned — §7, including the
   cubic-only `F @ cell` latent bug (applies here too; `sitraj.xyz` is cubic, so it's silent).
4. **Missing section: compile + compare FM vs toy vs student, on accuracy AND speed.** This is what
   makes the point land, and it is the payoff the section currently only asserts. Needs (1)'s DFT
   holdout for the accuracy half, and a working compile for the speed half — §6.

---

## 11. Datamodule port + generation refactor

*Rewritten 2026-09-18. Supersedes the 2026-09-16 version of this section, which described the
same destination in terms the user could not build intuition from ("generator only delivers the
next frame"). The structures below are close to what that version proposed; the ownership story
is the part that changed.*

### 11.1 The boundary, in one picture

The old `Generator` (429 lines) held two unrelated jobs. Almost all of its bulk is job 1:

- **Physical** — "does the byte content on disk match the record?" Offsets, sizes, truncation,
  digests, atomic write. *Zero generator knowledge.* Universal for every procedure, forever.
- **Semantic** — "would this config have produced these frames, and where in the procedure am I?"
  Provenance rules, resume position, finished-ness. *Entirely generator-owned.*

Everything below follows from cutting there and nowhere else.

**The generator's size was never the problem.** `rattle.py` is 212 lines and is essentially pure
science; `generator.py` is 429 and is essentially all file mechanics. Extract the physical layer and
the generator is already small. Do NOT invent further abstractions to shrink it — in particular,
do not take its state away from it. A generator owning a rich, procedure-specific state dict is
correct; MD's state (positions, velocities, RNG bit generator, step count) is irreducible
complexity that belongs to MD.

### 11.1a Portability invariant (user, 2026-09-18)

**Generation must never require a datamodule.** The datamodule is a caller, not the owner.

Three reviewable properties, in order of how easy they are to check:

1. `sample/`, `data/store.py`, `data/state.py` and `data/paths.py` import **no** nequip, hydra,
   lightning or omegaconf. Grep-checkable, so it cannot rot quietly. Only `data/datamodule.py`
   is allowed those.
2. `Generator.generate()` is the pipeline and stays there. A standalone script constructs a
   generator and calls it — that is exactly the script used to verify the byte-identity claims
   in Phases A and B, and it never touched the datamodule.
3. Wherever coordination code sits, **the pipeline half must contain no `self`.** A
   `prepare_data()` that spells the steps out is fine provided its body splits visibly into an
   adapter half (hydra config -> objects, teacher lifecycle: legitimately datamodule-specific,
   a script replaces it with direct construction) and a pipeline half that touches only local
   variables and is therefore paste-able verbatim.

**Known wart:** a standalone script must set `generator.generation_config` by hand or resume refuses
with "was not given the config". That field exists only because hydra's `instantiate` would build
the teacher twice if it were a constructor argument -- a datamodule concern leaking into the
standalone path. The clean fix is for a generator to derive its own provenance from its
constructor arguments instead of being handed a config dict. Not scheduled.

### 11.2 The record on disk — three sections, one writer each

**The governing rule: each section has exactly one writer, and that writer is also its checker.**
This is the intuition the whole design hangs on.

```python
{
    "version": 2,          # store writes, store checks
    "provenance": {...},   # generator writes, generator checks
    "contents":   {...},   # store writes, store checks
    "progress":   {...},   # generator writes, generator restores
}
```

| section | contents | writer / checker | why there |
|---|---|---|---|
| `version` | record format | store | the format of the file the store itself writes; nothing can be interpreted until it is trusted |
| `provenance` | generator class name, immutable settings, base-frame digest, teacher identity | generator | only the generator knows which knobs are load-bearing |
| `contents` | per-split byte `offsets`, per-split `digests`, `n_written`, `split_counts` | store | only the store touches bytes |
| `progress` | procedure position (rattle `n_steps`; MD positions/velocities/RNG) | generator | procedure-specific by nature |

**The store does not check which procedure produced a dataset** (user, 2026-09-18). An earlier
draft gave it a `generator_class` header field to guard. That is provenance: the generator carries
its own class name there and refuses a mismatch itself, which removes a whole compatibility layer
from the store and gives a better message, since only the generator can say *why* two procedures
are incompatible. Order makes it safe -- `check_compatible` runs before anything asks a generator
to restore another procedure's `progress`.

Offsets and digests are deliberately **not** part of generator progress. If they were, `state()`
would have to produce them, which forces generator code to know about byte offsets — precisely the
leak being removed. They are also not provenance: provenance is fixed for the dataset's life,
contents change at every checkpoint. Different lifetimes, different sections.

`n_written` / `split_counts` are store counters, because the store is what appends. Generators may
keep their own counters, meaning "where I am in the procedure". For rattle the two agree by
construction; a disagreement is a detectable bug (optional check, not load-bearing).

**One file, not several** (settled). The sections must be written atomically *together* — a torn
pair of files is strictly worse than a torn single file. `store.save_record(provenance=...,
progress=...)` injects `contents` itself and does one `torch.save` + `os.replace`.

### 11.3 The three components

```text
nequip_academy/
  data/
    paths.py        SPLITS, split_file()                    [exists]
    state.py        record I/O, diff mechanism, truncation  [exists, needs one fix — 11.8]
    store.py        SampleStore                             [to add]
    datamodule.py   DistillationDataModule                  [exists, thin]
  sample/
    generator.py      Generator base                          [to shrink]
    rattle.py       RattleGenerator                         [science only]
    md.py           MDGenerator                             [science only]
    split.py        split assignment policy                 [unchanged]
```

**1. `SampleStore`** — the thin I/O object. Owns `dataset_path` and, conceptually, nothing else.

```python
append(atoms, split)                 # write, track offsets + counts
n_written / split_counts             # counters
offsets() / digests()
save_record(provenance, progress)    # atomic; injects `contents`
load_record()                        # None | record; checks `version` only
reconcile(contents)                  # disk + counters <- record
refuse_orphan_files()                # split files with no record
```

Three things must agree: the bytes on disk, the record, and the store's own counters. On a fresh
run all three start empty and stay in step as `append` runs; on a resume the process is new, so
the counters are zero while disk and record may be well along. `reconcile(contents)` puts them
back in step -- refuse a short file, truncate a torn tail, compare digests, then seed the
counters. One call, because the order is forced (a torn tail must be cut *before* digests can
match) and because seeding is only correct *after* the repair.

The store's only state is those counters. Offsets are free from `stat()`, but `n_written` and
`split_counts` cannot be recovered without parsing every structure, so they are carried.

`contents` is an argument rather than a cached record, so the sequencing stays visible in the
datamodule and `reconcile` has no hidden dependency on `load_record` having run. `load_record`
deliberately does not seed the counters: the provenance check sits between the two, and a run
refused there must leave the store untouched.

**2. `Generator`** — the science, and its own state.

```python
provenance()          -> dict    # what defines "the same run"
immutable_keys()      -> set     # which of those may not change
state()               -> dict    # resume position
restore(state)        -> None    # put me back there
finished              -> bool
attach_calculator(c)  -> None
step(store)           -> None    # produce next structure(s); store.append(...)
```

A generator never implements a safeguard. It only *declares* which knobs are load-bearing; the
data layer does the comparing and all the yelling.

**3. `DistillationDataModule`** — owns the restart algorithm and nothing else. Writes nothing,
checks nothing; it only sequences.

### 11.4 The restart algorithm — the acceptance test for this design

Shown below as a free function to make 11.1a's point: it names only `generator` and `store`, so
it is paste-able into a script. Today it lives as `Generator.generate()` and `prepare_data()` calls
it; either placement is fine so long as no `self` appears in these lines.

```python
def prepare_data(self):
    generator = instantiate(self.generation)        # teacher NOT loaded yet
    store = SampleStore(self.dataset_path)
    record = store.load_record()

    if record is None:
        store.refuse_orphan_files()
    else:
        generator.check_compatible(record["provenance"])   # (a) semantic, cheapest, first
        store.reconcile(record["contents"])                # (b)+(c) physical: trim, then digest
        generator.restore(record["progress"])              # (d)
        if generator.finished:
            return                                         # teacher never loaded

    generator.attach_calculator(instantiate(self.teacher))
    run(generator, store, self.state_interval)
```

Order is load-bearing: the semantic check is cheap and runs first, so a config mismatch never
causes a truncation. **If this function stops reading like that checklist, the boundary is wrong.**

### 11.5 Splitting the checks — mechanism vs policy

The apparent tension ("some checks are generator-specific") dissolves one level down:

| check | mechanism (data layer) | policy (generator) |
|---|---|---|
| config compatibility | `diff_configs(stored, live, immutable_keys)` | supplies `immutable_keys`, incl. its own class name |
| base-frame identity | `frames_digest()` | decides that base frames *are* provenance |
| size vs offsets, truncation | total | none |
| content digests | total | none |
| resume position | serialize/restore contract | supplies + consumes the dict |
| record format version | total | none |
| generator class match | none -- it is one provenance key | the generator's own |

### 11.6 Target user-facing shape (settled, already implemented)

Primary path is ordinary NequIP training — no bespoke CLI:

```bash
nequip-train -cp /abs/path/to/configs -cn distill
```

```yaml
run: [train, val, test]

data:
  _target_: nequip_academy.data.DistillationDataModule
  _recursive_: false
  seed: 1
  dataset_path: /scratch/me/cdp_distill_samples
  state_interval: 50

  teacher:
    _target_: nequip.integrations.ase.NequIPCalculator._from_saved_model
    model_path: /scratch/me/teacher.nequip.zip
    device: cuda
    chemical_species_to_atom_type_map: true

  generation:
    _target_: nequip_academy.sample.RattleGenerator
    base_frames: /scratch/me/base_frames.xyz
    split: {train: 0.8, val: 0.1, test: 0.1}
    split_policy: scattered
    split_seed: 0
    strain_magnitudes: [-0.05, 0.0, 0.05]
    n_random_strain_samples: 2
    anisotropic_strain_magnitude: 0.05
    max_displacement_ang: 0.25
    seed: 1

  transforms: [...]          # ordinary NequIP
  train_dataloader: {...}    # ordinary NequIP
```

- `_recursive_: false` is load-bearing: without it Hydra instantiates `data.teacher` during
  DataModule construction, loading the teacher onto a GPU possibly for a dataset that is already
  complete.
- Measured (NequIP 0.17.1): `ASEDataModule.__init__` only converts file paths into `ASEDataset`
  configs; it does not read them. So the subclass can compute split paths and call
  `super().__init__(train_file_path=..., ...)` before the files exist, then generate in
  `prepare_data()` before `setup()` builds datasets.
- Generation finishes entirely in `prepare_data()`, before dataloader workers exist. No public
  "distillation Dataset" with generation side effects — `ASEDataset` stays boring.

Config migration removes: top-level `dataset_path`, top-level `generator`, the `sample` run type, code
that patches `data.*_file_path`, and the hard requirement that `data._target_` is `ASEDataModule`.
Keeps: three pre-split extxyz files, no NequIP `split_dataset` for generated data, split membership
frozen at generation time, restart/provenance guardrails.

### 11.7 Settled sub-decisions

1. **One record file, three sections** (11.2). Atomicity.
2. **Digests are stored and verified on resume.** Offsets alone catch the realistic failure (torn
   write of an append-only file we wrote ourselves); digests additionally catch mid-file edits and
   wrong-dataset. At 50–200 structures the extra read is free. Revisit if datasets reach ~10^5.
3. **`attach_calculator()` replaces `calculator=None`.** The no-teacher fast path is currently a
   hack — construct with a `None` calculator and hope nothing touches it. Attaching makes the fast
   path the normal construction rather than a special case.
4. **`label()` moves out of `RattleGenerator` to a shared module function.** Teacher call +
   `SinglePointCalculator` has nothing to do with rattling. Caveat: generators cannot uniformly
   yield *unlabeled* frames — MD needs the calculator during integration and gets its labels free
   from the dynamics, while rattle must ask. So: shared helper, generator decides when to call it.
5. **Enumerable vs sequential generators is a real property, deferred to 11.10.** Rattle can name
   every structure it will make up front; MD cannot. Design must not foreclose it.

### 11.8 Current implementation state (2026-09-18)

Committed (`9a6f92f`, `b02b74c`):
- `DistillationDataModule` subclasses `ASEDataModule`, computes split paths in `__init__`,
  generates in `prepare_data()`, has a no-teacher fast path for complete datasets.
- `data.generation._target_` still points at the existing generator classes.
- Tracked example `examples/rattle_train_datamodule.yaml`.
- Generation still driven by the old `Generator.generate()`; restart still owned by generator
  machinery (`generation_state.pt`, config diff, base-frame digest, byte offsets, truncation,
  `restore_progress()`).

Uncommitted working tree — the old plan's "Phase 1", mechanical extraction, complete and green
(`pytest -q` → 46 passed):
- `data/paths.py`, `data/state.py` added; `generator.py` −130 lines, now delegating wrappers;
  `datamodule.py` uses the shared refusal helper; `tests/unit/data/test_state.py` (17 tests).

**One correction this rewrite forces:** `data/state.py::check_goal()` takes `base_frames` and
hashes them. Base frames are a generator input — that is semantic policy sitting in the physical
layer. It moves to the generator; `state.py` keeps only the shared diff mechanism. Everything
else in `paths.py`/`state.py` survives.

### 11.9 Phase plan

Each phase is one approved step, runnable at its end, with the full suite green. No bundling.

**Phase A — `SampleStore`.** Add `data/store.py` wrapping the existing `state.py` free functions in
an object holding `dataset_path` + counters. `Generator` uses it internally and delegates; public
behavior and record shape (v1) unchanged. Purely mechanical.
- Check: `pytest -q`; generated split files byte-identical to before.

**Phase B — provenance moves to the generator.** Add `Generator.provenance()` /
`immutable_keys()` / `check_compatible()`. Remove the `base_frames` leak from `state.py`, leaving
a generic `diff_configs()`. Record still v1 shape; the `goal` section is now assembled by the
generator.
- Check: `pytest -q`; resume-refusal messages unchanged in substance.

**Phase C — record v2.** DONE 2026-09-18. Three sections per 11.2, flat; the v1 translation is
deleted and a format-1 record gets a refusal that says to regenerate rather than a generic
unknown-format message. `contents` gains a per-split `digests`, checked by `reconcile` after
truncation.

Two things learned here:
- **An empty split file and an absent one must digest the same.** Truncating back to offset zero
  leaves a file that exists and is empty, while a record written before that split was ever
  touched has `None` for it. Hashing the empty file gives sha256("") and a perfectly good resume
  gets refused as a content mismatch. `digests()` returns `None` for zero bytes, absent or not.
  Found by the existing resume tests, not by inspection.
- **`save_record` re-reads all three files.** Quadratic in the number of checkpoints. At the scale
  here -- hundreds of structures, a megabyte or two, a teacher call per structure -- it is
  microseconds. A large dataset should raise `state_interval`. An incremental hash carried across
  appends would fix the asymptotics and was rejected as optimising for a size nobody runs.

**Phase D — remove the duplicated restore path.** DONE 2026-09-19. The `label()` extraction
this phase also listed was REJECTED (user, 2026-09-19) — see §5.

`Generator.calculator` is now optional and `generate(teacher_factory=...)` calls the factory only
after establishing there is something left to produce. `_finished_without_teacher` is deleted, as
is `_instantiate_generator`'s `load_teacher` flag.

What this did and did not change: the laziness itself already worked -- that was the point of
`_finished_without_teacher`. What it removes is doing it with TWO generators. On the resume path
the old code built the generator, read the record, checked settings and reconciled (which since
Phase C hashes all three split files), then threw that generator away and did all of it again
inside `generate()`. It also removes `calculator=None` as an invalid state the surrounding code
merely tiptoed around: a generator without a calculator is now documented as able to inspect and
resume but not step.

Care taken: an early `return` for a finished dataset would have skipped the final `write_state()`
that `generate()` always did. The guard is on acquiring the teacher only, so everything else below
still runs and a finished run rewrites its record exactly as before.

**`Generator.generate()` STAYS** (user, 2026-09-18). An earlier draft of this phase had
`prepare_data()` absorb the loop and `generate()` deleted; that would make nequip a hard
dependency of generating data and breaks 11.1a. The datamodule remains a caller.
- Check: `pytest -q`; `pytest -m e2e`; 11.1a property 1 still greps clean.

**Phase E — rename.** DONE 2026-09-19. `Generator` → `Generator`, `RattleGenerator` →
`RattleGenerator`, `MDGenerator` → `MDGenerator`, module `sample/generator.py` → `sample/generator.py`,
`dataset_path` → `dataset_path` (a user-facing yaml key), `generation_config` → `generation_config`,
`generation_state.pt` → `generation_state.pt` (free: format 2 already refuses every older record, so
the filename carries no compatibility weight). Error messages say "generator".

Two departures from the phase as originally written, both the user's call (2026-09-19):
- **No aliases.** The plan said keep the old names as aliases. The package is unreleased and has
  no external importers, so an alias is pure debt on a repo being cleaned up for shipping.
- **`docs/tutorial/` NOT updated.** It still documents the deleted `nequip-distill` CLI and is
  already broken; renaming classes inside it is churn on a file that needs a rewrite anyway. It
  stays on the known-debt list until the port is decided.

The package directory stays `sample/` and `SampleStore` keeps its name — "sampling" is still the
right word for what rattle and MD do; only the class that drives the procedure became a generator.
- Check: `pytest -q` (72 passed); `pytest -m e2e` (7 passed); 11.1a property 1 greps clean.

**Phase 0 (done, 2026-09-18) — `nequip-distill` deleted.** User's call: delete outright, with no
generate-only replacement for now. Removed `scripts/distill.py` and its entry point, `configs/`,
and the three CLI-driven examples; `tests/e2e/test_distill_cli.py` became
`tests/e2e/test_datamodule_e2e.py`, 16 registry cases down to 7 ordinary pytest tests.

Consequences, accepted knowingly:
- **The D11 guard is gone.** Growing a dataset and passing `ckpt_path` now silently under-trains
  the new structures until the checkpoint fingerprint lands (11.10). User accepted the window.
- **No sample-without-training path exists.** Revisit if generating on a GPU node and sweeping
  students elsewhere becomes a real workflow.
- **`docs/tutorial/` and `README.md` still document the deleted CLI.** The tutorial is shipped via
  a public Colab link, so it is broken until ported. Not yet decided.

Two things the rewrite verified rather than assumed:
- **nequip validates the student config before Lightning calls `prepare_data()`**, so the old
  CLI's "fail before burning teacher calls" guarantee survives for free. Covered by
  `test_broken_student_config_fails_before_generating`.
- **The teacher config is provenance**, so the no-teacher fast path cannot be tested by breaking
  the teacher's `_target_` — that trips the settings-changed refusal instead. It is tested with a
  LennardJones subclass that touches a marker file when constructed.

### 11.10 Deferred, after the boundary is stable

Principle: move the architectural boundary first, then improve restart semantics. Do not rewrite
restart logic and the NequIP integration boundary at the same time.

**Compatibility classification.** Replace "any config difference is fatal" with immutable/mutable,
via `immutable_keys()`.
- *Immutable:* generator class, base-frame contents, teacher identity, every sampling knob that
  affects already-written structures, split policy/seed/fractions (unless membership is stored
  explicitly), label schema and key mapping.
- *Mutable:* teacher device, `state_interval`, dataloader settings, transforms/stats, output path
  if the whole directory moved coherently.
- *Conditionally mutable:* sample budget, added rattle variants, added base frames, teacher path
  when a strong hash proves it is the same teacher.
- Default rule: a new or unclassified knob is **fatal** until explicitly classified.
- Messages must name exactly which immutable settings changed and which mutable ones were ignored.

**Teacher identity.** Record identity separately from runtime plumbing. `strict_hash` (hash the
artifact; preferred for science) / `metadata` (path + size + mtime) / explicit user identifier for
non-file calculators. **Device is never part of identity** — `cuda` → `cpu` must not invalidate a
finished dataset.

**Enumerable generators (rattle extension).** Rattle has a stable identity per structure:
`item_id = base_frame_key | variant_label`. If an enumerable generator declares its manifest up
front, progress becomes "how far through the manifest" — universal machinery, and rattle's own
state drops to nearly nothing. Then adding a strain magnitude appends new variants, old structures
stay byte-identical, and new ones inherit their base frame's split. MD stays a sequential generator
with an opaque blob. This asymmetry is a property of the procedures, not an implementation
accident — do not design against it.

Adding *base frames* stays fatal regardless until split membership is stored per `base_frame_key`,
or split assignment becomes a stable hash of the key. Otherwise `random_split(n)` moves existing
base frames across train/val/test.

**MD resume.** Needs procedure-specific trajectory checkpointing: positions, cell,
velocities/momenta, MD step count, snapshot count, NumPy bit-generator state, any Langevin
thermostat state not covered, split assignment. Until then MD refuses partial resume. The
datamodule may own the refusal message; the scientific state belongs to `MDGenerator`.

**Training checkpoint fingerprint.** Move the old CLI `ckpt_path` guard into native Lightning
datamodule state:

```python
def state_dict(self):
    sd = super().state_dict()
    sd["distillation_dataset_fingerprint"] = self.dataset_fingerprint()
    return sd

def load_state_dict(self, state_dict):
    if state_dict.get("distillation_dataset_fingerprint") != self.dataset_fingerprint():
        raise ValueError("checkpoint was trained with a different generated dataset")
    super().load_state_dict(state_dict)
```

Semantics: training crash with unchanged dataset resumes fine; sampling crash resumes generation
then trains; dataset extended + Lightning `ckpt_path` refuses; dataset extended + warm start is an
explicit model-from-checkpoint path, never `ckpt_path`.

**Lock.** `dataset_path/.generation.lock` around datamodule generation. Acquire, re-read record
after acquiring, generate or no-op, release. Lightning's rank-zero `prepare_data()` is not
sufficient — it does not protect against two separate jobs pointing at one `dataset_path`.

### 11.11 Test plan

1. **Store unit tests** — atomic write; append tracks offsets; truncation of excess; refusal on
   short file; refusal on missing file with recorded bytes; digest mismatch; orphan split files.
2. **Generator unit tests** — provenance round-trip; immutable-key diff names the changed key;
   `state()`/`restore()` round-trip; `finished` transitions.
3. **Datamodule unit tests** — constructor wires split paths into `ASEDataModule` configs;
   `prepare_data()` generates; second `prepare_data()` no-ops *without instantiating the teacher*;
   mutable settings do not invalidate a complete dataset.
4. **Rattle integration** — interrupted generation resumes to the same split-file digests as an
   uninterrupted run; split counts preserved; immutable change refused.
5. **Training integration** — `nequip-train` with `DistillationDataModule` runs train/val/test;
   checkpoint resume with the same dataset works; changed fingerprint refuses.
6. **Migration** — a v1 record produces an actionable refusal; an old config translates or fails
   with an actionable error.

Tests stay CPU-only with no teacher (`ase.calculators.lj.LennardJones`) — a correctness decision,
not a speed one. Byte-identity is a valid criterion *only* because no GPU is involved; see §6 for
the measured GPU nondeterminism and the tolerance-based check that replaces it there.

### 11.12 Operational facts established while smoke-testing the datamodule

- **`ckpt_path: null` must NOT appear in a direct `nequip-train` config.** Unlike `nequip-distill`,
  NequIP treats the mere presence of `ckpt_path` as a restart even when the value is `null`, then
  tries to load checkpoint `None` → `NoneType`/`seek` traceback. The tracked datamodule example
  omits it entirely.
- Hydra `_target_` values are import paths, independent of cwd.
- Hydra `-cp` must be absolute; relative `-cp ../configs` gets read as a package path such as
  `nequip.scripts.configs`.
- Ordinary file paths in YAML (`model_path`, `base_frames`, `dataset_path`) resolve against process
  cwd, not the config location.
- The untracked sandbox-local config `sandbox/configs/rattle_train_datamodule_local.yaml` differs
  from the tracked example only in path locality, and is launched from `sandbox/out`:

```bash
cd /n/home12/lsteinberger/code/nequip-academy/sandbox/out
PATH="/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311/bin:$PATH" \
CONDA_PREFIX="/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311" \
HYDRA_FULL_ERROR=1 \
nequip-train -cp /n/home12/lsteinberger/code/nequip-academy/sandbox/configs \
             -cn rattle_train_datamodule_local
```

  Kept untracked deliberately: user-side smoke runs stay contained in `sandbox/` instead of baking
  sandbox paths into the general example.
