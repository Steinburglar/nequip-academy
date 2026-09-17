# TuneandDistill — design record

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
> Sampler will require the most design, most open ended. How should it know which model to use in a script
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

**D1. Sampler owns calculator AND labeling, exactly one calculator, no separate labeler.**
Separate `labeler` slot REJECTED (user). Labeling itself is procedure-specific, not on the base
class: `label()` lives on `RattleSampler` only (static-structure eval); `MDSampler` gets labels
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
`assign_splits(n, ...)` labels `n` *items*; each sampler decides what `n` counts. Rattle → n =
base frames (rattles of one base frame are near-duplicates, must stay together). MD → n =
snapshots (assumes `sample_interval` decorrelates — UNVERIFIED, see §8).
Apportionment itself = `torch.utils.data.random_split` (nequip uses this too,
`nequip/data/dataset/utils.py:55` — don't reimplement). `split.py` only adds: (a) index→label
inversion (need labels before a dataset object exists, since frames are written straight to 3
files as generated), (b) empty-split → hard `ValueError` where torch only warns.
`split_policy` defaults differ ON PURPOSE: `MDSampler` → `"blocked"` (time-ordered, contiguous
tail holdout stays honest if interval under-decorrelates); `RattleSampler` → `"scattered"` (base
frames have no meaningful order). The `examples/*.yaml` configs set it explicitly anyway.

**D5. Output = three files** `train.extxyz` / `val.extxyz` / `test.extxyz` in `sample_path`, NOT
one `samples.extxyz`. Splits are frozen at generation, never recomputed at train time — this is
what keeps growing the dataset from silently moving val/test frames into train (the fraction-
based leakage risk that a single-file + `data.split_dataset`-fractions design would have).

**D6. Restart precedence for sampling: REVERSED from §1's plan.** Live config wins; sampler
diffs stored goal vs. live goal, does not silently ignore the live config. Three-way distinction:
progress state (n_written, offsets, procedure blob) / stored goal / live goal — stored goal is a
diff baseline + legality guard, not behavior-driving.

**PARTLY IMPLEMENTED**: the diff exists (`Sampler.check_goal`) but ANY difference is FATAL,
nothing is a warning yet and nothing is legal yet. Classification = §9 next-step #1.

**D6a. Stored goal = a COPY OF THE `sampler` CONFIG, not a hand-picked list of params (user,
settled).** A hand-written `goal()` per sampler duplicates every ctor kwarg and silently omits any
new one — the exact shape of the `anisotropic_strain_magnitude` bug (§7). `distill.py` hands over
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

**D7. Never pickle sampler/calculator.** Three reasons, in order: (a) a pickled sampler carries
its OLD config, and D6 says the live config wins — so the pickle buys nothing and makes it easy
to miss an attribute; (b) compiled/CUDA-bound torch models are non-portable across nodes/GPU
archs; (c) a pickle is coupled to class layout, so renaming an attribute silently breaks every
existing `sample_path` — a plain dict breaks loudly via the `version` check instead.

**Resume, as implemented.** `sampler_state.pt` beside the dataset holds `{version,
sampler_class, goal: {config, base_frames}, progress: {n_written, split_counts, offsets,
procedure}}`. `offsets` = byte length of each split file. Written every `state_interval`
structures (default 1) + at end, atomically (`.pt.tmp` + `os.replace`). `generate()` order on
resume: read → restore counters → `check_goal` → `truncate_to` → `restore_progress`.
Refuses: unknown `version`, different `sampler_class`, split files with NO record, file shorter
than recorded offset, `sampler_config is None`, ANY config difference. File LONGER than offset →
truncated + warned (fires even at `state_interval=1`, measured).

**D7a. RNG derivation is ASYMMETRIC between samplers — load-bearing for resume.**
- `RattleSampler`: no streaming RNG. Each structure seeded by `derive_seed(seed,
  frame_key(base_frame), variant_label)`, both module-level fns in `rattle.py`. `frame_key` =
  sha256 of numbers + positions (rounded 8) + cell (rounded 8) + pbc, truncated 16 hex chars.
  Structure depends ONLY on its own identity, never generation order — **nothing to checkpoint
  for rattle's RNG.** Variant identity is a LABEL not a position (`"iso:-0.05"`, `"aniso:0"`) so
  adding a strain magnitude doesn't renumber existing variants. `atoms.info` carries
  `base_frame`, `base_frame_key`, `variant` (MD carries `md_step` instead).
- `MDSampler`: KEEPS one streaming `np.random.default_rng(seed)` — must, trajectory is
  sequential, snapshot n only exists via integrating 1..n-1. **Resuming MD requires
  checkpointing `self.rng.bit_generator.state` alongside positions/velocities** (documented in
  `md.py` docstring, not implemented — §9 next-step #2).

---

## 3. Settled design decisions — student side

**D8. `sample` stays in `run`, earns its place by being omissible.**

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
template's `ModelCheckpoint.dirpath`/`logger.save_dir` land in the one folder, no rewriting by us.
Config handed over: strip `sample` from `run`, drop `sampler`/`sample_path`, point
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
  `ckpt_path` → sampler appends frames, nequip restarts the OLD run: it resumes at the restored
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
`sample_path`: stable home for checkpoints + D13 versioning off → training resume becomes
automatic (default = resume from last/best checkpoint in `student_path`; new start = hand a new
`student_path`). Cost accepted: we set `ModelCheckpoint.dirpath` ourselves — first real
intervention in nequip's territory (everything else in D9 passes the trainer config through
untouched). Record next to checkpoint: `student_state.json` w/ `{sample_path, sample_n_frames,
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
- `sample_path` lives ELSEWHERE, side by side with the hydra folder, not nested either direction —
  different lifetimes (dataset outlives any command, run folder is per-command). Nesting the hydra
  folder INSIDE `sample_path` is REJECTED (tested): hydra creates its run dir at command START, so
  `hydra.run.dir: ${sample_path}/...` makes hydra create `sample_path` itself → sampler sees a dir
  with contents but no state file → hard error every fresh run. Reverse (sample_path as subdir of
  hydra folder) is SAFE, tested fine — just never point `hydra.run.dir` at/inside `sample_path`.
- Interruption during SAMPLING: no checkpoint to hand back, `ckpt_path` stays null, resume driven
  entirely by `sample_path`. Interruption during TRAINING: new hydra folder on resume unless
  `student_path` (D15) makes `dirpath` stable — this was the "resume asymmetry", resolved on paper
  by D15, not yet in code.
- Hydra does NOT chdir (`hydra.job.chdir` unset → False, confirmed installed 1.3.2) → every
  relative path in a nequip config resolves against LAUNCH CWD, not the hydra run dir. Why
  `${hydra:runtime.output_dir}` is used explicitly rather than relative paths.
- `--multirun` groups siblings under `multirun/<date>/<time>/<job_num>/` instead of `outputs/`.

### Repository layout policy
Keep four different kinds of files separate:

- `nequip_extension_template/` is package code.
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
- **`warm_start_from` as a second checkpoint key** (user). D11.
- **Single `runs.jsonl` provenance file** — split-brain. Settled instead: each stage's record
  lives with its own artifact — sampling's is the state file in `sample_path`; training's is the
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
No sampler, rattle, active learning, MD driver. `nequip/data/_sampler.py::PartialSampler` is an
unrelated torch DataLoader sampler. Only MD code is `nequip/ase/nosehoover.py::NoseHoover` — an
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

**Anisotropic strain bound decoupled from `strain_magnitudes`.** `RattleSampler` kwarg
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

### ASE gotchas that bit the samplers
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
   lock file in `sample_path`, undesigned. (Two-command workaround rejected, §5.)
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
`colab.research.google.com/github/Steinburglar/TuneandDistill/blob/main/docs/tutorial/NequIP_Distill_Tutorial.ipynb`.
Two URLs hardcode `Steinburglar/TuneandDistill@main` (the `pip install git+` cell and the `wget` of
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

## 11. Proposed datamodule port (2026-09-16)

Prompt from user/lab discussion: the distillation package may fit NequIP better if sampling lives
inside the `data:` section as a custom datamodule, rather than as a separate top-level `sampler`
driven by `nequip-distill` before handing paths to `ASEDataModule`.

**Assessment.** This is likely the more natural long-term boundary. A Dataset could technically
yield generated frames, but the lifecycle is too broad for `__getitem__`: teacher loading/release,
materialization, restart safety, train/val/test ownership, provenance checks, distributed setup,
and checkpoint compatibility all belong at the DataModule layer. The Dataset should stay boring:
read already-materialized data and apply normal NequIP transforms.

### Target user-facing shape

Primary path should become ordinary NequIP training:

```bash
nequip-train -cp /abs/path/to/configs -cn distill
```

with distillation configured under `data:`:

```yaml
run: [train, val, test]

data:
  _target_: nequip_extension_template.data.DistillationDataModule
  _recursive_: false
  seed: 1
  sample_path: /scratch/me/cdp_distill_samples
  state_interval: 50

  teacher:
    _target_: nequip.integrations.ase.NequIPCalculator._from_saved_model
    model_path: /scratch/me/teacher.nequip.zip
    device: cuda
    chemical_species_to_atom_type_map: true

  generation:
    _target_: nequip_extension_template.sample.RattleGenerator
    base_frames: /scratch/me/base_frames.xyz
    split: {train: 0.8, val: 0.1, test: 0.1}
    split_policy: scattered
    split_seed: 0
    strain_magnitudes: [-0.05, 0.0, 0.05]
    n_random_strain_samples: 2
    anisotropic_strain_magnitude: 0.05
    max_displacement_ang: 0.25
    seed: 1

  transforms:
    - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
      model_type_names: ${model_type_names}
    - _target_: nequip.data.transforms.NeighborListTransform
      r_max: ${cutoff_radius}

  train_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 5
    num_workers: 4
    shuffle: true
  val_dataloader:
    _target_: torch.utils.data.DataLoader
    batch_size: 10
    num_workers: 4
  test_dataloader: ${data.val_dataloader}
```

`_recursive_: false` is load-bearing: without it Hydra may instantiate `teacher` during
DataModule construction, which loads the teacher too early and possibly when generation is already
complete. The datamodule should instantiate the teacher only after it has decided more generation
is needed.

### Boundary split

**`DistillationDataModule` owns orchestration/lifecycle:**
- output path and file naming (`train.extxyz`, `val.extxyz`, `test.extxyz`)
- generation lock file
- reading/writing durable generation state
- config/provenance compatibility checks
- torn-write truncation using recorded byte offsets
- deciding whether the teacher must be loaded
- instantiating and releasing the teacher
- exposing ordinary NequIP datasets to the trainer
- recording dataset fingerprint in Lightning datamodule state

**Generator/procedure classes own scientific sampling:**
- how samples are enumerated or integrated
- how one generated structure gets labeled
- procedure-specific state for resume
- what unit is split: base frame for rattle, snapshot for MD

**ASEDataset remains the actual training Dataset:**
- read the generated extxyz files
- apply transforms
- participate in NequIP statistics/dataloaders normally

Do not make a public "distillation Dataset" responsible for generation side effects. Internal
helper classes can be dataset-like, but generation should finish before dataloader workers exist.

### Implementation sketch

Add:

```text
nequip_extension_template/
  data/
    __init__.py
    datamodule.py          # DistillationDataModule
    state.py               # state read/write/check/truncate/lock helpers
    paths.py               # split_file(), state_file(), constants
```

Refactor:

```text
nequip_extension_template/sample/sampler.py
```

Current `Sampler` mixes two responsibilities:
1. filesystem lifecycle manager
2. sampling procedure abstraction

Move (1) to `data/state.py` and `DistillationDataModule`. Keep (2), probably renamed to
`SampleGenerator`, `GenerationProcedure`, or similar. `RattleSampler` becomes `RattleGenerator`;
`MDSampler` becomes `MDGenerator`. Compatibility aliases can remain while configs migrate.

Method mapping:

```text
Sampler.generate()
  -> DistillationDataModule.prepare_data()

Sampler.read_state/check_goal/truncate_to/write_state
  -> data/state.py helpers used by the datamodule

Sampler.finished/step/procedure_state/restore_progress
  -> generator/procedure interface

Sampler.append()
  -> GenerationContext.append(atoms, split), owned by datamodule/state layer
```

Preferred generator API:

```python
class GenerationProcedure:
    @property
    def finished(self) -> bool: ...
    def step(self, context: GenerationContext) -> None: ...
    def procedure_state(self) -> dict: ...
    def restore_progress(self, procedure_state: dict) -> None: ...
```

`GenerationContext` provides `append(atoms, split)`, counters, split file paths, and state writes.
This keeps persistence policy out of procedure code.

### DataModule lifecycle

`DistillationDataModule` should likely subclass `nequip.data.datamodule.ASEDataModule`.
Measured fact from installed NequIP 0.17.1: `ASEDataModule.__init__` only converts file paths into
`ASEDataset` configs; it does not read them. Therefore the subclass can compute final split paths
at construction and call `super().__init__(train_file_path=..., val_file_path=..., test_file_path=...)`
before files exist, then generate them in `prepare_data()` before `setup()` instantiates datasets.

Lifecycle:

```text
__init__
  store raw teacher/generation config without recursive instantiation
  compute split file paths from sample_path
  call ASEDataModule.__init__ with those split paths and normal transforms/loaders/stats

prepare_data
  acquire sample_path lock
  read generation_state.pt if present
  if complete and compatible:
      return without loading teacher
  if no state but split files exist:
      hard error
  restore counters and truncate torn writes
  instantiate teacher
  instantiate generator/procedure with teacher/context
  generate until complete
  write final state
  release teacher and torch.cuda.empty_cache()

setup(stage)
  call super().setup(stage)
  normal ASEDataModule behavior loads ASEDataset from generated files
```

Add a generate-only utility only if needed:

```bash
nequip-distill-generate -cp /abs/path -cn distill
```

This should instantiate `config.data` and call `prepare_data()` without training. The current
`nequip-distill` orchestration should become compatibility sugar or a migration error, not the
primary workflow.

### Config migration

Old:

```yaml
run: [sample, train, val, test]
sample_path: ...
sampler:
  _target_: nequip_extension_template.sample.RattleSampler
  calculator: ...
data:
  _target_: nequip.data.datamodule.ASEDataModule
```

New:

```yaml
run: [train, val, test]
data:
  _target_: nequip_extension_template.data.DistillationDataModule
  _recursive_: false
  sample_path: ...
  teacher: ...
  generation: ...
```

Remove:
- top-level `sample_path`
- top-level `sampler`
- special `sample` run type
- code that patches `data.train_file_path` / `val_file_path` / `test_file_path`
- hard requirement that `data._target_` is `ASEDataModule`

Keep:
- three pre-split extxyz files
- no NequIP `split_dataset` for generated data
- frozen split membership at generation time
- restart/provenance guardrails

### Resume semantics to keep

Keep the core safety model:
- generation state lives beside generated data
- no state + existing split files = hard error
- state version mismatch = hard error
- generator class mismatch = hard error
- file shorter than recorded byte offset = hard error
- file longer than recorded byte offset = truncate and regenerate tail
- state writes are atomic (`.tmp` + `os.replace`)
- progress is separate from goal/provenance

Rename `sampler_state.pt` to `generation_state.pt` for new datasets. Compatibility can read the
old name during migration.

Suggested state shape:

```python
{
    "version": 2,
    "generator_class": "nequip_extension_template.sample.RattleGenerator",
    "goal": {
        "immutable_config": {...},
        "teacher_identity": {...},
        "base_frames_digest": "...",
    },
    "progress": {
        "n_written": 50,
        "split_counts": {"train": 40, "val": 5, "test": 5},
        "offsets": {"train": 123, "val": 45, "test": 45},
        "procedure": {...},
    },
    "dataset_fingerprint": "...",
}
```

### Resume semantics to improve

Replace current "any config difference is fatal" with classified compatibility:

**Immutable by default:**
- generator class
- base frame contents, not just path
- teacher identity
- sampling knobs that affect already-written structures
- split policy/seed/fractions, unless split membership is stored explicitly
- label schema and key mapping

**Mutable:**
- teacher device
- `state_interval`
- dataloader settings
- transforms/stats settings applied after generated files are read
- output path, if the whole dataset directory was moved coherently

**Conditionally mutable:**
- requested sample budget
- added rattle variants
- added base frames
- teacher path, if a strong artifact hash proves it is the same teacher

Default rule: a new or unclassified generation knob is fatal until explicitly classified.

### Teacher identity

Current code stores calculator config but does not strongly identify the teacher artifact.
Datamodule state should record a teacher identity separate from runtime plumbing:

```yaml
teacher_identity_policy: strict_hash
```

Possible policies:
- `strict_hash`: hash teacher file contents; strongest and preferred for science
- `metadata`: path + size + mtime; faster but weaker
- explicit user-supplied identifier for non-file calculators

Device must not be part of identity. Moving from `cuda` to `cpu` should not invalidate a finished
dataset.

### Rattle resume/extension

Current rattle resume is close to ideal because each generated structure has a stable identity:

```text
base_frame_key + variant_label + seed -> geometry
```

Initial datamodule port can preserve existing conservative behavior: same base frame digest, same
variant set, same split assignment, continue from `n_steps`.

Later improvement: switch from pure count-based progress to item IDs:

```text
item_id = base_frame_key | variant_label
```

Then state can know which item IDs are complete, allowing natural extension:
- adding a new strain magnitude appends new variants
- existing structures remain bit-identical
- new structures inherit the base frame split

Adding base frames is harder. If split membership is recomputed with `random_split(n)`, adding
frames can move old base frames across train/val/test. Legal base-frame extension requires either:
- store split membership per `base_frame_key` and append new keys without moving old ones, or
- use a stable hash-based split assignment per base-frame key

Until one of those exists, changing base frame contents/count should remain fatal.

### MD resume

Moving generation into the DataModule does not by itself solve MD resume. MD still needs
procedure-specific trajectory checkpointing:
- positions
- cell
- velocities/momenta
- MD step count
- snapshot count
- NumPy RNG bit generator state
- any ASE Langevin/thermostat state not captured above
- split assignment

Until that is implemented, MD generation should still refuse resume from partial state. The
datamodule can own the refusal message and cleanup, but the scientific resume state belongs to
the MD generator.

### Training checkpoint semantics

The current CLI guard refuses "new structures added + `ckpt_path`" because Lightning checkpoint
resume may not train the new data. In the datamodule design, use the datamodule checkpoint state
instead:

```python
def state_dict(self):
    sd = super().state_dict()
    sd["distillation_dataset_fingerprint"] = self.dataset_fingerprint()
    return sd

def load_state_dict(self, state_dict):
    old = state_dict.get("distillation_dataset_fingerprint")
    new = self.dataset_fingerprint()
    if old != new:
        raise ValueError("checkpoint was trained with a different generated dataset")
    super().load_state_dict(state_dict)
```

Semantics:
- training crash, dataset unchanged: resume OK
- sampling crash before training: generation resumes, then training starts normally
- dataset extended and user tries Lightning resume from old checkpoint: refuse
- dataset extended and user wants warm start: use explicit model-from-checkpoint initialization,
  not Lightning `ckpt_path`

This is more native than `distill.py` inspecting `ckpt_path`.

### Concurrency

Add a lock file under `sample_path`, even though Lightning usually runs `prepare_data()` on rank
zero. This protects against two separate jobs pointing at the same dataset:

```text
sample_path/.generation.lock
```

Process:
1. acquire lock
2. re-read state after lock acquisition
3. generate or no-op
4. release lock

Do not rely only on Lightning rank-zero behavior.

### Test plan

1. State helper unit tests:
   - atomic write
   - compatible complete state no-ops
   - config differences classified correctly
   - torn writes truncate
   - no state + existing split file refuses

2. Datamodule unit tests:
   - constructor wires split paths into ASEDataModule configs
   - `prepare_data()` generates files
   - second `prepare_data()` no-ops without instantiating teacher
   - mutable settings do not invalidate a complete dataset

3. Rattle integration tests:
   - interrupted datamodule generation resumes to same digests as uninterrupted
   - split counts preserved
   - immutable setting change refused
   - eventually, added variant extension works without changing old files

4. Training integration tests:
   - `nequip-train` with `DistillationDataModule` runs train/val/test
   - checkpoint resume with same dataset works
   - checkpoint resume with changed dataset fingerprint refuses
   - warm-start path is explicit and does not masquerade as resume

5. Migration tests:
   - old config either translates or fails with actionable error

### Migration order

1. Add `DistillationDataModule` that reuses current `Sampler.generate()` with minimal changes.
2. Add examples that run through `nequip-train`.
3. Move state/lifecycle code out of `Sampler` into `data/state.py`.
4. Rename/refactor sampler classes into generator/procedure classes.
5. Replace "any config difference is fatal" with immutable/mutable classification.
6. Add datamodule checkpoint fingerprint protection.
7. Decide whether to support manifest/item-ID based rattle extension.

Principle: move the public architectural boundary first, then improve restart semantics. Do not
rewrite all restart logic at the same time as the NequIP integration boundary.

### Handoff after first datamodule slice (2026-09-16)

Implemented and committed:
- `b02b74c Organize sampler tests and design notes`
- `9a6f92f Add distillation data module path`

Current implementation state:
- `nequip_extension_template.data.DistillationDataModule` exists and subclasses NequIP's
  `ASEDataModule`.
- It computes split file paths from `data.sample_path` and passes them to `ASEDataModule` as
  `train_file_path` / `val_file_path` / `test_file_path`.
- It accepts `data.teacher` and `data.generation`; `data.generation` currently points to the
  existing sampler classes, e.g. `nequip_extension_template.sample.RattleSampler`.
- It deliberately requires `_recursive_: false` in configs so Hydra does not eagerly instantiate
  `data.teacher`.
- `prepare_data()` materializes all generated frames before NequIP loads datasets/training starts.
  This is expected behavior for now.
- Restart/provenance are still owned by the existing sampler state machinery:
  `sampler_state.pt`, config diff, base-frame digest, byte offsets, truncation, and
  `restore_progress()`.
- The datamodule has a no-teacher fast path for already-complete sampled datasets: it instantiates
  the sampler with `calculator=None`, checks/restores state, and returns without loading the
  teacher if `sampler.finished` is true.
- No `data/state.py`, lock file, `generation_state.pt`, teacher artifact hash, dataset
  fingerprint, or generator/procedure refactor exists yet.

New tracked example:
- `examples/rattle_train_datamodule.yaml`
- Primary invocation target is now direct NequIP:

```bash
cd /n/home12/lsteinberger/code/TuneandDistill
nequip-train -cp "$PWD/examples" -cn rattle_train_datamodule
```

Important config detail:
- `ckpt_path: null` must NOT appear in direct `nequip-train` configs. Unlike `nequip-distill`,
  NequIP treats the presence of `ckpt_path` as a restart, even when the value is `null`, and then
  tries to load checkpoint `None`, producing the `NoneType`/`seek` traceback. The tracked
  datamodule example now omits `ckpt_path` entirely.

Sandbox-local smoke-test setup:
- An untracked, gitignored local config was copied to:

```text
sandbox/configs/rattle_train_datamodule_local.yaml
```

- It differs from the tracked example only by path locality and by omitting `ckpt_path`:

```yaml
data:
  sample_path: rattle_train_datamodule_dataset
  teacher:
    model_path: ../inputs/teacher.nequip.zip
  generation:
    base_frames: ../inputs/base_frames.xyz
```

- It is intended to be launched from `sandbox/out`, not from repo root:

```bash
cd /n/home12/lsteinberger/code/TuneandDistill/sandbox/out

PATH="/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311/bin:$PATH" \
CONDA_PREFIX="/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311" \
HYDRA_FULL_ERROR=1 \
nequip-train \
  -cp /n/home12/lsteinberger/code/TuneandDistill/sandbox/configs \
  -cn rattle_train_datamodule_local
```

Path gotchas established during smoke testing:
- Hydra `_target_` values are Python import paths and are independent of cwd.
- Hydra `-cp` must be absolute for `nequip-train`; relative `-cp ../configs` can be interpreted as
  a package/module path such as `nequip.scripts.configs`.
- Ordinary file paths in YAML (`model_path`, `base_frames`, `sample_path`) are interpreted relative
  to the process cwd. The sandbox-local config only works from `sandbox/out`.
- Running from `sandbox/out` gives default Hydra output under `sandbox/out/outputs/...` and sampled
  frames under `sandbox/out/rattle_train_datamodule_dataset/`.

Verification already run:
- `python -m pytest tests/unit/data/test_distillation_datamodule.py -q`
- `python -m pytest tests/e2e/test_distill_cli.py -k nequip_train_with_distillation_datamodule -m e2e -q`
- `python -m pytest -q`

Known follow-ups:
1. The sandbox-local config is intentionally untracked. This is less ideal for version management,
   but it keeps user-side smoke runs contained in `sandbox/` without baking sandbox paths into the
   general example.
2. Next architecture step is not more config work; it is splitting the sampler's two roles:
   datamodule/state layer owns mechanical persistence, generator owns semantic provenance/progress.
