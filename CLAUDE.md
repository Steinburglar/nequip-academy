# TuneandDistill — Agent Context

Durable state only. **Design rationale, measured facts about nequip/lightning/ASE, rejected
designs, open problems and the ordered next steps all live in `planning.md`** — read it on
demand, don't duplicate it here.

## Purpose

`nequip` extension package. Ships ONE new CLI: **`nequip-distill`**. Given a teacher artifact
loadable as an ASE calculator + base frame(s): **sample** frames (rattle or ASE MD) → **label**
with the teacher → **train** a student via `nequip.scripts.train.main()`.

Finetuning is OUT OF SCOPE (nequip core does it). Scope is a hackathon one-off being cleaned up
for shipping, NOT a broad extensible framework — reject abstraction that only pays off
hypothetically. One command, one config; a design needing two of either is rejected.

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

`nequip_extension_template/sample/`
- `sampler.py` (429) — base `Sampler`. Holds calculator, `base_frames`, `sample_path`,
  `n_written`, `n_resumed`, `split_counts`, `state_interval`, `sampler_config`. Abstract surface
  = `finished` (property) + `step()`. Overridable: `procedure_state()` (default `{}`),
  `restore_progress(blob)` (default RAISES `NotImplementedError` → a sampler not taught to resume
  refuses instead of duplicating everything). Concrete: `split_file(s)`, `append(atoms, split)`,
  `state_file`, `goal_state()`, `progress_state()`, `state_payload()`, `write_state()`,
  `read_state()`, `check_goal(stored)`, `truncate_to(offsets)`, `generate()`. Module fns:
  `frames_digest`, `flatten`, `split_file(sample_path, split)` (also exported from
  `sample/__init__` with `SPLITS` — `distill.py` needs the 3 paths in runs that build no sampler).
- `rattle.py` (212) — `RattleSampler`. Owns `label()` (teacher call + `SinglePointCalculator`).
  Resumes: `procedure_state()` = `{"n_steps": ...}`, nothing else — order is variant-major and
  fixed, and the goal must be unchanged, so the count IS the position. Module fns: `frame_key`,
  `derive_seed`, `isotropic_strain_matrix`, `random_anisotropic_strain_matrix`, `rattle_positions`.
- `md.py` (164) — `MDSampler`. Langevin NVT, ONE trajectory from `base_frames[0]`, rest ignored
  (warns). CANNOT resume — inherits the base refusal. SCAFFOLDING, not production.
- `split.py` (91) — free fns, NOT base-class methods. `assign_splits(n, split, seed, policy)`,
  `policy` = `"scattered"` (torch shuffle) or `"blocked"` (contiguous train→val→test ranges).
- `__init__.py` — exports `Sampler`, `RattleSampler`, `MDSampler`.

`nequip_extension_template/scripts/distill.py` (221) — hydra entry,
`@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")`, same shape as
`nequip.scripts.train.main`. `run` = optional `sample` (first, at most once) + any of
`train`/`val`/`test`; the tail goes to `nequip.scripts.train.main(train_config)` in-process.
Helpers: `_split_run_list`, `_check_train_config` (runs BEFORE sampling), `_dataset_files`,
`_require_dataset` (no-`sample` runs need 3 non-empty files; the sampler is NOT built — would
load the teacher onto a GPU for nothing), `_train_config`, `_release_teacher`.
Sets `sampler.sampler_config = OmegaConf.to_container(config.sampler, resolve=True)` AFTER
`instantiate` — NOT as a ctor arg, `instantiate` recurses into args hunting `_target_` and would
build the teacher twice.

Leftovers from the upstream template, delete or replace when touched: `_keys.py` registers two
unused placeholder fields; `model/`, `nn/`, `train/` are README-only stub dirs.

## What works / what refuses

| | State |
|---|---|
| rattle sampling | works, GPU-validated, deterministic per structure |
| 3-file split output | works; splits frozen at generation |
| student training in-process | works, GPU-validated end to end |
| rattle resume | works (truncate to recorded offset, continue) |
| MD sampling | runs, but scaffolding — see `planning.md` §8 |
| MD resume | REFUSES (base-class `NotImplementedError`) |
| resume w/ ANY sampler-config difference | REFUSES, incl. `calculator.device` — deliberate, §9 #1 |
| growing a dataset + `ckpt_path` | REFUSES (D11 trap guard) |
| warm start on grown data | not implemented (D11 Path B) |
| `student_path` / stable checkpoint dir | not implemented (D15) |
| parallel sweeps into one `sample_path` | unsafe, no lock |
| compiled `.pt2` teacher | segfaults; use packaged `.nequip.zip` |

## Run commands

```bash
CONDA=/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311
export LD_LIBRARY_PATH=$CONDA/lib:${LD_LIBRARY_PATH:-}
# from repo root; hydra does NOT chdir, so relative paths resolve to launch cwd
$CONDA/bin/nequip-distill -cp $PWD/examples -cn rattle_train   # sample + train + val + test
$CONDA/bin/nequip-distill -cp $PWD/examples -cn rattle_only    # sample only
$CONDA/bin/nequip-distill -cp $PWD/examples -cn md_only
```

- **`-cp` MUST be absolute.** Hydra resolves a relative `--config-path` against the DECORATED
  FUNCTION'S MODULE, so `-cp testartifacts` fails with `Primary config module
  'nequip_extension_template.scripts.testartifacts' not found`. `nequip-train` has the same trap.
  Auto-absolutizing it in a wrapper was proposed and REJECTED (user).
- `nequip-distill --help` FAILS with no `config.yaml` in cwd (hydra composes before help); use
  `-cp ... -cn ... --cfg job` to inspect a resolved config instead.
- On PATH because the repo is `pip install -e . --no-deps --no-build-isolation` into nequip311.
  Editable, so edits are live; re-run only if entry points change. Needed `license = {file =
  "LICENSE"}` dropped from `pyproject.toml` first — no LICENSE file exists.
- **All real runs go through Slurm (user explicit, never the login node.)** Smoke-scale sampler
  tests are fine on login.
- `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` fixes `GLIBCXX_3.4.31 not found`.

## Test suite

```bash
pytest                                      # default suite, excludes e2e
pytest tests/integration                   # resume/integration tests
pytest -m e2e                              # CLI subprocess suite, 15 cases, ~4 min
```

Both CPU only, NO teacher, NO GPU — a correctness decision, not just speed (user explicit).
Calculator is `ase.calculators.lj.LennardJones`, so "resumed == uninterrupted" is a comparison of
file hashes, which also catches a dropped, duplicated or mis-filed structure. **Byte-identity is
NOT a valid criterion whenever the teacher runs on GPU** — see `planning.md` §6 for the measured
nondeterminism and the tolerance-based check to use instead.

- `tests/integration/sample/test_resume.py` — resume machinery. Helpers: `build(...)` mirrors what `distill.py` does
  (the same settings both construct the sampler AND become `sampler_config`); `kill_after(sampler,
  n)` wraps `step` to raise after exactly n appends, so the kill lands at a known point.
  Mutation-checked: `truncate_to` → no-op ⇒ 2 failures; rattle `restore_progress` forgets
  `n_steps` ⇒ 3; `check_goal` early-return ⇒ 3.
- `tests/e2e/test_distill_cli.py` — the CLI end to end, marked `e2e` and `slow`.
  Each case writes a config and runs `python -m nequip_extension_template.scripts.distill` as a
  SUBPROCESS — `@hydra.main` owns global state and does not survive two calls in one interpreter.
  `hydra.run.dir` is pinned per case so assertions can look inside it. 10 synthetic 32-atom fcc
  argon cells (fcc, NOT random positions — random points in a box overlap atoms and LJ energies
  explode); student = 1 layer, `l_max: 0`, `num_features: 8`, 2 epochs, `accelerator: cpu`.
  Covers 7 refusals, sample-only + byte-identical re-run, full pipeline, two students on one
  dataset, train-only with a deliberately broken calculator target (proves no sampler is built),
  `ckpt_path: null`, checkpoint restart, the D11 refusal. Mutation-checked, 3 mutations.

## Examples And Local Artifacts

Tracked example configs: `examples/rattle_only.yaml`, `examples/md_only.yaml`,
`examples/rattle_train.yaml`. Local-only inputs and generated outputs live under gitignored
`sandbox/` paths in those examples. `testartifacts/` has been retired.
- `rattle_train.yaml` — the FULL config: `run: [sample, train, val, test]`, rattle half copied
  from `rattle_only.yaml`, student half = `ASEDataModule` + `EMALightningModule` +
  `NequIPGNNModel` (2 layers, `l_max: 1`, `num_features: [32, 16]`, `r_max: 6.0` — the CDP student
  arch from `../distillation/config/base.yaml`), CSVLogger, `max_epochs: 20`, `ModelCheckpoint
  (dirpath=${hydra:runtime.output_dir}, filename=best, save_last=true)`. Its `data:` block
  deliberately has NO `*_file_path` and NO `split_dataset` — `distill.py` fills those in.
  50 structures → 40/5/5.
- `sandbox/inputs/teacher.nequip.zip` — THE teacher, packaged (not compiled). Copied from
  `../distillation/results/CDP/student_direct/S1b/n200_seed1/model.nequip.zip`.
- `sandbox/inputs/teacher.nequip.pt2` — the segfaulting compiled one, kept as evidence only.
- `sandbox/inputs/base_frames.xyz` — 10 CsH2PO4 frames (64 atoms, periodic), first 10 of a 200-frame REAL
  DFT subset. DFT energy/forces DISCARDED, the teacher relabels. 10 chosen so 0.8/0.1/0.1
  apportions to 8/1/1. Named "base frames" not "seed frames" — collided with `n200_seed1` naming
  AND with the RNG seed.

Old generated outputs were moved from `testartifacts/out/` to `sandbox/out/`; safe to delete when
no longer useful.

## Known debt (the shippable-repo cleanup list)

- Package still named `nequip_extension_template`. Rename touches `pyproject.toml` (`name`,
  `packages.find.include`, entry-points, `version.attr`) + every intra-package import.
- `pyproject.toml` placeholders: `description = "TODO"`, `authors = [{name = "your name here"}]`.
  No LICENSE file.
- `README.md` is a scaffold with empty sections — **user writes the prose, do not fill it in.**
- `configs/distill_template.yaml` is STALE (predates the `split_policy`/3-file design; don't trust
  its `sampler:` shape). `examples/*.yaml` are the current source of truth for runnable configs.
- `tests/e2e/test_distill_cli.py` still carries a case registry from the old smoke script; split it
  into ordinary pytest tests when touched.
- `.pre-commit-config.yaml` exists (ruff line-length 88 double quotes, yamllint, whitespace,
  `fail_fast: true`) but no hook is installed in `.git/hooks/`, and ruff is not in nequip311
  (`No module named ruff`). Lint by hand before committing.
- Latent bug: `rattle.py` strains with `F @ cell`; ASE stores lattice vectors as ROWS so it should
  be `cell @ F.T`. Silent on cubic cells (everything used so far). `planning.md` §7.

## Where things are documented

- `planning.md` — design record. §1 original intent, §2–3 decisions D1–D15, §4 on-disk layout,
  §5 rejected designs (**check before re-proposing anything**), §6 measured nequip/lightning/hydra
  facts, §7 rattle + ASE gotchas, §8 open problems, §9 ordered next steps, §10 the tutorial.
- `README.md` — human-facing, scaffold only.
- `docs/tutorial/` — vendored upstream notebook + our distillation section. **NEVER RUN LOCALLY**;
  the user tests it in Colab. `planning.md` §10 for the file map and the build step.
