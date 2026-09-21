> **DEPRECATED — DOES NOT RUN.** This section drives `nequip-distill`, a CLI that was
> deleted on 2026-09-18. The package now contributes a `data:` datamodule to plain
> `nequip-train` instead; see `planning.md` §11. Port this section and `distill.yaml`
> once the refactor settles.

# Distillation section — draft

Cells to append to `NequIP_Tutorial.ipynb` after the fine-tuning section. Kept
separate so the vendored notebook stays byte-identical to upstream.

> Upstream bug: cell 29 compiles `/content/results/best.ckpt`, the model trained from
> scratch, and cell 30 plots it as "Fine-tuned model". The fine-tuned checkpoint is
> `/content/results_ft/best.ckpt`. This section uses the correct path.

---

## [markdown]

# Distilling into a fast, specialized model

While fine-tuning a foundation model can increase your domain accuracy, the resulting model is still very large,
and thus slow to evaluate, which can make long production MD simulations difficult.



To solve this issue, we can instead use it as a *teacher*. The teacher labels structures relatively cheaply, so we
can generate far more data than we have DFT for and train a small silicon-only *student* on those labels.

This is `nequip-distill`, a NequIP extension. Sampling, labeling and training happen
in one command.

## [code]

```python
# @title Installing nequip-distill
!pip install --quiet git+https://github.com/Steinburglar/nequip-academy.git
```

<!-- TODO: repo private; make public or use a token-authenticated clone. -->

## [code]

```python
# @title Packaging the fine-tuned model as our teacher
!nequip-package build /content/results_ft/best.ckpt /content/teacher.nequip.zip
!wget --quiet https://raw.githubusercontent.com/Steinburglar/nequip-academy/main/docs/tutorial/distill.yaml
```

## [markdown]

A distillation config is an ordinary NequIP config with two additions. First, `run`
gains a `sample` stage, which must come first:

```yaml
run: [sample, train, val, test]
sample_path: ./distill_dataset
```

`sample_path` is where the generated structures go, as `train.extxyz`,
`val.extxyz`, and `test.extxyz`, plus a record of what has been sampled so far. The split is
fixed as structures are written, not re-derived from fractions at training time, so
that growing the dataset later cannot move a test structure into training or vice versa. Re-running the
same command resumes sampling rather than starting over. Importantly, in contrast to a normal config,yaml, `data` has no
`split_dataset` and no `*_file_path`; `nequip-distill` fills those in.

Second, a `sampler` block says how to generate structures and which teacher labels
them:

```yaml
sampler:
  _target_: nequip_academy.sample.RattleSampler
  calculator:
    _target_: nequip.integrations.ase.NequIPCalculator._from_saved_model
    model_path: ./teacher.nequip.zip
    device: cuda
    chemical_species_to_atom_type_map: true
  base_frames: ./sitraj.xyz
  strain_magnitudes: [-0.05, 0.0, 0.05]
  n_random_strain_samples: 2
  max_displacement_ang: 0.25
```

With this particular sampler, each base frame is strained — an isotropic volume scan plus a few random anisotropic
strains — and then rattled. That is 5 variants of each of our 110 frames, so 550
structures from 110. The DFT labels in `sitraj.xyz` are discarded; the teacher labels
everything. See the comments in `distill.yaml` for the rest of the options.
Different ways of sampling synthetically labelled frames is an active area of research, and the Nequip team plans to add more samplers in the future, such as using MD to sample new structures. For now, we provide a simple rattle sampler as a starting point.

## [code]

```python
# @title Run the distillation
!nequip-distill -cn distill
```

## [markdown]

The command samples and labels the 550 structures, releases the teacher from the GPU,
and then hands the config to NequIP's own trainer — everything from `data` down is
stock NequIP, and checkpoints land in `./results_distill` as usual.

Note that the training, validation, and test metrics are all relative to the teacher's labels, and not DFT. To properly evaluate the student, one should run a separate evaluation on a held-out DFT dataset, which is not yet included in this tutorial.



<!-- TODO: two cells still to write -- student vs teacher energy-volume curve (reuse
     `scaling_factors`/`lattice_constants`/`ft_energies` from the notebook), and a
     timing comparison after compiling the student. -->

<!-- TODO: a "growing the dataset" cell would be the natural close, but resume
     currently refuses ANY config change (Next steps #1) and split assignment on a
     grown dataset is unsolved (#3). Do not write it until those land. -->
