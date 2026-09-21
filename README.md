<!--
  SCAFFOLD ONLY. Section order/shape follows what nequip extension packages ship
  (see allegro, NequIP-LES). Prose is deliberately left for a human to write —
  each comment says what belongs in the section under it. Delete the comments as
  you fill them in. Facts that are mechanical rather than editorial (the install
  line, the invocation form) are already filled in; check them if the package is
  renamed or the entry point moves.
-->

# NequIP Academy

<!-- One or two sentences: what this package is, and the one thing it does.
     Name the CLI (`nequip-distill`) here. Say it is a `nequip` extension package. -->
     THIS is a `nequip` extension package that provides easy distillation of large, accurate MLIP's into smaller, faster Nequip models within the nequip-train framework.

<!-- Then a short paragraph on the idea: a trained teacher model labels structures
        cheaply, and a small student model is trained on those labels. The student
        is much smaller and faster to evaluate than the teacher, so it can be used in
        production MD simulations. -->
        When training and deploying MLIP's, one may find that they have a large, accurate model that is too slow for production use, as is often the case with foundation models that have been fine-tuned to a specific domain. This package allows you to easily "distill" the knowledge of the large model into a smaller, faster Nequip model that retains the accuracy of the teacher model. The user can distill from any teacher architecture that implements an ASE calculator interface, and is not limited to Nequip teachers. 


## Status

<!-- Be honest about maturity here. What is validated, what is scaffolding, what is
     not implemented. See `planning.md` §8/§9 for the current list. -->
Current status is messing, to be cleaned before any shipment. 

## Installation

Requires [`nequip`](https://github.com/mir-group/nequip) `>=0.17.1`.

```bash
git clone https://github.com/Steinburglar/nequip-academy.git
cd nequip-academy
pip install -e .
```

<!-- Add anything environment-specific a new user genuinely needs (GPU/torch notes,
     known-good versions). Do not turn this into a cluster-specific runbook —
     machine-specific setup belongs in CLAUDE.md, not here. -->


## What you need

<!-- The prerequisites, as a short list. Roughly:
       - a teacher model artifact loadable as an ASE calculator
       - one or more base frames to sample around
       - a normal nequip student config
       - a sampling section describing the procedure
     Write it as prose or a list, whichever reads better. -->


## Usage

'nequistill' works by providing a new DataModule class that generates and labels a "synthetic" dataset with a provided teacher model. To use it, simply write a normal nequip config, but with the DistillDataModule as the data target. The Datamodule requires some additional parameters to specify the teacher model, location of the base frames, and the sampling procedure (see example below). A full example config is provided in `configs/distill_example.yaml`. 



<!-- copy here over the data section of the example config, with brief explanation of the fields in comments -->
```yaml
data:
  # The path to the teacher model
  teacher_model: /path/to/teacher_model.pth
  # The path to the base frames
  base_frames: /path/to/base_frames/
  # The sampling procedure
  sampling:
    # The type of generator to use
    type: RattleGenerator
    # The parameters for the generator
    params:
      # The magnitude of the rattle
      rattle_magnitude: 0.1
      # The number of samples to generate
      num_samples: 1000
```

Once your config is ready, you can run the distillation process with the following command:

```bash
nequip-train -cp /absolute/path/to/config/dir -cn config_name
```

The `-cp` path must be **absolute** — hydra resolves a relative `--config-path`
against the decorated function's module. `nequip-train` behaves the same way.




### Configuration

<!-- Document the config sections this package adds on top of a normal nequip
     config: `generator`, `dataset_path`, and how `run:` is extended with `sample`.
     Point at the example configs rather than duplicating them. -->


### Generators

<!-- One short subsection per generator, with its knobs:
       - RattleGenerator — strain + per-atom rattle around each base frame
       - MDGenerator — Langevin NVT trajectory from one base frame
     Say plainly which one is production-ready. `planning.md` §7 has the rattle
     algorithm and the magnitude lesson worth repeating to users. -->


## Examples

<!-- Point at `docs/tutorial/` (the Colab notebook) and at the example configs.
     A one-line description of each is enough. -->


## Contributing

<!-- Development install, how to run the tests, the pre-commit setup. -->


## Citing

<!-- Citation for this package if there is one, plus the nequip citation users
     should include. -->


## License

<!-- No LICENSE file exists yet. `nequip` is MIT; the upstream extension template
     recommends MIT, Apache 2.0, or BSD 3-Clause. Pick one, add the file, and note
     it here. -->


## Join the NequIP Community

Extension package developers are invited to join the NequIP community chat server,
hosted on [Zulip](https://zulip.com/). Zulip is organized a little differently from
Slack or Discord — please review [their introduction](https://zulip.com/help/introduction-to-topics)
before posting. [Fill out the interest form here](https://forms.gle/mEuonVCHdsgTtLXy7).
