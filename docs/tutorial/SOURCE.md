# Vendored upstream tutorial

Everything else in this directory is a **verbatim copy** of the upstream NequIP tutorial
repository. Do not hand-edit those files: keeping them byte-identical to upstream is what
lets our own additions show up as a clean diff and lets us re-pull a newer upstream later.

- Upstream: https://github.com/mir-group/nequip-tutorial
- Commit: `8f90935ba42fd9e03df323cf03428c456d87b881` (2026-03-09)
- License: MIT (see `LICENSE`, copied unchanged)
- Colab: https://colab.research.google.com/github/mir-group/nequip-tutorial/blob/main/NequIP_Tutorial.ipynb

Vendored files: `NequIP_Tutorial.ipynb`, `config.yaml`, `config_finetuning.yaml`,
`sitraj.xyz`, `README.md`, `LICENSE`.

To refresh from upstream:

```bash
SHA=<new commit sha>
for f in LICENSE README.md NequIP_Tutorial.ipynb config.yaml config_finetuning.yaml sitraj.xyz; do
  curl -sSLf -o "docs/tutorial/$f" "https://raw.githubusercontent.com/mir-group/nequip-tutorial/$SHA/$f"
done
```

## Why it is here

Planned (NOT yet written): a new section appended to this tutorial covering `nequip-distill`
— how to run it, what `sample_path` is and does, and what a distillation config must add on
top of a normal NequIP training config. The upstream copy is the baseline that section
attaches to.

## Generated

`NequIP_Distill_Tutorial.ipynb` is BUILT, not hand-written:

```bash
python docs/tutorial/build_notebook.py    # upstream notebook + distill_section.md
```

Edit `distill_section.md` (and `distill.yaml`) and regenerate; never edit the output
notebook directly. `build_notebook.py` copies the upstream cells verbatim and appends
ours, so the two notebooks share a common prefix cell-for-cell.

Colab link, once pushed to a public `main`:
https://colab.research.google.com/github/Steinburglar/nequip-academy/blob/main/docs/tutorial/NequIP_Distill_Tutorial.ipynb
