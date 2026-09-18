# End-To-End Tests

Subprocess tests for user-facing workflows.

These tests should exercise the package through the command a user runs, which is
`nequip-train` (or `python -m nequip.scripts.train`) with a config whose `data:`
section is a `DistillationDataModule`. Mark them with `e2e` and `slow` when they
run training or take more than a few seconds.
