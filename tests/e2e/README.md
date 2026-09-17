# End-To-End Tests

Subprocess tests for user-facing workflows.

These tests should exercise the package through the command a user runs, usually
`python -m nequip_extension_template.scripts.distill` or `nequip-distill`. Mark
them with `e2e` and `slow` when they run training or take more than a few seconds.
