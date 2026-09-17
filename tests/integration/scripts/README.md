# Script Integration Tests

Integration tests for script-level config wiring that does not need a full
subprocess run.

Use this for interactions between `distill.py`, sampler paths, and NequIP config
objects when those can be checked in-process.
