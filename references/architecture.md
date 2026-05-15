# Architecture

TorchBridgeBench measures migration burden for PyTorch-to-MindSpore/Ascend
bridges and translators.

## Tracks

- `intercept`: PyTorch code remains mostly intact while adapter or bridge
  preamble redirects execution to another backend.
- `translate`: PyTorch code is translated to target-framework code before
  execution.

Report track-specific rankings separately. Migration effort is not identical
across tracks.

## Layers

The target architecture has three verification tiers:

1. Tier-1 Tensor: final outputs match within tolerance.
2. Tier-2 Activation + Gradient: first diverging forward activation or backward
   gradient is identified.
3. Tier-3 Task: training or task metrics remain within accepted drift.

The current plugin core implements Tier-1 and a generic Tier-2 channel contract:
test code may expose `ACTIVATIONS` and/or `GRADIENTS`, and the core compares
those structures after Tier-1. Tier-3 has a generic `TASK_METRICS` contract for
loss curves, accuracy, and task-level values. Framework hook and training-loop
automation can populate these fields in a later iteration.

## Agent Boundary

Claude Code agents represent the migration user. They may:

- read bridge documentation,
- write adapter preambles,
- diagnose failures,
- repair user-side migration code,
- record effort.

They should not silently patch bridge internals. Bridge fixes belong to a
different project.

## Dual-System Plan

This plugin is one of two intended products:

- Claude Code Plugin: native skills and agents orchestrate evaluation.
- Python Agent package: future pip-installable system with equivalent behavior.

The deterministic core in this plugin is new and should inform the future
Python package. It does not depend on `../-demo`.
