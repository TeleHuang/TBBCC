# Implementation Status

This file maps `../ClaudeCodePluginDesign.md` to the current Claude Code plugin
implementation.

## Completed in This Plugin

- Claude Code plugin manifest:
  - `.claude-plugin/plugin.json`
  - Validated with `claude plugin validate`.
- Primary user workflows:
  - `/torchbridgebench:eval`
  - `/torchbridgebench:inspect`
- Plugin agents:
  - `evaluator`
  - `diagnostician`
  - `repairer`
- Independent deterministic core:
  - `scripts/tbbcc.py`
  - no import-time dependency on `../-demo`
  - standard-library-only MVP path
- Data contracts:
  - TestCase JSON
  - AdapterSpec JSON
  - Suite JSON
  - per-run report JSON/Markdown
  - suite summary JSON/Markdown
- Tier implementation:
  - Tier-1 Tensor output comparison implemented.
  - Tier-2 generic `ACTIVATIONS` / `GRADIENTS` channel comparison implemented.
  - Tier-3 generic `TASK_METRICS` comparison implemented.
- Failure classification:
  - Environment/setup and execution failure patterns.
  - NumericMismatch and ShapeMismatch from deterministic comparison.
- Local examples:
  - passing no-op case/adapter,
  - intentional numeric drift case/adapter,
  - 2x2 local smoke suite,
  - minimal PyTorch tensor smoke suite,
  - Tier-2 activation drift smoke suite,
  - Tier-3 task metric drift smoke suite.
- Runtime documentation:
  - development loading with `--plugin-dir`,
  - plugin validation,
  - local marketplace caveats,
  - cache/path behavior.

## Verified Commands

```bash
cd /home/ma-user/work/torchbridgebenchCCplugin

claude plugin validate .
claude --plugin-dir . plugin details torchbridgebench

python -m py_compile scripts/tbbcc.py

python scripts/tbbcc.py validate-inputs \
  --case examples/cases/pure_python_vector.json \
  --case examples/cases/scale_sensitive_vector.json \
  --adapter examples/adapters/noop.json \
  --adapter examples/adapters/numeric-drift.json

python scripts/tbbcc.py eval \
  --case examples/cases/pure_python_vector.json \
  --adapter examples/adapters/noop.json \
  --out reports/final_smoke

python scripts/tbbcc.py eval-suite \
  --suite examples/suites/local_smoke.json \
  --out reports/final_suite

python scripts/tbbcc.py eval-suite \
  --suite examples/suites/torch_smoke.json \
  --out reports/final_torch_suite

python scripts/tbbcc.py eval-suite \
  --suite examples/suites/tier2_smoke.json \
  --out reports/final_tier2_suite

python scripts/tbbcc.py eval-suite \
  --suite examples/suites/tier3_smoke.json \
  --out reports/final_tier3_suite
```

Expected `local_smoke` result: `3/4` pass and one `NumericMismatch`.
Expected `torch_smoke` result: `1/2` pass and one `NumericMismatch`.
Expected `tier2_smoke` result: `1/2` pass and one Tier-2 `NumericMismatch`.
Expected `tier3_smoke` result: `1/2` pass and one `TrainingDivergence`.

## Deliberate Gaps

- Automated framework hook capture for Tier-2 activation and gradient comparison
  is not implemented yet; the generic comparison channel is implemented.
- Automated framework training loops for Tier-3 are not implemented yet; the
  generic `TASK_METRICS` comparison channel is implemented.
- Monte Carlo repair sampling and ME/AR/migrate@k aggregation are not automated
  in the deterministic core yet.
- Real PyTorch/MindSpore framework cases are not bundled yet; the current cases
  are pure Python smoke tests for harness validation.
- Marketplace packaging is documented but not materialized as a separate
  marketplace root, because this directory is the plugin root and marketplace
  relative sources must not use `../`.

## Next Implementation Targets

1. Add framework-backed TestCase examples using the validated
   `activate_torch4ms_ms272_cann85.sh` environment.
2. Add timeout, stdout/stderr artifact retention policies for heavy runs.
3. Implement Tier-2 hooks for small `torch.nn.Module` cases.
4. Add an Agent effort ledger format consumed by the `repairer` workflow.
5. Mirror this core into a future pip-installable Python Agent package.
