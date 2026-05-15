# Agent Workflow

The agentic loop follows the design document while keeping deterministic work in
the local core.

## Phase 0: Adapt or Translate

- Intercept track: read adapter docs and create/choose adapter preamble.
- Translate track: create target code or translated case.

Effort here counts as `effort_adapt`.

## Phase 1: Deterministic Verification

Run the core:

```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/tbbcc.py eval --case CASE --adapter ADAPTER --out OUT
```

Tier-1 currently compares baseline and target results. Tier-2 and Tier-3 are
reserved in the report schema.

## Phase 2: Classification

Use automatic classification first, then ask a diagnostician agent to confirm.
Confirmation is separate from repair and should not count toward repair effort.

## Phase 3: Repair

Repair only migration-side code or adapter specifications. Record:

- attempt count,
- changed files,
- diff size,
- strategy,
- rerun result,
- context summary.

After every repair, rerun from Tier-1.

## Stop Conditions

Stop when:

- all enabled tiers pass,
- max attempts is reached,
- the failure is environment-only,
- repair requires bridge internals,
- the case is outside current implementation scope.
