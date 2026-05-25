"""Shared TorchBridgeBench effort and sampling metrics."""

from __future__ import annotations

import math
import statistics
from typing import Any


EFFORT_FORMULA_VERSION = "shared-effort-v1"
EFFORT_FORMULA = "rounds + (prompt_chars + completion_chars + edit_units) / 1000.0"


def calc_effort_total(*, rounds: int, prompt_chars: int = 0, completion_chars: int = 0, edit_units: int = 0) -> float:
    return float(rounds + (prompt_chars + completion_chars + edit_units) / 1000.0)


def calc_entry_effort(entry: dict[str, Any]) -> float:
    if entry.get("effort") is not None:
        return float(entry["effort"])
    return calc_effort_total(
        rounds=int(entry.get("rounds") or 0),
        prompt_chars=int(entry.get("prompt_chars") or 0),
        completion_chars=int(entry.get("completion_chars") or 0),
        edit_units=int(entry.get("edit_units", entry.get("diff_size") or 0) or 0),
    )


def confidence_interval(values: list[float], confidence: float = 0.95) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {
            "confidence": confidence,
            "n": 0,
            "mean": None,
            "std": None,
            "stderr": None,
            "half_width": None,
            "low": None,
            "high": None,
        }
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    stderr = std / math.sqrt(n) if n > 0 else None
    # Normal approximation is intentional; it is stable and dependency-free.
    z = 1.96 if confidence == 0.95 else 1.96
    half_width = z * stderr if stderr is not None else None
    return {
        "confidence": confidence,
        "n": n,
        "mean": mean,
        "std": std,
        "stderr": stderr,
        "half_width": half_width,
        "low": mean - half_width if half_width is not None else mean,
        "high": mean + half_width if half_width is not None else mean,
    }


def variance_control(values: list[float], *, threshold_cv: float, current_n: int, max_n: int) -> dict[str, Any]:
    ci = confidence_interval(values)
    mean = ci["mean"]
    std = ci["std"]
    cv = (std / mean) if mean not in (None, 0) and std is not None else None
    needs_more = bool(cv is not None and cv > threshold_cv and current_n < max_n)
    return {
        "threshold_cv": threshold_cv,
        "current_n": current_n,
        "max_n": max_n,
        "cv": cv,
        "needs_more_samples": needs_more,
        "recommended_next_n": min(current_n + 1, max_n) if needs_more else current_n,
        "confidence_interval": ci,
    }


def calc_ar(me: float | None, baseline_effort: float | None) -> float | None:
    if me is None or baseline_effort is None:
        return None
    if baseline_effort <= 0:
        return 1.0
    return max(0.0, 1.0 - (me / baseline_effort))


def extract_baseline_metadata(baseline: dict[str, Any] | None) -> dict[str, Any] | None:
    if not baseline:
        return None
    keys = [
        "baseline_effort",
        "mean_effort",
        "std_effort",
        "cv_effort",
        "confidence_interval",
        "variance_control",
        "model",
        "model_observed",
        "provider",
        "protocol",
        "agent_system",
        "agent_system_version",
        "claude_version",
        "effort_formula_version",
        "effort_formula",
        "task_set_digest",
        "calibration_prompt_digest",
        "sample_count",
        "expected_sample_count",
        "batch_count",
        "expected_batch_count",
        "case_sample_count",
        "expected_case_sample_count",
        "error_count",
        "reroll_count",
        "total_cost_usd",
    ]
    return {key: baseline.get(key) for key in keys if key in baseline}


def summarize_effort_ledger(
    entries: list[dict[str, Any]],
    baseline_effort: float | None = None,
    baseline_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    adapt_entries = [entry for entry in entries if str(entry.get("phase")) == "adapt"]
    repair_entries = [entry for entry in entries if str(entry.get("phase")) == "repair"]

    def phase_total(items: list[dict[str, Any]]) -> float:
        total = 0.0
        for item in items:
            total += calc_entry_effort(item)
        return total

    effort_adapt = phase_total(adapt_entries)
    effort_repair = phase_total(repair_entries)
    effort_total = effort_adapt + effort_repair
    repair_attempts = sum(int(entry.get("attempts") or 1) for entry in repair_entries)
    return {
        "effort_formula_version": EFFORT_FORMULA_VERSION,
        "effort_formula": EFFORT_FORMULA,
        "source": "agent_effort_ledger" if entries else "deterministic_no_agent",
        "effort_adapt": effort_adapt,
        "effort_repair": effort_repair,
        "effort_total": effort_total,
        "me": effort_total,
        "baseline_effort": baseline_effort,
        "baseline_metadata": extract_baseline_metadata(baseline_metadata),
        "ar": calc_ar(effort_total, baseline_effort),
        "repair_attempts": repair_attempts,
        "entries": entries,
    }


def summarize_suite_effort(
    runs: list[dict[str, Any]],
    ledger_entries: list[dict[str, Any]],
    *,
    reroll_entries: list[dict[str, Any]] | None = None,
    baseline_effort: float | None,
    baseline_metadata: dict[str, Any] | None,
    evaluated_case_ids: list[str],
) -> dict[str, Any]:
    effort_adapt = sum(float(run.get("effort_adapt") or 0.0) for run in runs)
    effort_repair = sum(float(run.get("effort_repair") or 0.0) for run in runs)
    effort_total = effort_adapt + effort_repair
    run_efforts = [float(run.get("effort_total") or 0.0) for run in runs]

    reroll_groups: dict[int, list[dict[str, Any]]] = {}
    for entry in reroll_entries if reroll_entries is not None else ledger_entries:
        if "reroll_index" not in entry and "reroll" not in entry:
            continue
        reroll_index = int(entry.get("reroll_index", entry.get("reroll")) or 1)
        reroll_groups.setdefault(reroll_index, []).append(entry)
    reroll_efforts = [sum(calc_entry_effort(entry) for entry in entries) for _, entries in sorted(reroll_groups.items())]
    if not reroll_efforts and effort_total:
        reroll_efforts = [effort_total]

    baseline_meta = extract_baseline_metadata(baseline_metadata)
    source_cases = set((baseline_metadata or {}).get("source_cases") or [])
    baseline_case_count = int((baseline_metadata or {}).get("case_count") or (baseline_metadata or {}).get("case_sample_count") or 0)
    unique_cases = sorted(set(evaluated_case_ids))
    matched_cases = [case_id for case_id in unique_cases if case_id in source_cases] if source_cases else []
    scope = {
        "evaluated_case_count": len(unique_cases),
        "baseline_case_count": baseline_case_count or None,
        "matched_baseline_case_count": len(matched_cases) if source_cases else None,
        "matches_baseline_task_set": bool(source_cases and set(unique_cases) == source_cases),
        "is_subset_of_baseline_task_set": bool(source_cases and set(unique_cases).issubset(source_cases)),
    }

    effective_baseline = baseline_effort
    scaling_policy = "full_baseline_constant"
    if baseline_effort is not None and source_cases and set(unique_cases) != source_cases and len(source_cases) > 0:
        matched_ratio = len(matched_cases) / len(source_cases)
        effective_baseline = baseline_effort * matched_ratio
        scaling_policy = "linear_case_count_subset"
    scope["effective_baseline_effort"] = effective_baseline
    scope["baseline_scaling_policy"] = scaling_policy

    ar = calc_ar(effort_total, baseline_effort)
    ar_samples = [calc_ar(value, baseline_effort) for value in reroll_efforts]
    ar_samples = [value for value in ar_samples if value is not None]
    scope_adjusted_ar = calc_ar(effort_total, effective_baseline)
    scope_adjusted_ar_samples = [calc_ar(value, effective_baseline) for value in reroll_efforts]
    scope_adjusted_ar_samples = [value for value in scope_adjusted_ar_samples if value is not None]

    return {
        "effort_formula_version": EFFORT_FORMULA_VERSION,
        "effort_formula": EFFORT_FORMULA,
        "source": "agent_effort_ledger" if ledger_entries else "deterministic_no_agent",
        "effort_adapt": effort_adapt,
        "effort_repair": effort_repair,
        "effort_total": effort_total,
        "me": effort_total,
        "ar": ar,
        "scope_adjusted_ar": scope_adjusted_ar,
        "baseline_effort": baseline_effort,
        "baseline_metadata": baseline_meta,
        "baseline_scope": scope,
        "repair_attempts": sum(int(run.get("repair_attempts") or 0) for run in runs),
        "run_effort_confidence_interval": confidence_interval(run_efforts),
        "reroll_effort_confidence_interval": confidence_interval(reroll_efforts),
        "ar_confidence_interval": confidence_interval(ar_samples),
        "scope_adjusted_ar_confidence_interval": confidence_interval(scope_adjusted_ar_samples),
        "variance_control": variance_control(reroll_efforts, threshold_cv=0.2, current_n=len(reroll_efforts), max_n=max(len(reroll_efforts), 10)),
        "reroll_efforts": reroll_efforts,
    }


def migrate_at_k(samples: list[dict[str, Any]], k_values: list[int] | None = None) -> dict[str, Any]:
    if k_values is None:
        k_values = [1, 3, 5, 10]

    by_task: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        task_id = str(sample.get("task_id") or sample.get("case_id") or "global")
        by_task.setdefault(task_id, []).append(sample)

    def sample_passed(sample: dict[str, Any], mode: str) -> bool:
        if mode == "exec":
            if "exec_passed" in sample:
                return bool(sample["exec_passed"])
            if "t1_passed" in sample:
                return bool(sample["t1_passed"])
        if mode == "full":
            if "full_passed" in sample:
                return bool(sample["full_passed"])
            if "all_tiers_passed" in sample:
                return bool(sample["all_tiers_passed"])
        return str(sample.get("final_state")) == "ALL_PASS"

    result: dict[str, Any] = {"k_values": k_values, "task_count": len(by_task), "exec": {}, "full": {}}
    for mode in ("exec", "full"):
        previous_rate: float | None = None
        saturation_k: int | None = None
        for k in k_values:
            successes = 0
            for task_samples in by_task.values():
                ordered = sorted(task_samples, key=lambda item: int(item.get("reroll_index", item.get("attempt_index", 0)) or 0))
                if any(sample_passed(sample, mode) for sample in ordered[:k]):
                    successes += 1
            rate = successes / len(by_task) if by_task else None
            delta = None if previous_rate is None or rate is None else rate - previous_rate
            result[mode][f"migrate@{k}"] = {
                "k": k,
                "successes": successes,
                "total": len(by_task),
                "rate": rate,
                "delta_from_previous": delta,
            }
            if delta is not None and delta < 0.05 and saturation_k is None:
                saturation_k = k
            previous_rate = rate
        result[mode]["saturation_k"] = saturation_k
    return result
