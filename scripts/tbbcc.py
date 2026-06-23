#!/usr/bin/env python3
"""TorchBridgeBench Claude Code plugin deterministic core.

This MVP intentionally uses only the Python standard library. Framework-specific
cases can still run because user code is executed in an isolated Python process;
the harness only requires the case to expose a JSON-normalizable RESULT or run().
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import importlib.util
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any

from tbbcc_metrics import calc_ar, confidence_interval, migrate_at_k, summarize_effort_ledger, summarize_suite_effort
from tbbcc_report_plots import generate_report_plots


FAILURE_CLASSES = {
    "EnvironmentFailure",
    "DependencyMissing",
    "ImportOrderError",
    "OperatorNotFound",
    "TypeMismatch",
    "ShapeMismatch",
    "DeviceMismatch",
    "AutogradFailure",
    "NumericMismatch",
    "TrainingDivergence",
    "RuntimeCrash",
    "TranslationError",
    "Unknown",
}


@dataclasses.dataclass
class TestCase:
    id: str
    level: str
    track: str
    code: str
    expected_ops: list[str]
    training_config: dict[str, Any]
    ground_truth: dict[str, Any]
    difficulty: str | None
    failure_mode: str | None
    seed: int | None
    source_path: str


@dataclasses.dataclass
class AdapterSpec:
    bridge_id: str
    track: str
    preamble: str
    source_preamble: str
    env: dict[str, str]
    docs: str
    known_gaps: list[str]
    op_api_map: dict[str, str]
    atol: float
    rtol: float
    timeout_seconds: int
    fne_threshold: float | None
    gc_threshold: float | None
    tca_threshold: float | None
    dtw_threshold: float | None
    source_path: str


@dataclasses.dataclass
class ExecResult:
    role: str
    ok: bool
    result: Any | None
    activations: Any | None
    gradients: Any | None
    task_metrics: Any | None
    returncode: int
    duration_seconds: float
    stdout: str
    stderr: str
    traceback: str | None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return data


def load_case(path: Path) -> TestCase:
    data = _load_json(path)
    required = ["id", "level", "track", "code"]
    missing = [k for k in required if k not in data]
    if missing:
        raise SystemExit(f"TestCase {path} missing required field(s): {', '.join(missing)}")
    return TestCase(
        id=str(data["id"]),
        level=str(data["level"]),
        track=str(data["track"]),
        code=str(data["code"]),
        expected_ops=list(data.get("expected_ops") or []),
        training_config=dict(data.get("training_config") or {}),
        ground_truth=dict(data.get("ground_truth") or {}),
        difficulty=data.get("difficulty"),
        failure_mode=data.get("failure_mode"),
        seed=data.get("seed"),
        source_path=str(path),
    )


def load_adapter(path: Path) -> AdapterSpec:
    data = _load_json(path)
    required = ["bridge_id", "track"]
    missing = [k for k in required if k not in data]
    if missing:
        raise SystemExit(f"AdapterSpec {path} missing required field(s): {', '.join(missing)}")
    timeout = int(data.get("timeout_seconds") or 60)
    if timeout <= 0:
        raise SystemExit(f"AdapterSpec {path} timeout_seconds must be positive")
    env = data.get("env") or {}
    if not isinstance(env, dict):
        raise SystemExit(f"AdapterSpec {path} env must be an object")
    return AdapterSpec(
        bridge_id=str(data["bridge_id"]),
        track=str(data["track"]),
        preamble=str(data.get("preamble") or ""),
        source_preamble=str(data.get("source_preamble") or ""),
        env={str(k): str(v) for k, v in env.items()},
        docs=str(data.get("docs") or ""),
        known_gaps=list(data.get("known_gaps") or []),
        op_api_map=dict(data.get("op_api_map") or {}),
        atol=float(data.get("atol", 1e-5)),
        rtol=float(data.get("rtol", 1e-5)),
        timeout_seconds=timeout,
        fne_threshold=_optional_float(data.get("fne_threshold")),
        gc_threshold=_optional_float(data.get("gc_threshold")),
        tca_threshold=_optional_float(data.get("tca_threshold")),
        dtw_threshold=_optional_float(data.get("dtw_threshold")),
        source_path=str(path),
    )


def load_effort_ledger(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "entries": [],
            "baseline_effort": None,
        }
    data = _load_json(path)
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise SystemExit(f"Effort ledger {path} must contain an entries array")
    return {
        "entries": entries,
        "reroll_entries": data.get("reroll_entries") or entries,
        "baseline_effort": data.get("baseline_effort"),
        "migration_samples": data.get("migration_samples") or data.get("samples") or [],
    }


def load_ar_baseline(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = _load_json(path)
    if data.get("baseline_effort") is None:
        raise SystemExit(f"AR baseline {path} missing baseline_effort")
    if str(data.get("effort_formula_version") or "") != "shared-effort-v1":
        raise SystemExit(f"AR baseline {path} has unsupported effort_formula_version: {data.get('effort_formula_version')}")
    data = dict(data)
    data["source_path"] = str(path)
    return data


_CURRENT_EFFORT_LEDGER: dict[str, Any] = {
    "entries": [],
    "baseline_effort": None,
    "source_path": None,
    "baseline_metadata": None,
}


def set_current_effort_ledger(ledger: dict[str, Any]) -> None:
    global _CURRENT_EFFORT_LEDGER
    _CURRENT_EFFORT_LEDGER = ledger


def build_effort_ledger(case: TestCase, adapter: AdapterSpec) -> dict[str, Any]:
    entries = []
    for entry in _CURRENT_EFFORT_LEDGER.get("entries", []):
        entry_case = entry.get("case_id")
        entry_bridge = entry.get("bridge_id")
        if entry_case not in (None, case.id):
            continue
        if entry_bridge not in (None, adapter.bridge_id):
            continue
        entries.append(entry)
    return {
        "entries": entries,
        "baseline_effort": _CURRENT_EFFORT_LEDGER.get("baseline_effort"),
        "source_path": _CURRENT_EFFORT_LEDGER.get("source_path"),
        "baseline_metadata": _CURRENT_EFFORT_LEDGER.get("baseline_metadata"),
    }


def prepare_effort_context(effort_ledger_path: str | None, ar_baseline_path: str | None) -> dict[str, Any]:
    ledger = load_effort_ledger(Path(effort_ledger_path)) if effort_ledger_path else load_effort_ledger(None)
    if effort_ledger_path:
        ledger["source_path"] = str(Path(effort_ledger_path).resolve())
    baseline = load_ar_baseline(Path(ar_baseline_path)) if ar_baseline_path else None
    if baseline is not None:
        ledger["baseline_effort"] = baseline["baseline_effort"]
        ledger["baseline_metadata"] = baseline
    ledger.setdefault("migration_samples", [])
    return ledger


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _runner_source(preamble: str, code: str) -> str:
    preamble_block = textwrap.indent(preamble, "    ") if preamble.strip() else "    pass"
    code_block = textwrap.indent(code, "    ")
    return f'''import json
import math
import sys
import traceback

def _normalize(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float):
            if math.isnan(value):
                return {{"__float__": "nan"}}
            if math.isinf(value):
                return {{"__float__": "inf" if value > 0 else "-inf"}}
        return value
    if isinstance(value, dict):
        return {{str(k): _normalize(v) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return _normalize(value.tolist())
    if hasattr(value, "item"):
        try:
            return _normalize(value.item())
        except Exception:
            pass
    return repr(value)

try:
{preamble_block}

{code_block}

    if "RESULT" in globals():
        result = globals()["RESULT"]
    elif "run" in globals() and callable(globals()["run"]):
        result = globals()["run"]()
    else:
        raise RuntimeError("Test code must define RESULT or run().")
    payload = {{"result": _normalize(result)}}
    if "ACTIVATIONS" in globals():
        payload["activations"] = _normalize(globals()["ACTIVATIONS"])
    if "GRADIENTS" in globals():
        payload["gradients"] = _normalize(globals()["GRADIENTS"])
    if "TASK_METRICS" in globals():
        payload["task_metrics"] = _normalize(globals()["TASK_METRICS"])
    print("TBBCC_PAYLOAD_JSON=" + json.dumps(payload, sort_keys=True))
except Exception:
    print("TBBCC_TRACEBACK_START", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print("TBBCC_TRACEBACK_END", file=sys.stderr)
    sys.exit(1)
'''


def execute_python(role: str, preamble: str, code: str, env: dict[str, str], timeout: int) -> ExecResult:
    source = _runner_source(preamble, code)
    start = _dt.datetime.now(_dt.UTC)
    with tempfile.TemporaryDirectory(prefix="tbbcc_") as td:
        script = Path(td) / f"{role}.py"
        script.write_text(source, encoding="utf-8")
        merged_env = os.environ.copy()
        merged_env.update(env)
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                text=True,
                capture_output=True,
                env=merged_env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            end = _dt.datetime.now(_dt.UTC)
            return ExecResult(
                role=role,
                ok=False,
                result=None,
                activations=None,
                gradients=None,
                task_metrics=None,
                returncode=124,
                duration_seconds=(end - start).total_seconds(),
                stdout=_coerce_text(exc.stdout),
                stderr=_coerce_text(exc.stderr) or f"Timeout after {timeout}s",
                traceback=f"Timeout after {timeout}s",
            )
    end = _dt.datetime.now(_dt.UTC)
    payload = _extract_payload(proc.stdout)
    tb = _extract_traceback(proc.stderr)
    return ExecResult(
        role=role,
        ok=proc.returncode == 0 and payload is not _MISSING,
        result=None if payload is _MISSING else payload.get("result"),
        activations=None if payload is _MISSING else payload.get("activations"),
        gradients=None if payload is _MISSING else payload.get("gradients"),
        task_metrics=None if payload is _MISSING else payload.get("task_metrics"),
        returncode=proc.returncode,
        duration_seconds=(end - start).total_seconds(),
        stdout=proc.stdout,
        stderr=proc.stderr,
        traceback=tb,
    )


def execute_python_pair(source_preamble: str, target_preamble: str, code: str, env: dict[str, str], timeout: int) -> tuple[ExecResult, ExecResult]:
    """Execute source and target in separate Python subprocesses.

    Bridge adapters can be import-order sensitive. Keeping both roles in one
    interpreter lets source-side PyTorch imports, sys.modules mutations, global
    device state, or environment edits leak into target initialization. The
    benchmark measures bridge compatibility, not contamination by the harness,
    so isolation is the default.
    """
    source = execute_python("source", source_preamble, code, {}, timeout)
    target = execute_python("target", target_preamble, code, env, timeout)
    return source, target


class _Missing:
    pass


_MISSING = _Missing()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _extract_payload(stdout: str) -> Any:
    prefix = "TBBCC_PAYLOAD_JSON="
    for line in reversed(stdout.splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    legacy_prefix = "TBBCC_RESULT_JSON="
    for line in reversed(stdout.splitlines()):
        if line.startswith(legacy_prefix):
            return {"result": json.loads(line[len(legacy_prefix) :])}
    return _MISSING


def _extract_traceback(stderr: str) -> str | None:
    if "TBBCC_TRACEBACK_START" not in stderr:
        return stderr.strip() or None
    m = re.search(r"TBBCC_TRACEBACK_START\n(?P<body>.*?)\nTBBCC_TRACEBACK_END", stderr, re.S)
    if m:
        return m.group("body").strip()
    return stderr.strip() or None


def _flatten_numbers(value: Any, prefix: str = "$") -> tuple[list[float], list[str]]:
    numbers: list[float] = []
    paths: list[str] = []
    if isinstance(value, dict):
        if set(value.keys()) == {"__float__"}:
            marker = value["__float__"]
            if marker == "nan":
                numbers.append(float("nan"))
            elif marker == "inf":
                numbers.append(float("inf"))
            elif marker == "-inf":
                numbers.append(float("-inf"))
            paths.append(prefix)
            return numbers, paths
        for key in sorted(value):
            child_numbers, child_paths = _flatten_numbers(value[key], f"{prefix}.{key}")
            numbers.extend(child_numbers)
            paths.extend(child_paths)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            child_numbers, child_paths = _flatten_numbers(item, f"{prefix}[{idx}]")
            numbers.extend(child_numbers)
            paths.extend(child_paths)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numbers.append(float(value))
        paths.append(prefix)
    return numbers, paths


def compare_values(src: Any, tgt: Any, atol: float, rtol: float) -> dict[str, Any]:
    if _shape_signature(src) != _shape_signature(tgt):
        return {
            "passed": False,
            "reason": "ShapeMismatch",
            "source_shape": _shape_signature(src),
            "target_shape": _shape_signature(tgt),
        }

    src_nums, paths = _flatten_numbers(src)
    tgt_nums, _ = _flatten_numbers(tgt)
    if not src_nums and not tgt_nums:
        return {
            "passed": src == tgt,
            "reason": None if src == tgt else "SemanticMismatch",
            "exact_match": src == tgt,
        }
    if len(src_nums) != len(tgt_nums):
        return {
            "passed": False,
            "reason": "ShapeMismatch",
            "source_count": len(src_nums),
            "target_count": len(tgt_nums),
        }

    diffs: list[float] = []
    failed_paths: list[str] = []
    dot = 0.0
    src_norm = 0.0
    tgt_norm = 0.0
    for idx, (a, b) in enumerate(zip(src_nums, tgt_nums)):
        if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
            close = a == b
            diff = 0.0 if close else float("inf")
        else:
            diff = abs(a - b)
            close = diff <= (atol + rtol * abs(b))
            dot += a * b
            src_norm += a * a
            tgt_norm += b * b
        diffs.append(diff)
        if not close:
            failed_paths.append(paths[idx])

    finite_diffs = [d for d in diffs if math.isfinite(d)]
    mae = statistics.fmean(finite_diffs) if finite_diffs else float("inf")
    max_error = max(diffs) if diffs else 0.0
    p95 = _percentile(finite_diffs, 95.0) if finite_diffs else float("inf")
    cosine = None
    if src_norm > 0 and tgt_norm > 0:
        cosine = dot / (math.sqrt(src_norm) * math.sqrt(tgt_norm))
    return {
        "passed": not failed_paths,
        "reason": None if not failed_paths else "NumericMismatch",
        "count": len(src_nums),
        "mae": mae,
        "max_error": max_error,
        "p95": p95,
        "cosine": cosine,
        "failed_paths": failed_paths[:20],
        "failed_count": len(failed_paths),
        "atol": atol,
        "rtol": rtol,
    }


def _shape_signature(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {"__float__"}:
            return "number"
        return {k: _shape_signature(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return ["list", len(value), _shape_signature(value[0]) if value else "empty"]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    return type(value).__name__


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def compare_optional_channel(source_value: Any, target_value: Any, atol: float, rtol: float) -> dict[str, Any]:
    if source_value is None and target_value is None:
        return {"implemented": False, "passed": None, "metrics": {}}
    if source_value is None or target_value is None:
        return {
            "implemented": True,
            "passed": False,
            "metrics": {
                "reason": "MissingChannel",
                "source_present": source_value is not None,
                "target_present": target_value is not None,
            },
        }
    metrics = compare_values(source_value, target_value, atol, rtol)
    return {"implemented": True, "passed": bool(metrics.get("passed")), "metrics": metrics}


def build_t2_result(source: ExecResult, target: ExecResult, atol: float, rtol: float) -> dict[str, Any]:
    activations = compare_optional_channel(source.activations, target.activations, atol, rtol)
    gradients = compare_optional_channel(source.gradients, target.gradients, atol, rtol)
    implemented = bool(activations["implemented"] or gradients["implemented"])
    if not implemented:
        return {
            "name": "ActivationGradient",
            "implemented": False,
            "passed": None,
            "metrics": {},
        }
    channel_results = {
        "activations": activations,
        "gradients": gradients,
    }
    passed = all(
        result["passed"] is not False
        for result in channel_results.values()
        if result["implemented"]
    )
    first_fail = None
    for name, result in channel_results.items():
        if result["implemented"] and result["passed"] is False:
            first_fail = name
            break
    return {
        "name": "ActivationGradient",
        "implemented": True,
        "passed": passed,
        "metrics": {
            "channels": channel_results,
            "first_fail": first_fail,
        },
    }


def build_t3_result(source: ExecResult, target: ExecResult, atol: float, rtol: float) -> dict[str, Any]:
    task = compare_optional_channel(source.task_metrics, target.task_metrics, atol, rtol)
    if not task["implemented"]:
        return {
            "name": "Task",
            "implemented": False,
            "passed": None,
            "metrics": {},
        }
    return {
        "name": "Task",
        "implemented": True,
        "passed": task["passed"],
        "metrics": task["metrics"],
    }


def auto_classify(
    source: ExecResult,
    target: ExecResult,
    comparison: dict[str, Any] | None,
    t2_result: dict[str, Any] | None = None,
    t3_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = []
    tb = "\n".join(x for x in [source.traceback, target.traceback, source.stderr, target.stderr] if x)
    lower = tb.lower()
    cls = "NoFailure"

    if not source.ok:
        cls = "EnvironmentFailure"
        evidence.append("baseline execution failed")
    elif not target.ok:
        if "no module named" in lower or "modulenotfounderror" in lower:
            cls = "DependencyMissing"
        elif "import" in lower and ("before" in lower or "order" in lower):
            cls = "ImportOrderError"
        elif "has no attribute" in lower or "operator not found" in lower or "unsupported" in lower:
            cls = "OperatorNotFound"
        elif "dtype" in lower or "type mismatch" in lower or "expected scalar type" in lower:
            cls = "TypeMismatch"
        elif "shape" in lower or "broadcast" in lower or "size mismatch" in lower:
            cls = "ShapeMismatch"
        elif "device" in lower or "cuda" in lower or "npu" in lower or "ascend" in lower:
            cls = "DeviceMismatch"
        elif "grad" in lower or "autograd" in lower or "backward" in lower:
            cls = "AutogradFailure"
        else:
            cls = "RuntimeCrash"
        evidence.append("target execution failed")
    elif comparison and not comparison.get("passed", False):
        reason = comparison.get("reason")
        if reason in FAILURE_CLASSES:
            cls = reason
        elif reason == "SemanticMismatch":
            cls = "NumericMismatch"
        else:
            cls = "NumericMismatch"
        evidence.append(f"comparison failed: {reason}")
    elif t2_result and t2_result.get("implemented") and not t2_result.get("passed"):
        first_fail = t2_result.get("metrics", {}).get("first_fail")
        cls = "AutogradFailure" if first_fail == "gradients" else "NumericMismatch"
        evidence.append(f"Tier-2 failed: {first_fail}")
    elif t3_result and t3_result.get("implemented") and not t3_result.get("passed"):
        cls = "TrainingDivergence"
        evidence.append("Tier-3 task metrics failed")
    else:
        cls = "NoFailure"
        evidence.append("no failure detected")

    return {
        "class": cls,
        "evidence": evidence,
        "counts_toward_compatibility": cls not in {"EnvironmentFailure", "DependencyMissing", "ImportOrderError"},
    }


def build_report(case: TestCase, adapter: AdapterSpec, source: ExecResult, target: ExecResult, comparison: dict[str, Any]) -> dict[str, Any]:
    atol = float(comparison.get("atol", adapter.atol))
    rtol = float(comparison.get("rtol", adapter.rtol))
    t2_result = build_t2_result(source, target, atol, rtol) if source.ok and target.ok else {
        "name": "ActivationGradient",
        "implemented": False,
        "passed": None,
        "metrics": {},
    }
    t3_result = build_t3_result(source, target, atol, rtol) if source.ok and target.ok else {
        "name": "Task",
        "implemented": False,
        "passed": None,
        "metrics": {},
    }
    classification = auto_classify(source, target, comparison, t2_result, t3_result)
    implemented_tiers_passed = bool(comparison.get("passed")) and (
        not t2_result.get("implemented") or bool(t2_result.get("passed"))
    ) and (
        not t3_result.get("implemented") or bool(t3_result.get("passed"))
    )
    final_state = "ALL_PASS" if source.ok and target.ok and implemented_tiers_passed else "MARK_UNFIXABLE"
    effort_ledger = build_effort_ledger(case, adapter)
    effort = summarize_effort_ledger(
        effort_ledger.get("entries", []),
        baseline_effort=effort_ledger.get("baseline_effort"),
        baseline_metadata=effort_ledger.get("baseline_metadata"),
    )
    if not effort_ledger.get("entries"):
        effort["effort_adapt"] = 0.0
        effort["effort_repair"] = 0.0
        effort["effort_total"] = 0.0
        effort["me"] = 0.0
        effort["ar"] = calc_ar(0.0, effort_ledger.get("baseline_effort"))
        effort["source"] = "deterministic_no_agent"
        effort["repair_attempts"] = 0
    effort["confidence_interval"] = confidence_interval([effort["effort_total"] or 0.0])
    effort["migrate_at_k"] = migrate_at_k(
        [
            {
                "case_id": case.id,
                "task_id": case.id,
                "reroll_index": 1,
                "final_state": "ALL_PASS" if source.ok and target.ok and comparison.get("passed") else "MARK_UNFIXABLE",
                "exec_passed": bool(source.ok and target.ok and comparison.get("passed")),
                "full_passed": bool(source.ok and target.ok and comparison.get("passed")),
                "all_tiers_passed": bool(source.ok and target.ok and comparison.get("passed")),
            }
        ]
    )
    return {
        "schema_version": "tbbcc.report.v0.1",
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "final_state": final_state,
        "case": dataclasses.asdict(case),
        "adapter": dataclasses.asdict(adapter),
        "environment": inspect_environment(),
        "timing": {
            "source_duration_seconds": source.duration_seconds,
            "target_duration_seconds": target.duration_seconds,
            "combined_execution_seconds": source.duration_seconds + target.duration_seconds,
        },
        "tiers": {
            "T1": {
                "name": "Tensor",
                "implemented": True,
                "passed": bool(source.ok and target.ok and comparison.get("passed")),
                "source": dataclasses.asdict(source),
                "target": dataclasses.asdict(target),
                "metrics": comparison,
                "auto_classification": classification,
            },
            "T2": {
                **t2_result,
            },
            "T3": {
                **t3_result,
            },
        },
        "effort": {
            **effort,
        },
    }


def inspect_environment(import_versions: bool = False) -> dict[str, Any]:
    modules = [
        "torch",
        "torch_npu",
        "mindspore",
        "torch4ms",
        "mindtorch",
        "mindtorch_v2",
        "mindnlp",
        "numpy",
    ]
    info: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "version_imports_enabled": import_versions,
        "modules": {},
    }
    for name in modules:
        spec = importlib.util.find_spec(name)
        if spec is None:
            info["modules"][name] = {"available": False}
            continue
        module_info = {
            "available": True,
            "origin": spec.origin,
        }
        if import_versions:
            try:
                mod = __import__(name)
                module_info["version"] = getattr(mod, "__version__", "present")
            except Exception as exc:
                module_info["version"] = f"import_error: {exc}"
        info["modules"][name] = module_info
    info["local_bridge_sources"] = discover_local_bridge_sources(Path.cwd())
    return info


def discover_local_bridge_sources(cwd: Path) -> dict[str, list[str]]:
    """Find nearby bridge source trees without importing them."""
    roots = []
    for candidate in (cwd, cwd.parent, cwd.parent.parent):
        if candidate not in roots and candidate.exists():
            roots.append(candidate)
    patterns = {
        "torch4ms": (("torch4ms", "ascend-torch4ms"), ("torch4ms",)),
        "mindtorch": (("mindtorch",), ("mindtorch",)),
        "mindtorch_v2": (("mindtorch", "mindtorch-v2", "mindtorch_v2"), ("mindtorch", "mindtorch_v2")),
        "mindnlp": (("mindnlp",), ("mindnlp",)),
    }
    found: dict[str, list[str]] = {name: [] for name in patterns}
    for root in roots:
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            lowered = child.name.lower()
            for bridge, (needles, package_names) in patterns.items():
                if any(needle in lowered for needle in needles):
                    if not _looks_like_source_tree(child, package_names):
                        continue
                    resolved = str(child.resolve())
                    if resolved not in found[bridge]:
                        found[bridge].append(resolved)
    return {name: paths for name, paths in found.items() if paths}


def _looks_like_source_tree(path: Path, package_names: tuple[str, ...]) -> bool:
    for package in package_names:
        if (path / package).is_dir() or (path / "src" / package).is_dir():
            return True
    for marker in ("pyproject.toml", "setup.py", "setup.cfg"):
        if (path / marker).exists():
            return True
    if any(path.glob("README*")):
        return True
    if (path / "docs").is_dir() or (path / "examples").is_dir():
        return True
    return any(path.glob("test_*.py"))


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def write_suite_summary(summary: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_suite_markdown(summary), encoding="utf-8")
    return json_path, md_path


def _safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _safe_max(values: list[float]) -> float | None:
    return max(values) if values else None


def _safe_min(values: list[float]) -> float | None:
    return min(values) if values else None


def summarize_quality_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    counted = [run for run in runs if run.get("counts_toward_compatibility")]
    passed = [run for run in counted if run.get("final_state") == "ALL_PASS"]
    first_passed = [run for run in passed if int(run.get("repair_attempts") or 0) == 0]
    t1 = [run.get("t1_metrics") or {} for run in runs]
    timing = [run.get("timing") or {} for run in runs]
    source_durations = [float(item["source_duration_seconds"]) for item in timing if item.get("source_duration_seconds") is not None]
    target_durations = [float(item["target_duration_seconds"]) for item in timing if item.get("target_duration_seconds") is not None]
    combined_durations = [float(run["duration_seconds"]) for run in runs if run.get("duration_seconds") is not None]
    ratios = []
    for item in timing:
        source = item.get("source_duration_seconds")
        target = item.get("target_duration_seconds")
        if source not in (None, 0) and target is not None:
            ratios.append(float(target) / float(source))
    t2_implemented = [run for run in runs if run.get("t2_implemented")]
    t2_passed = [run for run in t2_implemented if run.get("t2_passed") is True]
    t3_implemented = [run for run in runs if run.get("t3_implemented")]
    t3_passed = [run for run in t3_implemented if run.get("t3_passed") is True]
    return {
        "compatibility_rate": (len(passed) / len(counted)) if counted else None,
        "first_pass_rate": (len(first_passed) / len(counted)) if counted else None,
        "first_passed": len(first_passed),
        "compatibility_counted_total": len(counted),
        "numeric": {
            "mean_mae": _safe_mean([float(item["mae"]) for item in t1 if item.get("mae") is not None]),
            "max_p95": _safe_max([float(item["p95"]) for item in t1 if item.get("p95") is not None]),
            "max_error": _safe_max([float(item["max_error"]) for item in t1 if item.get("max_error") is not None]),
            "min_cosine": _safe_min([float(item["cosine"]) for item in t1 if item.get("cosine") is not None]),
            "failed_count": sum(int(item.get("failed_count") or 0) for item in t1),
        },
        "performance": {
            "mean_source_seconds": _safe_mean(source_durations),
            "mean_target_seconds": _safe_mean(target_durations),
            "mean_wall_seconds": _safe_mean(combined_durations),
            "target_source_ratio_mean": _safe_mean(ratios),
            "target_source_ratio_max": _safe_max(ratios),
        },
        "tiers": {
            "t1_pass_rate": (len([run for run in runs if run.get("t1_passed") is True]) / len(runs)) if runs else None,
            "t2_implemented": len(t2_implemented),
            "t2_pass_rate_when_implemented": (len(t2_passed) / len(t2_implemented)) if t2_implemented else None,
            "t3_implemented": len(t3_implemented),
            "t3_pass_rate_when_implemented": (len(t3_passed) / len(t3_implemented)) if t3_implemented else None,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    t1 = report["tiers"]["T1"]
    t2 = report["tiers"]["T2"]
    t3 = report["tiers"]["T3"]
    metrics = t1["metrics"]
    cls = t1["auto_classification"]
    timing = report.get("timing") or {}
    effort = report.get("effort") or {}
    lines = [
        "# TorchBridgeBench Report",
        "",
        f"- Case: `{report['case']['id']}`",
        f"- Adapter: `{report['adapter']['bridge_id']}`",
        f"- Final state: `{report['final_state']}`",
        f"- Tier-1 passed: `{t1['passed']}`",
        f"- Failure class: `{cls['class']}`",
        f"- Counts toward compatibility: `{cls['counts_toward_compatibility']}`",
        f"- Combined execution seconds: `{timing.get('combined_execution_seconds')}`",
        f"- ME: `{effort.get('me')}`",
        f"- AR: `{effort.get('ar')}`",
        f"- Effort source: `{effort.get('source')}`",
        "",
        "## Tier-1 Metrics",
        "",
    ]
    for key in ["count", "mae", "p95", "max_error", "cosine", "atol", "rtol", "failed_count"]:
        if key in metrics:
            lines.append(f"- {key}: `{metrics[key]}`")
    if metrics.get("failed_paths"):
        lines.append(f"- failed_paths: `{metrics['failed_paths']}`")
    lines.extend(["", "## Tier-2 Metrics", ""])
    if not t2.get("implemented"):
        lines.append("- implemented: `False`")
    else:
        lines.append(f"- passed: `{t2.get('passed')}`")
        lines.append(f"- first_fail: `{t2.get('metrics', {}).get('first_fail')}`")
        for channel, result in t2.get("metrics", {}).get("channels", {}).items():
            if result.get("implemented"):
                channel_metrics = result.get("metrics", {})
                lines.append(
                    f"- {channel}: passed=`{result.get('passed')}`, "
                    f"mae=`{channel_metrics.get('mae')}`, failed_count=`{channel_metrics.get('failed_count')}`"
                )
    lines.extend(["", "## Tier-3 Metrics", ""])
    if not t3.get("implemented"):
        lines.append("- implemented: `False`")
    else:
        t3_metrics = t3.get("metrics", {})
        lines.append(f"- passed: `{t3.get('passed')}`")
        for key in ["mae", "p95", "max_error", "cosine", "failed_count"]:
            if key in t3_metrics:
                lines.append(f"- {key}: `{t3_metrics[key]}`")
        if t3_metrics.get("failed_paths"):
            lines.append(f"- failed_paths: `{t3_metrics['failed_paths']}`")
    lines.extend(["", "## Evidence", ""])
    for item in cls.get("evidence", []):
        lines.append(f"- {item}")
    if report["tiers"]["T1"]["target"].get("traceback"):
        lines.extend(["", "## Target Traceback", "", "```text", report["tiers"]["T1"]["target"]["traceback"], "```"])
    lines.extend(["", "## Effort Metrics", ""])
    for key in ["effort_adapt", "effort_repair", "effort_total", "baseline_effort", "repair_attempts"]:
        lines.append(f"- {key}: `{effort.get(key)}`")
    baseline_meta = effort.get("baseline_metadata") or {}
    if baseline_meta:
        lines.append(f"- baseline model: `{baseline_meta.get('model')}`")
        lines.append(f"- baseline protocol: `{baseline_meta.get('protocol')}`")
        lines.append(f"- baseline task digest: `{baseline_meta.get('task_set_digest')}`")
    ci = effort.get("confidence_interval") or {}
    lines.append(f"- effort 95% CI: `{ci.get('low')}` to `{ci.get('high')}`")
    migrate = effort.get("migrate_at_k") or {}
    full = migrate.get("full") or {}
    exec_only = migrate.get("exec") or {}
    for key in ["migrate@1", "migrate@3", "migrate@5", "migrate@10"]:
        if key in full:
            lines.append(f"- {key}_full: `{full[key].get('rate')}`")
        if key in exec_only:
            lines.append(f"- {key}_exec: `{exec_only[key].get('rate')}`")
    lines.extend(
        [
            "",
            "## Scope Notes",
            "",
            "- Tier-2 is implemented when a case exposes ACTIVATIONS or GRADIENTS.",
            "- Tier-3 is implemented when a case exposes TASK_METRICS.",
            "- Agent effort is read from an optional effort ledger; deterministic runs without a ledger record zero agent effort.",
            "",
        ]
    )
    return "\n".join(lines)


def render_suite_markdown(summary: dict[str, Any]) -> str:
    totals = summary["totals"]
    compatibility_rate = totals.get("compatibility_rate")
    raw_pass_rate = totals.get("raw_pass_rate")
    compatibility_text = "n/a" if compatibility_rate is None else f"{compatibility_rate:.4f}"
    raw_pass_text = "n/a" if raw_pass_rate is None else f"{raw_pass_rate:.4f}"
    reroll_ci = (summary.get("effort") or {}).get("reroll_effort_confidence_interval") or {}
    run_ci = (summary.get("effort") or {}).get("run_effort_confidence_interval") or {}
    ar_ci = (summary.get("effort") or {}).get("ar_confidence_interval") or {}
    scope_ar_ci = (summary.get("effort") or {}).get("scope_adjusted_ar_confidence_interval") or {}
    migrate = (summary.get("effort") or {}).get("migrate_at_k") or {}
    effort_summary = summary.get("effort") or {}
    baseline_meta = effort_summary.get("baseline_metadata") or {}
    baseline_scope = effort_summary.get("baseline_scope") or {}
    variance = effort_summary.get("variance_control") or {}
    quality = summary.get("quality") or {}
    numeric = quality.get("numeric") or {}
    performance = quality.get("performance") or {}
    tiers = quality.get("tiers") or {}
    migrate_exec_1 = (((migrate.get("exec") or {}).get("migrate@1") or {}).get("rate"))
    migrate_full_1 = (((migrate.get("full") or {}).get("migrate@1") or {}).get("rate"))
    lines = [
        "# TorchBridgeBench Suite Summary",
        "",
        f"Suite `{summary['suite_id']}` finished `{totals['passed']}/{totals['total']}` runs. "
        "Compatibility excludes environment/configuration-only failures.",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value | Meaning |",
        "| --- | ---: | --- |",
        f"| Compatibility rate | `{compatibility_text}` | Bridge-relevant pass rate after excluding environment noise. |",
        f"| First-pass rate | `{quality.get('first_pass_rate')}` | Passed without counted repair effort. |",
        f"| Raw pass rate | `{raw_pass_text}` | All runs passed divided by all runs. |",
        f"| ME | `{effort_summary.get('me')}` | Total counted migration effort: adapt + repair. |",
        f"| AR | `{effort_summary.get('ar')}` | Work avoided versus calibrated no-bridge baseline. |",
        f"| Scope-adjusted AR | `{effort_summary.get('scope_adjusted_ar')}` | Subset-only diagnostic when not running the full baseline task set. |",
        f"| Effort_adapt / Effort_repair | `{effort_summary.get('effort_adapt')}` / `{effort_summary.get('effort_repair')}` | Agent work split by design phase. |",
        f"| Repair attempts | `{effort_summary.get('repair_attempts')}` | Counted repair attempts; environment remediation is excluded. |",
        f"| Numeric max error | `{numeric.get('max_error')}` | Worst Tier-1 numeric deviation. |",
        f"| Numeric mean MAE | `{numeric.get('mean_mae')}` | Mean absolute error across runs. |",
        f"| Numeric min cosine | `{numeric.get('min_cosine')}` | Worst cosine similarity across runs. |",
        f"| Mean wall seconds | `{performance.get('mean_wall_seconds')}` | Average end-to-end run time. |",
        f"| Target/source time ratio | `{performance.get('target_source_ratio_mean')}` | Mean target execution time divided by source time. |",
        f"| T2 pass rate | `{tiers.get('t2_pass_rate_when_implemented')}` | Activation/gradient pass rate where implemented. |",
        f"| T3 pass rate | `{tiers.get('t3_pass_rate_when_implemented')}` | Task-metric pass rate where implemented. |",
        f"| migrate@1 exec/full | `{migrate_exec_1}` / `{migrate_full_1}` | One reroll succeeds at T1 / all enabled tiers. |",
        f"| Reroll CV | `{variance.get('cv')}` | Stability of agent-related effort samples. |",
        "",
        "## Executive Summary",
        "",
    ]
    if compatibility_rate is None:
        lines.append("- No runs counted toward compatibility; review environment failures first.")
    else:
        lines.append(
            f"- Compatibility is measured after excluding environment/config-only failures; current rate is `{compatibility_text}`."
        )
    if summary["failure_classes"]:
        top_failure = sorted(summary["failure_classes"].items(), key=lambda item: (-item[1], item[0]))[0]
        lines.append(f"- Most frequent failure class: `{top_failure[0]}` x `{top_failure[1]}`.")
    else:
        lines.append("- No failure classes were recorded.")
    if baseline_scope:
        lines.append(
            f"- AR baseline scope: evaluated `{baseline_scope.get('evaluated_case_count')}` cases, "
            f"baseline has `{baseline_scope.get('baseline_case_count')}` cases, "
            f"policy=`{baseline_scope.get('baseline_scaling_policy')}`. "
            "The primary AR above always uses the calibrated full baseline; scope-adjusted AR is a subset-only diagnostic."
        )
    if variance:
        lines.append(
            f"- Reroll stability: cv=`{variance.get('cv')}`, needs_more_samples=`{variance.get('needs_more_samples')}`."
        )
    lines.append(
        f"- Performance summary: mean wall=`{performance.get('mean_wall_seconds')}`s, "
        f"target/source ratio mean=`{performance.get('target_source_ratio_mean')}`."
    )
    lines.append(
        f"- Numeric summary: max_error=`{numeric.get('max_error')}`, "
        f"mean_mae=`{numeric.get('mean_mae')}`, failed_count=`{numeric.get('failed_count')}`."
    )
    lines.extend(
        [
            "",
            "## Paper-Ready Values",
            "",
            f"- Compatibility rate: `{compatibility_text}`",
            f"- First-pass rate: `{quality.get('first_pass_rate')}`",
            f"- ME / AR: `{effort_summary.get('me')}` / `{effort_summary.get('ar')}`",
            f"- Numeric consistency: max_error=`{numeric.get('max_error')}`, min_cosine=`{numeric.get('min_cosine')}`",
            f"- Performance: mean_wall_seconds=`{performance.get('mean_wall_seconds')}`, target_source_ratio_mean=`{performance.get('target_source_ratio_mean')}`",
            f"- migrate@1_exec/full: `{migrate_exec_1}` / `{migrate_full_1}`",
        ]
    )
    if baseline_meta:
        lines.extend(["", "## AR Baseline", ""])
        lines.append(f"- model: `{baseline_meta.get('model')}`")
        lines.append(f"- observed model: `{baseline_meta.get('model_observed')}`")
        lines.append(f"- provider: `{baseline_meta.get('provider')}`")
        lines.append(f"- protocol: `{baseline_meta.get('protocol')}`")
        lines.append(f"- agent system: `{baseline_meta.get('agent_system')}`")
        lines.append(f"- agent system version: `{baseline_meta.get('agent_system_version')}`")
        lines.append(f"- effort formula version: `{baseline_meta.get('effort_formula_version')}`")
        lines.append(f"- task set digest: `{baseline_meta.get('task_set_digest')}`")
    lines.extend(["", "## Migrate@k", ""])
    for mode, label in [("exec", "exec only"), ("full", "full tiers")]:
        mode_data = migrate.get(mode) or {}
        for key in ["migrate@1", "migrate@3", "migrate@5", "migrate@10"]:
            if key in mode_data:
                lines.append(f"- {key} ({label}): `{mode_data[key].get('rate')}`")
        if "saturation_k" in mode_data:
            lines.append(f"- saturation_k ({label}): `{mode_data.get('saturation_k')}`")
    lines.extend(
        [
            "",
            "## Effort Detail",
            "",
            f"- Run-level effort mean: `{run_ci.get('mean')}`",
            f"- Run-level effort 95% CI: `{run_ci.get('low')}` to `{run_ci.get('high')}`",
            f"- Effort source: `{effort_summary.get('source')}`",
            f"- Ledger path: `{effort_summary.get('ledger_path')}`",
            "",
            "## Failure Classes",
            "",
        ]
    )
    if summary["failure_classes"]:
        for name, count in sorted(summary["failure_classes"].items()):
            lines.append(f"- {name}: `{count}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Run Details", ""])
    for item in summary["runs"]:
        counted = item.get("counts_toward_compatibility")
        counted_text = "yes" if counted else "no"
        lines.append(
            f"- `{item['case_id']}` x `{item['bridge_id']}`: "
            f"`{item['final_state']}` / `{item['failure_class']}` / counted=`{counted_text}` / duration=`{item.get('duration_seconds')}` "
            f"([json]({item['report_json']}), [md]({item['report_md']}))"
        )
    lines.append("")
    return "\n".join(lines)


def _slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return clean or "item"


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = (base / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def run_eval(case_path: Path, adapter_path: Path, out_dir: Path, timeout_override: int | None = None, atol_override: float | None = None, rtol_override: float | None = None) -> dict[str, Any]:
    started = _dt.datetime.now(_dt.UTC)
    case = load_case(case_path)
    adapter = load_adapter(adapter_path)
    timeout = int(timeout_override or adapter.timeout_seconds)
    atol = float(atol_override if atol_override is not None else case.ground_truth.get("atol", adapter.atol))
    rtol = float(rtol_override if rtol_override is not None else case.ground_truth.get("rtol", adapter.rtol))
    source, target = execute_python_pair(adapter.source_preamble, adapter.preamble, case.code, adapter.env, timeout)
    comparison: dict[str, Any]
    if source.ok and target.ok:
        comparison = compare_values(source.result, target.result, atol, rtol)
    else:
        comparison = {"passed": False, "reason": "ExecutionFailure", "atol": atol, "rtol": rtol}
    report = build_report(case, adapter, source, target, comparison)
    report["timing"]["wall_clock_seconds"] = (_dt.datetime.now(_dt.UTC) - started).total_seconds()
    json_path, md_path = write_report(report, out_dir)
    report["_paths"] = {"report_json": str(json_path), "report_md": str(md_path)}
    return report


def cmd_eval(args: argparse.Namespace) -> int:
    ledger = prepare_effort_context(args.effort_ledger, args.ar_baseline)
    set_current_effort_ledger(ledger)
    report = run_eval(
        Path(args.case),
        Path(args.adapter),
        Path(args.out),
        timeout_override=args.timeout,
        atol_override=args.atol,
        rtol_override=args.rtol,
    )
    paths = report.pop("_paths")
    print(json.dumps({"report_json": paths["report_json"], "report_md": paths["report_md"], "final_state": report["final_state"]}, indent=2))
    return 0 if report["final_state"] == "ALL_PASS" else 1


def cmd_eval_suite(args: argparse.Namespace) -> int:
    ledger = prepare_effort_context(args.effort_ledger, args.ar_baseline)
    set_current_effort_ledger(ledger)
    suite_started = _dt.datetime.now(_dt.UTC)
    suite_path = Path(args.suite).resolve()
    suite = _load_json(suite_path)
    suite_id = str(suite.get("suite_id") or suite_path.stem)
    cases = suite.get("cases") or []
    adapters = suite.get("adapters") or []
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"Suite {suite_path} must define a non-empty cases array")
    if not isinstance(adapters, list) or not adapters:
        raise SystemExit(f"Suite {suite_path} must define a non-empty adapters array")
    base = suite_path.parent
    out_dir = Path(args.out).resolve()
    runs: list[dict[str, Any]] = []
    failure_classes: dict[str, int] = {}
    passed = 0
    failed = 0
    compatibility_total = 0
    compatibility_passed = 0
    migration_samples: list[dict[str, Any]] = []
    for case_item in cases:
        case_path = _resolve_path(base, str(case_item))
        case_obj = load_case(case_path)
        for adapter_item in adapters:
            adapter_path = _resolve_path(base, str(adapter_item))
            adapter_obj = load_adapter(adapter_path)
            run_dir = out_dir / "runs" / f"{_slug(case_obj.id)}__{_slug(adapter_obj.bridge_id)}"
            report = run_eval(
                case_path,
                adapter_path,
                run_dir,
                timeout_override=args.timeout,
                atol_override=args.atol,
                rtol_override=args.rtol,
            )
            paths = report.pop("_paths")
            cls = report["tiers"]["T1"]["auto_classification"]["class"]
            counts_toward_compatibility = report["tiers"]["T1"]["auto_classification"]["counts_toward_compatibility"]
            run_duration = report.get("timing", {}).get("wall_clock_seconds")
            effort = report.get("effort", {})
            migration_samples.append(
                {
                    "case_id": case_obj.id,
                    "task_id": case_obj.id,
                    "bridge_id": adapter_obj.bridge_id,
                    "reroll_index": 1,
                    "final_state": report["final_state"],
                    "exec_passed": bool(report.get("tiers", {}).get("T1", {}).get("passed")),
                    "full_passed": report["final_state"] == "ALL_PASS",
                    "all_tiers_passed": report["final_state"] == "ALL_PASS",
                }
            )
            if report["final_state"] == "ALL_PASS":
                passed += 1
                if counts_toward_compatibility:
                    compatibility_passed += 1
            else:
                failed += 1
                failure_classes[cls] = failure_classes.get(cls, 0) + 1
            if counts_toward_compatibility:
                compatibility_total += 1
            runs.append(
                {
                    "case_id": case_obj.id,
                    "case_path": str(case_path),
                    "bridge_id": adapter_obj.bridge_id,
                    "adapter_path": str(adapter_path),
                    "final_state": report["final_state"],
                    "failure_class": cls,
                    "report_json": paths["report_json"],
                    "report_md": paths["report_md"],
                    "counts_toward_compatibility": counts_toward_compatibility,
                    "duration_seconds": run_duration,
                    "timing": report.get("timing", {}),
                    "t1_passed": report.get("tiers", {}).get("T1", {}).get("passed"),
                    "t1_metrics": report.get("tiers", {}).get("T1", {}).get("metrics", {}),
                    "t2_implemented": report.get("tiers", {}).get("T2", {}).get("implemented"),
                    "t2_passed": report.get("tiers", {}).get("T2", {}).get("passed"),
                    "t3_implemented": report.get("tiers", {}).get("T3", {}).get("implemented"),
                    "t3_passed": report.get("tiers", {}).get("T3", {}).get("passed"),
                    "effort_total": effort.get("effort_total"),
                    "effort_adapt": effort.get("effort_adapt"),
                    "effort_repair": effort.get("effort_repair"),
                    "repair_attempts": effort.get("repair_attempts"),
                    "ar": effort.get("ar"),
                }
            )
    total = passed + failed
    suite_duration = (_dt.datetime.now(_dt.UTC) - suite_started).total_seconds()
    ledger_migration_samples = ledger.get("migration_samples") or []
    migrate_samples = ledger_migration_samples if ledger_migration_samples else migration_samples
    suite_effort = summarize_suite_effort(
        runs,
        ledger.get("entries", []),
        reroll_entries=ledger.get("reroll_entries"),
        baseline_effort=ledger.get("baseline_effort"),
        baseline_metadata=ledger.get("baseline_metadata"),
        evaluated_case_ids=[run["case_id"] for run in runs],
    )
    suite_effort["ledger_path"] = ledger.get("source_path")
    suite_effort["migrate_at_k"] = migrate_at_k(migrate_samples)
    summary = {
        "schema_version": "tbbcc.suite.v0.1",
        "suite_id": suite_id,
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "suite_path": str(suite_path),
        "totals": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "compatibility_counted_total": compatibility_total,
            "compatibility_counted_passed": compatibility_passed,
            "compatibility_rate": (compatibility_passed / compatibility_total) if compatibility_total else None,
            "raw_pass_rate": (passed / total) if total else 0.0,
            "duration_seconds": suite_duration,
        },
        "effort": suite_effort,
        "quality": summarize_quality_metrics(runs),
        "failure_classes": failure_classes,
        "runs": runs,
    }
    json_path, md_path = write_suite_summary(summary, out_dir)
    print(json.dumps({"summary_json": str(json_path), "summary_md": str(md_path), "totals": summary["totals"]}, indent=2))
    return 0 if failed == 0 else 1


def cmd_validate(args: argparse.Namespace) -> int:
    errors: list[str] = []
    for case_path in args.case or []:
        try:
            load_case(Path(case_path))
        except SystemExit as exc:
            errors.append(str(exc))
    for adapter_path in args.adapter or []:
        try:
            load_adapter(Path(adapter_path))
        except SystemExit as exc:
            errors.append(str(exc))
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps({"ok": True}, indent=2))
    return 0


def cmd_inspect_env(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_environment(import_versions=bool(args.import_versions)), indent=2, ensure_ascii=False))
    return 0


def cmd_plot_reports(args: argparse.Namespace) -> int:
    kinds = []
    if args.failure_taxonomy:
        kinds.append("failure-taxonomy")
    if args.compatibility_overview:
        kinds.append("compatibility-overview")
    if args.tolerance_sweep:
        kinds.append("tolerance-sweep")
    if args.model_method_heatmap:
        kinds.append("model-method-heatmap")
    if args.design_space:
        kinds.append("design-space")
    if args.metric_scorecard:
        kinds.append("metric-scorecard")
    if args.ground_truth_coverage:
        kinds.append("ground-truth-coverage")
    result = generate_report_plots(
        [Path(item) for item in args.summary],
        Path(args.out).resolve(),
        kinds=kinds or None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _suite_case_ids(suite_path: Path) -> tuple[list[str], list[str]]:
    suite = _load_json(suite_path)
    cases = suite.get("cases") or []
    if not isinstance(cases, list):
        return [], [f"Suite {suite_path} cases must be a list"]
    ids: list[str] = []
    errors: list[str] = []
    for case_item in cases:
        case_path = _resolve_path(suite_path.parent, str(case_item))
        try:
            ids.append(load_case(case_path).id)
        except BaseException as exc:
            errors.append(f"{case_item}: {exc}")
    return ids, errors


def find_eval_caches(
    bridge_id: str,
    reports_root: Path,
    suite_path: Path | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    expected_ids: list[str] | None = None
    if suite_path is not None:
        expected_ids, errors = _suite_case_ids(suite_path)
        if errors:
            raise SystemExit(f"Cannot inspect suite {suite_path}: {errors[0]}")
    matches: list[dict[str, Any]] = []
    if not reports_root.exists():
        return matches
    for generated_suite in reports_root.rglob("suite.generated.json"):
        cache_dir = generated_suite.parent
        adapter_path = cache_dir / "adapter.generated.json"
        if not adapter_path.exists():
            continue
        try:
            adapter = load_adapter(adapter_path)
            case_ids, errors = _suite_case_ids(generated_suite)
        except BaseException:
            continue
        if errors or adapter.bridge_id != bridge_id:
            continue
        quality_warnings = _cache_quality_warnings(adapter)
        if quality_warnings:
            continue
        scope_match = True
        if expected_ids is not None:
            scope_match = case_ids == expected_ids
        summary_path = cache_dir / "summary.json"
        if not summary_path.exists() and (cache_dir / "eval" / "summary.json").exists():
            summary_path = cache_dir / "eval" / "summary.json"
        matches.append(
            {
                "cache_dir": str(cache_dir.resolve()),
                "adapter_path": str(adapter_path.resolve()),
                "suite_path": str(generated_suite.resolve()),
                "summary_path": str(summary_path.resolve()) if summary_path.exists() else None,
                "bridge_id": adapter.bridge_id,
                "case_count": len(case_ids),
                "scope_match": scope_match,
                "quality_warnings": quality_warnings,
                "mtime": generated_suite.stat().st_mtime,
            }
        )
    matches = [item for item in matches if item["scope_match"]]
    matches.sort(key=lambda item: item["mtime"], reverse=True)
    return matches[:limit]


def _cache_quality_warnings(adapter: AdapterSpec) -> list[str]:
    warnings: list[str] = []
    preamble = adapter.preamble or ""
    if adapter.bridge_id == "torch4ms":
        if "TORCH4MS_DEVICE_TARGET" not in preamble and "TORCH4MS_DEVICE_TARGET" not in adapter.env:
            warnings.append("torch4ms cache does not configure TORCH4MS_DEVICE_TARGET")
        if "Configuration" not in preamble or "default_device_target" not in preamble:
            warnings.append("torch4ms cache does not configure torch4ms Configuration.default_device_target")
    return warnings


def cmd_cache_status(args: argparse.Namespace) -> int:
    matches = find_eval_caches(
        bridge_id=str(args.bridge_id),
        reports_root=Path(args.reports_root).resolve(),
        suite_path=Path(args.suite).resolve() if args.suite else None,
        limit=int(args.limit),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "bridge_id": args.bridge_id,
                "suite": str(Path(args.suite).resolve()) if args.suite else None,
                "reports_root": str(Path(args.reports_root).resolve()),
                "match_count": len(matches),
                "matches": matches,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tbbcc", description="TorchBridgeBench Claude Code plugin core")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Run a Tier-1 benchmark evaluation")
    p_eval.add_argument("--case", required=True, help="Path to TestCase JSON")
    p_eval.add_argument("--adapter", required=True, help="Path to AdapterSpec JSON")
    p_eval.add_argument("--out", required=True, help="Output directory")
    p_eval.add_argument("--timeout", type=int, default=None, help="Override timeout seconds")
    p_eval.add_argument("--atol", type=float, default=None, help="Override absolute tolerance")
    p_eval.add_argument("--rtol", type=float, default=None, help="Override relative tolerance")
    p_eval.add_argument("--effort-ledger", default=None, help="Path to optional effort ledger JSON")
    p_eval.add_argument("--ar-baseline", default=None, help="Path to AR baseline.json calibration artifact")
    p_eval.set_defaults(func=cmd_eval)

    p_suite = sub.add_parser("eval-suite", help="Run a benchmark suite matrix")
    p_suite.add_argument("--suite", required=True, help="Path to suite JSON")
    p_suite.add_argument("--out", required=True, help="Output directory")
    p_suite.add_argument("--timeout", type=int, default=None, help="Override timeout seconds")
    p_suite.add_argument("--atol", type=float, default=None, help="Override absolute tolerance")
    p_suite.add_argument("--rtol", type=float, default=None, help="Override relative tolerance")
    p_suite.add_argument("--effort-ledger", default=None, help="Path to optional effort ledger JSON")
    p_suite.add_argument("--ar-baseline", default=None, help="Path to AR baseline.json calibration artifact")
    p_suite.set_defaults(func=cmd_eval_suite)

    p_validate = sub.add_parser("validate-inputs", help="Validate case and adapter JSON files")
    p_validate.add_argument("--case", action="append", default=[])
    p_validate.add_argument("--adapter", action="append", default=[])
    p_validate.set_defaults(func=cmd_validate)

    p_env = sub.add_parser("inspect-env", help="Print local benchmark environment summary")
    p_env.add_argument(
        "--import-versions",
        action="store_true",
        help="Import modules to read __version__. Slower and may initialize heavy frameworks.",
    )
    p_env.set_defaults(func=cmd_inspect_env)

    p_plot = sub.add_parser(
        "plot-reports",
        help="Generate optional compatibility-analysis figures from summary.json reports",
    )
    p_plot.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Path to a TorchBridgeBench or eval-migration summary.json. Repeat for comparisons.",
    )
    p_plot.add_argument("--out", required=True, help="Output directory for SVG/PDF/PNG figures and source data")
    p_plot.add_argument(
        "--failure-taxonomy",
        action="store_true",
        help="Generate stacked failure-class distribution figure",
    )
    p_plot.add_argument(
        "--compatibility-overview",
        action="store_true",
        help="Generate compatibility-vs-raw-pass overview figure",
    )
    p_plot.add_argument(
        "--tolerance-sweep",
        action="store_true",
        help="Generate tolerance sensitivity figure from numeric-error fields",
    )
    p_plot.add_argument(
        "--model-method-heatmap",
        action="store_true",
        help="Generate bridge x suite/category pass-rate heatmap",
    )
    p_plot.add_argument(
        "--design-space",
        action="store_true",
        help="Generate 2D model/component/failure-mode design-space chart",
    )
    p_plot.add_argument(
        "--metric-scorecard",
        action="store_true",
        help="Generate metric availability and scorecard figure",
    )
    p_plot.add_argument(
        "--ground-truth-coverage",
        action="store_true",
        help="Generate GPU ground-truth coverage and artifact-completeness figure",
    )
    p_plot.set_defaults(func=cmd_plot_reports)

    p_cache = sub.add_parser("cache-status", help="Find reusable generated adapter/suite caches")
    p_cache.add_argument("--bridge-id", required=True, help="Bridge id to match in adapter.generated.json")
    p_cache.add_argument("--suite", default=None, help="Optional benchmark suite whose case ids must match")
    p_cache.add_argument("--reports-root", default="reports", help="Reports directory to scan")
    p_cache.add_argument("--limit", type=int, default=10, help="Maximum matching caches to return")
    p_cache.set_defaults(func=cmd_cache_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
