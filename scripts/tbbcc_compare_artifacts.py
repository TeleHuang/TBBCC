#!/usr/bin/env python3
"""Compare canonical GPU reference artifacts against NPU bridge artifacts."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

import tbbcc
from tbbcc_gpu_reference import _case_slug


SCHEMA = "tbbcc.artifact_comparison.v1"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return data


def _case_map(root: Path, filename: str) -> dict[str, Path]:
    manifest = _load_json(root / "manifest.json")
    cases = manifest.get("cases") or []
    mapping: dict[str, Path] = {}
    for item in cases:
        if not isinstance(item, dict) or not item.get("case_id"):
            continue
        case_id = str(item["case_id"])
        explicit = item.get("reference_json") if filename == "reference.json" else item.get("target_json")
        path = Path(str(explicit)) if explicit else root / "cases" / _case_slug(case_id) / filename
        if not path.is_absolute():
            path = (root / path).resolve()
        mapping[case_id] = path
    return mapping


def _absolutize_artifacts(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "artifact_path" and isinstance(item, str):
                path = Path(item)
                out[key] = str(path if path.is_absolute() else (root / path).resolve())
            else:
                out[key] = _absolutize_artifacts(item, root)
        return out
    if isinstance(value, list):
        return [_absolutize_artifacts(item, root) for item in value]
    return value


def _channel_compare(gpu_case: dict[str, Any], npu_case: dict[str, Any], gpu_root: Path, npu_root: Path, channel: str, atol: float, rtol: float) -> dict[str, Any]:
    gpu_value = (gpu_case.get("channels") or {}).get(channel)
    npu_value = (npu_case.get("channels") or {}).get(channel)
    if gpu_value is None and npu_value is None:
        return {"implemented": False, "passed": None, "metrics": {}}
    if gpu_value is None or npu_value is None:
        return {
            "implemented": True,
            "passed": False,
            "metrics": {
                "reason": "MissingChannel",
                "gpu_present": gpu_value is not None,
                "npu_present": npu_value is not None,
            },
        }
    metrics = tbbcc.compare_values(
        _absolutize_artifacts(gpu_value, gpu_root),
        _absolutize_artifacts(npu_value, npu_root),
        atol,
        rtol,
    )
    return {"implemented": True, "passed": bool(metrics.get("passed")), "metrics": metrics}


def _case_tolerances(suite_path: Path | None) -> dict[str, tuple[float, float]]:
    if suite_path is None:
        return {}
    suite = _load_json(suite_path)
    cases = suite.get("cases") or []
    out: dict[str, tuple[float, float]] = {}
    for item in cases:
        case_path = tbbcc._resolve_path(suite_path.parent, str(item))
        case = tbbcc.load_case(case_path)
        out[case.id] = (float(case.ground_truth.get("atol", 1e-5)), float(case.ground_truth.get("rtol", 1e-5)))
    return out


def compare_artifacts(args: argparse.Namespace) -> int:
    gpu_root = Path(args.gpu_reference).resolve()
    npu_root = Path(args.npu_bridge).resolve()
    out_dir = Path(args.out).resolve()
    suite_path = Path(args.suite).resolve() if args.suite else None
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_cases = _case_map(gpu_root, "reference.json")
    npu_cases = _case_map(npu_root, "target.json")
    tolerances = _case_tolerances(suite_path)
    common_ids = sorted(set(gpu_cases) & set(npu_cases))
    missing_gpu = sorted(set(npu_cases) - set(gpu_cases))
    missing_npu = sorted(set(gpu_cases) - set(npu_cases))

    runs: list[dict[str, Any]] = []
    for case_id in common_ids:
        gpu_case = _load_json(gpu_cases[case_id])
        npu_case = _load_json(npu_cases[case_id])
        atol, rtol = tolerances.get(case_id, (float(args.atol), float(args.rtol)))
        hash_match = gpu_case.get("case_sha256") == npu_case.get("case_sha256")
        channels = {
            name: _channel_compare(gpu_case, npu_case, gpu_root, npu_root, name, atol, rtol)
            for name in ("result", "activations", "gradients", "task_metrics")
        }
        implemented = [item for item in channels.values() if item.get("implemented")]
        passed = bool(hash_match and gpu_case.get("status") == "passed" and npu_case.get("status") == "passed" and implemented and all(item.get("passed") is not False for item in implemented))
        first_fail = None
        for name, item in channels.items():
            if item.get("implemented") and item.get("passed") is False:
                first_fail = name
                break
        runs.append(
            {
                "case_id": case_id,
                "passed": passed,
                "hash_match": hash_match,
                "gpu_status": gpu_case.get("status"),
                "npu_status": npu_case.get("status"),
                "first_fail": first_fail,
                "channels": channels,
                "atol": atol,
                "rtol": rtol,
            }
        )

    passed = sum(1 for item in runs if item["passed"])
    failed = len(runs) - passed
    summary = {
        "schema_version": SCHEMA,
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "gpu_reference": str(gpu_root),
        "npu_bridge": str(npu_root),
        "suite": str(suite_path) if suite_path else None,
        "totals": {
            "gpu_cases": len(gpu_cases),
            "npu_cases": len(npu_cases),
            "overlap": len(common_ids),
            "missing_gpu": len(missing_gpu),
            "missing_npu": len(missing_npu),
            "passed": passed,
            "failed": failed,
            "compatibility_rate": (passed / len(runs)) if runs else None,
        },
        "missing_gpu_case_ids": missing_gpu[:100],
        "missing_npu_case_ids": missing_npu[:100],
        "runs": runs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary_json": str((out_dir / "summary.json").resolve()), "totals": summary["totals"]}, indent=2))
    return 0 if failed == 0 and not missing_gpu and not missing_npu else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare GPU reference artifacts against NPU bridge artifacts")
    parser.add_argument("--gpu-reference", required=True, help="GPU reference artifact root")
    parser.add_argument("--npu-bridge", required=True, help="NPU bridge artifact root")
    parser.add_argument("--out", required=True, help="Comparison output directory")
    parser.add_argument("--suite", default=None, help="Optional canonical suite for per-case tolerances")
    parser.add_argument("--atol", type=float, default=1e-5, help="Default absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Default relative tolerance")
    return parser


def main(argv: list[str] | None = None) -> int:
    return compare_artifacts(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
