#!/usr/bin/env python3
"""Collect canonical bridge target artifacts for TorchBridgeBench cases."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import tbbcc
from tbbcc_gpu_reference import _case_slug, _relativize_artifacts, _resolve_suite_cases


SCHEMA_MANIFEST = "tbbcc.bridge_artifacts.manifest.v1"
SCHEMA_CASE = "tbbcc.bridge_artifacts.case.v1"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _environment(bridge_id: str) -> dict[str, Any]:
    info = tbbcc.inspect_environment(import_versions=False)
    info["comparison_role"] = "npu-bridge"
    info["bridge_id"] = bridge_id
    info["python_executable"] = sys.executable
    info["python"] = platform.python_version()
    return info


def _load_existing(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data["_resumed"] = True
            return data
    except Exception:
        return None
    return None


def _write_case_target(
    *,
    suite_id: str,
    suite_path: Path,
    case_path: Path,
    adapter: tbbcc.AdapterSpec,
    out_dir: Path,
    timeout: int,
    resume: bool,
) -> dict[str, Any]:
    case = tbbcc.load_case(case_path)
    case_dir = out_dir / "cases" / _case_slug(case.id)
    target_path = case_dir / "target.json"
    if resume:
        existing = _load_existing(target_path)
        if existing is not None:
            return existing

    case_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = case_dir / "artifacts"
    started = _dt.datetime.now(_dt.UTC)
    result = tbbcc.execute_python(
        "npu_bridge",
        adapter.preamble,
        case.code,
        adapter.env,
        timeout,
        artifact_dir=artifact_dir,
    )
    finished = _dt.datetime.now(_dt.UTC)
    channels = {
        "result": result.result,
        "activations": result.activations,
        "gradients": result.gradients,
        "task_metrics": result.task_metrics,
    }
    payload = {
        "schema_version": SCHEMA_CASE,
        "created_at": finished.isoformat(),
        "suite_id": suite_id,
        "suite_path": str(suite_path),
        "case_id": case.id,
        "case_path": str(case_path),
        "case_sha256": _sha256_file(case_path),
        "level": case.level,
        "track": case.track,
        "seed": case.seed,
        "expected_ops": case.expected_ops,
        "bridge_id": adapter.bridge_id,
        "adapter_path": adapter.source_path,
        "status": "passed" if result.ok else "failed",
        "duration_seconds": (finished - started).total_seconds(),
        "environment": _environment(adapter.bridge_id),
        "channels": _relativize_artifacts(channels, out_dir),
        "stdout": tbbcc._trim_text(result.stdout),
        "stderr": tbbcc._trim_text(result.stderr),
        "traceback": tbbcc._trim_text(result.traceback or ""),
        "returncode": result.returncode,
        "_resumed": False,
    }
    target_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def collect_bridge_artifacts(args: argparse.Namespace) -> int:
    suite_path = Path(args.suite).resolve()
    adapter_path = Path(args.adapter).resolve()
    out_dir = Path(args.out).resolve()
    suite_id, case_paths = _resolve_suite_cases(suite_path)
    adapter = tbbcc.load_adapter(adapter_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = _dt.datetime.now(_dt.UTC)
    cases: list[dict[str, Any]] = []
    for case_path in case_paths:
        item = _write_case_target(
            suite_id=suite_id,
            suite_path=suite_path,
            case_path=case_path,
            adapter=adapter,
            out_dir=out_dir,
            timeout=int(args.timeout or adapter.timeout_seconds),
            resume=not bool(args.no_resume),
        )
        cases.append(
            {
                "case_id": item.get("case_id"),
                "case_path": item.get("case_path"),
                "case_sha256": item.get("case_sha256"),
                "level": item.get("level"),
                "status": item.get("status"),
                "duration_seconds": item.get("duration_seconds"),
                "target_json": str((out_dir / "cases" / _case_slug(str(item.get("case_id"))) / "target.json").resolve()),
                "resumed_from_cache": bool(item.get("_resumed")),
            }
        )

    passed = sum(1 for item in cases if item.get("status") == "passed")
    failed = sum(1 for item in cases if item.get("status") == "failed")
    resumed = sum(1 for item in cases if item.get("resumed_from_cache"))
    manifest = {
        "schema_version": SCHEMA_MANIFEST,
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "suite_id": suite_id,
        "suite_path": str(suite_path),
        "canonical_case_system": "benchmarks/v1.0.0 case.id",
        "comparison_role": "npu-bridge",
        "bridge_id": adapter.bridge_id,
        "adapter_path": str(adapter_path),
        "environment": _environment(adapter.bridge_id),
        "totals": {
            "total": len(cases),
            "passed": passed,
            "failed": failed,
            "resumed": resumed,
            "executed": len(cases) - resumed,
            "duration_seconds": (_dt.datetime.now(_dt.UTC) - started).total_seconds(),
        },
        "cases": cases,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest_json": str((out_dir / "manifest.json").resolve()), "totals": manifest["totals"]}, indent=2))
    return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect canonical NPU/bridge target artifacts")
    parser.add_argument("--suite", required=True, help="Suite JSON from benchmarks/v1.0.0/suites")
    parser.add_argument("--adapter", required=True, help="AdapterSpec JSON for the bridge target")
    parser.add_argument("--out", required=True, help="Output directory for bridge target artifacts")
    parser.add_argument("--timeout", type=int, default=None, help="Per-case timeout override")
    parser.add_argument("--no-resume", action="store_true", help="Re-run cases even when target.json already exists")
    return parser


def main(argv: list[str] | None = None) -> int:
    return collect_bridge_artifacts(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
