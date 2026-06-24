#!/usr/bin/env python3
"""Collect canonical PyTorch GPU reference artifacts for TorchBridgeBench cases."""

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


SCHEMA_MANIFEST = "tbbcc.gpu_reference.manifest.v1"
SCHEMA_CASE = "tbbcc.gpu_reference.case.v1"


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return data


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _relativize_artifacts(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "artifact_path" and isinstance(item, str):
                try:
                    out[key] = str(Path(item).resolve().relative_to(root.resolve()))
                except ValueError:
                    out[key] = item
            else:
                out[key] = _relativize_artifacts(item, root)
        return out
    if isinstance(value, list):
        return [_relativize_artifacts(item, root) for item in value]
    return value


def _environment(device: str) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device_requested": device,
    }
    try:
        import torch  # type: ignore

        env["torch"] = getattr(torch, "__version__", "present")
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda"] = getattr(torch.version, "cuda", None)
        env["cudnn"] = torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None
        if torch.cuda.is_available():
            env["gpu_count"] = int(torch.cuda.device_count())
            env["gpu"] = torch.cuda.get_device_name(0)
            try:
                env["cuda_device_capability"] = list(torch.cuda.get_device_capability(0))
            except Exception:
                pass
    except Exception as exc:
        env["torch_import_error"] = repr(exc)
    return env


def _resolve_suite_cases(suite_path: Path) -> tuple[str, list[Path]]:
    suite = _json_load(suite_path)
    suite_id = str(suite.get("suite_id") or suite_path.stem)
    cases = suite.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"Suite {suite_path} must contain a non-empty cases array")
    resolved = [tbbcc._resolve_path(suite_path.parent, str(item)) for item in cases]
    return suite_id, resolved


def _case_slug(case_id: str) -> str:
    return tbbcc._slug(case_id)


def _write_case_reference(
    *,
    suite_id: str,
    suite_path: Path,
    case_path: Path,
    out_dir: Path,
    device: str,
    timeout: int,
    env: dict[str, str],
    resume: bool,
) -> dict[str, Any]:
    case = tbbcc.load_case(case_path)
    case_dir = out_dir / "cases" / _case_slug(case.id)
    reference_path = case_dir / "reference.json"
    if resume and reference_path.exists():
        existing = _json_load(reference_path)
        existing["_resumed"] = True
        return existing

    case_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = case_dir / "artifacts"
    started = _dt.datetime.now(_dt.UTC)
    result = tbbcc.execute_python(
        "gpu_reference",
        "",
        case.code,
        env,
        timeout,
        artifact_dir=artifact_dir,
    )
    finished = _dt.datetime.now(_dt.UTC)
    status = "passed" if result.ok else "failed"
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
        "status": status,
        "duration_seconds": (finished - started).total_seconds(),
        "environment": _environment(device),
        "channels": _relativize_artifacts(channels, out_dir),
        "stdout": tbbcc._trim_text(result.stdout),
        "stderr": tbbcc._trim_text(result.stderr),
        "traceback": tbbcc._trim_text(result.traceback or ""),
        "returncode": result.returncode,
        "_resumed": False,
    }
    reference_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def collect_gpu_reference(args: argparse.Namespace) -> int:
    suite_path = Path(args.suite).resolve()
    out_dir = Path(args.out).resolve()
    suite_id, case_paths = _resolve_suite_cases(suite_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = str(args.device)
    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif device == "cuda":
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available() and not args.allow_cpu_fallback:
                raise SystemExit("CUDA is not available. Use --allow-cpu-fallback only for local smoke tests.")
        except ImportError as exc:
            raise SystemExit(f"PyTorch is not importable: {exc}") from exc

    started = _dt.datetime.now(_dt.UTC)
    cases: list[dict[str, Any]] = []
    for case_path in case_paths:
        item = _write_case_reference(
            suite_id=suite_id,
            suite_path=suite_path,
            case_path=case_path,
            out_dir=out_dir,
            device=device,
            timeout=int(args.timeout),
            env=env,
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
                "reference_json": str((out_dir / "cases" / _case_slug(str(item.get("case_id"))) / "reference.json").resolve()),
                "resumed_from_cache": bool(item.get("_resumed")),
            }
        )

    passed = sum(1 for item in cases if item.get("status") == "passed")
    failed = sum(1 for item in cases if item.get("status") == "failed")
    skipped = sum(1 for item in cases if item.get("resumed_from_cache"))
    manifest = {
        "schema_version": SCHEMA_MANIFEST,
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "suite_id": suite_id,
        "suite_path": str(suite_path),
        "canonical_case_system": "benchmarks/v1.0.0 case.id",
        "comparison_role": "gpu-reference",
        "device": device,
        "environment": _environment(device),
        "totals": {
            "total": len(cases),
            "passed": passed,
            "failed": failed,
            "resumed": skipped,
            "executed": len(cases) - skipped,
            "duration_seconds": (_dt.datetime.now(_dt.UTC) - started).total_seconds(),
        },
        "cases": cases,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest_json": str((out_dir / "manifest.json").resolve()), "totals": manifest["totals"]}, indent=2))
    return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect TorchBridgeBench canonical GPU PyTorch reference artifacts")
    parser.add_argument("--suite", required=True, help="Suite JSON from benchmarks/v1.0.0/suites")
    parser.add_argument("--out", required=True, help="Output directory for GPU reference artifacts")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Reference device. Use cpu only for smoke tests.")
    parser.add_argument("--timeout", type=int, default=120, help="Per-case timeout seconds")
    parser.add_argument("--no-resume", action="store_true", help="Re-run cases even when reference.json already exists")
    parser.add_argument("--allow-cpu-fallback", action="store_true", help="Allow --device cuda to proceed without CUDA for local smoke testing")
    return parser


def main(argv: list[str] | None = None) -> int:
    return collect_gpu_reference(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
