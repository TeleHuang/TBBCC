#!/usr/bin/env python3
"""Calibrate AR baseline effort through Claude Code.

The calibration target is the pure manual PyTorch-to-MindSpore migration
baseline. This script intentionally drives `claude -p` instead of calling an LLM
provider SDK directly, so the result is tied to Claude Code behavior and
metadata.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tbbcc_metrics import EFFORT_FORMULA_VERSION, calc_effort_total, confidence_interval, migrate_at_k, variance_control


SYSTEM_PROMPT = """You are translating PyTorch code to MindSpore code.
Return only the translated code, no markdown, no explanation.
Preserve behavior as closely as possible.
"""

AGENT_SYSTEM_VERSION = "claude-code-2.1.150-bare-print-json-deepseek-v4-pro"
SCHEMA_VERSION = "tbbcc.ar_baseline.v0.1"


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases(case_root: Path, suite: Path | None) -> list[dict[str, Any]]:
    paths: list[Path]
    if suite is not None:
        suite_data = load_json(suite)
        suite_cases = suite_data.get("cases")
        if not isinstance(suite_cases, list):
            raise SystemExit(f"Suite {suite} must contain a cases list")
        paths = [(suite.parent / str(item)).resolve() for item in suite_cases]
    else:
        paths = sorted(case_root.rglob("*.json"))

    cases: list[dict[str, Any]] = []
    for path in paths:
        data = load_json(path)
        if not isinstance(data, dict) or "id" not in data or "code" not in data:
            continue
        cases.append(
            {
                "case_id": str(data["id"]),
                "level": str(data.get("level") or ""),
                "track": str(data.get("track") or ""),
                "difficulty": data.get("difficulty"),
                "expected_ops": list(data.get("expected_ops") or []),
                "source_path": str(path),
                "code": str(data["code"]),
                "code_chars": len(str(data["code"])),
                "code_digest": sha256_text(str(data["code"])),
            }
        )
    return cases


def read_existing_samples(samples_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    existing: dict[tuple[str, int], dict[str, Any]] = {}
    if not samples_path.exists():
        return existing
    for line in samples_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        existing[(str(sample["case_id"]), int(sample["reroll_index"]))] = sample
    return existing


def observed_model_name(model_usage: dict[str, Any]) -> str | None:
    if not model_usage:
        return None
    names = sorted(model_usage)
    if len(names) == 1:
        return names[0]
    return ",".join(names)


def run_claude(
    *,
    claude_bin: str,
    cwd: Path,
    case: dict[str, Any],
    reroll_index: int,
    timeout: int,
    attempts: int,
    bare: bool,
    settings: str | None,
    model: str | None,
) -> dict[str, Any]:
    user_prompt = str(case["code"])
    prompt_chars = len(SYSTEM_PROMPT) + len(user_prompt)
    started_at = utc_now()
    base_cmd = [
        claude_bin,
        "-p",
        user_prompt,
        "--output-format",
        "json",
        "--no-session-persistence",
        "--system-prompt",
        SYSTEM_PROMPT,
        "--tools",
        "",
    ]
    if bare:
        base_cmd.append("--bare")
    if settings:
        base_cmd.extend(["--settings", settings])
    if model:
        base_cmd.extend(["--model", model])

    last_error: str | None = None
    for attempt_index in range(1, attempts + 1):
        attempt_started = time.monotonic()
        try:
            proc = subprocess.run(
                base_cmd,
                cwd=str(cwd),
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = f"timeout after {timeout}s"
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            duration_ms = int((time.monotonic() - attempt_started) * 1000)
            payload: dict[str, Any] | None = None
            returncode = 124
        else:
            stdout = proc.stdout
            stderr = proc.stderr
            duration_ms = int((time.monotonic() - attempt_started) * 1000)
            returncode = proc.returncode
            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError as exc:
                payload = None
                last_error = f"invalid Claude JSON: {exc}"

        if payload is not None and returncode == 0 and not payload.get("is_error"):
            translation = str(payload.get("result") or "")
            completion_chars = len(translation)
            translation_chars = len(translation.strip())
            total_effort = calc_effort_total(
                rounds=1,
                prompt_chars=prompt_chars,
                completion_chars=completion_chars,
                edit_units=translation_chars,
            )
            model_usage = payload.get("modelUsage") if isinstance(payload.get("modelUsage"), dict) else {}
            return {
                "schema_version": SCHEMA_VERSION,
                "status": "success",
                "case_id": case["case_id"],
                "level": case["level"],
                "track": case["track"],
                "difficulty": case["difficulty"],
                "expected_ops": case["expected_ops"],
                "source_path": case["source_path"],
                "code_chars": case["code_chars"],
                "code_digest": case["code_digest"],
                "reroll_index": reroll_index,
                "attempt_index": attempt_index,
                "started_at": started_at,
                "completed_at": utc_now(),
                "rounds": 1,
                "prompt_chars": prompt_chars,
                "completion_chars": completion_chars,
                "translation_chars": translation_chars,
                "translation_digest": sha256_text(translation),
                "translation_text": translation,
                "total_effort": total_effort,
                "claude": {
                    "returncode": returncode,
                    "duration_ms_wall": duration_ms,
                    "type": payload.get("type"),
                    "subtype": payload.get("subtype"),
                    "is_error": payload.get("is_error"),
                    "stop_reason": payload.get("stop_reason"),
                    "terminal_reason": payload.get("terminal_reason"),
                    "session_id": payload.get("session_id"),
                    "uuid": payload.get("uuid"),
                    "total_cost_usd": payload.get("total_cost_usd"),
                    "usage": payload.get("usage"),
                    "modelUsage": model_usage,
                    "observed_model": observed_model_name(model_usage),
                    "permission_denials": payload.get("permission_denials"),
                },
            }

        if payload is not None:
            last_error = json.dumps(
                {
                    "returncode": returncode,
                    "is_error": payload.get("is_error"),
                    "subtype": payload.get("subtype"),
                    "errors": payload.get("errors"),
                    "result": payload.get("result"),
                },
                ensure_ascii=False,
            )
        if attempt_index < attempts:
            time.sleep(min(2 * attempt_index, 10))

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "case_id": case["case_id"],
        "level": case["level"],
        "track": case["track"],
        "difficulty": case["difficulty"],
        "expected_ops": case["expected_ops"],
        "source_path": case["source_path"],
        "code_chars": case["code_chars"],
        "code_digest": case["code_digest"],
        "reroll_index": reroll_index,
        "attempt_index": attempts,
        "started_at": started_at,
        "completed_at": utc_now(),
        "rounds": 1,
        "prompt_chars": prompt_chars,
        "completion_chars": 0,
        "translation_chars": 0,
        "translation_digest": None,
        "translation_text": "",
        "total_effort": None,
        "error": last_error,
        "claude": {
            "returncode": returncode,
            "duration_ms_wall": duration_ms,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        },
    }


def summarize(
    *,
    samples: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    rerolls: int,
    cv_threshold: float,
    suite: Path | None,
    case_root: Path,
    claude_version: str,
    bare: bool,
    settings: str | None,
    model: str | None,
) -> dict[str, Any]:
    ok_samples = [s for s in samples if s.get("status") == "success"]
    error_samples = [s for s in samples if s.get("status") != "success"]
    reroll_summaries: list[dict[str, Any]] = []
    reroll_totals: list[float] = []
    for reroll_index in range(1, rerolls + 1):
        reroll_samples = [s for s in ok_samples if int(s["reroll_index"]) == reroll_index]
        total_effort = sum(float(s["total_effort"]) for s in reroll_samples)
        if len(reroll_samples) == len(cases):
            reroll_totals.append(total_effort)
        reroll_summaries.append(
            {
                "reroll_index": reroll_index,
                "case_count": len(reroll_samples),
                "expected_case_count": len(cases),
                "total_effort": total_effort,
                "mean_case_effort": total_effort / len(reroll_samples) if reroll_samples else 0.0,
            }
        )

    mean_effort = statistics.mean(reroll_totals) if reroll_totals else None
    std_effort = statistics.pstdev(reroll_totals) if len(reroll_totals) > 1 else 0.0 if reroll_totals else None
    cv_effort = (std_effort / mean_effort) if mean_effort and std_effort is not None else None
    ci_effort = confidence_interval(reroll_totals)
    variance_state = variance_control(reroll_totals, threshold_cv=cv_threshold, current_n=rerolls, max_n=max(rerolls, 10))
    observed_models = sorted(
        {
            str(s.get("claude", {}).get("observed_model"))
            for s in ok_samples
            if s.get("claude", {}).get("observed_model")
        }
    )
    total_cost_usd = sum(
        float(s.get("claude", {}).get("total_cost_usd") or 0.0)
        for s in ok_samples
    )
    migrate_samples = [
        {
            "case_id": sample["case_id"],
            "task_id": sample["case_id"],
            "reroll_index": sample["reroll_index"],
            "final_state": "ALL_PASS" if sample.get("status") == "success" else "MARK_UNFIXABLE",
            "exec_passed": sample.get("status") == "success",
            "full_passed": sample.get("status") == "success",
            "all_tiers_passed": sample.get("status") == "success",
        }
        for sample in samples
    ]
    compact_samples = []
    for sample in sorted(ok_samples, key=lambda item: (int(item["reroll_index"]), str(item["case_id"]))):
        compact_samples.append(
            {
                "case_id": sample["case_id"],
                "level": sample["level"],
                "reroll_index": sample["reroll_index"],
                "prompt_chars": sample["prompt_chars"],
                "completion_chars": sample["completion_chars"],
                "translation_chars": sample["translation_chars"],
                "rounds": sample["rounds"],
                "total_effort": sample["total_effort"],
                "translation_digest": sample["translation_digest"],
                "session_id": sample.get("claude", {}).get("session_id"),
                "uuid": sample.get("claude", {}).get("uuid"),
                "total_cost_usd": sample.get("claude", {}).get("total_cost_usd"),
                "observed_model": sample.get("claude", {}).get("observed_model"),
            }
        )

    task_digest_payload = [
        {
            "case_id": case["case_id"],
            "code_digest": case["code_digest"],
            "level": case["level"],
            "track": case["track"],
            "source_path": case["source_path"],
        }
        for case in cases
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "baseline_effort": mean_effort,
        "mean_effort": mean_effort,
        "std_effort": std_effort,
        "cv_effort": cv_effort,
        "confidence_interval": ci_effort,
        "variance_control": variance_state,
        "sample_count": len(ok_samples),
        "expected_sample_count": len(cases) * rerolls,
        "error_count": len(error_samples),
        "reroll_count": rerolls,
        "auto_expand": bool(cv_effort is not None and cv_effort > cv_threshold),
        "auto_expand_threshold": cv_threshold,
        "source_cases": [case["case_id"] for case in cases],
        "model": "deepseek-v4-pro",
        "model_observed": observed_models,
        "provider": "claude-code",
        "protocol": "claude-code-cli-print-json-bare" if bare else "claude-code-cli-print-json",
        "agent_system": "Claude Code",
        "agent_system_version": AGENT_SYSTEM_VERSION,
        "claude_version": claude_version,
        "effort_formula_version": EFFORT_FORMULA_VERSION,
        "effort_formula": "rounds + (prompt_chars + completion_chars + edit_units) / 1000.0",
        "migrate_at_k": migrate_at_k(migrate_samples),
        "task_set_digest": sha256_json(task_digest_payload),
        "calibration_prompt_digest": sha256_text(SYSTEM_PROMPT),
        "calibration_prompt": SYSTEM_PROMPT,
        "run_config": {
            "case_root": str(case_root),
            "suite": str(suite) if suite else None,
            "rerolls": rerolls,
            "bare": bare,
            "settings": settings,
            "model_arg": model,
            "tools": "",
            "temperature": None,
        },
        "total_cost_usd": total_cost_usd,
        "rerolls": reroll_summaries,
        "samples": compact_samples,
        "errors": [
            {
                "case_id": sample["case_id"],
                "reroll_index": sample["reroll_index"],
                "error": sample.get("error"),
                "claude": sample.get("claude"),
            }
            for sample in error_samples
        ],
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# AR Baseline Calibration",
        "",
        f"- model: `{summary['model']}`",
        f"- observed model: `{', '.join(summary['model_observed'])}`",
        f"- agent system: `{summary['agent_system']}`",
        f"- Claude Code version: `{summary['claude_version']}`",
        f"- protocol: `{summary['protocol']}`",
        f"- baseline_effort: `{summary['baseline_effort']}`",
        f"- sample_count: `{summary['sample_count']}/{summary['expected_sample_count']}`",
        f"- error_count: `{summary['error_count']}`",
        f"- total_cost_usd: `{summary['total_cost_usd']:.6f}`",
        "",
        "## Rerolls",
        "",
        "| reroll | cases | total_effort | mean_case_effort |",
        "| --- | ---: | ---: | ---: |",
    ]
    for reroll in summary["rerolls"]:
        lines.append(
            f"| {reroll['reroll_index']} | {reroll['case_count']}/{reroll['expected_case_count']} | "
            f"{reroll['total_effort']:.6f} | {reroll['mean_case_effort']:.6f} |"
        )
    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        for error in summary["errors"]:
            lines.append(f"- `{error['case_id']}` reroll `{error['reroll_index']}`: {error['error']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate AR baseline effort through Claude Code.")
    parser.add_argument("--case-root", type=Path, default=Path("benchmarks/v1.0.0/cases"))
    parser.add_argument("--suite", type=Path, default=Path("benchmarks/v1.0.0/suites/all_noop.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/ar_baseline/deepseek-v4-pro-cc-v1"))
    parser.add_argument("--rerolls", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--auto-expand", action="store_true", help="Continue adding rerolls while CV exceeds threshold.")
    parser.add_argument("--max-rerolls", type=int, default=10, help="Upper bound for auto expansion.")
    parser.add_argument("--cv-threshold", type=float, default=0.2, help="Coefficient of variation threshold for auto expansion.")
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--settings", default=os.environ.get("TBBCC_CLAUDE_SETTINGS"))
    parser.add_argument("--model", default="deepseek-v4-pro[1m]")
    parser.add_argument("--no-bare", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    case_root = args.case_root if args.case_root.is_absolute() else repo_root / args.case_root
    suite = args.suite if args.suite is None or args.suite.is_absolute() else repo_root / args.suite
    out_dir = args.out if args.out.is_absolute() else repo_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    summary_path = out_dir / "baseline.json"
    markdown_path = out_dir / "baseline.md"

    cases = load_cases(case_root, suite)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No calibration cases found")
    if args.rerolls <= 0:
        raise SystemExit("--rerolls must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_rerolls < args.rerolls:
        raise SystemExit("--max-rerolls must be >= --rerolls")

    try:
        claude_version = subprocess.check_output([args.claude_bin, "--version"], text=True).strip()
    except Exception as exc:
        raise SystemExit(f"Unable to run {args.claude_bin} --version: {exc}") from exc

    existing = {} if args.no_resume else read_existing_samples(samples_path)
    planned_rerolls = args.rerolls
    while True:
        existing = {} if args.no_resume else read_existing_samples(samples_path)
        todo = [
            (case, reroll_index)
            for reroll_index in range(1, planned_rerolls + 1)
            for case in cases
            if (case["case_id"], reroll_index) not in existing
        ]

        print(
            f"AR calibration via Claude Code: cases={len(cases)} rerolls={planned_rerolls} "
            f"todo={len(todo)} workers={args.workers} out={out_dir}",
            flush=True,
        )
        print(f"Claude Code: {claude_version}", flush=True)

        completed = 0
        if todo:
            with samples_path.open("a", encoding="utf-8") as samples_file:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = {
                        executor.submit(
                            run_claude,
                            claude_bin=args.claude_bin,
                            cwd=repo_root,
                            case=case,
                            reroll_index=reroll_index,
                            timeout=args.timeout,
                            attempts=args.attempts,
                            bare=not args.no_bare,
                            settings=args.settings,
                            model=args.model,
                        ): (case, reroll_index)
                        for case, reroll_index in todo
                    }
                    for future in concurrent.futures.as_completed(futures):
                        sample = future.result()
                        samples_file.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
                        samples_file.flush()
                        completed += 1
                        status = sample.get("status")
                        effort = sample.get("total_effort")
                        print(
                            f"[{completed}/{len(todo)}] {status} reroll={sample['reroll_index']} "
                            f"{sample['case_id']} effort={effort}",
                            flush=True,
                        )

        all_samples = list(read_existing_samples(samples_path).values())
        summary = summarize(
            samples=all_samples,
            cases=cases,
            rerolls=planned_rerolls,
            cv_threshold=args.cv_threshold,
            suite=suite,
            case_root=case_root,
            claude_version=claude_version,
            bare=not args.no_bare,
            settings=args.settings,
            model=args.model,
        )
        if args.auto_expand and summary["variance_control"].get("needs_more_samples"):
            next_rerolls = min(args.max_rerolls, planned_rerolls + 1)
            if next_rerolls > planned_rerolls:
                planned_rerolls = next_rerolls
                print(f"Auto-expanding rerolls to {planned_rerolls}", flush=True)
                continue
        break
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, markdown_path)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {markdown_path}", flush=True)
    if summary["error_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
