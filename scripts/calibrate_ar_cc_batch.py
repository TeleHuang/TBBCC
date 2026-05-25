#!/usr/bin/env python3
"""Batch AR baseline calibration through Claude Code.

This version is optimized for throughput. It sends a batch of benchmark cases
to a single `claude -p` call, requests structured JSON output, and records the
raw batch payload for auditability.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from tbbcc_metrics import EFFORT_FORMULA_VERSION, calc_effort_total, confidence_interval, migrate_at_k, variance_control


SYSTEM_PROMPT = """You are translating PyTorch code to MindSpore code.
Return only the translated code for each case, no markdown, no explanation.
Preserve behavior as closely as possible.
"""

MODEL_NAME = "deepseek-v4-pro"
MODEL_ARG = "deepseek-v4-pro[1m]"
PROVIDER = "claude-code"
AGENT_SYSTEM = "Claude Code"
AGENT_SYSTEM_VERSION = "claude-code-2.1.150-batch-deepseek-v4-pro"
PROTOCOL = "claude-code-cli-print-json-batch"
SCHEMA_VERSION = "tbbcc.ar_baseline.batch.v0.1"


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def calc_effort_total(*, rounds: int, prompt_chars: int, completion_chars: int, edit_units: int) -> float:
    return float(rounds + (prompt_chars + completion_chars + edit_units) / 1000.0)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases(case_root: Path, suite: Path | None) -> list[dict[str, Any]]:
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
        code = str(data["code"])
        cases.append(
            {
                "case_id": str(data["id"]),
                "level": str(data.get("level") or ""),
                "track": str(data.get("track") or ""),
                "difficulty": data.get("difficulty"),
                "expected_ops": list(data.get("expected_ops") or []),
                "source_path": str(path),
                "code": code,
                "code_chars": len(code),
                "code_digest": sha256_text(code),
            }
        )
    return cases


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def read_existing(samples_path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    existing: dict[tuple[int, int], dict[str, Any]] = {}
    if not samples_path.exists():
        return existing
    for line in samples_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "batch_index" not in record:
            continue
        try:
            key = (int(record["reroll_index"]), int(record["batch_index"]))
        except (KeyError, TypeError, ValueError):
            continue
        existing[key] = record
    return existing


def build_user_prompt(*, reroll_index: int, batch_index: int, batch_cases: list[dict[str, Any]]) -> str:
    payload = {
        "reroll_index": reroll_index,
        "batch_index": batch_index,
        "cases": [
            {
                "case_id": case["case_id"],
                "code": case["code"],
                "source_path": case["source_path"],
            }
            for case in batch_cases
        ],
    }
    return (
        "Translate every case independently from PyTorch to MindSpore.\n"
        "Return only structured output matching the JSON schema.\n"
        "Do not omit or reorder cases.\n\n"
        f"INPUT_JSON:\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def claude_json_schema() -> str:
    schema = {
        "type": "object",
        "properties": {
            "cases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "string"},
                        "translation": {"type": "string"},
                    },
                    "required": ["case_id", "translation"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["cases"],
        "additionalProperties": False,
    }
    return json.dumps(schema, ensure_ascii=False)


def run_batch(
    *,
    claude_bin: str,
    cwd: Path,
    batch_cases: list[dict[str, Any]],
    reroll_index: int,
    batch_index: int,
    timeout: int,
    attempts: int,
    bare: bool,
    model: str,
) -> dict[str, Any]:
    user_prompt = build_user_prompt(reroll_index=reroll_index, batch_index=batch_index, batch_cases=batch_cases)
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
        "--model",
        model,
        "--json-schema",
        claude_json_schema(),
    ]
    if bare:
        base_cmd.append("--bare")

    last_error: str | None = None
    last_stdout = ""
    last_stderr = ""
    last_returncode = 0
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
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            returncode = 124
            last_error = f"timeout after {timeout}s"
        else:
            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError as exc:
                payload = None
                last_error = f"invalid Claude JSON: {exc}"
            if payload is not None and returncode == 0 and not payload.get("is_error"):
                structured = payload.get("structured_output") or {}
                translated_cases = structured.get("cases") if isinstance(structured, dict) else None
                if not isinstance(translated_cases, list):
                    last_error = "Claude response missing structured_output.cases"
                else:
                    by_id = {str(item.get("case_id")): str(item.get("translation") or "") for item in translated_cases}
                    missing = [case["case_id"] for case in batch_cases if case["case_id"] not in by_id]
                    if not missing:
                        completion_json = json.dumps(structured, ensure_ascii=False, sort_keys=True)
                        translations = []
                        translation_chars = 0
                        for case in batch_cases:
                            translation = by_id[case["case_id"]]
                            translation_chars += len(translation.strip())
                            translations.append(
                                {
                                    "case_id": case["case_id"],
                                    "source_path": case["source_path"],
                                    "code_chars": case["code_chars"],
                                    "code_digest": case["code_digest"],
                                    "translation_chars": len(translation.strip()),
                                    "translation_digest": sha256_text(translation),
                                    "translation": translation,
                                }
                            )
                        model_usage = payload.get("modelUsage") if isinstance(payload.get("modelUsage"), dict) else {}
                        rounds = len(batch_cases)
                        total_effort = calc_effort_total(
                            rounds=rounds,
                            prompt_chars=prompt_chars,
                            completion_chars=len(completion_json),
                            edit_units=translation_chars,
                        )
                        return {
                            "schema_version": SCHEMA_VERSION,
                            "status": "success",
                            "reroll_index": reroll_index,
                            "batch_index": batch_index,
                            "batch_case_count": len(batch_cases),
                            "batch_case_ids": [case["case_id"] for case in batch_cases],
                            "case_count": len(batch_cases),
                            "started_at": started_at,
                            "completed_at": utc_now(),
                            "rounds": rounds,
                            "prompt_chars": prompt_chars,
                            "completion_chars": len(completion_json),
                            "translation_chars": translation_chars,
                            "completion_digest": sha256_text(completion_json),
                            "structured_output": structured,
                            "translations": translations,
                            "total_effort": total_effort,
                            "claude": {
                                "returncode": returncode,
                                "duration_ms_wall": int((time.monotonic() - attempt_started) * 1000),
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
                                "observed_model": ",".join(sorted(model_usage)) if model_usage else None,
                            },
                        }

        last_stdout = stdout
        last_stderr = stderr
        last_returncode = returncode
        if attempt_index < attempts:
            time.sleep(min(2 * attempt_index, 10))

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "reroll_index": reroll_index,
        "batch_index": batch_index,
        "batch_case_count": len(batch_cases),
        "batch_case_ids": [case["case_id"] for case in batch_cases],
        "case_count": len(batch_cases),
        "started_at": started_at,
        "completed_at": utc_now(),
        "rounds": len(batch_cases),
        "prompt_chars": prompt_chars,
        "completion_chars": 0,
        "translation_chars": 0,
        "completion_digest": None,
        "structured_output": None,
        "translations": [],
        "total_effort": None,
        "error": last_error,
        "claude": {
            "returncode": last_returncode,
            "duration_ms_wall": int((time.monotonic() - attempt_started) * 1000),
            "stdout_tail": last_stdout[-4000:],
            "stderr_tail": last_stderr[-4000:],
        },
    }


def summarize(
    *,
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    rerolls: int,
    batch_size: int,
    suite: Path | None,
    case_root: Path,
    claude_version: str,
    model: str,
    bare: bool,
) -> dict[str, Any]:
    successes = [record for record in records if record.get("status") == "success"]
    failures = [record for record in records if record.get("status") != "success"]
    reroll_summaries: list[dict[str, Any]] = []
    reroll_totals: list[float] = []
    for reroll_index in range(1, rerolls + 1):
        reroll_records = [record for record in successes if int(record["reroll_index"]) == reroll_index]
        total_effort = sum(float(record["total_effort"]) for record in reroll_records)
        reroll_totals.append(total_effort)
        reroll_summaries.append(
            {
                "reroll_index": reroll_index,
                "batch_count": len(reroll_records),
                "case_count": sum(int(record["case_count"]) for record in reroll_records),
                "total_effort": total_effort,
                "mean_batch_effort": total_effort / len(reroll_records) if reroll_records else 0.0,
            }
        )

    mean_effort = statistics.mean(reroll_totals) if reroll_totals else None
    std_effort = statistics.pstdev(reroll_totals) if len(reroll_totals) > 1 else 0.0 if reroll_totals else None
    cv_effort = (std_effort / mean_effort) if mean_effort and std_effort is not None else None
    ci_effort = confidence_interval(reroll_totals)
    variance_state = variance_control(reroll_totals, threshold_cv=0.2, current_n=rerolls, max_n=max(rerolls, 10))
    model_observed = sorted(
        {
            str(record.get("claude", {}).get("observed_model"))
            for record in successes
            if record.get("claude", {}).get("observed_model")
        }
    )
    total_cost_usd = sum(float(record.get("claude", {}).get("total_cost_usd") or 0.0) for record in successes)
    migrate_samples = []
    for record in successes + failures:
        status_ok = record.get("status") == "success"
        for case_id in record.get("batch_case_ids") or [f"batch-{record['batch_index']}"]:
            migrate_samples.append(
                {
                    "case_id": case_id,
                    "task_id": case_id,
                    "reroll_index": record["reroll_index"],
                    "final_state": "ALL_PASS" if status_ok else "MARK_UNFIXABLE",
                    "exec_passed": status_ok,
                    "full_passed": status_ok,
                    "all_tiers_passed": status_ok,
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
        "sample_count": len(successes),
        "expected_sample_count": rerolls,
        "batch_count": len(successes),
        "expected_batch_count": rerolls * ((len(cases) + batch_size - 1) // batch_size),
        "case_sample_count": sum(int(record["case_count"]) for record in successes),
        "expected_case_sample_count": len(cases) * rerolls,
        "case_count": len(cases),
        "error_count": len(failures),
        "reroll_count": rerolls,
        "batch_size": batch_size,
        "source_cases": [case["case_id"] for case in cases],
        "model": MODEL_NAME,
        "model_observed": model_observed,
        "provider": PROVIDER,
        "protocol": PROTOCOL,
        "agent_system": AGENT_SYSTEM,
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
            "batch_size": batch_size,
            "bare": bare,
            "model_arg": model,
        },
        "total_cost_usd": total_cost_usd,
        "rerolls": reroll_summaries,
        "samples": successes,
        "errors": [
            {
                "reroll_index": record["reroll_index"],
                "batch_index": record["batch_index"],
                "batch_case_ids": record["batch_case_ids"],
                "error": record.get("error"),
                "claude": record.get("claude"),
            }
            for record in failures
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
        f"- case_count: `{summary['case_count']}`",
        f"- batch_count: `{summary['batch_count']}`",
        f"- error_count: `{summary['error_count']}`",
        f"- total_cost_usd: `{summary['total_cost_usd']:.6f}`",
        "",
        "## Rerolls",
        "",
        "| reroll | batches | cases | total_effort | mean_batch_effort |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for reroll in summary["rerolls"]:
        lines.append(
            f"| {reroll['reroll_index']} | {reroll['batch_count']} | {reroll['case_count']} | "
            f"{reroll['total_effort']:.6f} | {reroll['mean_batch_effort']:.6f} |"
        )
    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        for error in summary["errors"]:
            lines.append(
                f"- reroll `{error['reroll_index']}` batch `{error['batch_index']}` "
                f"({', '.join(error['batch_case_ids'])}): {error['error']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate AR baseline effort through Claude Code in batches.")
    parser.add_argument("--case-root", type=Path, default=Path("benchmarks/v1.0.0/cases"))
    parser.add_argument("--suite", type=Path, default=Path("benchmarks/v1.0.0/suites/all_noop.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/ar_baseline/deepseek-v4-pro-cc-batch-v1"))
    parser.add_argument("--rerolls", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--model", default=MODEL_ARG)
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
    if args.rerolls <= 0 or args.batch_size <= 0 or args.workers <= 0:
        raise SystemExit("rerolls, batch-size, and workers must all be positive")

    try:
        claude_version = subprocess.check_output([args.claude_bin, "--version"], text=True).strip()
    except Exception as exc:
        raise SystemExit(f"Unable to run {args.claude_bin} --version: {exc}") from exc

    batches = chunked(cases, args.batch_size)
    existing = {} if args.no_resume else read_existing(samples_path)
    todo: list[tuple[int, int, list[dict[str, Any]]]] = []
    for reroll_index in range(1, args.rerolls + 1):
        for batch_index, batch_cases in enumerate(batches, start=1):
            record = existing.get((reroll_index, batch_index))
            if record is None or record.get("status") != "success":
                todo.append((reroll_index, batch_index, batch_cases))

    print(
        f"AR calibration via Claude Code: cases={len(cases)} batches={len(batches)} rerolls={args.rerolls} "
        f"todo={len(todo)} workers={args.workers} batch_size={args.batch_size} out={out_dir}",
        flush=True,
    )
    print(f"Claude Code: {claude_version}", flush=True)

    with samples_path.open("a", encoding="utf-8") as samples_file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_batch,
                    claude_bin=args.claude_bin,
                    cwd=repo_root,
                    batch_cases=batch_cases,
                    reroll_index=reroll_index,
                    batch_index=batch_index,
                    timeout=args.timeout,
                    attempts=args.attempts,
                    bare=not args.no_bare,
                    model=args.model,
                ): (reroll_index, batch_index)
                for reroll_index, batch_index, batch_cases in todo
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                samples_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                samples_file.flush()
                completed += 1
                print(
                    f"[{completed}/{len(todo)}] {record['status']} reroll={record['reroll_index']} "
                    f"batch={record['batch_index']} cases={record['batch_case_count']} effort={record.get('total_effort')}",
                    flush=True,
                )

    # Retried batches append a new record; only the most recent record for each
    # reroll/batch key participates in the final summary.
    records = list(read_existing(samples_path).values())
    summary = summarize(
        records=records,
        cases=cases,
        rerolls=args.rerolls,
        batch_size=args.batch_size,
        suite=suite,
        case_root=case_root,
        claude_version=claude_version,
        model=args.model,
        bare=not args.no_bare,
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, markdown_path)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {markdown_path}", flush=True)
    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
