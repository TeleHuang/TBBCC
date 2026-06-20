"""Built-in compatibility-analysis visualizations for evaluation summaries."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#2E9E44",
    "green_soft": "#AADCA9",
    "red": "#B64342",
    "red_soft": "#E9A6A1",
    "orange": "#E28E2C",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#8F8F8F",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
}

FAILURE_PALETTE = {
    "DependencyMissing": "#8c8c8c",
    "EnvironmentFailure": "#bdbdbd",
    "ImportOrderError": "#969696",
    "OperatorNotFound": "#d73027",
    "TypeMismatch": "#fc8d59",
    "DeviceMismatch": "#fee08b",
    "ShapeMismatch": "#4575b4",
    "RuntimeCrash": "#313695",
    "NumericMismatch": "#f46d43",
    "AutogradFailure": "#7b3294",
    "TrainingDivergence": "#1a9850",
    "TranslationError": "#66c2a5",
    "Unknown": "#636363",
}

DEFAULT_KINDS = [
    "compatibility-overview",
    "failure-taxonomy",
    "tolerance-sweep",
    "model-method-heatmap",
    "design-space",
    "metric-scorecard",
]

TOLERANCE_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def _slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return clean or "report"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _label_from_path(path: Path, data: dict[str, Any]) -> str:
    suite_id = data.get("suite_id")
    if suite_id:
        return str(suite_id)
    adapter = data.get("adapter") or {}
    if isinstance(adapter, dict) and adapter.get("library_name"):
        return str(adapter["library_name"])
    parent = path.parent.name
    if parent in {"eval", "reports"} and path.parent.parent.name:
        return path.parent.parent.name
    return parent or path.stem


def _bridge_from_payload(path: Path, data: dict[str, Any]) -> str:
    adapter = data.get("adapter") or {}
    if isinstance(adapter, dict) and adapter.get("library_name"):
        return str(adapter["library_name"])
    label = _label_from_path(path, data)
    for prefix in ("reports_", "report_", "eval_"):
        if label.startswith(prefix):
            return label[len(prefix):]
    return label


def _infer_failure_class(error: str | None) -> str:
    text = (error or "").lower()
    if not text:
        return "Unknown"
    if "no module named" in text or "modulenotfounderror" in text or "importerror" in text:
        return "DependencyMissing"
    if "shape" in text or "size mismatch" in text or "mat1 and mat2" in text:
        return "ShapeMismatch"
    if "device" in text or "cuda" in text or "npu" in text or "ascend" in text:
        return "DeviceMismatch"
    if "dtype" in text or "type" in text:
        return "TypeMismatch"
    if "grad" in text or "backward" in text or "autograd" in text:
        return "AutogradFailure"
    if "not implemented" in text or "unsupported" in text:
        return "OperatorNotFound"
    return "RuntimeCrash"


def _failure_classes_from_demo_payload(data: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    suites = data.get("suites") or {}
    if not isinstance(suites, dict):
        return counts
    diagnostics: dict[str, str] = {}
    agent = data.get("agent_report") or {}
    for item in agent.get("diagnostics") or []:
        if isinstance(item, dict) and item.get("case_id") and item.get("failure_class"):
            diagnostics[str(item["case_id"])] = str(item["failure_class"])
    for suite in suites.values():
        if not isinstance(suite, dict):
            continue
        for case in suite.get("cases") or []:
            if not isinstance(case, dict):
                continue
            if case.get("ok") or case.get("skipped"):
                continue
            cls = diagnostics.get(str(case.get("case_id"))) or _infer_failure_class(case.get("error"))
            counts[cls] = counts.get(cls, 0) + 1
    return counts


def _iter_demo_cases(data: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for suite_name, suite in (data.get("suites") or {}).items():
        if not isinstance(suite, dict) or suite.get("skipped"):
            continue
        for case in suite.get("cases") or []:
            if not isinstance(case, dict):
                continue
            item = dict(case)
            item.setdefault("suite", suite_name)
            cases.append(item)
    return cases


def _iter_plugin_runs(data: dict[str, Any]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for run in data.get("runs") or []:
        if not isinstance(run, dict):
            continue
        item = dict(run)
        item.setdefault("case_id", run.get("case_id"))
        item.setdefault("ok", run.get("final_state") == "ALL_PASS")
        item.setdefault("category", run.get("level") or run.get("tier") or "benchmark")
        item.setdefault("suite", data.get("suite_id") or "suite")
        item.setdefault("details", run)
        item.setdefault("error", run.get("error") or run.get("failure_class"))
        item.setdefault("skipped", False)
        item.setdefault("counts_toward_adaptation", not bool(run.get("is_environment_or_config")))
        runs.append(item)
    return runs


def _all_cases(data: dict[str, Any]) -> list[dict[str, Any]]:
    cases = _iter_demo_cases(data)
    if cases:
        return cases
    return _iter_plugin_runs(data)


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    return None


def _nested_get(data: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _case_numeric_error(case: dict[str, Any]) -> float | None:
    details = case.get("details") if isinstance(case.get("details"), dict) else {}
    candidates = [
        _nested_get(details, ["t1_metrics", "max_error"]),
        _nested_get(details, ["t1_metrics", "max_abs_err"]),
        details.get("max_error"),
        details.get("max_abs_err"),
        details.get("max_abs_err_y"),
        details.get("max_abs_err_g"),
        _nested_get(details, ["numeric_fidelity", "max_abs_err_y"]),
        _nested_get(details, ["numeric_fidelity", "max_abs_err_g"]),
    ]
    values = [_safe_float(v) for v in candidates]
    values = [v for v in values if v is not None]
    if values:
        return max(values)
    error = str(case.get("error") or "")
    match = re.search(r"(?:max(?:imum)?(?: absolute)? error|mae|max_abs_err|max_error)[=: ]+([0-9.eE+-]+)", error)
    if match:
        return _safe_float(float(match.group(1)))
    return None


def _rate_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.0f}%"


def _first_pass_rate_from_cases(cases: list[dict[str, Any]]) -> float | None:
    counted = [c for c in cases if not c.get("skipped") and c.get("counts_toward_adaptation", True)]
    if not counted:
        return None
    passed = 0
    for case in counted:
        details = case.get("details") if isinstance(case.get("details"), dict) else {}
        if case.get("ok") and not details.get("agent_successful_repair"):
            passed += 1
    return passed / len(counted)


def _read_summary(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    failures = data.get("failure_classes") if isinstance(data.get("failure_classes"), dict) else None
    if failures is None:
        failures = _failure_classes_from_demo_payload(data)

    totals = data.get("totals") if isinstance(data.get("totals"), dict) else {}
    eval_summary = data.get("evaluation_summary") if isinstance(data.get("evaluation_summary"), dict) else {}
    compatibility_rate = totals.get("compatibility_rate", eval_summary.get("adaptation_rate_overall"))
    raw_pass_rate = totals.get("raw_pass_rate")
    if raw_pass_rate is None and isinstance(eval_summary.get("counts"), dict):
        counts = eval_summary["counts"]
        total = counts.get("total") or 0
        raw_pass_rate = (counts.get("passed") or 0) / total if total else None
    total_runs = totals.get("total")
    if total_runs is None and isinstance(eval_summary.get("counts"), dict):
        total_runs = eval_summary["counts"].get("total")

    quality = data.get("quality") if isinstance(data.get("quality"), dict) else {}
    numeric = quality.get("numeric") if isinstance(quality.get("numeric"), dict) else {}
    numeric_fidelity = data.get("numeric_fidelity") if isinstance(data.get("numeric_fidelity"), dict) else {}
    benchmark = data.get("benchmark_summary") if isinstance(data.get("benchmark_summary"), dict) else {}
    agent = data.get("agent_report") if isinstance(data.get("agent_report"), dict) else {}
    effort = data.get("effort") if isinstance(data.get("effort"), dict) else {}
    repair_metrics = agent.get("repair_metrics") if isinstance(agent.get("repair_metrics"), dict) else {}
    effort_breakdown = agent.get("effort_breakdown") if isinstance(agent.get("effort_breakdown"), dict) else {}

    cases = _all_cases(data)
    passed_cases = [c for c in cases if c.get("ok") and not c.get("skipped")]
    failed_cases = [c for c in cases if not c.get("ok") and not c.get("skipped")]

    first_pass = _safe_float(quality.get("first_pass_rate"))
    if first_pass is None:
        first_pass = _first_pass_rate_from_cases(cases)

    return {
        "path": str(path),
        "label": _label_from_path(path, data),
        "bridge": _bridge_from_payload(path, data),
        "failure_classes": {str(k): int(v) for k, v in (failures or {}).items()},
        "compatibility_rate": _safe_float(compatibility_rate),
        "raw_pass_rate": _safe_float(raw_pass_rate),
        "total": int(total_runs or len(cases) or 0),
        "cases": cases,
        "passed_cases": len(passed_cases),
        "failed_cases": len(failed_cases),
        "by_suite": eval_summary.get("by_suite") or {},
        "by_category": eval_summary.get("by_category") or {},
        "numeric_fidelity": numeric_fidelity,
        "quality_numeric": numeric,
        "benchmark_summary": benchmark,
        "agent_report": agent,
        "effort": effort,
        "metrics": {
            "compatibility": _safe_float(compatibility_rate),
            "raw_pass": _safe_float(raw_pass_rate),
            "first_pass": first_pass,
            "numeric": _safe_float(numeric.get("min_cosine")),
            "numeric_within_tolerance": 1.0 if numeric_fidelity.get("within_tolerance_all") is True else (0.0 if numeric_fidelity.get("within_tolerance_all") is False else None),
            "performance": _safe_float(_nested_get(benchmark, ["aggregate", "throughput_iter_s"])),
            "ME": _safe_float(effort.get("me", effort_breakdown.get("ME", repair_metrics.get("effort_total")))),
            "AR": _safe_float(effort.get("ar", repair_metrics.get("AR"))),
        },
        "source_type": "summary_json",
    }


def load_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    summaries = [_read_summary(path) for path in paths]
    seen: dict[str, int] = {}
    for item in summaries:
        base = item["label"]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            item["label"] = f"{base}_{seen[base]}"
    return summaries


def _require_matplotlib():
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional package
        raise SystemExit(
            "Matplotlib is required for report plotting. Install matplotlib or run evaluation without plot-reports."
        ) from exc
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    return plt


def _save_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _save_figure(fig: Any, out_base: Path) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    svg = out_base.with_suffix(".svg")
    pdf = out_base.with_suffix(".pdf")
    png = out_base.with_suffix(".png")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return [svg, pdf, png]


def _add_panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_compatibility_overview(summaries: list[dict[str, Any]], out_base: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    plt = _require_matplotlib()
    rows = [
        {
            "label": item["label"],
            "bridge": item["bridge"],
            "compatibility_rate": item.get("compatibility_rate"),
            "raw_pass_rate": item.get("raw_pass_rate"),
            "total": item.get("total"),
        }
        for item in summaries
    ]
    data_path = _save_rows(out_base.parent / "source_data" / f"{out_base.name}.csv", rows)
    labels = [item["label"] for item in summaries]
    compatibility = [item.get("compatibility_rate") for item in summaries]
    raw = [item.get("raw_pass_rate") for item in summaries]
    x = list(range(len(labels)))
    width = max(4.8, min(10.5, 0.62 * len(labels) + 2.9))
    fig, ax = plt.subplots(figsize=(width, 2.8))
    bar_width = 0.36
    comp_vals = [float(v) if v is not None else math.nan for v in compatibility]
    raw_vals = [float(v) if v is not None else math.nan for v in raw]
    ax.bar([i - bar_width / 2 for i in x], [0 if math.isnan(v) else v for v in comp_vals], width=bar_width, label="Compatibility", color=PALETTE["blue_main"])
    ax.bar([i + bar_width / 2 for i in x], [0 if math.isnan(v) else v for v in raw_vals], width=bar_width, label="Raw pass", color=PALETTE["blue_secondary"], alpha=0.45)
    for i, value in enumerate(compatibility):
        ax.text(i - bar_width / 2, 0.03 if value is None else min(float(value) + 0.035, 1.01), _rate_text(value), ha="center", va="bottom", fontsize=6, rotation=90 if value is None else 0)
    for i, value in enumerate(raw):
        ax.text(i + bar_width / 2, 0.03 if value is None else min(float(value) + 0.035, 1.01), _rate_text(value), ha="center", va="bottom", fontsize=6, rotation=90 if value is None else 0, color=PALETTE["neutral_dark"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate")
    ax.set_title("Bridge-relevant compatibility vs raw pass rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _add_panel_label(ax, "a")
    fig.tight_layout()
    figures = _save_figure(fig, out_base)
    return figures, [data_path], {"provenance": "real report summaries", "note": "Compatibility excludes environment/configuration-only failures when reports expose that denominator."}


def plot_failure_taxonomy(summaries: list[dict[str, Any]], out_base: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    plt = _require_matplotlib()
    classes = sorted({name for item in summaries for name in item["failure_classes"]})
    labels = [item["label"] for item in summaries]
    rows = []
    for item in summaries:
        for cls in classes:
            rows.append({"label": item["label"], "bridge": item["bridge"], "failure_class": cls, "count": item["failure_classes"].get(cls, 0)})
    data_path = _save_rows(out_base.parent / "source_data" / f"{out_base.name}.csv", rows)
    width = max(4.8, min(10.5, 0.62 * len(labels) + 3.2))
    fig, ax = plt.subplots(figsize=(width, 2.9))
    bottoms = [0] * len(labels)
    if not classes:
        classes = ["NoFailure"]
    for cls in classes:
        values = [item["failure_classes"].get(cls, 0) for item in summaries]
        if not any(values):
            continue
        ax.bar(labels, values, bottom=bottoms, label=cls, color=FAILURE_PALETTE.get(cls, "#9e9e9e"), linewidth=0)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    if not any(bottoms):
        ax.text(0.5, 0.5, "No failed compatibility-counted cases", ha="center", va="center", transform=ax.transAxes, color=PALETTE["neutral_dark"])
    ax.set_ylabel("Failed runs")
    ax.set_title("Failure taxonomy across evaluation reports")
    ax.tick_params(axis="x", rotation=32)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_ylim(0, max(1, math.ceil((max(bottoms) if bottoms else 0) * 1.18)))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _add_panel_label(ax, "b")
    fig.tight_layout()
    figures = _save_figure(fig, out_base)
    return figures, [data_path], {"provenance": "real report summaries", "note": "Failure class is taken from report diagnostics or inferred from captured error text."}


def plot_tolerance_sweep(summaries: list[dict[str, Any]], out_base: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    plt = _require_matplotlib()
    rows: list[dict[str, Any]] = []
    for item in summaries:
        cases = [c for c in item["cases"] if not c.get("skipped") and c.get("counts_toward_adaptation", True)]
        numeric_cases = []
        for case in cases:
            err = _case_numeric_error(case)
            if err is not None or str(case.get("error") or "").lower().find("numeric") >= 0:
                numeric_cases.append((case, err))
        if not numeric_cases and item["numeric_fidelity"]:
            nf = item["numeric_fidelity"]
            err_vals = [_safe_float(nf.get("max_abs_err_y")), _safe_float(nf.get("max_abs_err_g"))]
            err_vals = [v for v in err_vals if v is not None]
            if err_vals:
                numeric_cases.append(({"case_id": "numeric_fidelity", "ok": bool(nf.get("within_tolerance_all"))}, max(err_vals)))
        denominator = len(numeric_cases)
        has_numeric_suite = any(str(c.get("suite") or c.get("category") or "").lower().find("numeric") >= 0 for c in cases)
        status = "measured" if denominator else ("blocked_no_vector" if has_numeric_suite else "not_measured")
        for atol in TOLERANCE_GRID:
            if denominator == 0:
                rate = None
                passed = None
            else:
                passed_n = 0
                for case, err in numeric_cases:
                    if case.get("ok") and err is None:
                        passed_n += 1
                    elif err is not None and err <= atol:
                        passed_n += 1
                passed = passed_n
                rate = passed_n / denominator
            rows.append(
                {
                    "label": item["label"],
                    "bridge": item["bridge"],
                    "atol": atol,
                    "pass_rate": rate,
                    "passed": passed,
                    "numeric_case_count": denominator,
                    "status": status,
                }
            )
    data_path = _save_rows(out_base.parent / "source_data" / f"{out_base.name}.csv", rows)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    color_cycle = [PALETTE["blue_main"], PALETTE["green"], PALETTE["orange"], PALETTE["violet"], PALETTE["red"]]
    plotted = 0
    unavailable: list[str] = []
    for idx, item in enumerate(summaries):
        series = [r for r in rows if r["label"] == item["label"]]
        if not series or all(r["pass_rate"] is None for r in series):
            status = series[0]["status"] if series else "not_measured"
            if status != "not_measured":
                unavailable.append(item["label"])
            continue
        ax.semilogx(
            [r["atol"] for r in series],
            [r["pass_rate"] for r in series],
            marker="o",
            label=f"{item['label']} (n={series[0]['numeric_case_count']})",
            color=color_cycle[idx % len(color_cycle)],
            linewidth=1.6,
        )
        plotted += 1
    if plotted == 0:
        ax.text(0.5, 0.5, "No numeric-error vectors available\nfor tolerance recomputation", ha="center", va="center", transform=ax.transAxes, color=PALETTE["neutral_dark"])
    elif unavailable:
        wrapped = ", ".join(unavailable[:5])
        if len(unavailable) > 5:
            wrapped += f", +{len(unavailable) - 5} more"
        ax.text(
            0.02,
            0.08,
            "numeric suite failed before vector capture:\n" + wrapped,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6,
            color=PALETTE["neutral_dark"],
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PALETTE["neutral_light"], "alpha": 0.92},
        )
    ax.set_xlabel("Absolute tolerance")
    ax.set_ylabel("Numeric pass rate")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("Tolerance sensitivity from measured numeric errors")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(loc="lower right")
    _add_panel_label(ax, "c")
    fig.tight_layout()
    figures = _save_figure(fig, out_base)
    return figures, [data_path], {"provenance": "derived from real numeric error fields", "note": "This plot recomputes pass rates from recorded max errors. Reports that fail before vector capture are explicitly marked blocked_no_vector in source data."}


def plot_model_method_heatmap(summaries: list[dict[str, Any]], out_base: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    plt = _require_matplotlib()
    rows: list[dict[str, Any]] = []
    for item in summaries:
        by_suite = item.get("by_suite") or {}
        if by_suite:
            for suite_name, stats in by_suite.items():
                rows.append(
                    {
                        "bridge": item["bridge"],
                        "label": item["label"],
                        "component": suite_name,
                        "pass_rate": stats.get("adaptation_rate"),
                        "passed": stats.get("passed"),
                        "total": stats.get("total"),
                    }
                )
        else:
            grouped: dict[str, dict[str, int]] = {}
            for case in item["cases"]:
                if case.get("skipped") or not case.get("counts_toward_adaptation", True):
                    continue
                component = str(case.get("suite") or case.get("category") or "overall")
                grouped.setdefault(component, {"passed": 0, "total": 0})
                grouped[component]["total"] += 1
                if case.get("ok"):
                    grouped[component]["passed"] += 1
            if not grouped:
                grouped["overall"] = {"passed": item.get("passed_cases", 0), "total": item.get("total", 0)}
            for component, stats in grouped.items():
                total = stats["total"]
                rows.append(
                    {
                        "bridge": item["bridge"],
                        "label": item["label"],
                        "component": component,
                        "pass_rate": (stats["passed"] / total) if total else None,
                        "passed": stats["passed"],
                        "total": total,
                    }
                )
    data_path = _save_rows(out_base.parent / "source_data" / f"{out_base.name}.csv", rows)
    bridges = sorted({r["bridge"] for r in rows})
    components = sorted({r["component"] for r in rows})
    fig_w = max(4.8, min(10.5, 1.1 + 0.72 * len(components) + 0.55 * len(bridges)))
    fig_h = max(2.6, min(6.4, 1.4 + 0.42 * len(bridges)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    matrix = [[None for _ in components] for _ in bridges]
    totals = [[None for _ in components] for _ in bridges]
    for r in rows:
        i = bridges.index(r["bridge"])
        j = components.index(r["component"])
        matrix[i][j] = _safe_float(r.get("pass_rate"))
        totals[i][j] = r.get("total")
    for i, bridge in enumerate(bridges):
        for j, component in enumerate(components):
            value = matrix[i][j]
            if value is None:
                color = PALETTE["neutral_light"]
                text = "n/a"
            else:
                red = (182 / 255, 67 / 255, 66 / 255)
                blue = (15 / 255, 77 / 255, 146 / 255)
                mix = value
                color = tuple((1 - mix) * red[k] + mix * blue[k] for k in range(3))
                text = f"{value:.0%}\n(n={totals[i][j]})"
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white", linewidth=1.2))
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=6, color="white" if value is not None and value < 0.45 else PALETTE["neutral_black"])
    ax.set_xlim(0, len(components))
    ax.set_ylim(0, len(bridges))
    ax.set_xticks([i + 0.5 for i in range(len(components))])
    ax.set_xticklabels(components, rotation=30, ha="right")
    ax.set_yticks([i + 0.5 for i in range(len(bridges))])
    ax.set_yticklabels(bridges)
    ax.invert_yaxis()
    ax.set_title("Model/component pass-rate matrix")
    for spine in ax.spines.values():
        spine.set_visible(False)
    _add_panel_label(ax, "d")
    fig.tight_layout()
    figures = _save_figure(fig, out_base)
    return figures, [data_path], {"provenance": "real report suite/category breakdowns", "note": "Rows are bridge/report labels; columns are suites or categories available in each summary."}


def plot_design_space(summaries: list[dict[str, Any]], out_base: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    plt = _require_matplotlib()
    rows: list[dict[str, Any]] = []
    for item in summaries:
        for case in item["cases"]:
            if case.get("skipped"):
                continue
            component = str(case.get("suite") or case.get("category") or "case")
            case_id = str(case.get("case_id") or "")
            name = str(case.get("name") or case_id)
            text = f"{case_id} {name}".lower()
            if any(token in text for token in ["resnet", "mobilenet", "vit", "yolo", "unet", "model"]):
                model_family = "model"
            elif any(token in text for token in ["conv", "linear", "matmul", "layernorm", "dropout", "relu", "tensor", "math"]):
                model_family = "operator/subgraph"
            elif any(token in text for token in ["train", "optim", "loss", "grad", "backward", "numeric"]):
                model_family = "training/numeric"
            else:
                model_family = component
            failure = "pass" if case.get("ok") else _infer_failure_class(case.get("error"))
            rows.append(
                {
                    "bridge": item["bridge"],
                    "label": item["label"],
                    "model_family": model_family,
                    "component": component,
                    "failure_mode": failure,
                    "case_id": case_id,
                }
            )
    data_path = _save_rows(out_base.parent / "source_data" / f"{out_base.name}.csv", rows)
    families = sorted({r["model_family"] for r in rows}) or ["n/a"]
    components = sorted({r["component"] for r in rows}) or ["n/a"]
    failures = sorted({r["failure_mode"] for r in rows})
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for r in rows:
        key = (r["model_family"], r["component"])
        counts.setdefault(key, {})
        counts[key][r["failure_mode"]] = counts[key].get(r["failure_mode"], 0) + 1
    fig_w = max(5.2, min(10.8, 1.2 + 0.78 * len(components) + 0.45 * len(families)))
    fig_h = max(3.0, min(7.0, 1.6 + 0.55 * len(families)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for i, family in enumerate(families):
        for j, component in enumerate(components):
            dist = counts.get((family, component), {})
            total = sum(dist.values())
            if total == 0:
                ax.scatter(j, i, s=60, color=PALETTE["neutral_light"], marker="s", alpha=0.45)
                continue
            fail_total = total - dist.get("pass", 0)
            failure_share = fail_total / total
            top_failure = max((k for k in dist if k != "pass"), key=lambda k: dist[k], default="pass")
            color = FAILURE_PALETTE.get(top_failure, PALETTE["blue_main"]) if fail_total else PALETTE["blue_main"]
            ax.scatter(j, i, s=90 + 80 * total, color=color, alpha=0.82, edgecolor="white", linewidth=0.8)
            ax.text(j, i, f"{total}\n{failure_share:.0%}", ha="center", va="center", fontsize=6, color="white" if failure_share > 0.35 else PALETTE["neutral_black"])
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, rotation=30, ha="right")
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families)
    ax.set_title("Benchmark design space: model family x component x failure mode")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    legend_text = "Bubble size = case count; color = dominant failure mode; text = cases / failure share"
    ax.text(0.0, -0.22, legend_text, transform=ax.transAxes, ha="left", va="top", fontsize=6, color=PALETTE["neutral_dark"])
    _add_panel_label(ax, "e")
    fig.tight_layout()
    figures = _save_figure(fig, out_base)
    return figures, [data_path], {"provenance": "derived from report case metadata", "note": "This is a 2D replacement for the earlier 3D design-space sketch."}


def plot_metric_scorecard(summaries: list[dict[str, Any]], out_base: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    plt = _require_matplotlib()
    metric_defs = [
        ("compatibility", "Compat."),
        ("raw_pass", "Raw pass"),
        ("first_pass", "First pass"),
        ("numeric", "Numeric"),
        ("numeric_within_tolerance", "Tol. ok"),
        ("AR", "AR"),
        ("ME", "ME"),
    ]
    rows: list[dict[str, Any]] = []
    for item in summaries:
        for key, label in metric_defs:
            value = item["metrics"].get(key)
            display_value = value
            normalized = value
            if key == "ME" and value is not None:
                me_values = [s["metrics"].get("ME") for s in summaries if s["metrics"].get("ME") is not None]
                max_me = max(me_values) if me_values else value
                normalized = 1.0 - (value / max_me if max_me else 0.0)
            if key == "performance" and value is not None:
                perf_values = [s["metrics"].get("performance") for s in summaries if s["metrics"].get("performance") is not None]
                max_perf = max(perf_values) if perf_values else value
                normalized = value / max_perf if max_perf else None
            rows.append(
                {
                    "label": item["label"],
                    "bridge": item["bridge"],
                    "metric": key,
                    "metric_label": label,
                    "value": display_value,
                    "normalized_for_plot": normalized,
                    "available": value is not None,
                }
            )
    data_path = _save_rows(out_base.parent / "source_data" / f"{out_base.name}.csv", rows)
    labels = [item["label"] for item in summaries]
    fig_w = max(5.6, min(10.5, 1.2 + 0.72 * len(metric_defs) + 0.35 * len(labels)))
    fig_h = max(2.9, min(6.8, 1.4 + 0.42 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for i, item in enumerate(summaries):
        for j, (key, label) in enumerate(metric_defs):
            value = item["metrics"].get(key)
            row = next(r for r in rows if r["label"] == item["label"] and r["metric"] == key)
            norm = _safe_float(row.get("normalized_for_plot"))
            if value is None or norm is None:
                color = PALETTE["neutral_light"]
                text = "n/a"
            else:
                norm = max(0.0, min(1.0, norm))
                color = (1 - norm, 0.90 - 0.45 * norm, 0.80 - 0.45 * norm)
                if key == "ME":
                    text = f"{value:.1f}"
                else:
                    text = f"{value:.2f}" if value <= 1.2 else f"{value:.1f}"
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white", linewidth=1.2))
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=6, color=PALETTE["neutral_black"])
    ax.set_xlim(0, len(metric_defs))
    ax.set_ylim(0, len(labels))
    ax.set_xticks([i + 0.5 for i in range(len(metric_defs))])
    ax.set_xticklabels([label for _, label in metric_defs], rotation=30, ha="right")
    ax.set_yticks([i + 0.5 for i in range(len(labels))])
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Metric scorecard with explicit missingness")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.0, -0.24, "Higher color intensity is better after normalization; ME is inverted. Missing values are n/a, not zero.", transform=ax.transAxes, ha="left", va="top", fontsize=6, color=PALETTE["neutral_dark"])
    _add_panel_label(ax, "f")
    fig.tight_layout()
    figures = _save_figure(fig, out_base)
    return figures, [data_path], {"provenance": "real report summary fields", "note": "Replaces radar charts when metric availability is incomplete."}


PLOTTERS = {
    "compatibility-overview": plot_compatibility_overview,
    "failure-taxonomy": plot_failure_taxonomy,
    "tolerance-sweep": plot_tolerance_sweep,
    "model-method-heatmap": plot_model_method_heatmap,
    "design-space": plot_design_space,
    "metric-scorecard": plot_metric_scorecard,
}


def write_plot_manifest(out_dir: Path, summaries: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> Path:
    manifest = {
        "schema_version": "tbbcc.report_plots.v0.2",
        "style": {
            "backend": "python/matplotlib",
            "svg_text": "editable",
            "pdf_fonttype": 42,
        },
        "summaries": [
            {
                "label": item["label"],
                "bridge": item["bridge"],
                "path": item["path"],
                "compatibility_rate": item.get("compatibility_rate"),
                "raw_pass_rate": item.get("raw_pass_rate"),
                "total": item.get("total"),
            }
            for item in summaries
        ],
        "figures": artifacts,
    }
    path = out_dir / "plot_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def generate_report_plots(summary_paths: list[Path], out_dir: Path, kinds: list[str] | None = None) -> dict[str, Any]:
    if not summary_paths:
        raise ValueError("At least one summary JSON path is required")
    requested = kinds or DEFAULT_KINDS
    unknown = [kind for kind in requested if kind not in PLOTTERS]
    if unknown:
        raise ValueError(f"Unknown plot kind(s): {', '.join(unknown)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(summary_paths)
    artifacts: list[dict[str, Any]] = []
    figure_paths: list[Path] = []
    data_paths: list[Path] = []
    for kind in requested:
        files, sources, meta = PLOTTERS[kind](summaries, out_dir / _slug(kind))
        figure_paths.extend(files)
        data_paths.extend(sources)
        artifacts.append(
            {
                "kind": kind,
                "files": [str(path) for path in files],
                "source_data": [str(path) for path in sources],
                **meta,
            }
        )
    manifest = write_plot_manifest(out_dir, summaries, artifacts)
    return {
        "out_dir": str(out_dir),
        "manifest": str(manifest),
        "figures": [str(path) for path in figure_paths],
        "source_data": [str(path) for path in data_paths],
        "kinds": requested,
    }
