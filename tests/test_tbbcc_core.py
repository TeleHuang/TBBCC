from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import tbbcc  # noqa: E402
import tbbcc_bridge_artifacts  # noqa: E402
import tbbcc_compare_artifacts  # noqa: E402
import tbbcc_gpu_reference  # noqa: E402
import tbbcc_model_suite  # noqa: E402


def test_resolve_path_uses_suite_parent(tmp_path: Path) -> None:
    suite_dir = tmp_path / "reports" / "generated"
    suite_dir.mkdir(parents=True)
    case = tmp_path / "benchmarks" / "case.json"
    case.parent.mkdir(parents=True)
    case.write_text("{}", encoding="utf-8")

    resolved = tbbcc._resolve_path(suite_dir, "../../benchmarks/case.json")
    assert resolved == case.resolve()


def test_execute_python_pair_isolates_target_from_source_process_state() -> None:
    code = """
import sys
import types

if ROLE == "source":
    sys.modules["tbbcc_fake_pollution"] = types.ModuleType("tbbcc_fake_pollution")
    RESULT = {"polluted": True}
else:
    RESULT = {"target_sees_pollution": "tbbcc_fake_pollution" in sys.modules}
"""
    source, target = tbbcc.execute_python_pair(
        'ROLE = "source"',
        'ROLE = "target"',
        code,
        {},
        timeout=10,
    )
    assert source.ok, source.stderr
    assert target.ok, target.stderr
    assert source.result == {"polluted": True}
    assert target.result == {"target_sees_pollution": False}


def test_execute_python_pair_applies_env_only_to_target() -> None:
    code = """
import os
RESULT = os.environ.get("TBBCC_TARGET_ONLY")
"""
    source, target = tbbcc.execute_python_pair("", "", code, {"TBBCC_TARGET_ONLY": "yes"}, timeout=10)
    assert source.ok, source.stderr
    assert target.ok, target.stderr
    assert source.result is None
    assert target.result == "yes"


def test_discover_local_bridge_sources_without_importing(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    plugin = workspace / "torchbridgebenchCCplugin"
    source = workspace / "ascend-torch4ms-ms272-stable"
    artifact = workspace / "tmp_torch4ms_adapters"
    plugin.mkdir(parents=True)
    source.mkdir()
    artifact.mkdir()
    (source / "test_train_cnn.py").write_text("import torch4ms\n", encoding="utf-8")

    found = tbbcc.discover_local_bridge_sources(plugin)
    assert found["torch4ms"] == [str(source.resolve())]


def test_run_eval_with_generated_suite_relative_paths(tmp_path: Path) -> None:
    case = tmp_path / "cases" / "case.json"
    adapter = tmp_path / "adapters" / "adapter.json"
    suite = tmp_path / "reports" / "suite.generated.json"
    out = tmp_path / "reports" / "eval"
    case.parent.mkdir()
    adapter.parent.mkdir()
    suite.parent.mkdir()

    case.write_text(
        json.dumps(
                {
                    "id": "unit/constant",
                    "level": "L1",
                    "track": "unit",
                    "code": "RESULT = 1",
                    "expected_ops": [],
                    "ground_truth": {"atol": 0.0, "rtol": 0.0},
            }
        ),
        encoding="utf-8",
    )
    adapter.write_text(
        json.dumps(
            {
                "bridge_id": "identity",
                "track": "intercept",
                "preamble": "",
                "source_preamble": "",
                "env": {},
                "atol": 0.0,
                "rtol": 0.0,
                "timeout_seconds": 10,
            }
        ),
        encoding="utf-8",
    )
    suite.write_text(
        json.dumps(
            {
                "suite_id": "relative_path_test",
                "cases": ["../cases/case.json"],
                "adapters": ["../adapters/adapter.json"],
            }
        ),
        encoding="utf-8",
    )

    class Args:
        effort_ledger = None
        ar_baseline = None
        timeout = None
        atol = None
        rtol = None

    args = Args()
    args.suite = str(suite)
    args.out = str(out)

    assert tbbcc.cmd_eval_suite(args) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["passed"] == 1


def test_large_tensor_report_is_compact(tmp_path: Path) -> None:
    case = tmp_path / "case.json"
    adapter = tmp_path / "adapter.json"
    out = tmp_path / "out"
    case.write_text(
        json.dumps(
            {
                "id": "unit/large_tensor",
                "level": "L1",
                "track": "unit",
                "code": "\n".join(
                    [
                        "import torch",
                        "RESULT = torch.arange(200000, dtype=torch.float32).reshape(400, 500)",
                    ]
                ),
                "expected_ops": [],
                "ground_truth": {"atol": 0.0, "rtol": 0.0},
            }
        ),
        encoding="utf-8",
    )
    adapter.write_text(
        json.dumps(
            {
                "bridge_id": "identity",
                "track": "intercept",
                "preamble": "",
                "source_preamble": "",
                "env": {},
                "atol": 0.0,
                "rtol": 0.0,
                "timeout_seconds": 10,
            }
        ),
        encoding="utf-8",
    )

    report = tbbcc.run_eval(case, adapter, out)
    report_path = Path(report["_paths"]["report_json"])
    assert report["final_state"] == "ALL_PASS"
    assert report_path.stat().st_size < 250_000
    stored = json.loads(report_path.read_text(encoding="utf-8"))
    source_result = stored["tiers"]["T1"]["source"]["result"]
    target_result = stored["tiers"]["T1"]["target"]["result"]
    assert source_result["__tbbcc_tensor_summary__"] is True
    assert target_result["__tbbcc_tensor_summary__"] is True
    assert source_result["shape"] == [400, 500]
    assert len(source_result["sha256"]) == 64
    assert "artifact_path" not in json.dumps(stored)
    assert not (out / "artifacts").exists()
    assert "TBBCC_PAYLOAD_JSON" not in stored["tiers"]["T1"]["source"]["stdout"]


def test_complex_tensor_comparison_uses_imaginary_part(tmp_path: Path) -> None:
    case = tmp_path / "case.json"
    adapter = tmp_path / "adapter.json"
    out = tmp_path / "out"
    case.write_text(
        json.dumps(
            {
                "id": "unit/complex_tensor",
                "level": "L1",
                "track": "unit",
                "code": "import torch\nRESULT = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)",
                "expected_ops": [],
                "ground_truth": {"atol": 0.0, "rtol": 0.0},
            }
        ),
        encoding="utf-8",
    )
    adapter.write_text(
        json.dumps(
            {
                "bridge_id": "imaginary_drift",
                "track": "intercept",
                "preamble": "import torch\n_orig_tensor = torch.tensor\ndef _tensor(*args, **kwargs):\n    value = _orig_tensor(*args, **kwargs)\n    return value + _orig_tensor([0+1j, 0+0j], dtype=value.dtype) if value.is_complex() else value\ntorch.tensor = _tensor\n",
                "source_preamble": "",
                "env": {},
                "atol": 0.0,
                "rtol": 0.0,
                "timeout_seconds": 10,
            }
        ),
        encoding="utf-8",
    )

    report = tbbcc.run_eval(case, adapter, out)
    metrics = report["tiers"]["T1"]["metrics"]
    assert report["final_state"] == "MARK_UNFIXABLE"
    assert metrics["reason"] == "NumericMismatch"
    assert metrics["max_error"] == 1.0


def test_eval_suite_resumes_existing_reports_by_default(tmp_path: Path) -> None:
    case = tmp_path / "case.json"
    adapter = tmp_path / "adapter.json"
    suite = tmp_path / "suite.json"
    out = tmp_path / "out"
    marker = tmp_path / "executions.txt"
    case.write_text(
        json.dumps(
            {
                "id": "unit/resume",
                "level": "L1",
                "track": "unit",
                "code": f"from pathlib import Path\nPath({str(marker)!r}).write_text(Path({str(marker)!r}).read_text() + 'x' if Path({str(marker)!r}).exists() else 'x')\nRESULT = 1",
                "expected_ops": [],
                "ground_truth": {"atol": 0.0, "rtol": 0.0},
            }
        ),
        encoding="utf-8",
    )
    adapter.write_text(
        json.dumps(
            {
                "bridge_id": "identity",
                "track": "intercept",
                "preamble": "",
                "source_preamble": "",
                "env": {},
                "atol": 0.0,
                "rtol": 0.0,
                "timeout_seconds": 10,
            }
        ),
        encoding="utf-8",
    )
    suite.write_text(
        json.dumps({"suite_id": "resume_test", "cases": [str(case)], "adapters": [str(adapter)]}),
        encoding="utf-8",
    )

    class Args:
        effort_ledger = None
        ar_baseline = None
        timeout = None
        atol = None
        rtol = None
        no_resume = False

    args = Args()
    args.suite = str(suite)
    args.out = str(out)

    assert tbbcc.cmd_eval_suite(args) == 0
    first_marker = marker.read_text(encoding="utf-8")
    assert first_marker == "xx"
    assert tbbcc.cmd_eval_suite(args) == 0
    assert marker.read_text(encoding="utf-8") == first_marker
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["completed"] == 1
    assert summary["totals"]["executed"] == 0
    assert summary["totals"]["skipped"] == 1

    args.no_resume = True
    assert tbbcc.cmd_eval_suite(args) == 0
    assert marker.read_text(encoding="utf-8") == "xxxx"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["executed"] == 1
    assert summary["totals"]["skipped"] == 0


def test_eval_suite_ignores_oversized_legacy_report_for_resume(tmp_path: Path) -> None:
    case = tmp_path / "case.json"
    adapter = tmp_path / "adapter.json"
    suite = tmp_path / "suite.json"
    out = tmp_path / "out"
    marker = tmp_path / "executions.txt"
    case_id = "unit/oversized_resume"
    bridge_id = "identity"
    case.write_text(
        json.dumps(
            {
                "id": case_id,
                "level": "L1",
                "track": "unit",
                "code": f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\nRESULT = 1",
                "expected_ops": [],
                "ground_truth": {"atol": 0.0, "rtol": 0.0},
            }
        ),
        encoding="utf-8",
    )
    adapter.write_text(
        json.dumps(
            {
                "bridge_id": bridge_id,
                "track": "intercept",
                "preamble": "",
                "source_preamble": "",
                "env": {},
                "atol": 0.0,
                "rtol": 0.0,
                "timeout_seconds": 10,
            }
        ),
        encoding="utf-8",
    )
    suite.write_text(json.dumps({"suite_id": "oversized_resume", "cases": [str(case)], "adapters": [str(adapter)]}), encoding="utf-8")
    run_dir = out / "runs" / f"{tbbcc._slug(case_id)}__{tbbcc._slug(bridge_id)}"
    run_dir.mkdir(parents=True)
    report = run_dir / "report.json"
    with report.open("wb") as f:
        f.seek(tbbcc._MAX_RESUMABLE_REPORT_BYTES + 1)
        f.write(b"{}")

    class Args:
        effort_ledger = None
        ar_baseline = None
        timeout = None
        atol = None
        rtol = None
        no_resume = False

    args = Args()
    args.suite = str(suite)
    args.out = str(out)

    assert tbbcc.cmd_eval_suite(args) == 0
    assert marker.read_text(encoding="utf-8") == "executed"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["executed"] == 1
    assert summary["totals"]["skipped"] == 0


def test_eval_suite_uses_persistent_workers_by_default(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    adapter = tmp_path / "adapter.json"
    suite = tmp_path / "suite.json"
    out = tmp_path / "out"
    init_marker = tmp_path / "init_counts.json"
    body_marker = tmp_path / "body_counts.json"
    body_code = f"""
import json
from pathlib import Path
marker = Path({str(body_marker)!r})
data = json.loads(marker.read_text()) if marker.exists() else {{"count": 0}}
data["count"] += 1
marker.write_text(json.dumps(data))
RESULT = 1
"""
    case_paths = []
    for idx in range(2):
        case = cases_dir / f"case{idx}.json"
        case.write_text(
            json.dumps(
                {
                    "id": f"unit/persistent/{idx}",
                    "level": "L1",
                    "track": "unit",
                    "code": body_code,
                    "expected_ops": [],
                    "ground_truth": {"atol": 0.0, "rtol": 0.0},
                }
            ),
            encoding="utf-8",
        )
        case_paths.append(str(case))
    preamble = f"""
import json
from pathlib import Path
marker = Path({str(init_marker)!r})
data = json.loads(marker.read_text()) if marker.exists() else {{"source": 0, "target": 0}}
data[ROLE] += 1
marker.write_text(json.dumps(data))
"""
    adapter.write_text(
        json.dumps(
            {
                "bridge_id": "identity",
                "track": "intercept",
                "preamble": "ROLE = 'target'\n" + preamble,
                "source_preamble": "ROLE = 'source'\n" + preamble,
                "env": {},
                "atol": 0.0,
                "rtol": 0.0,
                "timeout_seconds": 10,
            }
        ),
        encoding="utf-8",
    )
    suite.write_text(json.dumps({"suite_id": "persistent_worker", "cases": case_paths, "adapters": [str(adapter)]}), encoding="utf-8")

    class Args:
        effort_ledger = None
        ar_baseline = None
        timeout = None
        atol = None
        rtol = None
        no_resume = True
        isolated_per_case = False

    args = Args()
    args.suite = str(suite)
    args.out = str(out)

    assert tbbcc.cmd_eval_suite(args) == 0
    assert json.loads(init_marker.read_text(encoding="utf-8")) == {"source": 1, "target": 1}
    assert json.loads(body_marker.read_text(encoding="utf-8")) == {"count": 4}
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["worker_mode"] == "persistent"


def test_persistent_worker_ignores_non_protocol_stdout(tmp_path: Path) -> None:
    worker = tbbcc.PersistentPythonWorker(
        "source",
        "print('mindspore runtime banner before init')",
        {},
        timeout=10,
        artifact_root=tmp_path / "workers",
    )
    try:
        result = worker.run_case(
            "print('torch4ms runtime banner before result')\nRESULT = {'ok': True}",
            tmp_path / "artifacts",
        )
    finally:
        worker.close()

    assert result.ok, result.traceback
    assert result.result == {"ok": True}
    assert "mindspore runtime banner before init" in result.stdout
    assert "torch4ms runtime banner before result" in result.stdout
    assert "Invalid worker response" not in (result.traceback or "")


def test_original_torchbridgebench_torch4ms_report_is_anti_regression_fixture() -> None:
    report = Path("/home/ma-user/work/torchbridgebench/artifacts/reports/report_torch4ms_ms272_cann85_npu_20260510_clean.json")
    if not report.exists():
        pytest.skip("original torchbridgebench anti-regression report is not present in this workspace")

    data = json.loads(report.read_text(encoding="utf-8"))
    results = data.get("results") or []
    assert data.get("backend") == "torch4ms"
    assert len(results) == 41
    assert all(item.get("compatibility") is True for item in results)
    assert all(item.get("correctness") is True for item in results)
    assert {item.get("layer") for item in results} >= {"operator", "module", "autograd", "model", "end2end"}
    assert "repo_training_regression" in {item.get("suite_name") for item in results}


def test_gpu_reference_status_reports_mapping_required_for_mismatched_case_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gpu_root = tmp_path / "gpu"
    gpu_root.mkdir()
    (gpu_root / "summary.json").write_text(
        json.dumps(
            {
                "passed_cases": [
                    {"test_case_id": "bench_v1.0.0/L1/conv2d/001"},
                    {"test_case_id": "bench_v1.0.0/L1/relu/001"},
                ],
                "failed_cases": [],
            }
        ),
        encoding="utf-8",
    )
    case = tmp_path / "case.json"
    suite = tmp_path / "suite.json"
    case.write_text(
        json.dumps({"id": "bench_v1.0.0/L1/conv/conv2d_fp32", "level": "L1", "track": "unit", "code": "RESULT = 1"}),
        encoding="utf-8",
    )
    suite.write_text(json.dumps({"suite_id": "mismatch", "cases": [str(case)], "adapters": []}), encoding="utf-8")

    class Args:
        pass

    Args.artifact_root = str(gpu_root)
    Args.suite = str(suite)

    assert tbbcc.cmd_gpu_reference_status(Args()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gpu_case_count"] == 2
    assert payload["suite_case_count"] == 1
    assert payload["direct_overlap_count"] == 0
    assert payload["mapping_required"] is True


def test_gpu_reference_collector_uses_canonical_case_id(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    case = tmp_path / "case.json"
    suite = tmp_path / "suite.json"
    out = tmp_path / "gpu_reference"
    case.write_text(
        json.dumps(
            {
                "id": "bench_v1.0.0/L1/unit/reference_case",
                "level": "L1",
                "track": "unit",
                "seed": 42,
                "code": "RESULT = {'value': 1.0}",
                "expected_ops": ["unit.op"],
            }
        ),
        encoding="utf-8",
    )
    suite.write_text(json.dumps({"suite_id": "unit_gpu_reference", "cases": [str(case)], "adapters": []}), encoding="utf-8")

    class Args:
        pass

    Args.suite = str(suite)
    Args.out = str(out)
    Args.device = "cpu"
    Args.timeout = 10
    Args.no_resume = False
    Args.allow_cpu_fallback = False

    assert tbbcc_gpu_reference.collect_gpu_reference(Args()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["totals"]["passed"] == 1
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "tbbcc.gpu_reference.manifest.v1"
    assert manifest["canonical_case_system"] == "benchmarks/v1.0.0 case.id"
    assert manifest["cases"][0]["case_id"] == "bench_v1.0.0/L1/unit/reference_case"
    reference = json.loads((out / "cases" / "bench_v1.0.0_L1_unit_reference_case" / "reference.json").read_text(encoding="utf-8"))
    assert reference["schema_version"] == "tbbcc.gpu_reference.case.v1"
    assert reference["case_id"] == "bench_v1.0.0/L1/unit/reference_case"
    assert reference["channels"]["result"] == {"value": 1.0}

    class StatusArgs:
        pass

    StatusArgs.artifact_root = str(out)
    StatusArgs.suite = str(suite)
    assert tbbcc.cmd_gpu_reference_status(StatusArgs()) == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["gpu_case_count"] == 1
    assert status_payload["suite_case_count"] == 1
    assert status_payload["direct_overlap_count"] == 1
    assert status_payload["mapping_required"] is False


def test_artifact_compare_matches_gpu_and_bridge_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    case = tmp_path / "case.json"
    suite = tmp_path / "suite.json"
    adapter = tmp_path / "adapter.json"
    gpu_out = tmp_path / "gpu_reference"
    npu_out = tmp_path / "npu_bridge"
    compare_out = tmp_path / "compare"
    case.write_text(
        json.dumps(
            {
                "id": "bench_v1.0.0/L1/unit/artifact_compare",
                "level": "L1",
                "track": "unit",
                "seed": 42,
                "code": "RESULT = {'value': 2.0}\nACTIVATIONS = {'a': 3.0}\nGRADIENTS = {'g': 4.0}",
                "ground_truth": {"atol": 0.0, "rtol": 0.0},
            }
        ),
        encoding="utf-8",
    )
    suite.write_text(json.dumps({"suite_id": "artifact_compare", "cases": [str(case)], "adapters": []}), encoding="utf-8")
    adapter.write_text(
        json.dumps(
            {
                "bridge_id": "identity",
                "track": "intercept",
                "preamble": "",
                "source_preamble": "",
                "env": {},
                "atol": 0.0,
                "rtol": 0.0,
                "timeout_seconds": 10,
            }
        ),
        encoding="utf-8",
    )

    class GpuArgs:
        pass

    GpuArgs.suite = str(suite)
    GpuArgs.out = str(gpu_out)
    GpuArgs.device = "cpu"
    GpuArgs.timeout = 10
    GpuArgs.no_resume = False
    GpuArgs.allow_cpu_fallback = False
    assert tbbcc_gpu_reference.collect_gpu_reference(GpuArgs()) == 0
    capsys.readouterr()

    class BridgeArgs:
        pass

    BridgeArgs.suite = str(suite)
    BridgeArgs.adapter = str(adapter)
    BridgeArgs.out = str(npu_out)
    BridgeArgs.timeout = 10
    BridgeArgs.no_resume = False
    assert tbbcc_bridge_artifacts.collect_bridge_artifacts(BridgeArgs()) == 0
    capsys.readouterr()

    class CompareArgs:
        pass

    CompareArgs.gpu_reference = str(gpu_out)
    CompareArgs.npu_bridge = str(npu_out)
    CompareArgs.out = str(compare_out)
    CompareArgs.suite = str(suite)
    CompareArgs.atol = 1e-5
    CompareArgs.rtol = 1e-5
    assert tbbcc_compare_artifacts.compare_artifacts(CompareArgs()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["totals"]["overlap"] == 1
    assert payload["totals"]["passed"] == 1
    summary = json.loads((compare_out / "summary.json").read_text(encoding="utf-8"))
    assert summary["runs"][0]["case_id"] == "bench_v1.0.0/L1/unit/artifact_compare"
    assert summary["runs"][0]["channels"]["result"]["passed"] is True
    assert summary["runs"][0]["channels"]["activations"]["passed"] is True
    assert summary["runs"][0]["channels"]["gradients"]["passed"] is True


def test_canonical_model_suite_compare_produces_figure_candidates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    np = pytest.importorskip("numpy")
    registry = ROOT / "benchmarks" / "model_zoo" / "registry.json"
    suite = ROOT / "benchmarks" / "model_zoo" / "suites" / "canonical_models.json"
    models = json.loads(registry.read_text(encoding="utf-8"))["models"]
    gpu_root = tmp_path / "gpu_models"
    npu_root = tmp_path / "npu_models"
    out = tmp_path / "compare"

    def write_tensor(root: Path, model_id: str, channel: str, name: str, value: float) -> dict[str, object]:
        path = root / model_id / "artifacts" / channel / f"{name}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray([value, value + 1.0], dtype=np.float32)
        np.save(path, arr, allow_pickle=False)
        return {
            "__tbbcc_model_tensor__": True,
            "shape": [2],
            "dtype": "float32",
            "numel": 2,
            "artifact_path": str(path.relative_to(root)),
        }

    for model in models:
        model_id = model["model_id"]
        activation_layers = model["hooks"]["activation_layers"]
        gradient_layers = model["hooks"]["gradient_layers"]
        for root, role, drift in [(gpu_root, "gpu-reference", 0.0), (npu_root, "npu-bridge", 0.0)]:
            channels = {
                "result": write_tensor(root, model_id, "result", "output", 1.0),
                "activations": {
                    layer: write_tensor(root, model_id, "activations", f"activation_{layer}", 1.0)
                    for layer in activation_layers
                },
                "gradients": {
                    layer: write_tensor(root, model_id, "gradients", f"gradient_{layer}", 1.0)
                    for layer in gradient_layers
                },
                "task_metrics": {"latency_ms": 1.0},
            }
            if role == "npu-bridge" and model_id == "resnet18_imagenet_224":
                # Drift the second registry-ordered activation to verify first-divergence ordering.
                channels["activations"][activation_layers[1]] = write_tensor(root, model_id, "activations", f"activation_{activation_layers[1]}", 9.0 + drift)
            artifact = {
                "schema_version": "tbbcc.model_artifact.v1",
                "model_id": model_id,
                "role": role,
                "status": "passed",
                "channels": channels,
                "figure_role": model["figure_role"],
            }
            artifact_path = root / model_id / "model_artifact.json"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    (gpu_root / "manifest.json").write_text(json.dumps({"schema_version": "tbbcc.model_artifact_manifest.v1"}), encoding="utf-8")
    (npu_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "tbbcc.model_artifact_manifest.v1",
                "cases": [
                    {
                        "model_id": "vit_tiny_imagenet_224",
                        "status": "failed",
                        "error": "OperatorNotFound('attention')",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class Args:
        pass

    Args.registry = str(registry)
    Args.suite = str(suite)
    Args.gpu_reference = str(gpu_root)
    Args.npu_bridge = str(npu_root)
    Args.out = str(out)
    Args.atol = 0.0
    Args.rtol = 0.0
    Args.cosine_threshold = 0.999
    Args.aligned_cosine = 0.9999
    Args.usable_cosine = 0.99
    Args.aligned_p95 = 1e-2
    Args.usable_p95 = 5e-2

    assert tbbcc_model_suite.cmd_compare(Args()) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["totals"]["models"] == 4
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert {item["figure_id"] for item in summary["figure_candidates"]} == {
        "canonical_layerwise_fne",
        "canonical_gradient_consistency",
    }
    resnet = next(item for item in summary["models"] if item["model_id"] == "resnet18_imagenet_224")
    assert resnet["first_divergence_layer"] == "layer1"
    assert resnet["numerical_verdict"] == "diverged"
    assert (out / "source_data" / "layerwise_fne.csv").is_file()
    assert (out / "source_data" / "model_summary.csv").is_file()


def test_canonical_model_suite_compare_skips_missing_models(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    np = pytest.importorskip("numpy")
    registry = ROOT / "benchmarks" / "model_zoo" / "registry.json"
    suite = ROOT / "benchmarks" / "model_zoo" / "suites" / "canonical_models.json"
    models = json.loads(registry.read_text(encoding="utf-8"))["models"]
    gpu_root = tmp_path / "gpu_models"
    npu_root = tmp_path / "npu_models"
    out = tmp_path / "compare"
    model_by_id = {item["model_id"]: item for item in models}

    def write_tensor(root: Path, model_id: str, channel: str, name: str) -> dict[str, object]:
        tensor_path = root / model_id / "artifacts" / channel / f"{name}.npy"
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(tensor_path, np.asarray([1.0], dtype=np.float32), allow_pickle=False)
        return {
            "__tbbcc_model_tensor__": True,
            "shape": [1],
            "dtype": "float32",
            "numel": 1,
            "artifact_path": str(tensor_path.relative_to(root)),
        }

    def write_artifact(root: Path, model_id: str) -> None:
        model = model_by_id[model_id]
        path = root / model_id / "model_artifact.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema_version": "tbbcc.model_artifact.v1",
            "model_id": model_id,
            "role": "gpu-reference" if root == gpu_root else "npu-bridge",
            "status": "passed",
            "channels": {
                "result": write_tensor(root, model_id, "result", "output"),
                "activations": {
                    layer: write_tensor(root, model_id, "activations", f"activation_{layer}")
                    for layer in model["hooks"]["activation_layers"]
                },
                "gradients": {
                    layer: write_tensor(root, model_id, "gradients", f"gradient_{layer}")
                    for layer in model["hooks"]["gradient_layers"]
                },
                "task_metrics": {},
            },
        }
        path.write_text(json.dumps(artifact), encoding="utf-8")

    for model in models:
        if model["model_id"] == "vit_tiny_imagenet_224":
            continue
        write_artifact(gpu_root, model["model_id"])
        write_artifact(npu_root, model["model_id"])
    (gpu_root / "manifest.json").write_text(json.dumps({"schema_version": "tbbcc.model_artifact_manifest.v1"}), encoding="utf-8")
    (npu_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "tbbcc.model_artifact_manifest.v1",
                "cases": [
                    {
                        "model_id": "vit_tiny_imagenet_224",
                        "status": "failed",
                        "error": "OperatorNotFound('attention')",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class Args:
        pass

    Args.registry = str(registry)
    Args.suite = str(suite)
    Args.gpu_reference = str(gpu_root)
    Args.npu_bridge = str(npu_root)
    Args.out = str(out)
    Args.atol = 0.0
    Args.rtol = 0.0
    Args.cosine_threshold = 0.999
    Args.aligned_cosine = 0.9999
    Args.usable_cosine = 0.99
    Args.aligned_p95 = 1e-2
    Args.usable_p95 = 5e-2

    assert tbbcc_model_suite.cmd_compare(Args()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["totals"]["models"] == 3
    assert payload["totals"]["missing"] == 1
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["missing_models"][0]["model_id"] == "vit_tiny_imagenet_224"
    assert summary["missing_models"][0]["npu_error"] == "OperatorNotFound('attention')"
    assert summary["benchmark_verdict"] == "usable_partial"
    assert len(summary["models"]) == 3


def test_model_suite_collect_can_skip_by_max_models(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    class Args:
        pass

    Args.registry = str(ROOT / "benchmarks" / "model_zoo" / "registry.json")
    Args.suite = str(ROOT / "benchmarks" / "model_zoo" / "suites" / "canonical_models.json")
    Args.out = str(tmp_path / "model_collect")
    Args.role = "gpu-reference"
    Args.device = "cpu"
    Args.adapter = None
    Args.input_root = None
    Args.strict_input_root = False
    Args.model_id = []
    Args.model_cache = str(tmp_path / "model_cache")
    Args.seed = 1
    Args.time_budget_seconds = None
    Args.max_models = 0
    Args.no_pretrained = True
    Args.allow_cpu_fallback = True
    Args.keep_going = False

    assert tbbcc_model_suite.cmd_collect(Args()) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["totals"]["passed"] == 0
    assert payload["totals"]["skipped"] == 4
    manifest = json.loads((tmp_path / "model_collect" / "manifest.json").read_text(encoding="utf-8"))
    assert {item["reason"] for item in manifest["cases"]} == {"MaxModelsReached"}


def test_model_suite_bridge_preamble_runs_after_model_setup() -> None:
    source = inspect.getsource(tbbcc_model_suite._run_one_model)
    adapter_exec = source.index("exec(adapter.preamble")
    assert source.index("_build_model(") < adapter_exec
    assert source.index("_prepare_input(") < adapter_exec
    assert source.index("_register_activation_hooks(") < adapter_exec
    assert source.index("_register_gradient_hooks(") < adapter_exec
    assert adapter_exec < source.index("_forward_model(")


def test_find_eval_caches_matches_bridge_and_suite_scope(tmp_path: Path) -> None:
    case = tmp_path / "benchmarks" / "case.json"
    suite = tmp_path / "benchmarks" / "suite.json"
    cache = tmp_path / "reports" / "identity_eval_cache"
    case.parent.mkdir(parents=True)
    cache.mkdir(parents=True)
    case.write_text(
        json.dumps(
            {
                "id": "unit/cache",
                "level": "L1",
                "track": "unit",
                "code": "RESULT = 1",
            }
        ),
        encoding="utf-8",
    )
    suite.write_text(json.dumps({"suite_id": "cache_scope", "cases": ["case.json"], "adapters": []}), encoding="utf-8")
    (cache / "adapter.generated.json").write_text(
        json.dumps({"bridge_id": "identity", "track": "intercept", "preamble": ""}),
        encoding="utf-8",
    )
    (cache / "suite.generated.json").write_text(
        json.dumps(
            {
                "suite_id": "cache_scope_generated",
                "cases": ["../../benchmarks/case.json"],
                "adapters": ["adapter.generated.json"],
            }
        ),
        encoding="utf-8",
    )

    matches = tbbcc.find_eval_caches("identity", tmp_path / "reports", suite)
    assert len(matches) == 1
    assert matches[0]["cache_dir"] == str(cache.resolve())
    assert matches[0]["case_count"] == 1

    assert tbbcc.find_eval_caches("other_bridge", tmp_path / "reports", suite) == []


def test_find_eval_caches_rejects_weak_torch4ms_cache(tmp_path: Path) -> None:
    case = tmp_path / "benchmarks" / "case.json"
    suite = tmp_path / "benchmarks" / "suite.json"
    cache = tmp_path / "reports" / "torch4ms_weak_cache"
    case.parent.mkdir(parents=True)
    cache.mkdir(parents=True)
    case.write_text(
        json.dumps({"id": "unit/cache", "level": "L1", "track": "unit", "code": "RESULT = 1"}),
        encoding="utf-8",
    )
    suite.write_text(json.dumps({"suite_id": "cache_scope", "cases": ["case.json"], "adapters": []}), encoding="utf-8")
    (cache / "adapter.generated.json").write_text(
        json.dumps({"bridge_id": "torch4ms", "track": "intercept", "preamble": "import torch4ms\n_env = torch4ms.default_env()\n_env.__enter__()\n"}),
        encoding="utf-8",
    )
    (cache / "suite.generated.json").write_text(
        json.dumps({"suite_id": "cache_scope_generated", "cases": ["../../benchmarks/case.json"], "adapters": ["adapter.generated.json"]}),
        encoding="utf-8",
    )

    assert tbbcc.find_eval_caches("torch4ms", tmp_path / "reports", suite) == []
