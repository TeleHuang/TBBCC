from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import tbbcc  # noqa: E402


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
