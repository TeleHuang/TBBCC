from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _frontmatter(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path} has no YAML frontmatter"
    block = text.split("---\n", 2)[1]
    data: dict[str, object] = {}
    for raw_line in block.splitlines():
        if not raw_line.strip() or raw_line.startswith("#"):
            continue
        key, sep, value = raw_line.partition(":")
        assert sep, f"invalid frontmatter line in {path}: {raw_line!r}"
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()]
            data[key.strip()] = items
        elif value.isdigit():
            data[key.strip()] = int(value)
        else:
            data[key.strip()] = value.strip("\"'")
    return data


def test_modern_plugin_structure_only() -> None:
    assert not (ROOT / "commands").exists(), "legacy commands/ directory must not exist"
    assert not (ROOT / "skills" / "eval").exists(), "empty skills/eval must not exist"

    expected = {
        "torchbridgebench": "eval",
        "inspect": "inspect",
        "ar-baseline": "ar-baseline",
    }
    for directory, name in expected.items():
        skill_path = ROOT / "skills" / directory / "SKILL.md"
        assert skill_path.is_file(), f"missing {skill_path}"
        assert _frontmatter(skill_path)["name"] == name


def test_tools_frontmatter_fields_are_lists() -> None:
    for path in list((ROOT / "skills").glob("*/SKILL.md")) + list((ROOT / "agents").glob("*.md")):
        meta = _frontmatter(path)
        for field in ("allowed-tools", "tools"):
            if field in meta:
                assert isinstance(meta[field], list), f"{path}:{field} must be a YAML list"


def test_agents_have_turn_limits_and_skill_links() -> None:
    expected_turns = {
        "evaluator.md": 10,
        "diagnostician.md": 8,
        "adapter-author.md": 15,
        "repairer.md": 20,
    }
    eval_skill_agents = {"evaluator.md", "adapter-author.md", "repairer.md"}
    for filename, max_turns in expected_turns.items():
        meta = _frontmatter(ROOT / "agents" / filename)
        assert meta.get("maxTurns") == max_turns
        if filename in eval_skill_agents:
            assert meta.get("skills") == ["eval"]


def test_plugin_metadata_is_complete() -> None:
    data = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    for field in ("name", "description", "version", "author", "homepage", "repository", "license", "keywords"):
        assert data.get(field), f"plugin.json missing {field}"
    assert data["name"] == "torchbridgebench"
    assert isinstance(data["keywords"], list)


def test_plugin_prompts_are_portable() -> None:
    prompt_paths = list((ROOT / "skills").glob("*/SKILL.md")) + list((ROOT / "agents").glob("*.md"))
    for path in prompt_paths:
        text = path.read_text(encoding="utf-8")
        assert "/home/ma-user/work" not in text, f"{path} contains machine-specific path"
        assert not re.search(r"python\s+scripts/", text), f"{path} uses plugin script without CLAUDE_PLUGIN_ROOT"
