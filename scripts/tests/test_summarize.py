"""Tests for the observation app's Summary bookkeeping, per version (#244)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "policy_finder/observation_app/summarize.py"
)


@pytest.fixture
def summ(tmp_path, monkeypatch):
    """summarize.py over a worktree root with one instance at version 3, and
    an observatory that is never reachable (local copies only)."""
    monkeypatch.setenv("PF_WORKTREE_ROOT", str(tmp_path))
    monkeypatch.setenv("PF_OBSERVATORY", "nohost.invalid:/x")
    spec = importlib.util.spec_from_file_location("summarize", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wt = tmp_path / "pf-1"
    (wt / ".claude").mkdir(parents=True)
    (wt / ".claude/policy_finder.json").write_text("{}")
    rules = wt / "configs/managers/rule_based"
    rules.mkdir(parents=True)
    for version in ("pf-1", "pf-1-v2", "pf-1-v10", "pf-1-vx"):
        (rules / f"{version}.yml").write_text(f"code: {version}\n")
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "categories": {
                    "fixed norm": {"kind": "bar", "meaning": "m"},
                    "end game": {"kind": "mechanism", "meaning": "m"},
                },
                "instances": {},
            }
        )
    )
    return module


def test_versions_in_order(summ):
    assert summ.versions("pf-1") == ["pf-1", "pf-1-v2", "pf-1-v10"]
    assert summ.instance_of("pf-1-v2") == "pf-1"
    assert summ.instance_of("pf-1") == "pf-1"
    assert summ.rule_path("pf-1-v2").name == "pf-1-v2.yml"
    assert summ.rule_path("pf-1-v2").parents[3].name == "pf-1"


def test_each_version_is_recorded_on_its_own(summ, monkeypatch, capsys):
    summ.status(summ.load())
    out = capsys.readouterr().out
    assert "3 to summarize" in out
    for version in ("pf-1", "pf-1-v2", "pf-1-v10"):
        assert f"\n{version} (new, not swept yet)" in out

    argv = [
        "summarize",
        "record",
        "pf-1-v2",
        "--bar",
        "fixed norm",
        "--mechanism",
        "end game",
        "--idea",
        "an idea",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    summ.main()
    entry = summ.load()["instances"]["pf-1-v2"]
    assert entry["rule_sha"] == summ.rule_sha("pf-1-v2")
    assert entry["mechanisms"] == ["end game"]
    summ.status(summ.load())
    assert "2 to summarize" in capsys.readouterr().out
