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
                    "fixed norm": {"kind": "rule", "meaning": "m"},
                    "end game": {"kind": "rule", "meaning": "m"},
                    "harsh": {"kind": "policy category", "meaning": "m"},
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
        "--rule",
        "fixed norm",
        "--rule",
        "end game",
        "--policy-category",
        "harsh",
        "--idea",
        "an idea",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    summ.main()
    entry = summ.load()["instances"]["pf-1-v2"]
    assert entry["rule_sha"] == summ.rule_sha("pf-1-v2")
    assert entry["rules_included"] == ["fixed norm", "end game"]
    assert entry["policy_category"] == "harsh"
    summ.status(summ.load())
    assert "2 to summarize" in capsys.readouterr().out


def test_a_started_version_counts_before_its_rule(summ):
    """The launcher names the new version in the instance file before the
    agent writes its rule: it is listed, and not yet to summarize."""
    wt = summ.WT_ROOT / "pf-1"
    (wt / ".claude/policy_finder.json").write_text(
        json.dumps({"name": "pf-1", "version": 11, "rule_name": "pf-1-v11"})
    )
    assert summ.versions("pf-1") == ["pf-1", "pf-1-v2", "pf-1-v10", "pf-1-v11"]
    assert summ.rule_sha("pf-1-v11") is None


def test_a_version_without_a_policy_category_is_listed(summ, capsys):
    """Versions recorded before policy categories get one on the next run."""
    data = summ.load()
    data["instances"]["pf-1"] = {
        "rule_sha": summ.rule_sha("pf-1"),
        "rules_included": ["fixed norm"],
        "idea": "x",
    }
    summ.status(data)
    assert "\npf-1 (no policy category yet, not swept yet)" in capsys.readouterr().out


def test_a_rule_must_be_a_rule(summ, monkeypatch):
    argv = [
        "summarize",
        "record",
        "pf-1",
        "--rule",
        "harsh",
        "--policy-category",
        "harsh",
        "--idea",
        "x",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="--rule takes a rule"):
        summ.main()
