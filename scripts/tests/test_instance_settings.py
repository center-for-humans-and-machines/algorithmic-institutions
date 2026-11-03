"""Tests for a policy-finder instance's settings per version (#236, #244)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "policy_finder/instance_settings.py"
SPEC = importlib.util.spec_from_file_location("instance_settings", SCRIPT)
inst = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inst)

NAME = "probe1"


def test_rule_name():
    assert inst.rule_name(NAME, 1) == NAME
    assert inst.rule_name(NAME, 3) == f"{NAME}-v3"


def test_write_paths():
    assert inst.write_paths(NAME, 2) == (
        f"configs/managers/rule_based/{NAME}-v2.yml",
        f"notes/policy_finder/{NAME}-v2.md",
        f"scripts/policy_finder/{NAME}",
    )


@pytest.fixture
def wt(tmp_path):
    wt = (tmp_path / NAME).resolve()
    wt.mkdir()
    return wt


def write(wt, monkeypatch, *flags):
    argv = ["settings", str(wt), NAME, "--start-commit", "abc", *flags]
    monkeypatch.setattr(sys, "argv", argv)
    inst.main()
    return json.loads((wt / ".claude/policy_finder.json").read_text())


def test_version_1(wt, monkeypatch):
    instance = write(wt, monkeypatch, "--max-params", "6", "--python", "/py")
    assert instance == {
        "name": NAME,
        "version": 1,
        "rule_name": NAME,
        "start_commit": "abc",
        "min_params": 1,
        "max_params": 6,
        "sobol_points": 256,
        "python": "/py",
    }
    # the write paths' folders exist, for the container's read-write mounts
    for path in ("configs/managers/rule_based", "notes/policy_finder",
                 f"scripts/policy_finder/{NAME}"):
        assert (wt / path).is_dir(), path


def test_version_2(wt, monkeypatch):
    instance = write(wt, monkeypatch, "--version", "2", "--max-params", "none")
    assert instance["rule_name"] == f"{NAME}-v2"
    assert instance["max_params"] is None
    assert instance["python"] == "python3"


def test_no_claude_settings(wt, monkeypatch):
    """The container is the sandbox: no permissions, sandbox or hooks."""
    write(wt, monkeypatch)
    assert sorted(p.name for p in (wt / ".claude").iterdir()) == ["policy_finder.json"]
