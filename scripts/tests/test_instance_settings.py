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


@pytest.fixture
def wt(tmp_path):
    """A worktree with version 1's rule, notes, scripts and sweep."""
    wt = (tmp_path / NAME).resolve()
    for path in (
        f"configs/managers/rule_based/{NAME}.yml",
        f"notes/policy_finder/{NAME}.md",
        f"scripts/policy_finder/{NAME}/a.py",
        f"plots/simulation/policy_finder/{NAME}_sweep/sweep.json",
        "src/x.py",
    ):
        (wt / path).parent.mkdir(parents=True, exist_ok=True)
        (wt / path).write_text("")
    (wt / ".claude").mkdir()
    return wt


def write(wt, monkeypatch, *flags):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "settings",
            str(wt),
            "/main",
            NAME,
            "--start-commit",
            "abc",
            "--python",
            "/py",
            "--uv-python",
            "/uv",
            *flags,
        ],
    )
    inst.main()
    instance = json.loads((wt / ".claude/policy_finder.json").read_text())
    settings = json.loads((wt / ".claude/settings.local.json").read_text())
    return instance, settings


def test_version_1(wt, monkeypatch):
    instance, settings = write(wt, monkeypatch, "--max-params", "6")
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
    assert settings["permissions"]["allow"] == [
        f"Edit(/configs/managers/rule_based/{NAME}.yml)",
        f"Edit(/notes/policy_finder/{NAME}.md)",
        f"Edit(/scripts/policy_finder/{NAME}/**)",
    ]
    deny = settings["sandbox"]["filesystem"]["denyWrite"]
    assert str(wt / "src") in deny and "/main/.git" in deny
    assert str(wt / f"configs/managers/rule_based/{NAME}.yml") not in deny


def test_version_2_makes_version_1_read_only(wt, monkeypatch):
    instance, settings = write(
        wt, monkeypatch, "--version", "2", "--max-params", "none"
    )
    assert instance["rule_name"] == f"{NAME}-v2"
    assert instance["max_params"] is None
    assert settings["permissions"]["allow"] == [
        f"Edit(/configs/managers/rule_based/{NAME}-v2.yml)",
        f"Edit(/notes/policy_finder/{NAME}-v2.md)",
        f"Edit(/scripts/policy_finder/{NAME}/**)",
    ]
    deny = settings["sandbox"]["filesystem"]["denyWrite"]
    for path in (
        f"configs/managers/rule_based/{NAME}.yml",
        f"notes/policy_finder/{NAME}.md",
        "plots",
    ):
        assert str(wt / path) in deny, path
    # the shared scripts folder stays writable
    assert not any(
        d.startswith(str(wt / f"scripts/policy_finder/{NAME}")) for d in deny
    )
