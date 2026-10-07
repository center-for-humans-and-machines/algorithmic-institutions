"""Tests for the policy-finder guard hook (#236).

Each test copies the hook into a fake worktree under tmp_path and runs it as
Claude Code does: hook input JSON on stdin, decision JSON (or nothing) on
stdout.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / ".claude/hooks/policy_finder_guard.py"
NAME = "probe1"


@pytest.fixture
def root(tmp_path):
    root = (tmp_path / "wt").resolve()
    (root / ".claude/hooks").mkdir(parents=True)
    shutil.copy(HOOK, root / ".claude/hooks/policy_finder_guard.py")
    (root / ".claude/policy_finder.json").write_text(json.dumps({"name": NAME}))
    (root / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n")
    (root / "configs/managers/rule_based").mkdir(parents=True)
    (root / "scripts/policy_finder").mkdir(parents=True)
    return root


def run(root, tool_name, tool_input, cwd=None, raw=None):
    data = {"tool_name": tool_name, "tool_input": tool_input, "cwd": str(cwd or root)}
    out = subprocess.run(
        [sys.executable, str(root / ".claude/hooks/policy_finder_guard.py")],
        input=raw if raw is not None else json.dumps(data),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if not out.strip():
        return None
    decision = json.loads(out)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    return decision["permissionDecisionReason"]


# -- writes ---------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        f"configs/managers/rule_based/{NAME}.yml",
        f"notes/policy_finder/{NAME}.md",
        f"scripts/policy_finder/{NAME}/analysis.py",
        f"scripts/policy_finder/{NAME}/sub/dir/test_rule.py",
    ],
)
@pytest.mark.parametrize("tool", ["Write", "Edit"])
def test_write_allowed(root, tool, path):
    assert run(root, tool, {"file_path": str(root / path)}) is None
    assert run(root, tool, {"file_path": path}) is None  # relative to cwd


def test_notebook_allowed(root):
    path = root / f"scripts/policy_finder/{NAME}/explore.ipynb"
    assert run(root, "NotebookEdit", {"notebook_path": str(path)}) is None


@pytest.mark.parametrize(
    "path",
    [
        "configs/managers/rule_based/other.yml",
        f"configs/managers/rule_based/{NAME}.json",
        f"configs/managers/rule_based/{NAME}.yml.bak",
        "scripts/policy_finder/new_instance.sh",
        "notes/policy_finder/other.md",
        f"notes/policy_finder/{NAME}.txt",
        f"notes/policy_finder/{NAME}/a.md",
        "notes/autoresearch.md",
        "scripts/policy_finder/check_instance.sh",
        f"scripts/policy_finder/{NAME}x/a.py",
        "scripts/policy_finder/other/a.py",
        f"scripts/policy_finder/{NAME}/../check_instance.sh",
        ".claude/hooks/policy_finder_guard.py",
        ".claude/settings.local.json",
        ".claude/policy_finder.json",
        "src/aimanager/manager/api_manager.py",
        "CLAUDE.md",
        "/tmp/elsewhere.py",
        "~/notes.md",
    ],
)
@pytest.mark.parametrize("tool", ["Write", "Edit"])
def test_write_rejected(root, tool, path):
    assert "writes are limited" in run(root, tool, {"file_path": path})


def test_write_through_symlink_rejected(root):
    scripts = root / f"scripts/policy_finder/{NAME}"
    scripts.mkdir()
    (scripts / "link").symlink_to(root / "src", target_is_directory=True)
    reason = run(root, "Write", {"file_path": str(scripts / "link/x.py")})
    assert "writes are limited" in reason


def test_write_needs_instance_file(root):
    (root / ".claude/policy_finder.json").unlink()
    path = root / f"scripts/policy_finder/{NAME}/a.py"
    assert "no instance file" in run(root, "Write", {"file_path": str(path)})


def test_bad_instance_name(root):
    (root / ".claude/policy_finder.json").write_text(json.dumps({"name": "../src"}))
    path = root / "src/a.py"
    assert "bad instance name" in run(root, "Write", {"file_path": str(path)})


# -- reads ----------------------------------------------------------------


@pytest.mark.parametrize(
    "tool, tool_input",
    [
        ("Read", {"file_path": "CLAUDE.md"}),
        ("Read", {"file_path": "experiments/2group_8agent_50ep.csv"}),
        ("Grep", {"pattern": "punish"}),
        ("Grep", {"pattern": "punish", "path": "src"}),
        ("Glob", {"pattern": "**/*.yml"}),
        ("Glob", {"pattern": "*.py", "path": "scripts"}),
        ("Read", {"file_path": ".gitignore"}),
    ],
)
def test_read_allowed(root, tool, tool_input):
    assert run(root, tool, tool_input) is None


@pytest.mark.parametrize(
    "tool, tool_input, match",
    [
        ("Read", {"file_path": "/etc/passwd"}, "limited to the worktree"),
        ("Read", {"file_path": "~/.ssh/config"}, "limited to the worktree"),
        ("Read", {"file_path": "../other/CLAUDE.md"}, "limited to the worktree"),
        ("Read", {"file_path": ".git"}, ".git is not readable"),
        ("Grep", {"pattern": "x", "path": ".git"}, ".git is not readable"),
        ("Grep", {"pattern": "x", "path": "/"}, "limited to the worktree"),
        ("Glob", {"pattern": "/Users/**/*.yml"}, "limited to the worktree"),
        ("Glob", {"pattern": "~/**"}, "limited to the worktree"),
        ("Glob", {"pattern": "../**/*.yml"}, "may not climb"),
        ("Glob", {"pattern": "*", "path": "/tmp"}, "limited to the worktree"),
    ],
)
def test_read_rejected(root, tool, tool_input, match):
    assert match in run(root, tool, tool_input)


def test_read_from_outside_cwd_rejected(root, tmp_path):
    reason = run(root, "Grep", {"pattern": "x"}, cwd=tmp_path)
    assert "limited to the worktree" in reason


# -- bash -----------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "ls configs",
        "python scripts/policy_finder/probe1/analysis.py",
        "PYTHONPATH=src python -m pytest scripts/policy_finder/probe1",
        "grep -r digit src | head",
        "echo github.com",
        "cat .gitignore",
        "python -c \"print('legit')\"",
    ],
)
def test_bash_allowed(root, command):
    assert run(root, "Bash", {"command": command}) is None


@pytest.mark.parametrize(
    "command",
    [
        "git log -p",
        "git",
        "gh issue view 235",
        "cd src && git show HEAD~3:configs/x.yml",
        "ls; git status",
        "echo $(git rev-parse HEAD)",
        "echo `git log`",
        "/usr/bin/git log",
        "env git log",
        "python -c \"import subprocess; subprocess.run(['git', 'log'])\"",
        "x=1 git log",
        "(git log)",
        "ls | gh api repos",
    ],
)
def test_bash_rejected(root, command):
    assert "git and gh" in run(root, "Bash", {"command": command})


# -- other tools and bad input -------------------------------------------


@pytest.mark.parametrize(
    "tool", ["WebFetch", "WebSearch", "mcp__claude_ai_Gmail__send", "mcp__x__y"]
)
def test_denied_tools(root, tool):
    assert "is not available" in run(root, tool, {})


@pytest.mark.parametrize("tool", ["TodoWrite", "Agent", "AskUserQuestion"])
def test_other_tools_pass(root, tool):
    assert run(root, tool, {}) is None


@pytest.mark.parametrize("raw", ["", "not json", "[]"])
def test_bad_input_fails_closed(root, raw):
    assert "guard error" in run(root, None, None, raw=raw)
