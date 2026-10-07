"""Tests for scripts/policy_finder/check_instance.sh (#236).

Each test builds a throwaway repo with a base branch and an instance worktree,
the layout new_instance.sh makes, and runs the check from the repo's copy.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

CHECK = Path(__file__).resolve().parents[1] / "policy_finder/check_instance.sh"
NAME = "probe1"
RULE = f"configs/managers/rule_based/{NAME}.yml"
NOTES = f"notes/policy_finder/{NAME}.md"
NOTES_TEXT = """# probe1

## Explorations

1. Contributions by round.

## Key findings

1. They rise.

## Hypothesis

Punish early.

### Justification

Finding 1.
"""


def git(cwd, *args):
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    main = tmp_path / "main"
    (main / "scripts/policy_finder").mkdir(parents=True)
    (main / "configs/managers/rule_based").mkdir(parents=True)
    (main / "configs/managers/rule_based/.gitkeep").write_text("")
    (main / "src.py").write_text("x = 1\n")
    (main / ".gitignore").write_text(".claude/policy_finder.json\n")
    shutil.copy(CHECK, main / "scripts/policy_finder/check_instance.sh")
    git(tmp_path, "init", "-q", "-b", "policy-finder-base", str(main))
    git(main, "config", "user.email", "t@t")
    git(main, "config", "user.name", "t")
    git(main, "add", "-A")
    git(main, "commit", "-q", "-m", "base")
    wt = tmp_path / "policy-finder-worktrees" / NAME
    git(main, "worktree", "add", "-q", "-b", f"policy-finder/{NAME}", str(wt))
    (wt / ".claude").mkdir()
    (wt / ".claude/policy_finder.json").write_text("{}")  # gitignored
    return main, wt


def check(main, *args):
    return subprocess.run(
        ["bash", str(main / "scripts/policy_finder/check_instance.sh"), NAME, *args],
        capture_output=True,
        text=True,
    )


def write_rule(wt, notes=NOTES_TEXT):
    (wt / RULE).write_text("params: {a: x}\ncode: punishment = a\n")
    if notes is not None:
        (wt / NOTES).parent.mkdir(parents=True, exist_ok=True)
        (wt / NOTES).write_text(notes)


def test_rule_and_scripts_pass(repo):
    main, wt = repo
    write_rule(wt)
    (wt / f"scripts/policy_finder/{NAME}/sub").mkdir(parents=True)
    (wt / f"scripts/policy_finder/{NAME}/sub/analysis.py").write_text("")
    result = check(main)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_no_rule_fails(repo):
    main, _ = repo
    result = check(main)
    assert result.returncode == 1
    assert f"no {RULE}" in result.stderr


def test_no_notes_fails(repo):
    main, wt = repo
    write_rule(wt, notes=None)
    result = check(main)
    assert result.returncode == 1
    assert f"no {NOTES}" in result.stderr


@pytest.mark.parametrize(
    "notes, missing",
    [
        (NOTES_TEXT.replace("## Key findings", "## Findings"), "## Key findings"),
        (NOTES_TEXT.replace("### Justification", "## Justification"), "### Just"),
        (NOTES_TEXT.replace("## Explorations", "### Explorations"), "## Explor"),
        (
            NOTES_TEXT.replace("## Hypothesis\n\nPunish early.\n\n", "")
            + "\n## Hypothesis\n",
            "### Justification",  # found, but before the moved Hypothesis
        ),
        ("", "## Explorations"),
    ],
)
def test_notes_sections_in_order(repo, notes, missing):
    main, wt = repo
    write_rule(wt, notes=notes)
    result = check(main)
    assert result.returncode == 1
    assert f"needs the section '{missing}" in result.stderr


@pytest.mark.parametrize(
    "path",
    [
        "newfile.txt",
        "configs/managers/rule_based/other.yml",
        "notes/policy_finder/other.md",
        "notes/other.md",
        f"scripts/policy_finder/{NAME}x/a.py",
        "scripts/policy_finder/check_instance.sh",
        "src.py",
    ],
)
def test_untracked_or_modified_outside_fails(repo, path):
    main, wt = repo
    write_rule(wt)
    (wt / path).parent.mkdir(parents=True, exist_ok=True)
    (wt / path).write_text("changed\n")
    result = check(main)
    assert result.returncode == 1
    assert f"outside the write paths: {path}" in result.stderr


def test_deleted_file_fails(repo):
    main, wt = repo
    write_rule(wt)
    (wt / "src.py").unlink()
    assert "outside the write paths: src.py" in check(main).stderr


def test_committed_change_outside_fails(repo):
    main, wt = repo
    write_rule(wt)
    (wt / "src.py").write_text("x = 2\n")
    git(wt, "commit", "-q", "-am", "sneak")
    result = check(main)
    assert result.returncode == 1
    assert "outside the write paths: src.py" in result.stderr


def test_commit_commits_only_write_paths(repo):
    main, wt = repo
    write_rule(wt)
    (wt / f"scripts/policy_finder/{NAME}").mkdir(parents=True)
    (wt / f"scripts/policy_finder/{NAME}/a.py").write_text("")
    result = check(main, "--commit")
    assert result.returncode == 0, result.stderr
    files = git(
        main, "diff", "--name-only", f"policy-finder-base...policy-finder/{NAME}"
    )
    assert files.split() == [RULE, NOTES, f"scripts/policy_finder/{NAME}/a.py"]
    assert git(wt, "status", "--porcelain") == ""
    assert check(main).returncode == 0  # still passes once committed


def test_commit_rule_only(repo):
    main, wt = repo
    write_rule(wt)
    result = check(main, "--commit")
    assert result.returncode == 0, result.stderr
    assert (
        git(wt, "log", "-1", "--format=%s")
        == f"policy-finder {NAME}: rule, notes and analysis\n"
    )
