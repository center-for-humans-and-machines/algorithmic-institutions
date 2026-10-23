"""Tests for scripts/raven_remote_dir.sh: which Raven dir the cluster
scripts use."""

import os
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "raven_remote_dir.sh"
SHARED = "~/algorithmic-institutions"


def resolve(root, ai_remote_dir=None):
    env = {k: v for k, v in os.environ.items() if k != "AI_REMOTE_DIR"}
    if ai_remote_dir is not None:
        env["AI_REMOTE_DIR"] = ai_remote_dir
    return subprocess.run(
        [
            "bash",
            "-c",
            f'source "{HELPER}" && raven_remote_dir "$1" "{SHARED}"',
            "_",
            str(root),
        ],
        capture_output=True,
        text=True,
        env=env,
    )


def git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q", "-b", "policy-finder/probe1")
    git(
        tmp_path,
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "base",
    )
    return tmp_path


def test_shared_without_the_file(repo):
    assert resolve(repo).stdout.strip() == SHARED


def test_one_dir_per_branch_with_the_file(repo):
    (repo / ".raven_remote_dir").write_text("~/ai-isolated/{branch}\n")
    assert resolve(repo).stdout.strip() == "~/ai-isolated/policy-finder--probe1"
    git(repo, "checkout", "-q", "-b", "227-pipeline")
    assert resolve(repo).stdout.strip() == "~/ai-isolated/227-pipeline"


def test_ai_remote_dir_wins(repo):
    (repo / ".raven_remote_dir").write_text("~/ai-isolated/{branch}\n")
    assert resolve(repo, "~/elsewhere").stdout.strip() == "~/elsewhere"


def test_detached_head_fails(repo):
    (repo / ".raven_remote_dir").write_text("~/ai-isolated/{branch}\n")
    git(repo, "checkout", "-q", "--detach")
    result = resolve(repo)
    assert result.returncode == 1 and "set AI_REMOTE_DIR" in result.stderr


def test_this_checkout_is_isolated():
    root = HELPER.parents[1]
    assert (root / ".raven_remote_dir").read_text() == "~/ai-isolated/{branch}\n"
