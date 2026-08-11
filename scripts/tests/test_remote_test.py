"""Tests for scripts/remote_test.sh

Tests the script's structure, argument parsing, and configuration
without depending on SSH connection state.
"""

import os
import stat
import subprocess

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "remote_test.sh")
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


def ssh_is_active():
    """Check if SSH ControlMaster to raven is active."""
    result = subprocess.run(
        ["ssh", "-O", "check", "raven"],
        capture_output=True,
        timeout=5,
    )
    return result.returncode == 0


def run_script(*args):
    """Run remote_test.sh with given args, return CompletedProcess."""
    return subprocess.run(
        [SCRIPT, *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        timeout=30,
    )


@pytest.fixture
def script_content():
    with open(SCRIPT) as f:
        return f.read()


class TestScriptSetup:
    def test_script_exists(self):
        assert os.path.isfile(SCRIPT)

    def test_script_is_executable(self):
        mode = os.stat(SCRIPT).st_mode
        assert mode & stat.S_IXUSR, "Script should be executable by owner"

    def test_shebang(self):
        with open(SCRIPT) as f:
            first_line = f.readline()
        assert first_line.startswith("#!/usr/bin/env bash")

    def test_uses_strict_mode(self, script_content):
        assert "set -euo pipefail" in script_content


class TestConfiguration:
    def test_remote_host(self, script_content):
        assert 'REMOTE_HOST="raven"' in script_content

    def test_remote_project_dir(self, script_content):
        assert "~/algorithmic-institutions" in script_content

    def test_log_dir_under_claude(self, script_content):
        assert ".claude/test-logs" in script_content


class TestRsyncExcludes:
    """Verify critical directories are excluded from sync."""

    # experiments/ is deliberately synced: the evaluation-suite tests read
    # the human reference CSV when the remote suite runs them
    MUST_EXCLUDE = [
        ".venv/",
        ".git/",
        "artifacts/",
        "data/",
        "__pycache__/",
        ".claude/test-logs/",
    ]

    def test_critical_excludes_present(self, script_content):
        for pattern in self.MUST_EXCLUDE:
            assert (
                f"--exclude='{pattern}'" in script_content
            ), f"Missing rsync exclude for {pattern}"

    def test_uses_delete_flag(self, script_content):
        assert "rsync -azP --delete" in script_content


class TestSSHCheck:
    @pytest.mark.skipif(ssh_is_active(), reason="SSH to raven is active")
    def test_exits_2_when_no_ssh_connection(self):
        result = run_script()
        assert result.returncode == 2

    @pytest.mark.skipif(ssh_is_active(), reason="SSH to raven is active")
    def test_error_message_mentions_ssh_raven(self):
        result = run_script()
        assert "ssh raven" in result.stderr

    @pytest.mark.skipif(ssh_is_active(), reason="SSH to raven is active")
    def test_error_message_mentions_controlpersist(self):
        result = run_script()
        assert "ControlPersist" in result.stderr

    def test_check_ssh_function_exists(self, script_content):
        assert "check_ssh()" in script_content
        assert "ssh -O check" in script_content


class TestArgumentParsing:
    def test_supports_sync_only_flag(self, script_content):
        assert "--sync-only)" in script_content
        assert "DO_TEST=false" in script_content

    def test_supports_test_only_flag(self, script_content):
        assert "--test-only)" in script_content
        assert "DO_SYNC=false" in script_content

    def test_supports_double_dash_separator(self, script_content):
        assert "--)" in script_content
        assert "PYTEST_ARGS" in script_content


class TestRemoteExecution:
    def test_activates_venv(self, script_content):
        assert "source .venv/bin/activate" in script_content

    def test_uses_python_m_pytest(self, script_content):
        assert "python -m pytest" in script_content

    def test_uses_login_shell(self, script_content):
        assert "bash -l -c" in script_content

    def test_default_pytest_args(self, script_content):
        assert "src/ -v --tb=short" in script_content

    def test_creates_log_directory(self, script_content):
        assert 'mkdir -p "${LOG_DIR}"' in script_content

    def test_creates_latest_symlink(self, script_content):
        assert "ln -sf" in script_content
        assert "LATEST_LOG" in script_content
