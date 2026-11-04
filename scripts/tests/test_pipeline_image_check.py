"""pipeline.py's image check: only validate-rule's own verdict fails a version."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "policy_finder"
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("pipeline", HERE / "pipeline.py")
pl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pl)


@pytest.fixture
def version(tmp_path, monkeypatch):
    monkeypatch.setenv("PF_RUNS", str(tmp_path))
    monkeypatch.delenv("PF_WORKTREE_ROOT", raising=False)
    conf = tmp_path / "worktrees/pf-x/.claude/policy_finder.json"
    conf.parent.mkdir(parents=True)
    conf.write_text(json.dumps({"rule_name": "pf-x", "version": 1, "min_params": 1,
                                "max_params": 5, "sobol_points": 256}))
    monkeypatch.setattr(pl.time, "sleep", lambda s: None)
    v = pl.Version("pf-x")
    v.data["sweep_sha"] = "a" * 40
    return v


def answers(monkeypatch, *logs):
    queue = list(logs)
    calls = []

    def fake(sha, rule, lo, hi, n):
        calls.append(rule)
        log = queue.pop(0)
        return log.startswith("Valid"), log

    monkeypatch.setattr(pl.cluster_proxy, "validate_rule", fake)
    return calls


def test_invalid_is_a_verdict(version, monkeypatch):
    answers(monkeypatch, "Traceback ...\nInvalid: the rule reads th.os\n")
    with pytest.raises(pl.Failed, match="th.os"):
        version.step_image_check()


def test_the_login_node_is_retried(version, monkeypatch):
    calls = answers(monkeypatch, "libgomp: Thread creation failed\n",
                    'FATAL: container creation failed: "fuseapps"\n', "Valid: 256 points\n")
    version.step_image_check()
    assert len(calls) == 3


def test_a_login_node_that_never_works_is_an_error(version, monkeypatch):
    answers(monkeypatch, *["  File numpy\nKeyboardInterrupt\n"] * pl.IMAGE_CHECK_TRIES)
    with pytest.raises(pl.cluster_proxy.ClusterError):
        version.step_image_check()
