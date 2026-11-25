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

    def fake(sha, rule, lo, hi, n, setting="exp2"):
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


def test_a_second_pipeline_waits_and_sees_the_state(version, monkeypatch):
    import fcntl
    import threading
    holder = open(version.wt_root / "pf-x.lock", "w")
    fcntl.flock(holder, fcntl.LOCK_EX)
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("state", version.run()))
    t.start()
    t.join(0.3)
    assert t.is_alive()  # waiting on the lock
    version.state_path.write_text(json.dumps({"status": "scored", "done": [], "data": {}}))
    fcntl.flock(holder, fcntl.LOCK_UN)
    t.join(5)
    assert result["state"]["status"] == "scored"


def test_a_fetch_during_an_outage_costs_no_job(version, monkeypatch):
    version.data["sweep_jobs"] = {"c.yml": ["1"]}
    submitted = []
    monkeypatch.setattr(pl.cluster_proxy, "submit", lambda *a, **k: submitted.append(a) or "2")

    def down(*a, **k):
        raise pl.cluster_proxy.ClusterError("could not look: Connection closed")

    monkeypatch.setattr(pl.cluster_proxy, "fetch", down)
    with pytest.raises(pl.cluster_proxy.ClusterError):
        version._fetch("sweep_jobs", "a" * 40, "c.yml", "plots/x", version.wt_root / "x")
    assert submitted == []

    def missing(*a, **k):
        raise pl.cluster_proxy.NoOutput("no plots/x on snapshot")

    monkeypatch.setattr(pl.cluster_proxy, "fetch", missing)
    monkeypatch.setattr(pl.cluster_proxy, "wait", lambda jobs: None)
    with pytest.raises(pl.Failed):
        version._fetch("sweep_jobs", "a" * 40, "c.yml", "plots/x", version.wt_root / "x")
    assert len(submitted) == pl.JOB_ATTEMPTS - 1


def test_transcripts_are_kept(version, tmp_path, monkeypatch):
    """Every agent run's session transcripts are copied next to the state file."""
    state = tmp_path / "spawner"
    projects = state / "runs/run-1/home/.claude/projects/-opt-repo"
    (projects / "sub").mkdir(parents=True)
    (projects / "a.jsonl").write_text("{}\n")
    (projects / "sub/b.jsonl").write_text("{}\n")
    (projects / "notes.txt").write_text("not a transcript\n")
    monkeypatch.setattr(pl.agent_runner, "SPAWNER_STATE", str(state))
    assert version.keep_transcripts("run-1") == 2
    kept = tmp_path / "worktrees/pf-x.transcripts/run-1/-opt-repo"
    assert (kept / "a.jsonl").exists() and (kept / "sub/b.jsonl").exists()
    assert not (kept / "notes.txt").exists()
    assert version.keep_transcripts("run-missing") == 0  # logged, not an error


def test_single_idea_version_1_is_the_seed(tmp_path, monkeypatch):
    """Version 1 of a single-idea instance: the seed rule and notes, no agent."""
    monkeypatch.setenv("PF_RUNS", str(tmp_path))
    monkeypatch.delenv("PF_WORKTREE_ROOT", raising=False)
    wt = tmp_path / "worktrees/pf-s"
    (wt / ".claude").mkdir(parents=True)
    (wt / ".claude/policy_finder.json").write_text(json.dumps(
        {"rule_name": "pf-s", "version": 1, "mode": "single_idea"}))
    v = pl.Version("pf-s")
    assert v.seeded
    monkeypatch.setattr(pl.agent_runner, "start", lambda *a: pytest.fail("agent started"))
    v.step_export()
    v.step_agent()
    v.step_collect()
    rule = (wt / "configs/managers/rule_based/pf-s.yml").read_text()
    assert "punishment = alpha * (20.0 - c[..., -1])" in rule
    notes = (wt / "notes/policy_finder/pf-s.md").read_text()
    assert notes.startswith("# pf-s\n") and "### Justification" in notes
    (wt / ".claude/policy_finder.json").write_text(json.dumps(
        {"rule_name": "pf-s-v2", "version": 2, "mode": "single_idea"}))
    assert not pl.Version("pf-s").seeded
