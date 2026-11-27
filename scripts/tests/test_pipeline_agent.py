"""pipeline.py's agent step for harness opencode: failed runs are counted
attempts with a bounded back-off, never the Claude plan's limit handling."""

import contextlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "policy_finder"
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("pipeline_agent", HERE / "pipeline.py")
pl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pl)


@pytest.fixture
def version(tmp_path, monkeypatch):
    monkeypatch.setenv("PF_RUNS", str(tmp_path))
    monkeypatch.delenv("PF_WORKTREE_ROOT", raising=False)
    conf = tmp_path / "worktrees/pf-x/.claude/policy_finder.json"
    conf.parent.mkdir(parents=True)
    conf.write_text(json.dumps({"name": "pf-x", "rule_name": "pf-x", "version": 1,
                                "harness": "opencode", "model": "q/c"}))
    sleeps = []
    monkeypatch.setattr(pl.time, "sleep", sleeps.append)
    monkeypatch.setattr(pl.agent_runner, "slot", lambda d: contextlib.nullcontext(0))
    monkeypatch.setattr(pl.export_instance, "build", lambda *a: None)
    monkeypatch.setattr(pl.usage, "limited", lambda *a: pytest.fail("plan limits"))
    monkeypatch.setattr(pl.usage, "gate", lambda *a, **k: pytest.fail("plan gate"))
    v = pl.Version("pf-x")
    v.data["prompt"] = "Design your rule."
    monkeypatch.setattr(v, "keep_transcripts", lambda run_id: 0)
    v.sleeps = sleeps
    return v


def runs(monkeypatch, version, records):
    queue = list(records)
    monkeypatch.setattr(pl.agent_runner, "start", lambda *a, **k: f"r{len(queue)}")

    def finish(run_id, log=print, harness="claude"):
        assert harness == "opencode"
        return {"status": queue[0][0], "report": queue.pop(0)[1], "result": {},
                "models": ["openrouter/q/c"]}
    monkeypatch.setattr(pl.agent_runner, "finish", finish)


def test_rate_limited_runs_are_counted_then_fail(version, monkeypatch):
    runs(monkeypatch, version, [("failed", "429 rate limit")] * pl.OPENCODE_ATTEMPTS)
    with pytest.raises(pl.Failed, match=f"failed {pl.OPENCODE_ATTEMPTS} times"):
        version.step_agent()
    assert len(version.data["agent_runs"]) == pl.OPENCODE_ATTEMPTS
    assert "limited_runs" not in version.data
    # a bounded back-off before each retry
    assert version.sleeps == [pl.OPENCODE_RETRY_WAIT * k
                              for k in range(1, pl.OPENCODE_ATTEMPTS)]


def test_a_failed_session_is_not_accepted_with_a_rule(version, monkeypatch):
    rule = version.exports / version.stem / version.rule
    rule.parent.mkdir(parents=True)
    rule.write_text("params: {}\n")
    # the runner's verdict arrives as status failed (agent_runner.finish)
    runs(monkeypatch, version, [("failed", "Let me look at the data."),
                                ("succeeded", "the rule and its notes")])
    version.step_agent()
    assert len(version.data["agent_runs"]) == 2
    assert (version.wt_root / "pf-x.report.md").read_text() == "the rule and its notes\n"
