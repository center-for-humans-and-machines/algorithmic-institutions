"""Tests for infra/dais/campaign_check.py's verdict on failed versions."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "infra/dais/campaign_check.py"
SPEC = importlib.util.spec_from_file_location("campaign_check", SCRIPT)
cc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cc)

DONE = ["export", "agent", "collect", "check", "sweep_config", "sweep_snap", "image_check"]


def test_failures_of_the_agent_and_of_the_infrastructure():
    check = {"status": "failed", "done": DONE[:3], "error": "check_instance.sh failed:\nx"}
    assert cc.failure(check) == ("check", "check_instance.sh failed:", True)
    sims = {"status": "failed", "done": DONE, "error": "x wrote no sweep.json in 2 jobs"}
    assert cc.failure(sims)[0] == "sweep_jobs" and not cc.failure(sims)[2]
    agent = {"status": "failed", "done": DONE[:1], "error": "the agent failed 3 times",
             "data": {"agent_runs": [{"result": {"errors": [
                 json.dumps({"name": "APIError", "statusCode": 502})]}}]}}
    assert cc.failure(agent)[:2] == ("agent", "the agent failed 3 times")
    assert not cc.failure(agent)[2]
    agent["data"]["agent_runs"][0]["result"]["errors"] = ["no rule written"]
    assert cc.failure(agent)[2]
    # no attempt of the agent's: all its runs were lost
    lost = {"status": "failed", "done": DONE[:1], "error": "4 agent runs lost: x",
            "data": {"lost_runs": [{}] * 4}}
    assert not cc.failure(lost)[2]
    # check_instance.sh could not read the worktree: the driver's side
    driver = {"status": "failed", "done": DONE[:3],
              "error": "check_instance.sh failed:\nFAIL: cannot diff a..b"}
    assert not cc.failure(driver)[2]


def test_limit_stops_are_reported_and_can_gate(tmp_path, monkeypatch, capsys):
    import sys
    wt = tmp_path / "worktrees"
    (wt / "pfx-1/.claude").mkdir(parents=True)
    (wt / "pfx-1/.claude/policy_finder.json").write_text(json.dumps({"version": 1}))
    runs = [{"run_id": "apt-1", "status": "expired", "time_limit_s": 5400,
             "opencode_time_limit_s": 4800, "result": {"seconds": 4800, "timed_out": True}},
            {"run_id": "apt-2", "status": "failed", "time_limit_s": 5400,
             "opencode_time_limit_s": 4800, "result": {"seconds": 900, "length_stops": 1}}]
    (wt / "pfx-1.state.json").write_text(json.dumps({
        "status": "failed", "done": DONE[:1], "error": "the agent failed 2 times",
        "data": {"agent_runs": runs}}))
    monkeypatch.setenv("PF_RUNS", str(tmp_path))
    out = tmp_path / "limits.json"
    monkeypatch.setattr(sys, "argv", ["campaign_check.py", "resume", "pfx-1", "--expect",
                                      "finished", "--no-limit-stops", "--json-out", str(out)])
    assert cc.main() == 1  # the gate: runs stopped at a limit
    report = json.loads(out.read_text())
    assert report["time_limit_stops"] == ["pfx-1 apt-1"]
    assert report["output_limit_stops"] == ["pfx-1 apt-2"]
    assert report["versions_lost_with_limit_stops"] == ["pfx-1 (output limit, time limit)"]
    assert report["agent_run_seconds"]["limits"][0]["longest_share_of_opencode_limit"] == 1.0
    assert "OpenCode stopped at 4800 s" in capsys.readouterr().out


def test_loop_stops_and_expired_runs_are_counted_in_a_scored_version(
        tmp_path, monkeypatch, capsys):
    import sys
    wt = tmp_path / "worktrees"
    (wt / "pfx-1/.claude").mkdir(parents=True)
    (wt / "pfx-1/.claude/policy_finder.json").write_text(json.dumps({"version": 1}))
    loop = {"tool": "write", "error": "SchemaError", "repeats": 20}
    runs = [{"run_id": "apt-1", "status": "failed", "result": {"tool_loop": loop}},
            {"run_id": "apt-2", "status": "expired", "result": {}},
            {"run_id": "apt-3", "status": "done", "result": {"seconds": 60}}]
    (wt / "pfx-1.state.json").write_text(json.dumps({
        "status": "scored", "done": DONE, "data": {"agent_runs": runs}}))
    monkeypatch.setenv("PF_RUNS", str(tmp_path))
    out = tmp_path / "limits.json"
    monkeypatch.setattr(sys, "argv", ["campaign_check.py", "resume", "pfx-1", "--expect",
                                      "scored", "--no-expired", "--json-out", str(out)])
    assert cc.main() == 1  # the expired run of a version that scored still counts
    assert json.loads(out.read_text())["tool_loop_stops"] == ["pfx-1 apt-1 (20x write)"]
    assert "runs stopped in a loop of failing tool calls: 1" in capsys.readouterr().out
