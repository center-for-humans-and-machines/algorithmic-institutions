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


def test_an_expired_run_is_a_counted_attempt(version, monkeypatch):
    runs(monkeypatch, version, [("expired", "")] * pl.OPENCODE_ATTEMPTS)
    with pytest.raises(pl.Failed):
        version.step_agent()
    assert len(version.data["agent_runs"]) == pl.OPENCODE_ATTEMPTS


def test_the_budget_cap_stops_the_version(version, monkeypatch):
    monkeypatch.setattr(pl.agent_runner, "start", lambda *a, **k: "r1")

    def finish(run_id, log=print, harness="claude"):
        raise pl.agent_runner.BudgetError("key usage 21.4 USD is at the cap")
    monkeypatch.setattr(pl.agent_runner, "finish", finish)
    monkeypatch.setattr(pl, "STEPS", ("agent",))
    state = version.run()
    assert state["status"] == "stopped" and "at the cap" in state["error"]
    assert "agent_current" not in version.data


def test_every_rerun_starts_from_a_fresh_export(version, monkeypatch):
    builds = []
    monkeypatch.setattr(pl.export_instance, "build", lambda *a: builds.append(1))
    outcomes = iter(["endpoint", "lost", "expired", "failed", "succeeded"])

    def start(*a, **k):
        return f"r{len(builds)}"

    def finish(run_id, log=print, harness="claude"):
        what = next(outcomes)
        record = {"status": what, "report": "done" if what == "succeeded" else "",
                  "result": {}, "models": []}
        if what == "endpoint":
            raise pl.agent_runner.EndpointError("server down", dict(record, status="failed"))
        return record
    monkeypatch.setattr(pl.agent_runner, "start", start)
    monkeypatch.setattr(pl.agent_runner, "finish", finish)
    rule = version.exports / version.stem / version.rule
    rule.parent.mkdir(parents=True)
    rule.write_text("params: {}\n")
    monkeypatch.setattr(pl, "OPENCODE_ATTEMPTS", 5)
    # the first run fails on the endpoint: the version stops (resumable)
    with pytest.raises(pl.agent_runner.EndpointError):
        version.step_agent()
    assert builds == [] and len(version.data["endpoint_runs"]) == 1
    # the resume: each of the next four runs (after an endpoint failure, a lost
    # run, an expired one and a failed one) starts from a fresh export
    version.step_agent()
    assert len(builds) == 4


def test_a_start_that_failed_is_followed_by_a_fresh_export(version, monkeypatch):
    builds = []
    monkeypatch.setattr(pl.export_instance, "build", lambda *a: builds.append(1))
    starts = iter([pl.agent_runner.AgentError("worker.json not written"), "r1"])

    def start(*a, **k):
        what = next(starts)
        if isinstance(what, Exception):
            raise what
        return what
    monkeypatch.setattr(pl.agent_runner, "start", start)
    monkeypatch.setattr(pl.agent_runner, "finish", lambda *a, **k: {
        "status": "succeeded", "report": "done", "result": {}, "models": []})
    rule = version.exports / version.stem / version.rule
    rule.parent.mkdir(parents=True)
    rule.write_text("params: {}\n")
    with pytest.raises(pl.agent_runner.AgentError):
        version.step_agent()
    version.step_agent()
    assert builds == [1]


@pytest.mark.parametrize("setting, mentions_switching", [("exp2", True), ("exp1", False)])
def test_the_seed_notes_name_the_settings_game(tmp_path, monkeypatch, setting,
                                               mentions_switching):
    monkeypatch.setenv("PF_RUNS", str(tmp_path))
    monkeypatch.delenv("PF_WORKTREE_ROOT", raising=False)
    conf = tmp_path / "worktrees/pf-s/.claude/policy_finder.json"
    conf.parent.mkdir(parents=True)
    conf.write_text(json.dumps({"name": "pf-s", "rule_name": "pf-s", "version": 1,
                                "mode": "single_idea", "setting": setting}))
    v = pl.Version("pf-s")
    v.step_export()
    notes = (v.wt / "notes/policy_finder/pf-s.md").read_text()
    assert notes.startswith("# pf-s\n")
    assert ("switch decisions" in notes) == mentions_switching
    assert (v.wt / v.rule).read_text() == (pl.SEEDS / "single_idea.yml").read_text()


def test_single_idea_notes_the_check_rejects_cost_an_attempt(version, monkeypatch):
    version.conf.update(mode="single_idea", base_version="pf-x")
    notes = version.exports / version.stem / f"notes/policy_finder/{version.stem}.md"
    notes.parent.mkdir(parents=True)
    rule = version.exports / version.stem / version.rule
    rule.parent.mkdir(parents=True)
    rule.write_text("params: {}\n")
    good = (Path(__file__).resolve().parent / "fixtures/single_idea_notes/si-ds-2.md"
            ).read_text().replace("pfsmoke-20261127-1", "pf-x")
    writes = [good.replace("- Operation: add", "- Operation: none"), good]

    def build(*a):  # the fresh export for the retry: the agent writes its notes
        pass
    monkeypatch.setattr(pl.export_instance, "build", build)
    queue = [("succeeded", "first report"), ("succeeded", "second report")]
    monkeypatch.setattr(pl.agent_runner, "start", lambda *a, **k: f"r{len(queue)}")

    def finish(run_id, log=print, harness="claude"):
        notes.write_text(writes.pop(0))
        status, report = queue.pop(0)
        return {"status": status, "report": report, "result": {}, "models": []}
    monkeypatch.setattr(pl.agent_runner, "finish", finish)
    version.step_agent()
    runs = version.data["agent_runs"]
    assert len(runs) == 2 and "no change stated" in runs[0]["note_rejected"][0]
    assert "note_rejected" not in runs[1]
    assert version.sleeps == []  # no rate-limit back-off after a rejected note
    assert (version.wt_root / "pf-x.report.md").read_text() == "second report\n"


def test_free_mode_takes_the_notes_as_they_are(version, monkeypatch):
    rule = version.exports / version.stem / version.rule
    rule.parent.mkdir(parents=True)
    rule.write_text("params: {}\n")
    runs(monkeypatch, version, [("succeeded", "the report")])
    version.step_agent()
    assert len(version.data["agent_runs"]) == 1


def test_notes_that_are_not_utf8_are_a_rejected_note(version, monkeypatch):
    version.conf.update(mode="single_idea", base_version="pf-x")
    notes = version.exports / version.stem / f"notes/policy_finder/{version.stem}.md"
    notes.parent.mkdir(parents=True)
    rule = version.exports / version.stem / version.rule
    rule.parent.mkdir(parents=True)
    rule.write_text("params: {}\n")
    notes.write_bytes(b"# pf-x\n\n## Explorations\n\xff\xfe broken\n")
    runs(monkeypatch, version, [("succeeded", "a report")] * pl.OPENCODE_ATTEMPTS)
    with pytest.raises(pl.Failed):
        version.step_agent()
    assert all(r.get("note_rejected") for r in version.data["agent_runs"])
