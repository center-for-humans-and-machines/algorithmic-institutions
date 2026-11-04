"""Tests for scripts/policy_finder/campaign.py, with new_instance.sh and pipeline.py faked."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "policy_finder/campaign.py"
SPEC = importlib.util.spec_from_file_location("campaign", SCRIPT)
cm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cm)


class Fake:
    """Plays new_instance.sh and pipeline.py: writes the instance file and state.

    `outcomes[stem]` is a list of what each run of that version ends in:
    ("scored", pool), ("failed",), ("error",) or ("crash",) (writes nothing).
    """

    def __init__(self, wt_root, outcomes, default_pool=None):
        self.wt_root, self.outcomes, self.default_pool = wt_root, outcomes, default_pool
        self.commands = []

    def __call__(self, args, out):
        args = [str(a) for a in args]
        self.commands.append(args)
        pipeline = len(args) > 1 and args[1].endswith("pipeline.py")
        if pipeline:
            name = args[3]
        elif "--iterate-on" in args:
            name = args[args.index("--iterate-on") + 1]
        else:
            name = args[1]
        conf_path = self.wt_root / name / ".claude/policy_finder.json"
        if not pipeline:
            version = json.loads(conf_path.read_text())["version"] + 1 if conf_path.exists() else 1
            conf_path.parent.mkdir(parents=True, exist_ok=True)
            stem = cm.stem_of(name, version)
            conf_path.write_text(json.dumps({"name": name, "version": version, "rule_name": stem}))
        stem = json.loads(conf_path.read_text())["rule_name"]
        queue = self.outcomes.get(stem) or [("scored", self.default_pool(stem))]
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if outcome[0] == "crash":
            return 1
        state_path = self.wt_root / f"{stem}.state.json"
        state = {"status": outcome[0], "done": ["export", "agent"], "data": {
            "agent_runs": [{"result": {"total_cost_usd": 2.0}}]}}
        if outcome[0] == "scored":
            state["data"]["score"] = {"pool": outcome[1], "pool_se": 1.0,
                                      "best_of_n": outcome[1] + 1, "best_of_n_se": 1.0,
                                      "n_points": 6}
        if outcome[0] == "error":
            state["error"] = f"network {len(self.commands)}"
        state_path.write_text(json.dumps(state))
        return 0 if outcome[0] == "scored" else 1


def make(tmp_path, monkeypatch, outcomes=None, pools=None, **settings):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "test-campaign.yml"
    path.write_text(json.dumps({"prefix": "pf-20990101", "n_fresh": 4, "generations": 2,
                                "top_k": 2, "parallel": 3, "retry_wait": 0, **settings}))
    campaign = cm.Campaign(path, pf_runs=tmp_path / "runs")
    pools = pools or {}
    fake = Fake(campaign.wt_root, outcomes or {},
                default_pool=lambda stem: pools.get(stem, 60.0))
    monkeypatch.setattr(campaign, "execute", fake)
    monkeypatch.setattr(cm.time, "sleep", lambda s: None)
    return campaign, fake


def stems(campaign):
    return [[cm.stem_of(m["name"], m["version"]) for m in g["members"]]
            for g in campaign.state["generations"]]


def test_selection_iterates_the_best_lineages(tmp_path, monkeypatch):
    pools = {"pf-20990101-1": 50, "pf-20990101-2": 70, "pf-20990101-3": 65,
             "pf-20990101-4": 55, "pf-20990101-2-v2": 60, "pf-20990101-3-v2": 72}
    campaign, fake = make(tmp_path, monkeypatch, pools=pools)
    campaign.run()
    assert stems(campaign) == [
        ["pf-20990101-1", "pf-20990101-2", "pf-20990101-3", "pf-20990101-4"],
        ["pf-20990101-2-v2", "pf-20990101-3-v2"],
        # lineage bests: 3 -> 72, 2 -> 70 (its v2 at 60 does not lower it)
        ["pf-20990101-3-v3", "pf-20990101-2-v3"],
    ]
    fresh = [c for c in fake.commands if c[1:2] == ["pf-20990101-1"]][0]
    assert fresh[2:] == ["--min-params", "1", "--max-params", "5", "--sobol-points", "256"]
    assert ["--iterate-on", "pf-20990101-2"] in [c[1:3] for c in fake.commands[4:6]]
    summary = (campaign.dir / "test-campaign.summary.md").read_text()
    assert "| 3 | pf-20990101-3-v3 | scored | 60.00 ± 1.00 | 61.00 ± 1.00 | 2.0 |" in summary


def test_a_failed_version_keeps_its_slot(tmp_path, monkeypatch):
    pools = {"pf-20990101-1": 70, "pf-20990101-2": 65, "pf-20990101-3": 50, "pf-20990101-4": 50}
    campaign, fake = make(tmp_path, monkeypatch, pools=pools,
                          outcomes={"pf-20990101-1-v2": [("failed",)]})
    campaign.run()
    assert stems(campaign)[2] == ["pf-20990101-1-v3", "pf-20990101-2-v3"]
    iterate = [c for c in fake.commands if c[1:3] == ["--iterate-on", "pf-20990101-1"]]
    assert "--after-failure" not in iterate[0] and "--after-failure" in iterate[1]
    rows = {r["stem"]: r for r in campaign.rows()}
    assert rows["pf-20990101-1-v2"]["failed_step"] == "collect"


def test_errors_are_resumed_then_stop_the_campaign(tmp_path, monkeypatch):
    campaign, fake = make(tmp_path, monkeypatch,
                          outcomes={"pf-20990101-2": [("error",), ("error",), ("scored", 61)]})
    campaign.run()
    resumes = [c for c in fake.commands if c[1].endswith("pipeline.py") and c[3] == "pf-20990101-2"]
    assert len(resumes) == 2

    campaign, fake = make(tmp_path / "b", monkeypatch, error_retries=1,
                          outcomes={"pf-20990101-3": [("error",)]})
    with pytest.raises(cm.CampaignError, match="pf-20990101-3: error after 1"):
        campaign.run()
    # the other members finished, and the generation is not marked finished
    assert len([r for r in campaign.rows() if r["status"] == "scored"]) == 3
    assert "finished" not in campaign.state["generations"][0]


def test_resume_after_a_kill(tmp_path, monkeypatch):
    campaign, fake = make(tmp_path, monkeypatch,
                          outcomes={"pf-20990101-4": [("crash",)]})
    with pytest.raises(cm.CampaignError, match="start failed"):
        campaign.run()
    # a new driver on the same state: finished versions are not run again
    campaign2, fake2 = make(tmp_path, monkeypatch)
    assert stems(campaign2)[0] == stems(campaign)[0]
    campaign2.run()
    # only the crashed lineage runs again, and its instance exists: the pipeline
    first = fake2.commands[0]
    assert first[1].endswith("pipeline.py") and first[3] == "pf-20990101-4"
    reruns = [c for c in fake2.commands if "--iterate-on" not in c
              and c[-1] in campaign2.lineages()[:3]]
    assert not reruns
    assert len(campaign2.state["generations"]) == 3


def test_a_running_version_resumes_with_the_pipeline(tmp_path, monkeypatch):
    campaign, fake = make(tmp_path, monkeypatch, generations=0)
    wt = campaign.wt_root / "pf-20990101-1"
    (wt / ".claude").mkdir(parents=True)
    (wt / ".claude/policy_finder.json").write_text(
        json.dumps({"version": 1, "rule_name": "pf-20990101-1"}))
    (campaign.wt_root / "pf-20990101-1.state.json").write_text(
        json.dumps({"status": "running", "done": ["export"], "data": {}}))
    campaign.run()
    assert [c for c in fake.commands if "pf-20990101-1" in c][0][1].endswith("pipeline.py")
