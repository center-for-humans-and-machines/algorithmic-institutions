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
