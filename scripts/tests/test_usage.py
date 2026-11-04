"""Tests for scripts/policy_finder/usage.py: the plan-usage gate and limit detection."""

import datetime as dt
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "policy_finder/usage.py"
SPEC = importlib.util.spec_from_file_location("usage", SCRIPT)
us = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(us)


def _in(minutes):
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=minutes)).isoformat()


def test_gate_waits_until_the_window_resets():
    readings = [{"session": (85.0, _in(30)), "weekly": (10.0, _in(3000))},
                {"session": (85.0, _in(20)), "weekly": (10.0, _in(3000))},
                {"session": (3.0, _in(300)), "weekly": (11.0, _in(3000))}]
    sleeps, logs = [], []
    usage = us.gate(log=logs.append, read_usage=lambda: readings.pop(0),
                    sleep=sleeps.append)
    assert usage["session"][0] == 3.0
    assert len(sleeps) == 2 and all(60 <= s <= us.POLL for s in sleeps)
    assert "session 85%" in logs[0]


def test_gate_passes_below_the_ceiling_and_when_unknown():
    sleeps = []
    assert us.gate(read_usage=lambda: {"session": (79.0, None)}, sleep=sleeps.append)
    assert us.gate(read_usage=lambda: None, sleep=sleeps.append) is None
    assert sleeps == []


def test_weekly_ceiling():
    assert us.over({"session": (10.0, None), "weekly": (95.0, _in(60))})[0][0] == "weekly"


def test_read_is_off_without_the_agents_credentials(monkeypatch):
    monkeypatch.setattr(us, "CREDENTIALS", "")
    assert us.read() is None


def test_reset_wait():
    assert us.reset_wait(["Claude AI usage limit reached|1793800600"], now=1793800000) == 660
    assert us.reset_wait(["You've hit your limit"], now=1793800000) == us.POLL
    assert us.reset_wait(["usage limit reached|1793799000"], now=1793800000) == us.POLL


def test_limited():
    failed = {"status": "failed", "result": {"is_error": True}, "report": ""}
    assert us.limited({**failed, "report": "Claude AI usage limit reached|1793802604"})
    assert us.limited({**failed, "stderr_tail": "API Error: 429 Too Many Requests"})
    assert us.limited({**failed, "report": "You've hit your limit · resets 1pm"})
    assert not us.limited({**failed, "report": "no rule: the sweep would take 0.429 h"})
    assert not us.limited({**failed, "stderr_tail": "Traceback ... ValueError"})


def test_limited_from_the_transcripts(tmp_path):
    session = tmp_path / ".claude/projects/-opt/s/subagents/a.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text(
        json.dumps({"message": {"content": "pool 0.429 at rate limit of 4"}}) + "\n"
        + json.dumps({"isApiErrorMessage": True,
                      "message": {"content": [{"text": "API Error: Rate limit reached"}]}})
        + "\n")
    record = {"status": "succeeded", "result": {}, "report": "the subagent stopped"}
    assert us.limited(record, home=str(tmp_path))
    session.write_text(json.dumps({"message": {"content": "rate limit of 4"}}) + "\n")
    assert not us.limited(record, home=str(tmp_path))
