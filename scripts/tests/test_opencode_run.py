"""Tests for scripts/policy_finder/opencode_run.py: the summary and the redaction."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "policy_finder/opencode_run.py"
SPEC = importlib.util.spec_from_file_location("opencode_run", SCRIPT)
oc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oc)


def _assistant(text, input_tokens, cached, tools=0):
    parts = [{"type": "tool"}] * tools + [{"type": "step-finish"}]
    if text:
        parts.append({"type": "text", "text": text})
    return {"info": {"role": "assistant", "modelID": "q/c", "providerID": "openrouter",
                     "cost": 0.5, "tokens": {"input": input_tokens, "output": 3,
                                             "reasoning": 1,
                                             "cache": {"read": cached, "write": 0}}},
            "parts": parts}


def test_summarize_sums_the_assistant_messages():
    session = {"messages": [
        {"info": {"role": "user"}, "parts": [{"type": "text", "text": "Design."}]},
        _assistant("thinking aloud", 100, 0, tools=2),
        _assistant("", 120, 90, tools=1),
        _assistant("the report", 130, 110),
    ]}
    s = oc.summarize(session)
    assert s["result"] == "the report"
    assert s["tokens"] == {"input": 350, "output": 9, "reasoning": 3, "cache_read": 200,
                           "cache_write": 0}
    assert s["cost_catalog_usd"] == 1.5 and s["models"] == ["openrouter/q/c"]
    assert (s["steps"], s["tool_calls"]) == (3, 3)


def test_redact_tree(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a/notes.md").write_text("key sk-or-v1-secret here\n")
    (tmp_path / "clean.txt").write_text("nothing\n")
    hits = oc.redact_tree(tmp_path, "sk-or-v1-secret")
    assert hits == [str(tmp_path / "a/notes.md")]
    assert (tmp_path / "a/notes.md").read_text() == "key [redacted] here\n"
    assert oc.redact_tree(tmp_path, "") == []


GOOD = {"result": "the report", "models": ["openrouter/q/c"], "agent": "policy-finder",
        "final_error": None}


def test_the_report_is_the_final_message():
    session = {"messages": [_assistant("Let me look at the data.", 10, 0, tools=1),
                            _assistant("", 10, 0, tools=1)]}
    assert oc.summarize(session)["result"] == ""


@pytest.mark.parametrize("change, code, stderr, event, reason", [
    ({}, 1, "", None, "exited with 1"),
    ({}, 0, 'agent "policy-finder" not found. Falling back to default agent', None,
     "did not load"),
    ({"agent": "build"}, 0, "", None, "ran agent build"),
    ({"final_error": {"name": "APIError"}}, 0, "", None, "ended in an error"),
    ({}, 0, "", '{"name": "APIError"}', "reported an error"),
    ({"result": "  "}, 0, "", None, "no text"),
    ({"models": ["openrouter/other/m"]}, 0, "", None, "not openrouter/q/c"),
])
def test_failures(change, code, stderr, event, reason):
    assert oc.failures(GOOD, 0, "", "openrouter/q/c", None) == []
    why = oc.failures({**GOOD, **change}, code, stderr, "openrouter/q/c", event)
    assert len(why) == 1 and reason in why[0]


def test_a_local_model_is_compared_by_its_served_name():
    good = {**GOOD, "models": ["local/Qwen/Qwen3-Coder-Next"]}
    assert oc.failures(good, 0, "", "local/Qwen/Qwen3-Coder-Next", None) == []


def test_last_event_error(tmp_path):
    events = tmp_path / "e.jsonl"
    events.write_text('{"type": "error", "error": {"name": "Retry"}}\n'
                      '{"type": "step_finish"}\n')
    assert oc.last_event_error(events) is None
    with open(events, "a") as f:
        f.write('{"type": "error", "error": {"name": "APIError"}}\n')
    assert "APIError" in oc.last_event_error(events)
