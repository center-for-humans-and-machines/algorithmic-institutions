"""Tests for scripts/policy_finder/opencode_run.py: the summary and the redaction."""

import importlib.util
from pathlib import Path

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
