"""Tests for scripts/policy_finder/opencode_run.py: the summary and the redaction."""

import importlib.util
import json
import os
import sys
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


# -- main(), with a fake `opencode` on PATH and a dummy key ---------------------

KEY = "sk-or-v1-" + "ab12" * 16
FAKE = r'''#!/usr/bin/env python3
import json, os, sys, time
mode = os.environ["FAKE_MODE"]
key = os.environ.get("OPENROUTER_API_KEY", "")
if sys.argv[1] == "run":
    print(json.dumps({"type": "step_start", "sessionID": "ses_1", "text": key}), flush=True)
    if mode == "hang":
        time.sleep(60)
    sys.exit(1 if mode == "exit1" else 0)
if sys.argv[1] == "export":
    if mode == "badexport":
        print("[]")
    elif mode != "noexport":
        print(json.dumps({"info": {"agent": "policy-finder"}, "messages": [
            {"info": {"role": "assistant", "providerID": "openrouter", "modelID": "q/c",
                      "tokens": {"input": 5, "output": 1}},
             "parts": [{"type": "text", "text": "report, key " + key}]}]}))
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "opencode").write_text(FAKE.replace("/usr/bin/env python3", sys.executable))
    (bin_dir / "opencode").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    repo = tmp_path / "repo"
    (repo / ".opencode").mkdir(parents=True)
    (repo / ".opencode/opencode.json").write_text('{"model": "openrouter/q/c"}')
    (repo / "notes").mkdir()
    (repo / "notes/n.md").write_text(f"my env said {KEY}\n")
    data = tmp_path / "home/.local/share/opencode"
    (data / "log").mkdir(parents=True)
    (data / "log/x.log").write_text(f"log {KEY}\n")
    (data / "opencode.db").write_text(KEY)
    monkeypatch.setattr(oc, "DATA", str(data))
    monkeypatch.setattr(oc.time, "sleep", lambda s: None)
    out = tmp_path / "home/opencode"

    def run(mode, *extra):
        monkeypatch.setenv("FAKE_MODE", mode)
        return oc.main(["Design.", "--title", "t", "--out", str(out), "--repo", str(repo),
                        "--write-dir", "notes", *extra])
    return run, out, repo, data


def _summary(capsys):
    printed = capsys.readouterr().out
    assert KEY not in printed
    return json.loads(printed.strip().splitlines()[-1])


def _clean(out, repo, data):
    for path in [*out.rglob("*"), *repo.rglob("*.md")]:
        if path.is_file():
            assert KEY not in path.read_text(errors="replace"), path
    assert not list(data.rglob("*.db")) and not list((data / "log").iterdir())


def test_main_success_redacts_everything(fake, capsys):
    run, out, repo, data = fake
    assert run("ok") == 0
    s = _summary(capsys)
    assert s["failures"] == [] and s["result"] == "report, key [redacted]"
    _clean(out, repo, data)
    # the salted hash beside the out folder, for the pipeline's scrub
    info = json.loads((out.parent / oc.KEY_HASH).read_text())
    assert oc.key_hash(KEY, info["salt"])["sha256"] == info["sha256"]


@pytest.mark.parametrize("mode, reason", [
    ("exit1", "exited with 1"),
    ("noexport", "no text"),
    ("badexport", "no text"),
])
def test_main_failures_exit_non_zero(fake, capsys, mode, reason):
    run, out, repo, data = fake
    assert run(mode) != 0
    s = _summary(capsys)
    assert s["is_error"] and any(reason in f for f in s["failures"])
    _clean(out, repo, data)


def test_main_timeout_kills_and_still_redacts(fake, capsys):
    run, out, repo, data = fake
    assert run("hang", "--timeout", "1") == 124
    s = _summary(capsys)
    assert s["timed_out"] and any("exited with 124" in f for f in s["failures"])
    _clean(out, repo, data)


def test_main_cap_with_unknown_usage_does_not_start(fake, capsys):
    run, out, repo, data = fake
    assert run("ok", "--usage-url", "http://127.0.0.1:9/key", "--max-usage", "5") == 3
    s = _summary(capsys)
    assert s["budget_exhausted"] and "unknown" in s["error"]
    assert not (out / "events.jsonl").exists()
