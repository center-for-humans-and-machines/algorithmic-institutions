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


def test_the_report_is_the_final_turns_last_text():
    user = {"info": {"role": "user"}, "parts": [{"type": "text", "text": "Design."}]}
    # the report, then a todowrite call, then an empty final message: a success
    session = {"messages": [user, _assistant("thinking aloud", 10, 0, tools=2),
                            _assistant("the report", 10, 0, tools=1),
                            _assistant("", 10, 0)]}
    s = oc.summarize(session)
    assert s["result"] == "the report"
    assert oc.failures({**s, "agent": "policy-finder"}, 0, "", "openrouter/q/c", None) == []
    # text of an earlier turn is not this turn's report
    session = {"messages": [_assistant("old report", 10, 0), user, _assistant("", 10, 0)]}
    assert oc.summarize(session)["result"] == ""


def test_interim_text_then_an_error_fails():
    failed = _assistant("", 10, 0)
    failed["info"]["error"] = {"name": "APIError", "data": {"message": "429"}}
    session = {"messages": [_assistant("Let me look at the data.", 10, 0, tools=1), failed]}
    s = {**oc.summarize(session), "agent": "policy-finder"}
    assert s["result"] == "Let me look at the data."
    why = oc.failures(s, 0, "", "openrouter/q/c", None)
    assert len(why) == 1 and "ended in an error" in why[0]


@pytest.mark.parametrize("change, code, stderr, event, reason", [
    ({}, 1, "", None, "exited with 1"),
    ({}, 0, 'agent "policy-finder" not found. Falling back to default agent', None,
     "did not load"),
    ({"agent": "build"}, 0, "", None, "ran agent build"),
    ({"final_error": {"name": "APIError"}}, 0, "", None, "ended in an error"),
    ({}, 0, "", '{"name": "APIError"}', "reported an error"),
    ({"result": "  "}, 0, "", None, "no assistant text"),
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
    ("noexport", "no assistant text"),
    ("badexport", "no assistant text"),
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



class _Answer:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps({"data": self.data}).encode()


def _endpoint(monkeypatch, *answers):
    """OpenRouter's key endpoint: each call takes the next answer (an
    exception is raised)."""
    queue = list(answers)
    calls = []

    def urlopen(req, timeout=None):
        calls.append(req.full_url)
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return _Answer(answer)
    monkeypatch.setattr(oc.urllib.request, "urlopen", urlopen)
    return calls


URL = ["--usage-url", "https://openrouter.ai/api/v1/key"]


def test_a_key_at_its_credit_limit_does_not_start(fake, capsys, monkeypatch):
    run, out, repo, data = fake
    _endpoint(monkeypatch, {"usage": 5.0, "limit": 5, "limit_remaining": 0})
    assert run("ok", *URL) == 3
    s = _summary(capsys)
    assert s["budget_exhausted"] and "credit limit" in s["error"]


def test_require_key_limit_refuses_a_non_numeric_limit(fake, capsys, monkeypatch):
    run, out, repo, data = fake
    _endpoint(monkeypatch, {"usage": 1.0, "limit": "abc", "limit_remaining": None})
    assert run("ok", *URL, "--require-key-limit") == 3
    assert "numeric credit limit" in _summary(capsys)["error"]


def test_a_network_blip_is_retried_and_an_unknown_limit_warns(fake, capsys, monkeypatch):
    run, out, repo, data = fake
    calls = _endpoint(monkeypatch, OSError("blip"),
                      {"usage": 1.0, "limit": None, "limit_remaining": None})
    assert run("ok", *URL, "--max-usage", "5") == 0
    assert len(calls) >= 2
    s = _summary(capsys)
    assert "no credit limit" in s["warnings"][0]
    _endpoint(monkeypatch, OSError("down"))
    assert oc.key_info(URL[1], wait=0) is None


def test_an_unwritable_file_does_not_stop_the_redaction(fake, capsys):
    run, out, repo, data = fake
    locked = repo / "notes/locked.md"
    locked.write_text(f"key {KEY}\n")
    locked.chmod(0o444)
    try:
        assert run("ok") == 0
    finally:
        locked.chmod(0o644)
    s = _summary(capsys)
    assert any("locked.md" in e for e in s["redaction_errors"])
    assert KEY not in (repo / "notes/n.md").read_text()
    assert not list(data.rglob("*.db"))


def test_a_local_server_key_is_redacted_and_openrouter_never_called(
        fake, capsys, monkeypatch):
    run, out, repo, data = fake
    monkeypatch.setattr(oc.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("called the network"))
    monkeypatch.setenv("PF_LLM_API_KEY", KEY)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    (repo / ".opencode/opencode.json").write_text('{"model": "openrouter/q/c"}')
    assert run("ok", "--key-env", "PF_LLM_API_KEY", "--max-usage", "5") == 0
    _summary(capsys)
    for path in [*out.rglob("*"), *repo.rglob("*.md")]:
        if path.is_file():
            assert KEY not in path.read_text(errors="replace")
