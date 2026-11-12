"""Annotation failure paths: HTTP client aborts, failure budgets, stale inputs."""

import json

import httpx
import pytest
import yaml

from aimanager.annotation.cli import main, review_begin
from aimanager.annotation.client import Call, LLMClient
from aimanager.annotation.cluster import to_codebook_entries
from aimanager.annotation.common import (
    Run,
    latest_codebook,
    load_config,
    write_json,
    write_jsonl,
    write_manifest,
)
from aimanager.annotation.expert import parse_present
from aimanager.annotation.schemas import FreeCodes, ProposedCode, assignment_model
from aimanager.annotation.steps import open_rules, what_codebook

GOOD = json.dumps({"mechanisms": [{"description": "d", "quote": "q"}]})


def _reply(content, reasoning=None):
    msg = {"content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {}}


def _client(tmp_path, handler, use_cache=True, **llm):
    cfg = load_config()
    cfg["llm"].update({"backoff": [0, 0, 0], "concurrency": 1, **llm})
    run = Run(tmp_path, "r")
    client = LLMClient(
        run,
        cfg,
        base_url="http://x/v1",
        model="m",
        transport=httpx.MockTransport(handler),
        use_cache=use_cache,
    )
    return client, cfg, run


def _call(doc, step="s"):
    return Call(step, doc, [{"role": "user", "content": doc}], FreeCodes, "open_rules")


def _cache_files(run):
    return list((run.dir / "cache").rglob("*.json"))


def test_exhausted_5xx_is_a_failed_result(tmp_path):
    seen = []

    def handler(request):
        seen.append(1)
        return httpx.Response(503)

    client, _, run = _client(tmp_path, handler, max_consecutive_server_errors=100)
    (res,) = client.map([_call("a")])
    assert res.error.startswith("server_error") and len(seen) == 4
    assert not _cache_files(run)


def test_consecutive_server_errors_abort_the_map(tmp_path):
    seen = []

    def handler(request):
        seen.append(1)
        raise httpx.ConnectError("refused", request=request)

    client, _, _ = _client(tmp_path, handler, max_consecutive_server_errors=3)
    with pytest.raises(SystemExit, match="in a row"):
        client.map([_call(str(i)) for i in range(20)])
    assert len(seen) == 3  # no queued call ran after the abort


def test_read_timeout_is_retried_once(tmp_path):
    seen = []

    def handler(request):
        seen.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    client, _, _ = _client(tmp_path, handler, max_consecutive_server_errors=100)
    (res,) = client.map([_call("a")])
    assert "ReadTimeout" in res.error and len(seen) == 2


def test_malformed_200_is_retried_and_invalid_replies_not_cached(tmp_path):
    seen = []

    def handler(request):
        seen.append(1)
        doc = json.loads(request.content)["messages"][-1]["content"]
        if doc == "html" and len(seen) == 1:
            return httpx.Response(200, text="<html>bad gateway</html>")
        if doc == "invalid":
            return httpx.Response(200, json=_reply('{"nope": 1}'))
        return httpx.Response(200, json=_reply("<think>hm</think>\n" + GOOD))

    client, _, run = _client(tmp_path, handler)
    html, invalid = client.map([_call("html"), _call("invalid")])
    assert html.ok and html.info["think_stripped"] and client.n_think_stripped == 1
    assert invalid.error.startswith("parse_error")
    assert len(_cache_files(run)) == 1  # only the valid reply


def test_empty_content_with_reasoning_is_a_hard_error(tmp_path):
    def handler(request):
        return httpx.Response(200, json=_reply("", reasoning="let me think"))

    client, _, _ = _client(tmp_path, handler)
    with pytest.raises(SystemExit, match="llm.extra_body"):
        client.map([_call("a")])


def test_client_errors_abort(tmp_path):
    def handler(request):
        return httpx.Response(400, text="bad schema")

    client, _, _ = _client(tmp_path, handler, max_client_errors=2)
    with pytest.raises(SystemExit, match="4xx"):
        client.map([_call(str(i)) for i in range(5)])


def _mini_run(tmp_path):
    run = Run(tmp_path, "m")
    versions = [
        {
            "stem": s,
            "campaign": "pf-1",
            "lineage": s,
            "version": 1,
            "parent": None,
            "rule": f"rule {s}",
            "notes": "n",
            "setting": "a setting",
        }
        for s in ("a", "b")
    ]
    files = [run.path("corpus", "versions.jsonl"), run.path("corpus", "split.json")]
    write_jsonl(files[0], versions)
    write_json(files[1], {"free_coding": ["a", "b"], "expert_check": []})
    write_manifest(run, "corpus", [], files)
    return run


def test_step_over_budget_raises_and_writes_nothing(tmp_path):
    def handler(request):
        doc = json.loads(request.content)["messages"][-1]["content"]
        return httpx.Response(200, json=_reply("garbage" if "rule a" in doc else GOOD))

    client, cfg, _ = _client(tmp_path, handler)
    run = _mini_run(tmp_path)
    client.run = run
    with pytest.raises(SystemExit, match="failure_budget.open_rules"):
        open_rules(run, cfg, client)
    assert not (run.dir / "open").exists()
    assert not (run.dir / "manifests" / "open-rules.json").exists()


def test_failed_assignments_do_not_empty_the_codebook(tmp_path):
    proposal = {
        "codes": [
            {"name": n, "label": n, "definition": n, "fits": [], "does_not_fit": []}
            for n in ("x", "y")
        ]
    }

    def handler(request):
        name = json.loads(request.content)["response_format"]["json_schema"]["name"]
        content = json.dumps(proposal) if name == "propose" else "{}"
        return httpx.Response(200, json=_reply(content))

    client, cfg, _ = _client(tmp_path, handler)
    run = _mini_run(tmp_path)
    client.run = run
    rows = [{"id": f"fc:a:{i}", "version": "a", "description": "d"} for i in range(5)]
    src = run.path("open", "rule_free_codes.jsonl")
    write_jsonl(src, rows)
    write_manifest(run, "open-rules", [], [src])
    with pytest.raises(SystemExit, match="failure_budget.assign"):
        what_codebook(run, cfg, client)
    assert not (run.dir / "codebook").exists()


def test_reserved_code_names():
    proposed = [
        ProposedCode(name=n, label="l", definition="d", fits=[], does_not_fit=[])
        for n in ("None", "failed", "ok")
    ]
    names = [c["name"] for c in to_codebook_entries(proposed, set(), "rules")]
    assert names == ["none_2", "failed_2", "ok"]
    with pytest.raises(ValueError):
        assignment_model(["none", "a"])


def test_review_against_regenerated_base_is_rejected(tmp_path):
    run = Run(tmp_path, "c")
    v1 = run.path("codebook", "what.v1.yml")
    v1.write_text(yaml.safe_dump({"codes": [{"name": "a"}]}))
    write_manifest(run, "coverage", [], [v1])
    review_begin(run, "what")
    assert latest_codebook(run, "what")[0] == 2
    # coverage reruns and writes a different v1: the review no longer applies
    v1.write_text(yaml.safe_dump({"codes": [{"name": "b"}]}))
    write_manifest(run, "coverage", [], [v1])
    with pytest.raises(SystemExit, match="reviewed against a different"):
        latest_codebook(run, "what")


def test_parse_present_is_strict():
    assert parse_present("Yes", "s") is True and parse_present(0, "s") is False
    assert parse_present("", "s") is None and parse_present(None, "s") is None
    with pytest.raises(SystemExit, match="sheet.yml"):
        parse_present("absent", "sheet.yml")


def test_preflight(tmp_path):
    from aimanager.annotation.preflight import preflight

    assert (
        main(
            ["preflight", "--run", "p", "--runs-root", str(tmp_path)]
            + ["--backend", "fake"]
        )
        == 0
    )

    def handler(request):
        return httpx.Response(200, json=_reply("not json"))

    client, _, _ = _client(tmp_path, handler, use_cache=False)
    assert preflight(client) == 5
