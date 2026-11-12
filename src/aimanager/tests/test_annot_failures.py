"""Annotation failure paths: HTTP client aborts, failure budgets, stale inputs."""

import json

import httpx
import pytest
import yaml

from aimanager.annotation.cli import main, review_begin
from aimanager.annotation.client import Call, LLMClient
from aimanager.annotation.cluster import (
    est_tokens,
    fit_items,
    is_negated,
    to_codebook_entries,
)
from aimanager.annotation.consolidate import UNPLACED, apply_consolidation
from aimanager.annotation.judge import code_order, resolve
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


def test_proposal_items_fit_the_token_budget():
    items = [(f"i{k}", "x" * 300) for k in range(500)]
    kept = fit_items(items, budget_tokens=5000, cap=400, seed=1)
    assert 0 < len(kept) < 500
    assert sum(est_tokens(t) + 3 for _, t in kept) <= 5000
    assert kept == [it for it in items if it in kept]  # input order kept
    assert fit_items(items[:5], 5000, 400, 1) == items[:5]
    assert len(fit_items(items, 10**6, 400, 1)) == 400


def test_negated_codes_are_recognised():
    assert is_negated("do_not_punish_based_on_history")
    assert is_negated("ignore_group_norm_erosion_triggers")
    assert is_negated("x", definition="The manager does not adjust to the opponent.")
    assert is_negated("x", label="No reaction to the opponent")
    assert is_negated("punishment_should_not_be_applied_to_timed_out_players")
    assert not is_negated("punish_only_low_contributors", "Punish low", "Punishes.")
    assert not is_negated("notify_players", definition="Notifies players.")


def test_apply_consolidation_merges_and_families():
    codes = [
        {"name": n, "label": n, "definition": n, "implemented": n != "c"}
        for n in ("a", "a_2", "b", "c", "d")
    ]
    merges = [
        {"codes": ["a", "a_2"], "keep": "a", "label": "A", "definition": "Does A."},
        {"codes": ["a_2", "b"], "keep": "b", "label": "x", "definition": "x"},
        {"codes": ["c", "d"], "keep": "zz", "label": "n", "definition": "Never d."},
    ]
    families = [{"name": "Fam One", "definition": "f1", "codes": ["a_2", "c"]}]
    surv, defs, audit, unplaced = apply_consolidation(codes, merges, families)
    by = {c["name"]: c for c in surv}
    assert list(by) == ["a", "b", "c"]
    assert by["a"]["definition"] == "Does A." and by["a"]["merged_from"] == ["a_2"]
    # a_2 was already merged; keep not in the group -> first name; negated
    # rewritten definition -> the old one stays; implemented if any member is
    assert by["c"]["merged_from"] == ["d"] and by["c"]["definition"] == "c"
    assert by["c"]["implemented"] is True
    assert [a["applied"] for a in audit] == [True, False, True]
    # a family listing a merged name places its survivor; unlisted -> unplaced
    assert by["a"]["family"] == "fam_one" and by["c"]["family"] == "fam_one"
    assert unplaced == ["b"] and by["b"]["family"] == UNPLACED
    assert set(defs) == {"fam_one", UNPLACED}


def test_judge_order_and_resolution():
    codes = [{"name": f"c{i}", "family": "f" + str(i % 2)} for i in range(6)]
    assert code_order(codes, 0, 1) == codes
    o1 = code_order(codes, 1, 1)
    assert o1 == code_order(codes, 1, 1) and sorted(map(str, o1)) == sorted(
        map(str, codes)
    )
    fams = [c["family"] for c in o1]
    assert fams == sorted(fams, key=fams.index)  # families stay together
    yes = {"present": True, "evidence": "x"}
    no = {"present": False, "evidence": ""}
    assert resolve([yes, dict(yes)]) == (yes, False)
    assert resolve([yes, no], tiebreak=no) == (no, True)
    with pytest.raises(ValueError):
        resolve([yes, no])


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
    write_manifest(run, "consolidate-what", [], [v1])
    review_begin(run, "what")
    assert latest_codebook(run, "what")[0] == 2
    # coverage reruns and writes a different v1: the review no longer applies
    v1.write_text(yaml.safe_dump({"codes": [{"name": "b"}]}))
    write_manifest(run, "consolidate-what", [], [v1])
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
    assert preflight(client) == 6
