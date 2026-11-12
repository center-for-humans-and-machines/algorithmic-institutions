"""Annotation core: real corpus counts, HTTP client behaviour, small helpers."""

import json
import subprocess
from pathlib import Path

import httpx
import pytest

from aimanager.annotation.client import Call, LLMClient
from aimanager.annotation.common import Run, load_config, quote_in_source
from aimanager.annotation.corpus import run_corpus
from aimanager.annotation.expert import kappa_stats
from aimanager.annotation.schemas import FreeCodes

REPO = Path(__file__).resolve().parents[3]


def _n_refs():
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(REPO),
                "for-each-ref",
                "refs/remotes/origin/policy-finder/",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0
    return sum("/pf-2026110" in line for line in out.splitlines())


@pytest.mark.skipif(_n_refs() < 36, reason="policy-finder refs not fetched")
def test_real_corpus_matches_plan(tmp_path):
    cfg = load_config()
    versions = run_corpus(Run(tmp_path, "c"), cfg, repo=REPO)
    assert len(versions) == 110
    assert sum(v["score_status"] == "ok" for v in versions) == 107
    assert all(v["rule"] and v["notes"] for v in versions)
    split = json.loads((tmp_path / "c" / "corpus" / "split.json").read_text())
    assert len(split["free_coding"]) == 36 and len(split["expert_check"]) == 30


def _reply(content, finish="stop"):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def test_http_client_retries_and_length(tmp_path):
    seen = []
    good = json.dumps({"mechanisms": [{"description": "d", "quote": "q"}]})

    def handler(request):
        body = json.loads(request.content)
        seen.append(body["max_tokens"])
        doc = body["messages"][-1]["content"]
        assert body["response_format"]["type"] == "json_schema"
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        if "server-flaky" in doc and len(seen) == 1:
            return httpx.Response(503)
        if "too-long" in doc and body["max_tokens"] == 2048:
            return httpx.Response(200, json=_reply('{"mech', "length"))
        if "bad-request" in doc:
            return httpx.Response(400, text="nope")
        return httpx.Response(200, json=_reply(good))

    cfg = load_config()
    cfg["llm"]["backoff"] = [0]
    client = LLMClient(
        Run(tmp_path, "r"),
        cfg,
        base_url="http://x/v1",
        model="m",
        transport=httpx.MockTransport(handler),
    )

    def call(doc):
        msgs = [{"role": "user", "content": doc}]
        return Call("s", doc, msgs, FreeCodes, "open_rules")

    (flaky,) = client.map([call("server-flaky")])
    assert flaky.ok and seen == [2048, 2048]
    seen.clear()
    (long_,) = client.map([call("too-long")])
    assert long_.ok and seen == [2048, 4096]
    (bad,) = client.map([call("bad-request")])
    assert bad.error.startswith("client_error")
    seen.clear()
    client.map([call("too-long")])  # both attempts cached now
    assert seen == []
    log = [json.loads(x) for x in open(tmp_path / "r" / "logs" / "calls.jsonl")]
    assert {r["status"] for r in log} == {"ok", "length", "4xx"}


def test_quote_check_and_kappa():
    src = "gap = th.relu(mates_mean -   cur)\npunishment = raw"
    assert quote_in_source("gap = th.relu(mates_mean - cur)", src)
    assert quote_in_source("gap = ... punishment = raw", src)
    assert not quote_in_source("gap = mean - cur", src)
    assert quote_in_source("", src) is None
    no_var = kappa_stats([(False, False)] * 4)
    assert no_var["agreement"] == 1.0 and no_var["kappa"] is None
    k = kappa_stats([(True, True), (False, False), (True, False), (False, False)])
    assert k["agreement"] == 0.75 and 0 < k["kappa"] < 1
