"""Preflight probe of an OpenAI-compatible vLLM endpoint. Standard library only.

Checks what the annotation pipeline relies on, then measures throughput:
  (a) a plain chat request returns non-empty content with finish_reason stop;
  (b) a request with top-level response_format json_schema returns JSON that
      parses and obeys the schema;
  (c) with thinking disabled no reasoning text reaches the content (and the
      reasoning field stays empty);
  (d) a ~12k-token prompt fits the context;
  (e) a burst of concurrent requests sharing a long prefix, for tokens/s.

Endpoint and model from ANNOT_BASE_URL / ANNOT_MODEL (set by the job), or
--base-url / --model. Writes probe.json to ANNOT_JOB_LOG_DIR (or --out).
Exit status 1 if (a), (b), (c) or (d) fails.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
THINK_MARKERS = ("<think>", "</think>", "<|begin_of_thought|>")

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "integer"},
        "mechanism": {"type": "string", "enum": ["sanction", "reward", "none"]},
        "quotes": {
            "type": "array",
            "items": {"type": "string", "maxLength": 120},
            "maxItems": 3,
        },
    },
    "required": ["answer", "mechanism", "quotes"],
    "additionalProperties": False,
}


def post(base, path, body, timeout=600):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:500]}") from None
    out["_latency_s"] = round(time.time() - t0, 3)
    return out


def chat(base, model, messages, max_tokens, **extra):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        **NO_THINK,
    }
    body.update(extra)
    return post(base, "/chat/completions", body)


def first(resp):
    choice = resp["choices"][0]
    msg = choice["message"]
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return msg.get("content") or "", reasoning, choice.get("finish_reason")


def check_schema(obj):
    """Small validator for SCHEMA; returns a list of violations."""
    errs = []
    if not isinstance(obj, dict):
        return ["not an object"]
    extra = set(obj) - set(SCHEMA["properties"])
    if extra:
        errs.append(f"extra keys {sorted(extra)}")
    for k in SCHEMA["required"]:
        if k not in obj:
            errs.append(f"missing {k}")
    if "answer" in obj and not (
        isinstance(obj["answer"], int) and not isinstance(obj["answer"], bool)
    ):
        errs.append("answer not integer")
    if obj.get("mechanism") not in SCHEMA["properties"]["mechanism"]["enum"]:
        errs.append(f"mechanism {obj.get('mechanism')!r} not in enum")
    q = obj.get("quotes")
    if not isinstance(q, list) or len(q) > 3:
        errs.append("quotes not a list of <= 3")
    elif any(not isinstance(x, str) or len(x) > 120 for x in q):
        errs.append("quote not a string of <= 120 chars")
    return errs


def run(base, model):
    res = {"base_url": base, "model": model, "checks": {}}
    models = json.load(urllib.request.urlopen(base + "/models", timeout=30))
    entry = next((m for m in models["data"] if m["id"] == model), None)
    res["max_model_len"] = entry and entry.get("max_model_len")
    if entry is None:
        res["checks"]["models"] = {
            "ok": False,
            "served": [m["id"] for m in models["data"]],
        }
        return res

    # (a) plain chat
    r = chat(
        base,
        model,
        [{"role": "user", "content": "What is 17*19? Return only the integer."}],
        32,
    )
    content, reasoning, fin = first(r)
    res["checks"]["plain"] = {
        "ok": bool(content.strip()) and fin == "stop",
        "content": content[:200],
        "correct_323": "323" in content,
        "finish_reason": fin,
        "latency_s": r["_latency_s"],
    }

    # (b) top-level response_format json_schema
    text = (
        "Rule: members who contribute less than half of the group average lose "
        "two points; everybody else keeps their payoff."
    )
    r = chat(
        base,
        model,
        [
            {"role": "system", "content": "Classify mechanisms. Answer in JSON."},
            {
                "role": "user",
                "content": f"{text}\nanswer = 6*7; mechanism = the kind of rule; "
                "quotes = up to three short quotes from the rule.",
            },
        ],
        256,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "probe", "schema": SCHEMA},
        },
    )
    content, reasoning, fin = first(r)
    try:
        obj = json.loads(content)
        errs = check_schema(obj)
    except ValueError as e:
        obj, errs = None, [f"json: {e}"]
    res["checks"]["json_schema"] = {
        "ok": not errs and fin == "stop",
        "errors": errs,
        "parsed": obj,
        "finish_reason": fin,
        "latency_s": r["_latency_s"],
    }

    # (c) thinking disabled: nothing of a reasoning trace in content
    r = chat(
        base,
        model,
        [
            {
                "role": "user",
                "content": "Think carefully step by step: how many letters r are in "
                "'strawberry'? Give the final number.",
            }
        ],
        512,
    )
    content, reasoning, fin = first(r)
    leaked = [m for m in THINK_MARKERS if m in content]
    res["checks"]["thinking_off"] = {
        "ok": not leaked and not reasoning.strip() and bool(content.strip()),
        "leaked_markers": leaked,
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "content_head": content[:200],
        "finish_reason": fin,
        "completion_tokens": r["usage"]["completion_tokens"],
    }

    # control, informational: with thinking on, a reasoning parser should
    # move the trace into the reasoning field (empty for instruct-only models)
    r = chat(
        base,
        model,
        [{"role": "user", "content": "How many letters r are in 'strawberry'?"}],
        1024,
        chat_template_kwargs={"enable_thinking": True, "thinking": True},
    )
    content, reasoning, fin = first(r)
    res["thinking_on_control"] = {
        "reasoning_chars": len(reasoning),
        "markers_in_content": [m for m in THINK_MARKERS if m in content],
        "finish_reason": fin,
    }

    # (d) a prompt of about 12k tokens (longest prompt incl. codebook)
    filler = " ".join(
        f"line {i}: members contribute and the manager decides." for i in range(1300)
    )
    r = chat(
        base,
        model,
        [
            {
                "role": "user",
                "content": filler + "\nHow many lines are there? One number.",
            }
        ],
        16,
    )
    res["checks"]["long_prompt"] = {
        "ok": bool(r["choices"][0]["message"].get("content")),
        "prompt_tokens": r["usage"]["prompt_tokens"],
        "latency_s": r["_latency_s"],
    }

    # (e) throughput: shared ~3k-token prefix, distinct suffixes, schema output
    prefix = "Codebook:\n" + "\n".join(
        f"- code_{i}: a rule that changes payoffs when condition {i} holds; "
        "fits: sanctions low contributors; does not fit: equal split."
        for i in range(120)
    )
    n, conc = 96, 48

    def one(i):
        return chat(
            base,
            model,
            [
                {"role": "system", "content": prefix},
                {
                    "role": "user",
                    "content": f"Document {i}: rule {i} punishes members below "
                    f"{i % 7} tokens. answer = {i}; pick the mechanism; quote it.",
                },
            ],
            256,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "probe", "schema": SCHEMA},
            },
        )

    one(-1)  # warm the prefix and the grammar
    t0 = time.time()
    with ThreadPoolExecutor(conc) as ex:
        outs = list(ex.map(one, range(n)))
    wall = time.time() - t0
    ptok = sum(o["usage"]["prompt_tokens"] for o in outs)
    ctok = sum(o["usage"]["completion_tokens"] for o in outs)
    bad = sum(bool(check_schema(json.loads(first(o)[0]))) for o in outs)
    res["throughput"] = {
        "requests": n,
        "concurrency": conc,
        "wall_s": round(wall, 2),
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "output_tok_per_s": round(ctok / wall, 1),
        "total_tok_per_s": round((ptok + ctok) / wall, 1),
        "requests_per_s": round(n / wall, 2),
        "schema_violations": bad,
        "finish_length": sum(first(o)[2] == "length" for o in outs),
    }

    # (f) decode throughput: fixed-length free text (min_tokens = max_tokens)
    n_dec, out_len = 128, 384

    def dec(i):
        return chat(
            base,
            model,
            [
                {
                    "role": "user",
                    "content": f"Write a long essay about public goods game {i}.",
                }
            ],
            out_len,
            min_tokens=out_len,
        )

    t0 = time.time()
    with ThreadPoolExecutor(n_dec) as ex:
        outs = list(ex.map(dec, range(n_dec)))
    wall = time.time() - t0
    ctok = sum(o["usage"]["completion_tokens"] for o in outs)
    res["decode_throughput"] = {
        "requests": n_dec,
        "concurrency": n_dec,
        "tokens_per_request": out_len,
        "wall_s": round(wall, 2),
        "completion_tokens": ctok,
        "output_tok_per_s": round(ctok / wall, 1),
    }
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=os.environ.get("ANNOT_BASE_URL"))
    ap.add_argument("--model", default=os.environ.get("ANNOT_MODEL"))
    ap.add_argument("--out", default=os.environ.get("ANNOT_JOB_LOG_DIR", "."))
    a = ap.parse_args()
    if not a.base_url or not a.model:
        sys.exit("need --base-url and --model (or ANNOT_BASE_URL / ANNOT_MODEL)")
    res = run(a.base_url.rstrip("/"), a.model)
    path = os.path.join(a.out, "probe.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    failed = [k for k, v in res["checks"].items() if not v["ok"]]
    verdict = "FAILED " + ",".join(failed) if failed else "all checks ok"
    print(f"probe: {verdict} -> {path}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
