"""Check an OpenAI-compatible vLLM endpoint the way an agent harness uses it.

Standard library only. Each check runs --trials times at a low temperature and
passes only if every trial does:

- chat: a plain completion with text
- tool: a tool call, not streamed
- tool_stream: the same streamed; the argument deltas must join to JSON
- tool_long: a tool call after one long user message
- multi_turn / multi_turn_stream: a tool call after a long history of earlier
  tool calls and tool results (the reported failure mode: the old qwen3_coder
  parser looped on '!!!' after long inputs containing tool calls)

A tool call passes if finish_reason is `tool_calls`, the tool is the expected
one, its arguments parse as JSON with every required key, and the output has
no degenerate repetition. Prints usage for each; exits 1 if a check fails.

A reasoning model thinks before it answers: --max-tokens gives room for that
(default 200 for the chat check, 600 for the tool checks) and --extra-body
adds request fields, such as '{"reasoning_effort": "low"}'.

    python3 probe_endpoint.py --base URL --key-file F --model M [--trials 3]
"""

import argparse
import json
import re
import sys
import time
import urllib.request

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text to a file in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path."},
                    "content": {"type": "string", "description": "File content."},
                    "overwrite": {"type": "boolean"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_s": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
]

WRITE_PROMPT = (
    "Create the file configs/rule.yml containing exactly these lines:\n"
    "params: {threshold: 0.5}\ncode: |\n  punishment = th.zeros_like(c[..., -1])\n"
    "Use the write_file tool, with overwrite true."
)
SAMPLING = {"temperature": 0.2, "top_p": 0.95, "top_k": 40}


def game_log(n_tokens, seed):
    return "\n".join(
        f"episode {seed} round {i}: contributions {i % 21} {(i * 7 + seed) % 21} "
        f"{(i * 3) % 21} 0, punishments 0 0 {i % 4} 3, pool {(i * 7 + seed) % 97}"
        for i in range(n_tokens // 30)
    )


def history(n_tokens):
    """A long agent history: three shell calls with long results."""
    msgs = [
        {"role": "system", "content": "You are a coding agent. Use the tools."},
        {"role": "user", "content": "Inspect the game logs, then write the rule."},
    ]
    for k, cmd in enumerate(["ls logs", "cat logs/a.txt", "cat logs/b.txt"]):
        call_id = f"call_{k}"
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "run_shell",
                            "arguments": json.dumps({"command": cmd}),
                        },
                    }
                ],
            }
        )
        out = "a.txt\nb.txt" if k == 0 else game_log(n_tokens // 2, k)
        msgs.append({"role": "tool", "tool_call_id": call_id, "content": out})
    msgs.append({"role": "user", "content": WRITE_PROMPT})
    return msgs


def post(a, key, body):
    req = urllib.request.Request(
        a.base + "/chat/completions",
        data=json.dumps(
            {"model": a.model, **SAMPLING, **a.extra_body, **body}
        ).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    resp = urllib.request.urlopen(req, timeout=900)
    if not body.get("stream"):
        return json.loads(resp.read())
    chunks = []
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks


def degenerate(text):
    return bool(text and re.search(r"(.)\1{40,}", text))


def judge_call(name, argtext, finish, content, expect="write_file"):
    """'' if the call is well formed, else the reason."""
    if finish != "tool_calls":
        return f"finish_reason={finish} content={str(content)[:200]!r}"
    if name != expect:
        return f"tool={name}"
    if degenerate(content) or degenerate(argtext):
        return "degenerate repetition"
    try:
        args = json.loads(argtext)
    except ValueError:
        return f"arguments not JSON: {argtext[:200]!r}"
    tool = next(t for t in TOOLS if t["function"]["name"] == name)
    missing = [k for k in tool["function"]["parameters"]["required"] if k not in args]
    return f"missing {missing}" if missing else ""


def unstreamed(r):
    choice = r["choices"][0]
    msg = choice["message"]
    calls = msg.get("tool_calls") or []
    name = calls[0]["function"]["name"] if calls else None
    argtext = calls[0]["function"]["arguments"] if calls else ""
    return name, argtext, choice["finish_reason"], msg.get("content"), r["usage"]


def streamed(chunks):
    name, argtext, usage, finish, content = None, "", None, None, ""
    for c in chunks:
        usage = c.get("usage") or usage
        for ch in c.get("choices", []):
            finish = ch.get("finish_reason") or finish
            delta = ch.get("delta") or {}
            content += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name") or name
                argtext += fn.get("arguments") or ""
    return name, argtext, finish, content, usage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--key-file", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--long-tokens", type=int, default=60000)
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--extra-body", type=json.loads, default={})
    a = p.parse_args()
    with open(a.key_file) as f:
        key = f.read().strip()

    write = [{"role": "user", "content": WRITE_PROMPT}]
    long_user = [
        {
            "role": "user",
            "content": "Here is a game log:\n" + game_log(a.long_tokens, 0),
        },
        {"role": "assistant", "content": "I have read the log."},
        {"role": "user", "content": WRITE_PROMPT},
    ]
    multi = history(a.long_tokens)
    checks = {
        "tool": ({"messages": write}, unstreamed),
        "tool_stream": ({"messages": write}, streamed),
        "tool_long": ({"messages": long_user}, unstreamed),
        "multi_turn": ({"messages": multi}, unstreamed),
        "multi_turn_stream": ({"messages": multi}, streamed),
    }
    failed = []

    r = post(
        a,
        key,
        {
            "max_tokens": a.max_tokens or 200,
            "messages": [
                {
                    "role": "user",
                    "content": "In one sentence: what is a public goods game?",
                }
            ],
        },
    )
    msg = r["choices"][0]["message"]
    text = msg["content"] or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    ok = bool(text.strip()) and not degenerate(text)
    print(
        f"[{'PASS' if ok else 'FAIL'}] chat: usage={r['usage']} text={text[:120]!r}"
        f" reasoning={len(reasoning)} chars"
    )
    if not ok:
        failed.append("chat")

    for name, (body, parse) in checks.items():
        body = {**body, "max_tokens": a.max_tokens or 600, "tools": TOOLS,
                "tool_choice": "auto"}
        stream = parse is streamed
        if stream:
            body.update(stream=True, stream_options={"include_usage": True})
        reasons, usage, seconds = [], None, []
        for _ in range(a.trials):
            t0 = time.time()
            tool, argtext, finish, content, usage = parse(post(a, key, body))
            seconds.append(time.time() - t0)
            reasons.append(judge_call(tool, argtext, finish, content))
        passed = sum(not r for r in reasons)
        ok = passed == a.trials
        print(
            f"[{'PASS' if ok else 'FAIL'}] {name}: {passed}/{a.trials} well formed,"
            f" {max(seconds):.1f}s max, usage={usage}"
            + ("" if ok else f" reasons={[r for r in reasons if r]}"),
            flush=True,
        )
        if not ok:
            failed.append(name)
    print("PROBE " + ("FAILED " + ",".join(failed) if failed else "OK"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
