"""Check an OpenAI-compatible vLLM endpoint the way an agent harness uses it.

Standard library only. Runs a plain chat completion, a tool call (non-streamed
and streamed), a tool call after a long context, and prints the usage the
server reports for each. Exit code 1 if a check fails.

    python3 probe_endpoint.py --base http://127.0.0.1:8000/v1 --model M
"""

import argparse
import json
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

TOOL_PROMPT = (
    "Create the file configs/rule.yml containing exactly these two lines:\n"
    "params: {threshold: 0.5}\ncode: |\n  punishment = th.zeros_like(c[..., -1])\n"
    "Use the write_file tool, with overwrite true."
)


def post(base, body, stream=False):
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"},
    )
    resp = urllib.request.urlopen(req, timeout=900)
    if not stream:
        return json.loads(resp.read())
    chunks = []
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks


def check_call(call):
    """A tool call is well formed if its arguments parse as JSON with the
    required keys of its tool."""
    fn = call["function"]
    args = json.loads(fn["arguments"])
    tool = next(t for t in TOOLS if t["function"]["name"] == fn["name"])
    missing = [k for k in tool["function"]["parameters"]["required"] if k not in args]
    return args, missing


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--long-tokens", type=int, default=60000)
    a = p.parse_args()
    ok = True
    sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 40}

    # 1. plain chat
    t0 = time.time()
    r = post(
        a.base,
        {
            "model": a.model,
            "max_tokens": 200,
            **sampling,
            "messages": [
                {
                    "role": "user",
                    "content": "In one sentence: what is a public goods game?",
                }
            ],
        },
    )
    ok &= report(
        "chat",
        bool(r["choices"][0]["message"]["content"]),
        f"{time.time() - t0:.1f}s usage={r['usage']} "
        f"text={r['choices'][0]['message']['content'][:160]!r}",
    )

    # 2. tool call, not streamed
    t0 = time.time()
    r = post(
        a.base,
        {
            "model": a.model,
            "max_tokens": 400,
            **sampling,
            "tools": TOOLS,
            "tool_choice": "auto",
            "messages": [{"role": "user", "content": TOOL_PROMPT}],
        },
    )
    msg = r["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if calls:
        args, missing = check_call(calls[0])
        ok &= report(
            "tool call",
            not missing and r["choices"][0]["finish_reason"] == "tool_calls",
            f"{time.time() - t0:.1f}s finish={r['choices'][0]['finish_reason']} "
            f"name={calls[0]['function']['name']} args={args} missing={missing} "
            f"usage={r['usage']}",
        )
    else:
        ok &= report(
            "tool call", False, f"no tool_calls; content={msg.get('content')!r:.300}"
        )

    # 3. tool call, streamed: arguments arrive in deltas and must join to JSON
    t0 = time.time()
    chunks = post(
        a.base,
        {
            "model": a.model,
            "max_tokens": 400,
            **sampling,
            "tools": TOOLS,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": TOOL_PROMPT}],
        },
        stream=True,
    )
    name, argtext, usage, finish = None, "", None, None
    for c in chunks:
        usage = c.get("usage") or usage
        for ch in c.get("choices", []):
            finish = ch.get("finish_reason") or finish
            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name") or name
                argtext += fn.get("arguments") or ""
    try:
        args, missing = check_call({"function": {"name": name, "arguments": argtext}})
        ok &= report(
            "streamed tool call",
            not missing,
            f"{time.time() - t0:.1f}s chunks={len(chunks)} finish={finish} "
            f"name={name} args={args} usage={usage}",
        )
    except Exception as e:  # noqa: BLE001
        ok &= report(
            "streamed tool call", False, f"{e!r} name={name} argtext={argtext[:300]!r}"
        )

    # 4. a tool call after a long context (a symptom reported with qwen3_coder:
    #    endless '!!!' after long inputs that contain tool calls)
    filler = "\n".join(
        f"round {i}: contributions 12 15 20 0, punishments 0 0 0 3, pool {i * 7 % 97}"
        for i in range(a.long_tokens // 25)
    )
    t0 = time.time()
    r = post(
        a.base,
        {
            "model": a.model,
            "max_tokens": 400,
            **sampling,
            "tools": TOOLS,
            "messages": [
                {"role": "user", "content": "Here is a game log:\n" + filler},
                {"role": "assistant", "content": "I have read the log."},
                {"role": "user", "content": TOOL_PROMPT},
            ],
        },
    )
    msg = r["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if calls:
        args, missing = check_call(calls[0])
        ok &= report(
            "long-context tool call",
            not missing,
            f"{time.time() - t0:.1f}s args={args} usage={r['usage']}",
        )
    else:
        ok &= report(
            "long-context tool call",
            False,
            f"no tool_calls; usage={r['usage']} content={msg.get('content')!r:.300}",
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
