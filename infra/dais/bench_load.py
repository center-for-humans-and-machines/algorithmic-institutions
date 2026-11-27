"""Generation throughput with N concurrent long-context sessions.

Standard library only. Each session sends a distinct prompt of about
--context tokens and streams --gen tokens back (ignore_eos, so every session
generates the same amount), first cold, then once more with the same prefix
plus a new turn, as an agent's next turn would (prefix cache warm). Prints
time to first token, decode rate per session and the aggregate.

    python3 bench_load.py --model M [--sessions 5] [--context 32000] [--gen 1024]
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request


def session_prompt(i, n_tokens):
    lines = [
        f"session {i} round {r}: contributions {r % 21} {(r * 3) % 21} "
        f"{(r * 7 + i) % 21} {(r * 11) % 21}, punishment {(r + i) % 31}"
        for r in range(n_tokens // 28)
    ]
    return "Game log:\n" + "\n".join(lines)


def stream(base, model, messages, gen, out):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": gen,
        "stream": True,
        "ignore_eos": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"},
    )
    t0 = time.time()
    first, usage = None, None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            if first is None and chunk.get("choices"):
                if chunk["choices"][0].get("delta", {}).get("content"):
                    first = time.time()
    end = time.time()
    n = usage["completion_tokens"]
    out.append(
        {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": n,
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            ),
            "ttft_s": round(first - t0, 2),
            "total_s": round(end - t0, 2),
            "decode_tok_s": round((n - 1) / (end - first), 1),
        }
    )


def run_round(a, label, messages_per_session):
    out, threads = [], []
    t0 = time.time()
    for msgs in messages_per_session:
        th = threading.Thread(target=stream, args=(a.base, a.model, msgs, a.gen, out))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.time() - t0
    gen_total = sum(r["completion_tokens"] for r in out)
    summary = {
        "round": label,
        "sessions": len(out),
        "wall_s": round(wall, 1),
        "ttft_s_median": statistics.median(r["ttft_s"] for r in out),
        "decode_tok_s_per_session_median": statistics.median(
            r["decode_tok_s"] for r in out
        ),
        "aggregate_gen_tok_s": round(gen_total / wall, 1),
        "prompt_tokens": [r["prompt_tokens"] for r in out],
        "cached_tokens": [r["cached_tokens"] for r in out],
    }
    print("BENCH " + json.dumps(summary), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--sessions", type=int, default=5)
    p.add_argument("--context", type=int, default=32000)
    p.add_argument("--gen", type=int, default=1024)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    convs = [
        # the tag goes first, so that a later run with another tag starts cold
        [{"role": "user", "content": f"[{a.tag}]\n" + session_prompt(i, a.context)}]
        for i in range(a.sessions)
    ]
    run_round(a, f"{a.tag} cold", convs)
    convs = [
        c
        + [
            {"role": "assistant", "content": "Noted."},
            {
                "role": "user",
                "content": "Now describe a punishment rule for this group.",
            },
        ]
        for c in convs
    ]
    run_round(a, f"{a.tag} warm", convs)


if __name__ == "__main__":
    sys.exit(main())
