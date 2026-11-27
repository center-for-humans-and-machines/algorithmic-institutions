"""Generation throughput with N concurrent long-context sessions.

Standard library only. Each session sends a distinct prompt of about
--context tokens and streams --gen tokens back (ignore_eos, so every request
generates the same amount).

Rounds mode (default): every session once cold, then once more with the same
prefix plus a new turn, as an agent's next turn would (prefix cache warm).
Duration mode (--duration S): every session repeats its warm request until S
seconds have passed or --stop-file exists, for throughput under a steady load.

Prints one `BENCH {json}` line per round; exits 1 if any request fails.

    python3 bench_load.py --base URL --key-file F --model M [--sessions 5]
        [--context 30000] [--gen 1024] [--tag T] [--duration S --stop-file F]
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request


def session_prompt(tag, i, n_tokens):
    lines = [
        f"session {i} round {r}: contributions {r % 21} {(r * 3) % 21} "
        f"{(r * 7 + i) % 21} {(r * 11) % 21}, punishment {(r + i) % 31}"
        for r in range(n_tokens // 28)
    ]
    # the tag goes first, so that a later run with another tag starts cold
    return f"[{tag}]\nGame log:\n" + "\n".join(lines)


def stream(a, key, messages):
    body = {
        "model": a.model,
        "messages": messages,
        "max_tokens": a.gen,
        "stream": True,
        "ignore_eos": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        a.base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
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
    if first is None or n < 2:
        raise RuntimeError(f"no content streamed: usage={usage}")
    details = usage.get("prompt_tokens_details") or {}
    return {
        "start": t0,
        "end": end,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": n,
        "cached_tokens": details.get("cached_tokens"),
        "ttft_s": first - t0,
        "decode_tok_s": (n - 1) / (end - first),
    }


def run(a, key, label, convs, until=None):
    """One request per session (until=None), or repeated until `until`."""
    out, errors, threads = [], [], []

    def worker(msgs):
        try:
            while True:
                out.append(stream(a, key, msgs))
                if until is None or until():
                    return
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    t0 = time.time()
    for msgs in convs:
        th = threading.Thread(target=worker, args=(msgs,))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.time() - t0
    if errors or not out:
        print(f"BENCH_FAILED {label}: {errors[:3]}", flush=True)
        sys.exit(1)
    summary = {
        "round": label,
        "sessions": len(convs),
        "requests": len(out),
        "wall_s": round(wall, 1),
        "ttft_s_median": round(statistics.median(r["ttft_s"] for r in out), 2),
        "ttft_s_max": round(max(r["ttft_s"] for r in out), 2),
        "decode_tok_s_per_session_median": round(
            statistics.median(r["decode_tok_s"] for r in out), 1
        ),
        "aggregate_gen_tok_s": round(
            sum(r["completion_tokens"] for r in out) / wall, 1
        ),
        "prompt_tokens_median": statistics.median(r["prompt_tokens"] for r in out),
        "cached_fraction": round(
            sum(r["cached_tokens"] or 0 for r in out)
            / sum(r["prompt_tokens"] for r in out),
            3,
        ),
    }
    print("BENCH " + json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--key-file", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--sessions", type=int, default=5)
    p.add_argument("--context", type=int, default=30000)
    p.add_argument("--gen", type=int, default=1024)
    p.add_argument("--tag", default="t")
    p.add_argument("--duration", type=float)
    p.add_argument("--stop-file")
    a = p.parse_args()
    with open(a.key_file) as f:
        key = f.read().strip()
    first = [
        [{"role": "user", "content": session_prompt(a.tag, i, a.context)}]
        for i in range(a.sessions)
    ]
    warm = [
        c
        + [
            {"role": "assistant", "content": "Noted."},
            {
                "role": "user",
                "content": "Now describe a punishment rule for this group.",
            },
        ]
        for c in first
    ]
    run(a, key, f"{a.tag} cold", first)
    if a.duration is None:
        run(a, key, f"{a.tag} warm", warm)
        return
    deadline = time.time() + a.duration

    def until():
        stop = a.stop_file and os.path.exists(a.stop_file)
        return stop or time.time() > deadline

    run(a, key, f"{a.tag} steady", warm, until)


if __name__ == "__main__":
    main()
