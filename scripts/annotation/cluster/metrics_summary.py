"""Summarise one job's log directory (written by common.sh) as JSON.

Reads timeline.txt, metrics_samples.txt (vLLM /metrics every few seconds),
gpu_samples.txt (nvidia-smi or rocm-smi CSV) and startup_lines.txt.
Usage: python3 metrics_summary.py <job log dir>
"""

import json
import re
import statistics
import sys
from pathlib import Path

LINE = re.compile(r"^vllm:([a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)")


def timeline(d):
    marks = {}
    for line in (d / "timeline.txt").read_text().splitlines():
        ts, _, rest = line.split(" ", 2)
        for key in (
            "job ",
            "starting server",
            "server ready",
            "START command",
            "END command",
            "server stopped",
        ):
            if rest.startswith(key) and key.strip() not in marks:
                marks[key.strip()] = int(ts)
    out = {}
    if "job" in marks and "server ready" in marks:
        out["startup_s"] = marks["server ready"] - marks["job"]
    if "starting server" in marks and "server ready" in marks:
        out["server_load_s"] = marks["server ready"] - marks["starting server"]
    if "START command" in marks and "END command" in marks:
        out["command_s"] = marks["END command"] - marks["START command"]
    return out


def samples(d):
    rows, cur = [], None
    path = d / "metrics_samples.txt"
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.startswith("### "):
            cur = {"t": int(line[4:])}
            rows.append(cur)
            continue
        m = LINE.match(line)
        if m and cur is not None:
            name = m.group(1)
            cur[name] = cur.get(name, 0.0) + float(m.group(3))
    return [r for r in rows if len(r) > 1]


def dist(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"median": round(statistics.median(xs), 3), "max": round(max(xs), 3)}


def metrics(rows):
    if not rows:
        return {}
    kv = [r.get("kv_cache_usage_perc", r.get("gpu_cache_usage_perc")) for r in rows]
    out = {
        "samples": len(rows),
        "kv_cache_usage": dist(kv),
        "running": dist([r.get("num_requests_running") for r in rows]),
        "waiting": dist([r.get("num_requests_waiting") for r in rows]),
    }
    busy = [r for r in rows if r.get("num_requests_running", 0) > 0]
    out["busy_fraction"] = round(len(busy) / len(rows), 3)
    first, last = rows[0], rows[-1]
    for k in (
        "prompt_tokens_total",
        "generation_tokens_total",
        "request_success_total",
        "num_preemptions_total",
    ):
        if k in last:
            out[k] = last[k] - first.get(k, 0.0)
    if "prefix_cache_queries_total" in last and last["prefix_cache_queries_total"]:
        q = last["prefix_cache_queries_total"] - first.get(
            "prefix_cache_queries_total", 0
        )
        h = last.get("prefix_cache_hits_total", 0) - first.get(
            "prefix_cache_hits_total", 0
        )
        out["prefix_cache_hit_rate"] = round(h / q, 3) if q else None
    rates = []
    for a, b in zip(rows, rows[1:]):
        dt = b["t"] - a["t"]
        if dt > 0 and "generation_tokens_total" in b:
            rates.append(
                (b["generation_tokens_total"] - a.get("generation_tokens_total", 0))
                / dt
            )
    busy_rates = [x for x in rates if x > 0]
    out["generation_tok_per_s"] = {
        "peak_window": round(max(rates), 1) if rates else None,
        "mean_while_busy": (
            round(statistics.mean(busy_rates), 1) if busy_rates else None
        ),
    }
    return out


def prom(path):
    vals = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = LINE.match(line)
            if m:
                vals[m.group(1)] = vals.get(m.group(1), 0.0) + float(m.group(3))
    return vals


def totals(d):
    """Counter deltas between the snapshots at server ready and at shutdown."""
    a, b = prom(d / "metrics_start.prom"), prom(d / "metrics_end.prom")
    if not b:
        return {}
    keys = (
        "prompt_tokens_total",
        "generation_tokens_total",
        "request_success_total",
        "num_preemptions_total",
        "prefix_cache_queries_total",
        "prefix_cache_hits_total",
    )
    out = {k: b[k] - a.get(k, 0.0) for k in keys if k in b}
    q = out.get("prefix_cache_queries_total")
    if q:
        out["prefix_cache_hit_rate"] = round(
            out.get("prefix_cache_hits_total", 0) / q, 3
        )
    return {"totals": out}


def gpu(d):
    path = d / "gpu_samples.txt"
    if not path.exists():
        return {}
    used = []
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 6 and parts[3].replace(".", "").isdigit():  # nvidia-smi
            used.append(float(parts[3]) / 1024)  # MiB -> GiB
    if not used:  # rocm-smi CSV: find the VRAM used column
        header = None
        for line in path.read_text().splitlines():
            if line.startswith("device,"):
                header = line.split(",")
                continue
            if header and line.startswith("card"):
                row = dict(zip(header, line.split(",")))
                for k, v in row.items():
                    if "VRAM Total Used Memory" in k and v.strip().isdigit():
                        used.append(int(v) / 2**30)
    return {"gpu_mem_used_gib": dist(used)} if used else {}


def main():
    d = Path(sys.argv[1])
    out = {"log_dir": str(d)}
    out.update(timeline(d))
    out.update(metrics(samples(d)))
    out.update(totals(d))
    out.update(gpu(d))
    sl = d / "startup_lines.txt"
    if sl.exists():
        out["startup_lines"] = [ln[-200:] for ln in sl.read_text().splitlines()]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
