"""Score the exp2_multi head-to-head (configs/simulation/manager_testing/33_MULTI_h2h,
docs/plans/policy-finder-multi-opponent.md, step 8).

Per row (a governor) and column (an opponent, or `self`): the group pool of the row
in every game against that opponent, both seats pooled (pool_scores.against_anchor),
over every seed found. Games of different seeds and pairings are independent, so a
mean's standard error is sd/sqrt(n), and a difference of two cells adds their
variances.

Aggregates per row:
- `training`: the plain mean over levin, ah, prev_best, zero (the campaign's objective);
- `held_out`: the mean over cem_best and thr9_p10;
- `self`: self-play.

Writes a JSON next to the data and prints a table.

    python scripts/data_analysis/exp2_multi_h2h.py [--root plots/simulation/33_MULTI_h2h]
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

from aimanager.simulation.pool_scores import add_pool, against_anchor, pairing_of

TRAINING = ["levin", "ah", "prev_best", "zero"]
HELD_OUT = ["cem_best", "thr9_p10"]
COLUMNS = TRAINING + HELD_OUT + ["self"]
ROWS = ["pf_9", "pf_11", "pf_7", "pf_9_w0", "prev_best", "cem_best", "levin", "ah", "zero"]
LABELS = {
    "pf_9": "pf-20261106-9-v5",
    "pf_11": "pf-20261106-11-v7",
    "pf_7": "pf-20261106-7-v5",
    "pf_9_w0": "pf-20261106-9-v5, w = 0",
}
COLS = ["episode", "round_number", "agent_group", "contribution", "punishment", "run"]


def cells(root):
    """(row, column) -> {seed: per-episode pools}."""
    out = {}
    for d in sorted(Path(root).glob("seed_*")):
        f = d / "per_round.parquet"
        if not f.exists():
            continue
        seed = d.name.split("_")[1]
        df = add_pool(pd.read_parquet(f, columns=COLS))
        runs = [pairing_of(r) for r in df["run"].unique()]
        for anchor in sorted({m for pair in runs for m in pair}):
            for manager, frame in against_anchor(df, anchor).items():
                col = "self" if manager == anchor else anchor
                if manager in ROWS and col in COLUMNS:
                    out.setdefault((manager, col), {}).setdefault(seed, []).append(
                        frame["pool"]
                    )
    return {k: {s: pd.concat(v, ignore_index=True) for s, v in d.items()}
            for k, d in out.items()}


def stat(series):
    x = pd.concat(series, ignore_index=True)
    return {"pool": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(len(x))),
            "n": int(len(x))}


def mean_of(stats):
    if any(s is None for s in stats):
        return None
    return {"pool": sum(s["pool"] for s in stats) / len(stats),
            "se": math.sqrt(sum(s["se"] ** 2 for s in stats)) / len(stats)}


def diff(a, b):
    if a is None or b is None:
        return None
    return {"pool": a["pool"] - b["pool"], "se": math.hypot(a["se"], b["se"])}


def summarize(data):
    seeds = sorted({s for d in data.values() for s in d})
    table = {}
    for r in ROWS:
        row = {}
        for c in COLUMNS:
            d = data.get((r, c))
            row[c] = stat(list(d.values())) if d else None
            row[c + "_by_seed"] = (
                {s: stat([v])["pool"] for s, v in d.items()} if d else None
            )
        row["training"] = mean_of([row[c] for c in TRAINING])
        row["held_out"] = mean_of([row[c] for c in HELD_OUT])
        table[r] = row
    best = table["prev_best"]
    for r in ROWS:
        for agg in ("training", "held_out", "self"):
            table[r][f"{agg}_vs_prev_best"] = diff(table[r][agg], best[agg])
    table["pf_9"]["training_vs_w0"] = diff(table["pf_9"]["training"],
                                           table["pf_9_w0"]["training"])
    return {"seeds": seeds, "labels": LABELS, "columns": COLUMNS, "rows": table}


def fmt(s):
    return "—" if s is None else f"{s['pool']:.2f} ± {s['se']:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="plots/simulation/33_MULTI_h2h")
    args = ap.parse_args()
    result = summarize(cells(args.root))
    Path(args.root, "h2h_summary.json").write_text(json.dumps(result, indent=1))
    print("seeds", result["seeds"])
    head = ["row"] + COLUMNS + ["training", "held_out", "training - prev_best"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r, row in result["rows"].items():
        vals = [fmt(row[c]) for c in COLUMNS + ["training", "held_out"]]
        vals.append(fmt(row["training_vs_prev_best"]))
        print(f"| {LABELS.get(r, r)} | " + " | ".join(vals) + " |")
    print("pf_9 - pf_9_w0 (training):", fmt(result["rows"]["pf_9"]["training_vs_w0"]))


if __name__ == "__main__":
    main()
