"""Score the exp2_diff head-to-head (34_DIFF_h2h, docs/plans/policy-finder-pool-difference.md
step 7): each row's lead in common pool over ah's group in the same game, both seats
pooled, with the two pools, the two group sizes, the row's fines, and the same per round
(for the page's charts). Writes plots/simulation/34_DIFF_h2h/h2h_summary.json.

    python scripts/data_analysis/exp2_diff_h2h.py
"""
import json
import math
from pathlib import Path

import pandas as pd

from aimanager.simulation.pool_scores import POOL, add_pool, pairing_of

ROOT = Path("plots/simulation/34_DIFF_h2h")
COLUMNS = ["episode", "round_number", "agent_group", "contribution", "punishment", "run"]


def seat_frames(df: pd.DataFrame, seed: int):
    """Per (row, seed, seat, episode, round): both groups' pool, size, fines and mean
    contribution, the row's group as `own`."""
    out = []
    for run in df["run"].unique():
        g0, g1 = pairing_of(run)
        row, own_group = (g0, 0) if g1 == "ah" else (g1, 1)
        sub = df[df["run"] == run]
        g = sub.groupby(["episode", "round_number", "agent_group"])
        agg = pd.DataFrame({
            "pool": g[POOL].sum(), "n": g.size(), "fines": g["punishment"].sum(),
            "contrib": g["contribution"].mean(),
        }).unstack("agent_group").fillna(0.0)
        frame = pd.DataFrame({
            f"{side}_{k}": agg[(k, grp)] if (k, grp) in agg.columns else 0.0
            for side, grp in (("own", own_group), ("other", 1 - own_group))
            for k in ("pool", "n", "fines", "contrib")
        }).reset_index()
        frame["row"], frame["seed"], frame["seat"] = row, seed, own_group
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def mean_se(x: pd.Series) -> dict:
    return {"mean": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(len(x)))}


def main():
    frames = []
    for d in sorted(ROOT.glob("seed_*")):
        seed = int(d.name.split("_")[1])
        df = add_pool(pd.read_parquet(d / "per_round.parquet", columns=COLUMNS[:-1] + ["run"]))
        frames.append(seat_frames(df, seed))
        print("read", d, flush=True)
    rounds = pd.concat(frames, ignore_index=True)
    rounds["lead"] = rounds["own_pool"] - rounds["other_pool"]
    # per game: means over the 24 rounds (an empty group-round is 0)
    games = rounds.groupby(["row", "seed", "seat", "episode"]).mean(numeric_only=True)
    games = games.reset_index()
    # fines per member-round of the row's group
    rows = {}
    for row, g in games.groupby("row"):
        r = rounds[rounds["row"] == row]
        rows[row] = {
            "lead": mean_se(g["lead"]),
            "own": mean_se(g["own_pool"]),
            "other": mean_se(g["other_pool"]),
            "own_members": float(g["own_n"].mean()),
            "other_members": float(g["other_n"].mean()),
            "fine_per_member_round": float(r["own_fines"].sum() / r["own_n"].sum()),
            "ah_fine_per_member_round": float(r["other_fines"].sum() / r["other_n"].sum()),
            "n": int(len(g)),
            "by_seed": {int(s): float(x["lead"].mean()) for s, x in g.groupby("seed")},
            "by_seat": {int(s): float(x["lead"].mean()) for s, x in g.groupby("seat")},
        }
    per_round = (rounds.groupby(["row", "round_number"])
                 [["own_pool", "other_pool", "own_n", "other_n", "own_fines", "other_fines",
                   "own_contrib", "other_contrib"]].mean().reset_index())
    out = {"rows": rows, "per_round": per_round.to_dict(orient="records")}
    (ROOT / "h2h_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'row':10s} {'lead':>13s} {'own':>6s} {'ah':>6s} {'n own':>6s} {'n ah':>6s}"
          f" {'fine/m':>7s} {'ah f/m':>7s}")
    for row, v in sorted(rows.items(), key=lambda kv: -kv[1]["lead"]["mean"]):
        print(f"{row:10s} {v['lead']['mean']:6.2f} ± {v['lead']['se']:.2f}"
              f" {v['own']['mean']:6.2f} {v['other']['mean']:6.2f} {v['own_members']:6.2f}"
              f" {v['other_members']:6.2f} {v['fine_per_member_round']:7.2f}"
              f" {v['ah_fine_per_member_round']:7.2f}  seeds "
              + " ".join(f"{x:.1f}" for x in v["by_seed"].values()))


if __name__ == "__main__":
    main()
