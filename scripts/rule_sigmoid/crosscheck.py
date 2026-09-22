"""Put the batched sweep and `simulate.py` side by side on the same rules.

The sweep is new code. It uses the same env, the same four artifacts and the
same protocol, but its own rollout loop and its own batched clone adapter, so
its numbers are only worth what a comparison against the established path says
they are. This reads the cross-check simulations' `per_round.parquet` and
recomputes exactly the quantities `paired_rollout.summarise` produces, with
the same conventions: seat totals per round, a timed-out player's cell zeroed
rather than read at the imputed default, the leaver gap restricted to valid
cells at the decision rounds.

Usage:
    PYTHONPATH=src python scripts/rule_sigmoid/crosscheck.py \
        --sim-dirs plots/simulation/27_rule_sigmoid_paired_s4{2,3,4} \
        --sweep runs/valid_clone --sweep-seeds 45,46,47 \
        --out plots/data_analysis/rule_sigmoid/crosscheck.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

from aggregate import load_sweep, per_point

#: `simulate.py` names a run "ah <humans> managed by <pairing>"; the pairing
#: is "<focal>_vs_<rival>".
FOCAL_GROUP, RIVAL_GROUP = 0, 1


def _seat(df, group):
    g = df[df["group_id"] == group].copy()
    g["c"] = np.where(g["contribution_valid"].astype(bool), g["contribution"], 0.0)
    g["p"] = np.where(g["contribution_valid"].astype(bool), g["punishment"], 0.0)
    per_round = g.groupby(["episode", "round_number"], as_index=False).agg(
        members=("participant_code", "size"),
        sum_c=("c", "sum"),
        sum_p=("p", "sum"),
        n_valid=("contribution_valid", "sum"),
    )
    per_round["pool"] = 1.6 * per_round["sum_c"] - per_round["sum_p"]
    return per_round


def _leaver_gap(df, switch_every=4):
    """The next round's membership has to be read BEFORE restricting to the
    seat, or a leaver's next row is the one that was dropped and every member
    reads as a stayer."""
    d = df.sort_values(["episode", "participant_code", "round_number"]).copy()
    key = ["episode", "participant_code"]
    d["next_group"] = d.groupby(key)["group_id"].shift(-1)
    d["leaves"] = (d["next_group"] != d["group_id"]) & d["next_group"].notna()
    d = d[d["group_id"] == FOCAL_GROUP]
    dec = d[
        ((d["round_number"] + 1) % switch_every == 0)
        & d["contribution_valid"].astype(bool)
        & d["next_group"].notna()
    ]
    if not len(dec):
        return np.nan
    lv, st = dec[dec["leaves"]], dec[~dec["leaves"]]
    if not len(lv) or not len(st):
        return np.nan
    return float(lv["contribution"].mean() - st["contribution"].mean())


def sim_table(sim_dirs):
    rows = []
    for d in sim_dirs:
        sim = pd.read_parquet(os.path.join(d, "per_round.parquet"))
        for run, g in sim.groupby("run"):
            pairing = run.split("managed by ")[-1].strip()
            focal = _seat(g, FOCAL_GROUP)
            valid = g[g["contribution_valid"].astype(bool) & (g["group_id"] == 0)]
            rows.append(
                {
                    "pairing": pairing,
                    "name": pairing.split("_vs_")[0],
                    "sim_dir": os.path.basename(d),
                    "focal_members": focal["members"].mean(),
                    "focal_contribution": focal["sum_c"].mean(),
                    "focal_punishment": focal["sum_p"].mean(),
                    "focal_pool": focal["pool"].mean(),
                    "mean_p": focal["sum_p"].sum() / focal["members"].sum(),
                    "mean_c_valid": valid["contribution"].mean(),
                    "c_gap": _leaver_gap(g),
                }
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dirs", nargs="+", required=True)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--sweep-seeds", default="45,46,47")
    ap.add_argument("--rival", default="ah_punisher")
    ap.add_argument("--baseline", default="never")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sim = sim_table(args.sim_dirs)
    sim = sim[sim["pairing"].str.endswith(f"_vs_{args.rival}")]
    # the sim path runs 100 episodes per seed against the sweep's thousands,
    # so its own spread is the yardstick any disagreement is judged against
    spread = (
        sim.groupby("name")[["focal_members", "focal_pool", "focal_contribution"]]
        .std(ddof=1)
        .add_prefix("sd_")
        .reset_index()
    )
    sim = sim.groupby("name", as_index=False).mean(numeric_only=True).merge(spread)

    seeds = [int(s) for s in args.sweep_seeds.split(",")]
    sweep = per_point(load_sweep(args.sweep), seeds=seeds)
    cols = [
        "focal_members",
        "focal_contribution",
        "focal_punishment",
        "focal_pool",
        "mean_p",
        "mean_c_valid",
        "c_gap",
    ]
    out = sim.merge(sweep[["name"] + cols], on="name", suffixes=("_sim", "_sweep"))
    for c in cols:
        out[f"d_{c}"] = out[f"{c}_sweep"] - out[f"{c}_sim"]
    keep = ["sd_focal_members", "sd_focal_pool", "sd_focal_contribution"]
    out = out[
        ["name"]
        + [f"{c}_{s}" for c in cols for s in ("sim", "sweep")]
        + [f"d_{c}" for c in cols]
        + keep
    ]
    out.to_csv(args.out, index=False)
    print(out.round(2).to_string(index=False))

    # The levels are what a common seat-size offset moves; the contrasts are
    # what every claim in this arm is actually made of, and they are the
    # comparison that decides whether the sweep can be trusted.
    base = args.baseline
    con = []
    for _, r in out.iterrows():
        if r["name"] == base:
            continue
        b = out[out["name"] == base].iloc[0]
        row = {"name": r["name"], "minus": base}
        for c in ("focal_contribution", "focal_pool", "focal_members"):
            row[f"{c}_sim"] = r[f"{c}_sim"] - b[f"{c}_sim"]
            row[f"{c}_sweep"] = r[f"{c}_sweep"] - b[f"{c}_sweep"]
            row[f"d_{c}"] = row[f"{c}_sweep"] - row[f"{c}_sim"]
        con.append(row)
    con = pd.DataFrame(con)
    con.to_csv(args.out.replace(".csv", "_contrasts.csv"), index=False)
    print("\n--- contrasts, the quantity every claim rests on ---")
    print(con.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
