"""Does the reward a manager is paid on explain the inverted targeting?

Reads one or two `per_round.parquet` files from
`configs/simulation/manager_testing/25_reward_targeting_cross_eval*.yml` and
turns them into the paired tables the question needs. The manager under test
always sits in group 0 and the clone in group 1, so every statistic below is
group 0's.

Three measurements, and each one is paired seed against seed:

  1. POLICY SHAPE. Mean punishment per RPA contribution bin, the evaluation
     suite's own bins, with the human and clone columns and the row counts.
     The headline is `shape_delta` = mean punishment at contribution 0 minus
     mean punishment at contribution 20. Humans run about +4.5, the clone
     about +3.0, an inverted manager negative.
  2. THE LEAVER DIAGNOSTIC. What leavers contributed minus what stayers
     contributed, from REALISED membership change -- `convert._derive_switching`
     compares a player's group at the decision round with its group at the next
     round. Nothing re-runs the switch predictor, which would disturb its
     recurrent state. A correctly targeted manager runs negative: the players
     it drives out are the ones who were giving less.
  3. BOTH OUTCOME MEASURES. The group's total pool and the pool per member,
     side by side, so the incentive claim is checked rather than assumed.

VALIDITY. The `contribution` column carries an imputed 9 for a player who gave
no input, and `env.punish` forces that player's punishment to 0. Averaging
those rows in would pull the 6-10 bin toward zero and invent contributions
nobody made, so every mean here is taken over `contribution_valid` rows only.
`convert.load_sim` does NOT mask them (`load_human` does, via
`player_no_input`), which is why this script works from the raw parquet and
applies the suite's switch labelling itself rather than taking the canonical
frame wholesale.

LEVEL VERSUS SHAPE. A manager that punishes half as hard has half the
`shape_delta` without having changed who it aims at, so `shape_delta_norm`
divides the shape by the manager's own mean punishment, and `targeting_rho`
-- the Spearman correlation between contribution and punishment -- drops the
level entirely, being invariant to any increasing rescaling. All three are
reported; a claim about targeting has to survive `targeting_rho`, which is the
one a level difference cannot fake.

THE NOISE FLOOR. With two parquets (the sim-seed 42 run and its 142 twin,
identical in every other key) every statistic is computed twice and the
difference between the two runs is this harness's own Monte-Carlo noise. A
paired arm difference means nothing until it clears that.

Usage:
    python scripts/rl_reward_targeting/measure_arms.py \\
        plots/simulation/25_reward_targeting_cross_eval/per_round.parquet \\
        --replicate \\
        plots/simulation/25_reward_targeting_cross_eval_s142/per_round.parquet
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aimanager.evaluation_suite.convert import (  # noqa: E402
    HUMAN_DATA_FILE,
    _derive_switching,
    load_human,
)
from aimanager.evaluation_suite.metrics import RPA_EDGES, RPA_LABELS  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "plots/data_analysis/evaluation/rl_manager_reward_targeting")

POOL = ["rl_s42", "rl_s43", "rl_s44"]
PERCAPITA = ["rl_pc_s42", "rl_pc_s43", "rl_pc_s44"]
SEEDS = [42, 43, 44]
CLONE = "lin_punisher"
MAX_CONTRIBUTION = 20.0
MPCR = 1.6


def short(name):
    """`ah group_switching managed by rl_pc_s42_vs_punisher` -> `rl_pc_s42`."""
    for a, b in (
        ("ah group_switching managed by ", ""),
        ("_vs_punisher", ""),
        ("_self", ""),
    ):
        name = name.replace(a, b)
    return name


def load_runs(path):
    """Raw parquet -> {manager: group-0 frame with the suite's switch labels}."""
    df = pd.read_parquet(path)
    df["run"] = df["run"].map(short)
    df = df.rename(columns={"episode": "episode_id"})
    out = {}
    for run, d in df.groupby("run"):
        d = _derive_switching(d.copy(), switch_every=4)
        out[run] = d[d["group_id"] == 0].reset_index(drop=True)
    return out


def bin_contribution(s):
    return pd.cut(s, RPA_EDGES, labels=RPA_LABELS).astype(str)


def shape_row(d):
    """Mean punishment per contribution bin, over valid rows only."""
    v = d[d["contribution_valid"].astype(bool)]
    return (
        v.groupby(bin_contribution(v["contribution"]))["punishment"]
        .mean()
        .reindex(RPA_LABELS)
    )


def shape_counts(d):
    v = d[d["contribution_valid"].astype(bool)]
    return bin_contribution(v["contribution"]).value_counts().reindex(RPA_LABELS)


def targeting_rho(contribution, punishment):
    """Spearman rho between contribution and punishment. Level cannot touch it.

    `shape_delta` is a difference of two punishment means, so a manager that
    punishes half as hard has half the shape_delta without having re-aimed at
    anybody. Dividing by the level fixes that only as long as the level is not
    near zero, and one of these managers punishes on 7% of rounds. A rank
    correlation is invariant to ANY increasing rescaling of punishment, so it
    separates who is aimed at from how hard, and it uses every row rather than
    the two extreme bins.

    Human sign convention: negative means punishment falls as contribution
    rises -- punish the free-rider, spare the full contributor. Positive is the
    inversion.
    """
    ok = contribution.notna() & punishment.notna()
    if ok.sum() < 2 or punishment[ok].nunique() < 2:
        return np.nan
    return float(st.spearmanr(contribution[ok], punishment[ok]).statistic)


def human_shape():
    """The human reference. `load_human` already NaNs both invalid columns."""
    h = load_human(os.path.join(ROOT, HUMAN_DATA_FILE))
    h = h.dropna(subset=["punishment", "contribution"])
    shape = h.groupby(bin_contribution(h["contribution"]))["punishment"].mean()
    counts = bin_contribution(h["contribution"]).value_counts()
    return shape.reindex(RPA_LABELS), counts.reindex(RPA_LABELS), h


def leaver_gap(d, valid_col="contribution_valid"):
    """Mean contribution of leavers minus that of stayers, at decision rounds.

    Two estimators. `pooled` throws every decision row into one contrast;
    `within` averages the contrast computed inside each (episode, round), which
    cannot be moved by a round that happens to hold many leavers or by drift
    over the episode. They should agree; where they do not, the pooled one is
    the one being confounded.
    """
    d = d[d["switch_mask"].astype(bool)]
    if valid_col in d.columns:
        d = d[d[valid_col].astype(bool)]
    d = d.dropna(subset=["contribution"])
    lv = d[d["does_switch"].astype(bool)]["contribution"]
    stay = d[~d["does_switch"].astype(bool)]["contribution"]
    pooled = float(lv.mean() - stay.mean()) if len(lv) and len(stay) else np.nan

    by_side = (
        d.groupby(["episode_id", "round_number", d["does_switch"].astype(bool)])[
            "contribution"
        ]
        .mean()
        .unstack()
    )
    if True in by_side.columns and False in by_side.columns:
        per_round = (by_side[True] - by_side[False]).dropna()
    else:
        per_round = pd.Series(dtype=float)
    within = float(per_round.mean()) if len(per_round) else np.nan
    within_ci = (
        1.96 * float(per_round.std()) / np.sqrt(len(per_round))
        if len(per_round) > 1
        else np.nan
    )
    return {
        "leaver_gap": pooled,
        "leaver_gap_within": within,
        "leaver_gap_within_ci95": within_ci,
        "n_leavers": int(len(lv)),
        "n_stayers": int(len(stay)),
        "n_contrast_rounds": int(len(per_round)),
    }


def outcomes(d):
    """Both pool measures, per episode so the interval is honest.

    `common_good` in the parquet is the env's state field, which is the pool
    ALREADY divided by the valid headcount -- the share one member receives,
    not the group total. (`payoff = 20 - c - p + common_good` in
    `simulate.mem_to_df` is what fixes that reading.) The group total is
    therefore `common_good * n_valid`, rebuilt per round below and
    cross-checked against `1.6 * sum(c) - sum(p)` by `pool_identity_residual`.
    """
    v = d[d["contribution_valid"].astype(bool)]
    per_round = v.groupby(["run", "episode_id", "round_number"]).agg(
        pool_per_member=("common_good", "mean"),
        n_valid=("participant_code", "size"),
        contribution=("contribution", "mean"),
        punishment=("punishment", "mean"),
    )
    per_round["group_total_pool"] = per_round["pool_per_member"] * per_round["n_valid"]
    size = (
        d.groupby(["run", "episode_id", "round_number"])["participant_code"]
        .size()
        .rename("group_size")
    )
    per_round = per_round.join(size)
    ep = per_round.groupby(["run", "episode_id"]).mean(numeric_only=True)
    n = ep.groupby("run").size()
    agg = ep.groupby("run").mean()
    sd = ep.groupby("run").std()
    for c in ("group_total_pool", "pool_per_member", "contribution", "group_size"):
        agg[c + "_ci95"] = 1.96 * sd[c] / np.sqrt(n)
    agg["n_episodes"] = n
    return agg, ep


OUTCOME_COLS = [
    "contribution",
    "group_size",
    "pool_per_member",
    "group_total_pool",
]


def contrasts(ep, baseline="never"):
    """Each manager minus `never`, on both pool measures, with intervals.

    This is the premise under test, re-measured in this run rather than
    quoted: correct targeting is supposed to raise contribution, cost members,
    and have those two cancel on the UNDIVIDED pool while leaving a gain on the
    per-member share. If that is so, a manager paid the undivided pool earns
    nothing for targeting correctly.

    The interval is an unpaired Welch interval over episode means. Episodes are
    not paired across runs: `reseed_per_run` gives every run the same episode-0
    draw, but a manager that punishes differently makes the players act
    differently, so the streams diverge from episode 1 on and a paired interval
    would claim a pairing the data does not have.
    """
    if baseline not in ep.index.get_level_values("run"):
        return pd.DataFrame()
    base = ep.xs(baseline, level="run")
    rows = []
    for run in ep.index.get_level_values("run").unique():
        if run == baseline:
            continue
        cur = ep.xs(run, level="run")
        rec = {"manager": run}
        for c in OUTCOME_COLS:
            a, b = cur[c], base[c]
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            diff = float(a.mean() - b.mean())
            rec[f"d_{c}"] = diff
            rec[f"d_{c}_lo"] = diff - 1.96 * se
            rec[f"d_{c}_hi"] = diff + 1.96 * se
        rows.append(rec)
    return pd.DataFrame(rows).set_index("manager")


def pool_identity_residual(d):
    """Guard: does `common_good * n_valid` really equal 1.6*sum(c) - sum(p)?

    If it does not, `group_total_pool` above is not the group total and every
    outcome claim built on it is wrong. Run over valid rows, per group-round.
    """
    v = d[d["contribution_valid"].astype(bool)]
    g = v.groupby(["run", "episode_id", "round_number"]).agg(
        cg=("common_good", "mean"),
        n=("participant_code", "size"),
        sc=("contribution", "sum"),
        sp=("punishment", "sum"),
    )
    return float((g["cg"] * g["n"] - (MPCR * g["sc"] - g["sp"])).abs().max())


def measure(path):
    """One parquet -> (per-manager statistics, shape table, count table)."""
    runs = load_runs(path)
    hshape, hcounts, human = human_shape()
    shapes = {"human managers": hshape}
    counts = {"human managers": hcounts}
    rows = []
    for name in sorted(runs):
        d = runs[name]
        s = shape_row(d)
        shapes[name] = s
        counts[name] = shape_counts(d)
        v = d[d["contribution_valid"].astype(bool)]
        level = float(v["punishment"].mean())
        delta = float(s["{0}"] - s["{20}"])
        rec = {
            "manager": name,
            "shape_delta": delta,
            "mean_punishment": level,
            "shape_delta_norm": delta / level if level else np.nan,
            "targeting_rho": targeting_rho(v["contribution"], v["punishment"]),
            "punish_rate": float((v["punishment"] > 0).mean()),
            "mean_given_positive": float(v[v["punishment"] > 0]["punishment"].mean()),
            "n_agent_rounds": int(len(v)),
        }
        rec.update(leaver_gap(d))
        rows.append(rec)
    stats = pd.DataFrame(rows).set_index("manager")

    hdelta = float(hshape["{0}"] - hshape["{20}"])
    hlevel = float(human["punishment"].mean())
    hrec = {
        "shape_delta": hdelta,
        "mean_punishment": hlevel,
        "shape_delta_norm": hdelta / hlevel,
        "targeting_rho": targeting_rho(human["contribution"], human["punishment"]),
        "punish_rate": float((human["punishment"] > 0).mean()),
        "mean_given_positive": float(
            human[human["punishment"] > 0]["punishment"].mean()
        ),
        "n_agent_rounds": int(len(human)),
    }
    hrec.update(leaver_gap(human))
    stats.loc["human managers"] = pd.Series(hrec)

    out, ep = outcomes(pd.concat(runs.values()))
    stats = stats.join(
        out[
            [
                "group_total_pool",
                "group_total_pool_ci95",
                "pool_per_member",
                "pool_per_member_ci95",
                "contribution",
                "contribution_ci95",
                "group_size",
                "group_size_ci95",
                "n_episodes",
            ]
        ]
    )

    shape = pd.DataFrame(shapes)
    shape.index.name = "contribution_bin"
    cnt = pd.DataFrame(counts)
    cnt.index.name = "contribution_bin"
    residual = pool_identity_residual(pd.concat(runs.values()))
    return stats, shape, cnt, residual, contrasts(ep)


PAIRED_COLS = [
    "shape_delta",
    "shape_delta_norm",
    "targeting_rho",
    "leaver_gap",
    "leaver_gap_within",
    "mean_punishment",
    "punish_rate",
    "group_total_pool",
    "pool_per_member",
    "contribution",
    "group_size",
]


def paired(stats):
    """per-capita minus pool, seed by seed. Three pairs, reported individually.

    Three seeds is only informative because the pairing holds everything but
    the reward fixed, so the mean of the three is not the result -- the three
    differences are, and they are printed as such.
    """
    rows = []
    for seed, a, b in zip(SEEDS, POOL, PERCAPITA):
        if a not in stats.index or b not in stats.index:
            continue
        rec = {"seed": seed}
        for c in PAIRED_COLS:
            rec[f"pool_{c}"] = stats.loc[a, c]
            rec[f"pc_{c}"] = stats.loc[b, c]
            rec[f"d_{c}"] = stats.loc[b, c] - stats.loc[a, c]
        rows.append(rec)
    if not rows:
        return pd.DataFrame(index=pd.Index([], name="seed"))
    return pd.DataFrame(rows).set_index("seed")


def power(pair, stats, floor=None):
    """Is a null here tight, or merely underpowered? They are not the same.

    Per statistic:
      * `mean_d`, `sd_d`, and the paired-t 95% interval on three pairs
        (t = 4.303 with 2 df, so the interval is wide by construction);
      * `mde80`, the smallest paired difference three seeds could have
        detected at 80% power given the spread the three pairs actually
        showed -- this is what makes a null readable. A null whose interval
        excludes the effect that would have mattered is tight; a null whose
        `mde80` is larger than that effect is an absence of evidence.
      * `seed_spread_pool` / `seed_spread_pc`, the range across the three
        seeds inside each arm. When the arm difference is small against the
        within-arm spread, the pairing is carrying the whole design.
      * `noise_floor`, the mean absolute movement of the six learned rows
        between the two sim seeds. A paired difference under this is not a
        difference at all.
    """
    t_crit = float(st.t.ppf(0.975, len(pair) - 1))
    t_pow = float(st.t.ppf(0.80, len(pair) - 1))
    rows = []
    for c in PAIRED_COLS:
        d = pair[f"d_{c}"].astype(float)
        sd = float(d.std(ddof=1))
        se = sd / np.sqrt(len(d))
        pool_v = [float(stats.loc[m, c]) for m in POOL if m in stats.index]
        pc_v = [float(stats.loc[m, c]) for m in PERCAPITA if m in stats.index]
        rec = {
            "statistic": c,
            "mean_d": float(d.mean()),
            "sd_d": sd,
            "ci95_lo": float(d.mean()) - t_crit * se,
            "ci95_hi": float(d.mean()) + t_crit * se,
            "mde80": sd * (t_crit + t_pow) / np.sqrt(len(d)),
            "seed_spread_pool": max(pool_v) - min(pool_v) if pool_v else np.nan,
            "seed_spread_pc": max(pc_v) - min(pc_v) if pc_v else np.nan,
        }
        if floor is not None and c in floor.columns:
            learned = [m for m in POOL + PERCAPITA if m in floor.index]
            rec["noise_floor"] = float(floor.loc[learned, c].mean())
        rows.append(rec)
    return pd.DataFrame(rows).set_index("statistic")


def noise_floor(stats_a, stats_b):
    """|seed 142 - seed 42| per manager per statistic: the harness's own noise."""
    common = stats_a.index.intersection(stats_b.index)
    d = (stats_b.loc[common, PAIRED_COLS] - stats_a.loc[common, PAIRED_COLS]).abs()
    d.index.name = "manager"
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("per_round")
    ap.add_argument("--replicate", default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    stats, shape, cnt, residual, contr = measure(args.per_round)
    stats.to_csv(os.path.join(args.out, "shape_level_leaver.csv"))
    shape.to_csv(os.path.join(args.out, "policy_shape.csv"))
    cnt.to_csv(os.path.join(args.out, "policy_shape_n.csv"))
    contr.to_csv(os.path.join(args.out, "incentive_contrasts.csv"))
    pd.DataFrame([{"pool_identity_max_residual": residual}]).to_csv(
        os.path.join(args.out, "guards.csv"), index=False
    )
    pair = paired(stats)
    pair.to_csv(os.path.join(args.out, "paired_differences.csv"))

    print(f"\npool identity max residual: {residual:.3e}  (want ~0)\n")
    print("--- policy shape: mean punishment per contribution bin ---")
    print(shape.round(3).to_string(), "\n")
    print("--- rows behind each mean ---")
    print(cnt.to_string(), "\n")
    print("--- shape, level, leaver gap, outcomes ---")
    cols = [
        "shape_delta",
        "shape_delta_norm",
        "targeting_rho",
        "mean_punishment",
        "punish_rate",
        "leaver_gap",
        "leaver_gap_within",
        "group_total_pool",
        "pool_per_member",
        "contribution",
        "group_size",
    ]
    print(stats[cols].round(3).to_string(), "\n")
    print("--- paired: per-capita minus pool, seed by seed ---")
    if len(pair):
        d = pair[[f"d_{c}" for c in PAIRED_COLS]]
        print(d.round(3).to_string(), "\n")
        print("mean of the three paired differences:")
        print(d.mean().round(3).to_string(), "\n")
    else:
        print("no paired seeds in this parquet\n")

    print("--- the premise, re-measured: each manager minus `never` ---")
    print(contr.round(3).to_string(), "\n")

    nf = None
    if args.replicate:
        stats_b, shape_b, cnt_b, residual_b, contr_b = measure(args.replicate)
        contr_b.to_csv(os.path.join(args.out, "incentive_contrasts_s142.csv"))
        stats_b.to_csv(os.path.join(args.out, "shape_level_leaver_s142.csv"))
        shape_b.to_csv(os.path.join(args.out, "policy_shape_s142.csv"))
        pair_b = paired(stats_b)
        pair_b.to_csv(os.path.join(args.out, "paired_differences_s142.csv"))
        nf = noise_floor(stats, stats_b)
        nf.to_csv(os.path.join(args.out, "noise_floor.csv"))
        print(f"replicate pool identity max residual: {residual_b:.3e}\n")
        print("--- noise floor: |sim seed 142 - sim seed 42|, per manager ---")
        print(nf.round(3).to_string(), "\n")
        print("--- paired differences, replicate ---")
        print(pair_b[[f"d_{c}" for c in PAIRED_COLS]].round(3).to_string(), "\n")

    if len(pair) > 1:
        pw = power(pair, stats, nf)
        pw.to_csv(os.path.join(args.out, "power.csv"))
        print("--- tight null or absence of evidence? ---")
        print(pw.round(3).to_string(), "\n")
        if args.replicate:
            pw_b = power(pair_b, stats_b, nf)
            pw_b.to_csv(os.path.join(args.out, "power_s142.csv"))
            print("--- the same, on the replicate ---")
            print(pw_b.round(3).to_string(), "\n")


if __name__ == "__main__":
    main()
