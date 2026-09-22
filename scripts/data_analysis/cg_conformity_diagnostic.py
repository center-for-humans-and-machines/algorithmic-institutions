"""Diagnostic behind the conformity-mixture contribution experiment.

Asks why human group means spread ~0.84x individual spread while the sim sits
at the independence floor (~0.58), i.e. what mechanism produces the CG deficit.

Probes, human vs sim (reference stack, lin_multinomial punisher run by
default):
  1. Spread-ratio trajectory by round  (does human spread GROW = amplification?)
  2. Conditional-cooperation slope     (contribution[t] ~ group_mean[t-1], own[t-1])
  3. Boundary stickiness               (P(0->0), P(20->20))
  4. Within-agent persistence          (lag-1 autocorr of contribution;
                                        and of residual after round means)
  5. Participant-level variance share  (ICC: between-participant / total)
  6. Peer slope with own-history controls (human only): contribution[t] on own
     lags 1-3, the cumulative mean of own prior rounds, and others_mean[t-1] --
     checks the peer channel is not a collinearity artifact.

Run from the repo root:
    .venv/bin/python scripts/data_analysis/cg_conformity_diagnostic.py [parquet]

The optional argument overrides the sim `per_round.parquet` path, so the same
diagnostic can be rerun on a candidate simulation.
"""

import sys

sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aimanager.evaluation_suite import convert  # noqa: E402

HUMAN_PATH = "experiments/2group_8agent_50ep.csv"
DEFAULT_SIM_PATH = (
    "plots/simulation/23_2g8a_self_gnn_contr_gnn_switch/per_round.parquet"
)
GROUP_KEY = ["episode_id", "group_id", "round_number"]
AGENT_KEY = ["episode_id", "participant_code"]


def add_others_mean(df):
    """Leave-one-out mean contribution of the agent's group in each round."""
    d = df.dropna(subset=["contribution"]).copy()
    d = d.sort_values(AGENT_KEY + ["round_number"])
    gsum = d.groupby(GROUP_KEY)["contribution"].transform("sum")
    gn = d.groupby(GROUP_KEY)["contribution"].transform("count")
    d["others_mean"] = (gsum - d["contribution"]) / (gn - 1).replace(0, np.nan)
    return d


def consecutive(d):
    """Keep rows whose previous observation is the immediately prior round."""
    d = d.copy()
    d["prev_round"] = d.groupby(AGENT_KEY)["round_number"].shift(1)
    return d[d["round_number"] == d["prev_round"] + 1]


def spread_ratio_by_round(df):
    d = df.dropna(subset=["contribution"])
    out = {}
    for r, g in d.groupby("round_number"):
        gm = g.groupby(["episode_id", "group_id"])["contribution"].mean()
        out[r] = gm.std() / g["contribution"].std()
    return pd.Series(out)


def cc_slope(df):
    """OLS of contribution[t] on own[t-1] and others_mean[t-1]."""
    d = add_others_mean(df)
    d["own_prev"] = d.groupby(AGENT_KEY)["contribution"].shift(1)
    d["others_prev"] = d.groupby(AGENT_KEY)["others_mean"].shift(1)
    d = consecutive(d).dropna(subset=["own_prev", "others_prev", "contribution"])
    X = np.column_stack(
        [np.ones(len(d)), d["own_prev"].values, d["others_prev"].values]
    )
    beta, *_ = np.linalg.lstsq(X, d["contribution"].values, rcond=None)
    return beta  # [intercept, own_prev, others_prev]


def cc_slope_own_controls(df):
    """Peer slope controlling for own lags 1-3 and own cumulative history.

    Returns (labels, coefficients, n_obs). The own cumulative mean uses only
    rounds strictly before t, so it adds long-run own disposition on top of the
    three short lags.
    """
    d = add_others_mean(df)
    for lag in (1, 2, 3):
        d[f"own_l{lag}"] = d.groupby(AGENT_KEY)["contribution"].shift(lag)
    d["own_cummean"] = d.groupby(AGENT_KEY)["contribution"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    d["others_l1"] = d.groupby(AGENT_KEY)["others_mean"].shift(1)
    cols = ["own_l1", "own_l2", "own_l3", "own_cummean", "others_l1"]
    d = consecutive(d).dropna(subset=cols + ["contribution"])
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
    beta, *_ = np.linalg.lstsq(X, d["contribution"].values, rcond=None)
    return ["intercept"] + cols, beta, len(d)


def stickiness(df, val):
    d = df.dropna(subset=["contribution"]).copy()
    d = d.sort_values(AGENT_KEY + ["round_number"])
    d["prev"] = d.groupby(AGENT_KEY)["contribution"].shift(1)
    d = consecutive(d)
    at = d[d["prev"] == val]
    return (at["contribution"] == val).mean(), len(at)


def lag1_autocorr(df, demean_round=False):
    d = df.dropna(subset=["contribution"]).copy()
    c = d["contribution"].astype(float)
    if demean_round:
        c = c - d.groupby("round_number")["contribution"].transform("mean")
    d["c"] = c
    d = d.sort_values(AGENT_KEY + ["round_number"])
    d["prev"] = d.groupby(AGENT_KEY)["c"].shift(1)
    d = consecutive(d).dropna(subset=["prev"])
    return np.corrcoef(d["c"], d["prev"])[0, 1]


def icc(df):
    """Between-participant share of the round-demeaned contribution variance."""
    d = df.dropna(subset=["contribution"]).copy()
    d["c"] = d["contribution"] - d.groupby("round_number")["contribution"].transform(
        "mean"
    )
    pm = d.groupby(AGENT_KEY)["c"].agg(["mean", "count"])
    return pm["mean"].var() / d["c"].var()


def main(argv):
    sim_path = argv[1] if len(argv) > 1 else DEFAULT_SIM_PATH
    human = convert.load_human(HUMAN_PATH)
    sims = convert.load_sim(sim_path)
    run = [r for r in sims if "multinomial" in r]
    run = run[0] if run else sorted(sims)[0]
    sim = sims[run]
    print(f"sim parquet: {sim_path}")
    print(f"sim run: {run}\n")

    rows = {}
    for name, df in [("human", human), ("sim", sim)]:
        sr = spread_ratio_by_round(df)
        beta = cc_slope(df)
        p00, n0 = stickiness(df, 0)
        p2020, n20 = stickiness(df, 20)
        rows[name] = dict(
            ratio_r0_3=sr.loc[0:3].mean(),
            ratio_r4_11=sr.loc[4:11].mean(),
            ratio_r12_23=sr.loc[12:23].mean(),
            cc_own_prev=beta[1],
            cc_others_prev=beta[2],
            p_0_to_0=p00,
            n_at_0=n0,
            p_20_to_20=p2020,
            n_at_20=n20,
            ac1_raw=lag1_autocorr(df),
            ac1_demeaned=lag1_autocorr(df, demean_round=True),
            icc_participant=icc(df),
        )

    print(pd.DataFrame(rows).round(3).to_string())

    print("\nspread ratio by round (human / sim):")
    sr_frame = pd.DataFrame(
        {"human": spread_ratio_by_round(human), "sim": spread_ratio_by_round(sim)}
    )
    print(sr_frame.round(3).T.to_string())

    print("\npeer slope with own-history controls (human):")
    labels, beta, n = cc_slope_own_controls(human)
    for label, coef in zip(labels, beta):
        print(f"  {label:<12} {coef: .4f}")
    print(f"  n = {n}")


if __name__ == "__main__":
    main(sys.argv)
