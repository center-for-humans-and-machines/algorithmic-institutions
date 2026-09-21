"""Rank the rule-based managers by the common good they produce.

Reads the sweep's `per_round.parquet` files and reports, for every manager
and for the human games, the common good on TWO accountings:

* **env** -- what `ArtificialHumanEnv` computes: a timed-out player's
  contribution and the punishment aimed at them are both dropped, and that
  player's own payoff is thrown away (review findings D1 / D2).
* **corrected** -- computed straight from contributions and punishments:
  punishment aimed at a timed-out player is charged to the pool (the
  artificial humans were shown it, so it was really spent) and the
  timed-out player is paid `20 - 0 - 0 + common_good`, which is what the
  real game paid them.

Usage:
    python scripts/data_analysis/rule_manager_sweep_report.py \\
        plots/simulation/24_rule_managers_s42_a ... --tag s42
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HUMAN_DATA = "experiments/2group_8agent_50ep.csv"
OUT_DIR = "plots/data_analysis/evaluation/rule_based_managers"
ENDOWMENT = 20.0
MPCR = 1.6

RUN_RE = re.compile(r"^ah .* managed by (?P<manager>.+)_self$")


def manager_name(run):
    m = RUN_RE.match(run)
    return m.group("manager") if m else run


def group_rounds(df, keys):
    """Per (…, episode, round, group) accounting on both conventions."""
    valid = df["contribution_valid"].astype(bool)
    df = df.assign(
        c_eff=df["contribution"].where(valid, 0.0).astype(float),
        p_valid=df["punishment"].where(valid, 0.0).astype(float),
        p_all=df["punishment"].astype(float),
        n_valid=valid.astype(float),
    )
    g = df.groupby(keys, as_index=False).agg(
        n=("c_eff", "size"),
        n_valid=("n_valid", "sum"),
        sum_c=("c_eff", "sum"),
        sum_p_env=("p_valid", "sum"),
        sum_p_all=("p_all", "sum"),
    )
    nv = g["n_valid"].clip(lower=1)
    g["pool_env"] = MPCR * g["sum_c"] - g["sum_p_env"]
    g["pool_corr"] = MPCR * g["sum_c"] - g["sum_p_all"]
    g.loc[g["n_valid"] == 0, ["pool_env", "pool_corr"]] = 0.0
    # env: only valid contributors are paid
    g["payoff_env"] = ENDOWMENT * g["n_valid"] + 0.6 * g["sum_c"] - 2 * g["sum_p_env"]
    # corrected: everyone in the group is paid, punishment on a timed-out
    # player is charged
    g["payoff_corr"] = (
        ENDOWMENT * g["n"] - g["sum_c"] - g["sum_p_all"] + g["n"] * g["pool_corr"] / nv
    )
    g.loc[g["n_valid"] == 0, ["payoff_env", "payoff_corr"]] = 0.0
    return g


def episode_rounds(g, keys):
    """Population totals: both groups of a round summed."""
    return g.groupby(keys, as_index=False)[
        ["pool_env", "pool_corr", "payoff_env", "payoff_corr", "sum_c", "sum_p_all"]
    ].sum()


def bootstrap_ci(per_episode, n_boot=2000, seed=0):
    """95% interval of the mean over episodes, resampling episodes."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(per_episode, dtype=float)
    draws = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise_sim(df, label):
    er = episode_rounds(
        group_rounds(df, ["episode", "round_number", "group_id"]),
        ["episode", "round_number"],
    )
    per_ep = er.groupby("episode")[
        ["pool_env", "pool_corr", "payoff_env", "payoff_corr"]
    ].mean()
    valid = df["contribution_valid"].astype(bool)
    # the policy statistics are about what a manager DOES, so they are taken
    # where the manager acted: simulated managers always act, human ones gave
    # no input on 4.2% of rows
    acted = df.get("manager_valid", pd.Series(True, index=df.index)).astype(bool)
    v = df[valid & acted]
    to = df[~valid & acted]
    lo_env, hi_env = bootstrap_ci(per_ep["pool_env"])
    lo_corr, hi_corr = bootstrap_ci(per_ep["pool_corr"])
    return {
        "manager": label,
        "cg_env": per_ep["pool_env"].mean(),
        "cg_env_lo": lo_env,
        "cg_env_hi": hi_env,
        "cg_corr": per_ep["pool_corr"].mean(),
        "cg_corr_lo": lo_corr,
        "cg_corr_hi": hi_corr,
        "payoff_env": per_ep["payoff_env"].mean(),
        "payoff_corr": per_ep["payoff_corr"].mean(),
        "mean_contribution": v["contribution"].mean(),
        "share_punished": float((v["punishment"] > 0).mean()),
        "mean_punishment": float(v["punishment"].mean()),
        "mean_punishment_pos": float(
            v.loc[v["punishment"] > 0, "punishment"].mean()
            if (v["punishment"] > 0).any()
            else 0.0
        ),
        "timeout_rate": float((~valid).mean()),
        "timeout_share_punished": float(
            (to["punishment"] > 0).mean() if len(to) else 0.0
        ),
        "timeout_mean_punishment": float(to["punishment"].mean() if len(to) else 0.0),
        "timeout_punishment_share": float(
            to["punishment"].sum() / max(df.loc[acted, "punishment"].sum(), 1e-9)
        ),
        "n_episodes": int(df["episode"].nunique()),
    }


def policy_shape(df, label):
    acted = df.get("manager_valid", pd.Series(True, index=df.index)).astype(bool)
    v = df[df["contribution_valid"].astype(bool) & acted]
    out = (
        v.groupby(v["contribution"].astype(int))["punishment"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "punishment", "size": "n"})
        .reset_index()
    )
    out["manager"] = label
    return out


def load_human():
    df = pd.read_csv(HUMAN_DATA)
    keep = df.groupby("pair_id")["episode_id"].transform("min")
    df = df[df["episode_id"] == keep].copy()
    df["contribution_valid"] = df["player_no_input"] == 0
    df["manager_valid"] = df["manager_no_input"] == 0
    df["punishment"] = df["punishment"].fillna(0.0)
    df["contribution"] = df["contribution"].fillna(0.0)
    # the raw file already carries an `episode` column (index within the
    # session); `episode_id` is the game and is what the sim's `episode` is
    return df.drop(columns=["episode"]).rename(columns={"episode_id": "episode"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dirs", nargs="+")
    ap.add_argument("--tag", default="s42")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    frames = []
    for d in args.sim_dirs:
        p = os.path.join(d, "per_round.parquet")
        df = pd.read_parquet(p)
        df["manager"] = df["run"].map(manager_name)
        frames.append(df)
    sim = pd.concat(frames, ignore_index=True)

    rows, shapes = [], []
    for label, sub in sim.groupby("manager"):
        rows.append(summarise_sim(sub, label))
        shapes.append(policy_shape(sub, label))

    human = load_human()
    rows.append(summarise_sim(human, "human managers (real)"))
    shapes.append(policy_shape(human, "human managers (real)"))

    table = pd.DataFrame(rows).sort_values("cg_env", ascending=False)
    table["rank_env"] = table["cg_env"].rank(ascending=False).astype(int)
    table["rank_corr"] = table["cg_corr"].rank(ascending=False).astype(int)
    table["rank_payoff_env"] = table["payoff_env"].rank(ascending=False).astype(int)
    table["rank_payoff_corr"] = table["payoff_corr"].rank(ascending=False).astype(int)

    os.makedirs(args.out_dir, exist_ok=True)
    table_path = os.path.join(args.out_dir, f"summary_{args.tag}.csv")
    table.to_csv(table_path, index=False)
    shape = pd.concat(shapes, ignore_index=True)
    shape_path = os.path.join(args.out_dir, f"policy_shape_{args.tag}.csv")
    shape.to_csv(shape_path, index=False)
    print(f"wrote {table_path}\nwrote {shape_path}\n")

    cols = [
        "manager",
        "cg_env",
        "cg_env_lo",
        "cg_env_hi",
        "cg_corr",
        "payoff_env",
        "payoff_corr",
        "mean_contribution",
        "share_punished",
        "mean_punishment",
        "mean_punishment_pos",
        "timeout_share_punished",
        "timeout_punishment_share",
        "rank_env",
        "rank_corr",
    ]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    make_figures(table, shape, args.out_dir, args.tag)


def make_figures(table, shape, out_dir, tag):
    sns.set_theme(style="whitegrid")

    order = table.sort_values("cg_env")["manager"].tolist()
    long = table.melt(
        id_vars="manager",
        value_vars=["cg_env", "cg_corr"],
        var_name="accounting",
        value_name="common_good",
    )
    long["accounting"] = long["accounting"].map(
        {"cg_env": "environment", "cg_corr": "corrected"}
    )
    plt.figure(figsize=(8, 0.34 * len(order) + 2.2))
    ax = sns.barplot(
        data=long, y="manager", x="common_good", hue="accounting", order=order
    )
    err = table.set_index("manager")
    for m in order:
        y = order.index(m)
        ax.plot(
            [err.loc[m, "cg_env_lo"], err.loc[m, "cg_env_hi"]],
            [y - 0.2, y - 0.2],
            color="black",
            lw=1.1,
        )
    ax.set_xlabel("common good per round, both groups (mean over episodes)")
    ax.set_ylabel("")
    ax.set_title(f"Common good by manager ({tag}); bars = 95% episode bootstrap")
    plt.tight_layout()
    p = os.path.join(out_dir, f"common_good_ranking_{tag}.jpg")
    plt.savefig(p, dpi=140)
    plt.close()
    print(f"wrote {p}")

    keep = shape[shape["n"] >= 20]
    plt.figure(figsize=(9, 5.5))
    ax = sns.lineplot(
        data=keep,
        x="contribution",
        y="punishment",
        hue="manager",
        marker="o",
        markersize=4,
    )
    ax.set_xlabel("contribution the manager is responding to")
    ax.set_ylabel("mean punishment")
    ax.set_title(f"Policy shape ({tag}); valid player-rounds, bins with n >= 20")
    ax.set_xticks(range(0, 21, 2))
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    plt.tight_layout()
    p = os.path.join(out_dir, f"policy_shape_{tag}.jpg")
    plt.savefig(p, dpi=140)
    plt.close()
    print(f"wrote {p}")

    plt.figure(figsize=(7, 5))
    ax = sns.scatterplot(
        data=table, x="mean_punishment", y="cg_env", hue="manager", s=60, legend=False
    )
    for _, r in table.iterrows():
        ax.annotate(
            r["manager"],
            (r["mean_punishment"], r["cg_env"]),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set_xlabel("mean punishment per valid agent-round")
    ax.set_ylabel("common good per round (environment accounting)")
    ax.set_title(f"Does punishing pay? ({tag})")
    plt.tight_layout()
    p = os.path.join(out_dir, f"punishment_vs_common_good_{tag}.jpg")
    plt.savefig(p, dpi=140)
    plt.close()
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
