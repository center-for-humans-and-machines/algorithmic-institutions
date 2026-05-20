"""Per-episode payoff vs the dummy-at-p=0 baseline.

Usage:
    python experiment_analysis/payoff_vs_dummy.py

For each dataset, compute per-episode mean per-agent payoff. Compare
against the corresponding AH-dummy-at-p=0 baseline (the no-manager
sim's payoff in PR #96). Surfaces whether the pilot contains episodes
where actual human management out-performed laissez-faire — i.e.
whether the training data demonstrates "management adds value".
"""

import pandas as pd


# Mean per-round per-agent payoff under sustained p=0 in PR #96 sims.
# 50ep AH:  ~25.0 (flat across 24 rounds in p0_vs_p0)
# v4 BC AH: ~24.0 (declines 26.7 -> 22.4 over 24 rounds; episode mean ~24)
DUMMY_BASELINE = {
    "Legacy": 24.0,
    "GS (50 ep)": 25.0,
}

DATASETS = [
    ("Legacy", "experiments/pilot_random1_player_round_slim.csv"),
    ("GS (50 ep)", "experiments/2group_8agent_50ep.csv"),
]


def compute_payoff(df):
    """Add a `payoff_calc` column matching simulate.py:load_pilot_data.

    `common_good` in both datasets is the group pool. Divide by valid
    contributors per (episode, round, group) to get per-capita, then
    payoff = 20 - contribution - punishment + per_capita_common_good.
    """
    df = df.copy()
    valid = (df["player_no_input"] == 0).astype(int)
    n_valid = (
        valid.groupby(
            [df["episode_id"], df["round_number"], df["group_id"]]
        )
        .transform("sum")
        .clip(lower=1)
    )
    cg_per_capita = df["common_good"] / n_valid
    df["payoff_calc"] = (
        20 - df["contribution"] - df["punishment"] + cg_per_capita
    )
    return df


def episode_payoff(df):
    """Per-episode mean per-agent payoff (computed)."""
    return compute_payoff(df).groupby("episode_id")["payoff_calc"].mean()


def main():
    for name, path in DATASETS:
        df = pd.read_csv(path)
        ep = episode_payoff(df)
        baseline = DUMMY_BASELINE[name]
        above = (ep > baseline).sum()
        n = len(ep)
        print(f"=== {name} ===")
        print(f"Episodes: {n}    Baseline (dummy p=0): {baseline:.2f}")
        print(
            f"Episodes above baseline: {above}/{n} "
            f"({above / n * 100:.1f}%)"
        )
        print(
            f"Payoff distribution (per-episode mean per-agent):"
        )
        print(ep.describe().to_string())
        print(f"Top 5 episodes: {ep.nlargest(5).round(2).to_dict()}")
        print(f"Bottom 5 episodes: {ep.nsmallest(5).round(2).to_dict()}")
        print()


if __name__ == "__main__":
    main()
