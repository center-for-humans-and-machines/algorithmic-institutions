"""Compare legacy pilot data vs group switching data.

Usage:
    python scripts/data_analysis/compare_datasets.py

Prints side-by-side statistics for the legacy and new datasets
to identify structural differences that may explain training gaps.
"""

import pandas as pd
import numpy as np


DATASETS = [
    ("Legacy", "experiments/pilot_random1_player_round_slim.csv"),
    (
        "GS (13 ep)",
        "experiments/group_switching_human_human_group_switching_8_agents.csv",
    ),
    ("GS (35 ep)", "experiments/2group_8agent_35ep.csv"),
]


def summarize(df, name):
    print(f"=== {name} ===")
    print(f"Shape: {df.shape}")
    print(f"Episodes: {df['episode_id'].nunique()}")
    players = df.groupby("episode_id")["player_id"].nunique()
    print(f"Players per episode: {sorted(players.unique())}")
    print(f"Rounds: {df['round_number'].nunique()}")
    print(f"Total agent-rounds: {len(df)}")
    print()

    print("contribution stats:")
    print(df["contribution"].describe().to_string())
    print()

    print("common_good stats:")
    print(df["common_good"].describe().to_string())
    print()

    print(
        f"player_no_input rate: {df['player_no_input'].mean():.4f}"
    )
    print(
        f"manager_no_input rate: "
        f"{df['manager_no_input'].mean():.4f}"
    )
    print(f"group_id values: {sorted(df['group_id'].unique())}")
    print()

    # Contribution distribution
    print("contribution value counts (top 10):")
    vc = df["contribution"].value_counts().sort_index()
    print(vc.to_string())
    print()

    # Per-round mean contribution
    round_mean = (
        df.groupby("round_number")["contribution"].mean()
    )
    print("mean contribution per round:")
    print(round_mean.to_string())
    print()


def _trajectory(df):
    """Return (first_round_mean, last_round_mean) of contribution."""
    per_round = df.groupby("round_number")["contribution"].mean()
    return per_round.iloc[0], per_round.iloc[-1]


def main():
    dfs = [(name, pd.read_csv(p)) for name, p in DATASETS]

    for name, df in dfs:
        summarize(df, name)

    # Direct comparison
    def fmt(v, is_int=False):
        if is_int:
            return f"{int(v):,}"
        return f"{v:.4f}"

    def players(df):
        return sorted(
            df.groupby("episode_id")["player_id"].nunique().unique()
        )[0]

    traj = {name: _trajectory(df) for name, df in dfs}

    rows = [
        ("Episodes", [df["episode_id"].nunique() for _, df in dfs], True),
        ("Total rows", [len(df) for _, df in dfs], True),
        ("Players/episode", [players(df) for _, df in dfs], True),
        (
            "Rounds",
            [df["round_number"].nunique() for _, df in dfs],
            True,
        ),
        (
            "Mean contribution",
            [df["contribution"].mean() for _, df in dfs],
            False,
        ),
        (
            "Contribution=20 share",
            [(df["contribution"] == 20).mean() for _, df in dfs],
            False,
        ),
        (
            "Contribution entropy (bits)",
            [_entropy(df["contribution"]) for _, df in dfs],
            False,
        ),
        (
            "Traj first-round mean",
            [traj[name][0] for name, _ in dfs],
            False,
        ),
        (
            "Traj last-round mean",
            [traj[name][1] for name, _ in dfs],
            False,
        ),
        (
            "Mean common_good",
            [df["common_good"].mean() for _, df in dfs],
            False,
        ),
        (
            "player_no_input rate",
            [df["player_no_input"].mean() for _, df in dfs],
            False,
        ),
    ]

    col_w = 14
    header = f"{'Metric':<30s}" + "".join(
        f"{name:>{col_w}s}" for name, _ in dfs
    )
    print("\n=== Comparison ===")
    print(header)
    print("-" * len(header))
    for label, vals, is_int in rows:
        line = f"{label:<30s}"
        for v in vals:
            line += f"{fmt(v, is_int):>{col_w}s}"
        print(line)


def _entropy(series):
    """Shannon entropy of a discrete distribution."""
    counts = series.value_counts(normalize=True)
    return -(counts * np.log2(counts + 1e-12)).sum()


if __name__ == "__main__":
    main()
