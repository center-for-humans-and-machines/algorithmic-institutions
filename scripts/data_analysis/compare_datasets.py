"""Compare legacy pilot data vs group switching data.

Usage:
    python scripts/data_analysis/compare_datasets.py

Prints side-by-side statistics for the legacy and new datasets
to identify structural differences that may explain training gaps.
"""

import pandas as pd
import numpy as np


LEGACY_PATH = "experiments/pilot_random1_player_round_slim.csv"
NEW_PATH = (
    "experiments/"
    "group_switching_human_human_group_switching_8_agents.csv"
)


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


def main():
    legacy = pd.read_csv(LEGACY_PATH)
    new = pd.read_csv(NEW_PATH)

    summarize(legacy, "Legacy pilot data")
    summarize(new, "Group switching data")

    # Direct comparison
    print("=== Comparison ===")
    print(
        f"{'Metric':<30s} {'Legacy':>10s} {'New':>10s}"
    )
    print("-" * 52)
    rows = [
        (
            "Episodes",
            legacy["episode_id"].nunique(),
            new["episode_id"].nunique(),
        ),
        ("Total rows", len(legacy), len(new)),
        (
            "Mean contribution",
            legacy["contribution"].mean(),
            new["contribution"].mean(),
        ),
        (
            "Std contribution",
            legacy["contribution"].std(),
            new["contribution"].std(),
        ),
        (
            "Mean common_good",
            legacy["common_good"].mean(),
            new["common_good"].mean(),
        ),
        (
            "player_no_input rate",
            legacy["player_no_input"].mean(),
            new["player_no_input"].mean(),
        ),
        (
            "Contribution entropy",
            _entropy(legacy["contribution"]),
            _entropy(new["contribution"]),
        ),
    ]
    for label, v1, v2 in rows:
        print(f"{label:<30s} {v1:>10.4f} {v2:>10.4f}")


def _entropy(series):
    """Shannon entropy of a discrete distribution."""
    counts = series.value_counts(normalize=True)
    return -(counts * np.log2(counts + 1e-12)).sum()


if __name__ == "__main__":
    main()
