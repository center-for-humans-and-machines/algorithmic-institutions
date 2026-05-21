"""Per-episode payoff vs the dummy-at-p=0 baseline.

Usage:
    python experiment_analysis/payoff_vs_dummy.py

For each dataset, compute per-episode focus-group payoff in two ways:

  avg: mean per-agent payoff over (rounds, agents in group 0)
  sum: mean per-round sum payoff over agents in group 0
       (group size can drift via switching in 2g8a, so the sum
        captures the manager's ability to attract agents in addition
        to per-agent welfare)

Compare each metric to the corresponding AH-dummy-at-p=0 baseline
(PR #96 sims, focus group only). Surfaces whether the pilot contains
episodes where human management out-performed laissez-faire on either
welfare or recruitment.
"""

import pandas as pd


# Dummy-at-p=0 baselines from PR #96 sims, restricted to the focus
# group (group_id == 0) and averaged across rounds.
#   avg: mean per-agent payoff in group 0
#   sum: mean per-round group-0 sum payoff (group size drifts via
#        switching in 2g8a runs, so this is well above n_agents/2 * avg)
DUMMY_BASELINE_AVG = {
    "Legacy": 24.0,
    "GS (50 ep)": 25.18,
}
DUMMY_BASELINE_SUM = {
    # legacy: 1g4a sim, no switching -> ~4 agents x ~24 avg
    "Legacy": 96.0,
    # 50ep: 2g8a sim, switching -> ~5 agents in group 0 on average
    "GS (50 ep)": 122.87,
}

DATASETS = [
    ("Legacy", "experiments/pilot_random1_player_round_slim.csv"),
    ("GS (50 ep)", "experiments/2group_8agent_50ep.csv"),
]


def compute_payoff(df):
    """Add a `payoff_calc` column matching simulate.py:load_pilot_data."""
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


def episode_metrics(df):
    """Per-episode (avg per-agent payoff, mean per-round sum payoff)
    restricted to group_id == 0."""
    df = compute_payoff(df)
    g0 = df[df["group_id"] == 0]
    avg = g0.groupby("episode_id")["payoff_calc"].mean()
    per_round_sum = g0.groupby(
        ["episode_id", "round_number"]
    )["payoff_calc"].sum()
    sum_mean = per_round_sum.groupby("episode_id").mean()
    return pd.DataFrame({"avg": avg, "sum": sum_mean})


def main():
    for name, path in DATASETS:
        df = pd.read_csv(path)
        m = episode_metrics(df)
        b_avg = DUMMY_BASELINE_AVG[name]
        b_sum = DUMMY_BASELINE_SUM[name]
        above_avg = (m["avg"] > b_avg).sum()
        above_sum = (m["sum"] > b_sum).sum()
        n = len(m)
        print(f"=== {name} ===")
        print(f"Episodes: {n}")
        print(
            f"Baseline avg: {b_avg:.2f}    "
            f"Baseline sum: {b_sum:.2f}"
        )
        print(
            f"Above baseline (avg): {above_avg}/{n} "
            f"({above_avg / n * 100:.1f}%)"
        )
        print(
            f"Above baseline (sum): {above_sum}/{n} "
            f"({above_sum / n * 100:.1f}%)"
        )
        print(
            "\nDistribution (per-episode group-0 metrics):"
        )
        print(m.describe().round(2).to_string())
        print(
            "\nTop 10 by avg: "
            f"{m['avg'].nlargest(10).round(2).to_dict()}"
        )
        print(
            "\nTop 10 by sum: "
            f"{m['sum'].nlargest(10).round(2).to_dict()}"
        )
        only_avg = set(m["avg"].nlargest(10).index) - set(
            m["sum"].nlargest(10).index
        )
        only_sum = set(m["sum"].nlargest(10).index) - set(
            m["avg"].nlargest(10).index
        )
        both = set(m["avg"].nlargest(10).index) & set(
            m["sum"].nlargest(10).index
        )
        print(f"\nTop-10 overlap avg ∩ sum: {sorted(both)}")
        print(f"Top-10 only-avg (not in top-sum): {sorted(only_avg)}")
        print(f"Top-10 only-sum (not in top-avg): {sorted(only_sum)}")
        print()


if __name__ == "__main__":
    main()
