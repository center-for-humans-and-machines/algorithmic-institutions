"""Empirical contribution change across sustained-low-punishment runs.

Usage:
    python experiment_analysis/decay_sequences.py
        [--max-prev-p N]  (default 2; includes prev_p in {0..N})
        [--min-len N]     (default 3; require run length >= N)

For each (episode, player), find maximal runs of consecutive rounds
where ``prev_punishment <= max_prev_p``. For each run with length
``>= min_len``, record:

    start_contribution  -> contribution at the first round of the run
    delta_c             -> contribution at last round - first round
    length              -> number of rounds in the run

Then bucket by starting contribution and report mean delta_c plus run
count side-by-side for each dataset. Partitions whether the failure
mode in PR #96 is data-side ("no decay sequences exist in training")
or model-side ("sequences exist but the model doesn't learn them").
"""

import argparse
import pandas as pd


DATASETS = [
    ("Legacy", "experiments/pilot_random1_player_round_slim.csv"),
    ("GS (50 ep)", "experiments/2group_8agent_50ep.csv"),
]

CONTRIBUTION_BUCKETS = [
    ("0", (0, 0)),
    ("1-5", (1, 5)),
    ("6-10", (6, 10)),
    ("11-15", (11, 15)),
    ("16-19", (16, 19)),
    ("20", (20, 20)),
]


def _bucket(value, buckets):
    for label, (lo, hi) in buckets:
        if lo <= value <= hi:
            return label
    return None


def extract_runs(df, max_prev_p, min_len):
    """Return a DataFrame of runs: one row per (episode, player, run).

    Columns: start_contribution, end_contribution, delta_c, length.
    """
    df = df.sort_values(["episode_id", "player_id", "round_number"]).copy()
    grp = df.groupby(["episode_id", "player_id"])
    df["prev_punishment"] = grp["punishment"].shift(1)
    df["prev_contribution"] = grp["contribution"].shift(1)
    df = df.dropna(subset=["prev_punishment", "prev_contribution"]).copy()

    df["low_p"] = df["prev_punishment"] <= max_prev_p
    # Within each (episode, player) trace, label each maximal run of
    # consecutive low_p=True rounds with a unique id.
    df["run_id"] = (
        df.groupby(["episode_id", "player_id"])["low_p"]
        .transform(lambda s: (s != s.shift()).cumsum())
    )
    df_low = df[df["low_p"]]

    runs = (
        df_low.groupby(["episode_id", "player_id", "run_id"])
        .agg(
            start_contribution=("contribution", "first"),
            end_contribution=("contribution", "last"),
            length=("contribution", "size"),
        )
        .reset_index(drop=True)
    )
    runs["delta_c"] = runs["end_contribution"] - runs["start_contribution"]
    return runs[runs["length"] >= min_len]


def summarize(runs):
    runs = runs.copy()
    runs["start_bucket"] = runs["start_contribution"].apply(
        lambda v: _bucket(v, CONTRIBUTION_BUCKETS)
    )
    return (
        runs.groupby("start_bucket")
        .agg(
            n_runs=("delta_c", "size"),
            mean_delta_c=("delta_c", "mean"),
            mean_length=("length", "mean"),
            mean_end_c=("end_contribution", "mean"),
        )
        .reindex([lbl for lbl, _ in CONTRIBUTION_BUCKETS])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-prev-p", type=int, default=2)
    parser.add_argument("--min-len", type=int, default=3)
    args = parser.parse_args()

    print(
        f"Runs: consecutive rounds with prev_punishment <= "
        f"{args.max_prev_p}, length >= {args.min_len}"
    )

    summaries = {}
    for name, path in DATASETS:
        df = pd.read_csv(path)
        runs = extract_runs(df, args.max_prev_p, args.min_len)
        summaries[name] = summarize(runs)
        print(f"\n=== {name} ===")
        print(f"Total runs: {len(runs)}")
        print(summaries[name].to_string(float_format=lambda v: f"{v:.2f}"))

    if len(DATASETS) == 2:
        a, b = DATASETS[0][0], DATASETS[1][0]
        a_s, b_s = summaries[a], summaries[b]
        print(f"\n=== {b} − {a} (mean delta_c) ===")
        diff = b_s["mean_delta_c"] - a_s["mean_delta_c"]
        for c in diff.index:
            line = (
                f"  {c:<10s}  Δ {diff[c]:>+6.2f}   "
                f"(n: {a}={int(a_s.loc[c, 'n_runs'] or 0)}, "
                f"{b}={int(b_s.loc[c, 'n_runs'] or 0)})"
            )
            print(line)


if __name__ == "__main__":
    main()
