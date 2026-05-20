"""Joint (prev_contribution, prev_punishment) cell coverage.

Usage:
    python experiment_analysis/joint_coverage.py

For each dataset, computes the lag-1 shift of contribution and
punishment per (episode_id, player_id), buckets both, and prints the
joint share of rows in each cell. Highlights the (high prev_c, low
prev_p) corner — the predecessor state of any "decay-under-no-
punishment" learning signal.
"""

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
PUNISHMENT_BUCKETS = [
    ("0", (0, 0)),
    ("1-3", (1, 3)),
    ("4-7", (4, 7)),
    ("8-15", (8, 15)),
    ("16+", (16, 30)),
]


def _bucket(series, buckets):
    """Return a pandas Series of bucket labels (NaN if outside any bucket)."""
    out = pd.Series(index=series.index, dtype="object")
    for label, (lo, hi) in buckets:
        mask = (series >= lo) & (series <= hi)
        out.loc[mask] = label
    return out


def derive_prev(df):
    """Add prev_contribution and prev_punishment columns (lag-1)."""
    df = df.sort_values(["episode_id", "player_id", "round_number"]).copy()
    grp = df.groupby(["episode_id", "player_id"])
    df["prev_contribution"] = grp["contribution"].shift(1)
    df["prev_punishment"] = grp["punishment"].shift(1)
    return df.dropna(subset=["prev_contribution", "prev_punishment"])


def _crosstab(df, normalize):
    """Bucketed crosstab of prev_c against prev_p. ``normalize`` follows
    pandas.crosstab semantics: True (over all), 'index' (rows sum to 1),
    'columns' (cols sum to 1)."""
    c_label = _bucket(df["prev_contribution"], CONTRIBUTION_BUCKETS)
    p_label = _bucket(df["prev_punishment"], PUNISHMENT_BUCKETS)
    ct = pd.crosstab(c_label, p_label, normalize=normalize) * 100
    c_order = [lbl for lbl, _ in CONTRIBUTION_BUCKETS]
    p_order = [lbl for lbl, _ in PUNISHMENT_BUCKETS]
    return ct.reindex(index=c_order, columns=p_order).fillna(0)


def joint_table(df):
    """Joint share (%): rows: prev_c, cols: prev_p; total sums to 100."""
    return _crosstab(df, normalize=True)


def conditional_table(df):
    """Row-conditional share (%): P(prev_p bucket | prev_c bucket).
    Each row sums to 100 (or 0 if the prev_c bucket has no rows)."""
    return _crosstab(df, normalize="index")


def _print_table(title, table, fmt="{:>7.2f}%", check_sum=True):
    print(f"\n=== {title} ===")
    header = "prev_c \\ prev_p  " + "".join(f"{c:>8s}" for c in table.columns)
    print(header)
    print("-" * len(header))
    for c, row in table.iterrows():
        line = f"{c:<16s}"
        for v in row:
            line += fmt.format(v)
        print(line)
    if check_sum:
        print(f"sum check: {table.sum().sum():.2f}")


def main():
    joint = {}
    cond = {}
    for name, path in DATASETS:
        df = pd.read_csv(path)
        df = derive_prev(df)
        joint[name] = joint_table(df)
        cond[name] = conditional_table(df)

    for name in [n for n, _ in DATASETS]:
        _print_table(
            f"{name} — joint (prev_c, prev_p) share (%)", joint[name]
        )

    for name in [n for n, _ in DATASETS]:
        _print_table(
            f"{name} — P(prev_p | prev_c) row-conditional (%)",
            cond[name],
            check_sum=False,
        )

    # Side-by-side diff for the joint and conditional tables.
    if len(DATASETS) == 2:
        a, b = DATASETS[0][0], DATASETS[1][0]
        _print_table(
            f"{b} − {a} joint (pp diff)",
            joint[b] - joint[a],
            fmt="{:>+6.2f}pp",
            check_sum=False,
        )
        _print_table(
            f"{b} − {a} conditional P(prev_p | prev_c) (pp diff)",
            cond[b] - cond[a],
            fmt="{:>+6.2f}pp",
            check_sum=False,
        )


if __name__ == "__main__":
    main()
