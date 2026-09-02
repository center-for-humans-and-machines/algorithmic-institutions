"""Punishment-policy fingerprint of top-quartile episodes by payoff.

Usage:
    python experiment_analysis/top_episode_policy.py

Splits each dataset's episodes by per-episode mean per-agent payoff
into top and bottom quartiles, then computes the manager's conditional
punishment policy P(prev_p bucket | prev_c bucket) restricted to each
slice. Surfaces whether the "successful" episodes used a structurally
different policy that could be a salvageable training signal.
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
    out = pd.Series(index=series.index, dtype="object")
    for label, (lo, hi) in buckets:
        out.loc[(series >= lo) & (series <= hi)] = label
    return out


def compute_payoff(df):
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


def derive_prev(df):
    df = df.sort_values(["episode_id", "player_id", "round_number"]).copy()
    grp = df.groupby(["episode_id", "player_id"])
    df["prev_contribution"] = grp["contribution"].shift(1)
    df["prev_punishment"] = grp["punishment"].shift(1)
    df["prev_player_no_input"] = grp["player_no_input"].shift(1)
    df = df.dropna(
        subset=["prev_contribution", "prev_punishment", "prev_player_no_input"]
    )
    # Drop rows where the prior contribution was imputed because the
    # player didn't actually input — there's no decision for the
    # manager to respond to, so these rows aren't policy signal.
    return df[df["prev_player_no_input"] == 0]


def conditional_policy(df):
    c_label = _bucket(df["prev_contribution"], CONTRIBUTION_BUCKETS)
    p_label = _bucket(df["prev_punishment"], PUNISHMENT_BUCKETS)
    ct = pd.crosstab(c_label, p_label, normalize="index") * 100
    c_order = [lbl for lbl, _ in CONTRIBUTION_BUCKETS]
    p_order = [lbl for lbl, _ in PUNISHMENT_BUCKETS]
    return ct.reindex(index=c_order, columns=p_order).fillna(0)


def _print_table(title, table):
    print(f"\n=== {title} ===")
    header = "prev_c \\ prev_p  " + "".join(
        f"{c:>8s}" for c in table.columns
    )
    print(header)
    print("-" * len(header))
    for c, row in table.iterrows():
        line = f"{c:<16s}"
        for v in row:
            line += f"{v:>7.1f}%"
        print(line)


def main():
    for name, path in DATASETS:
        df_raw = pd.read_csv(path)
        df = compute_payoff(df_raw)

        ep_payoff = df.groupby("episode_id")["payoff_calc"].mean()
        q_top = ep_payoff.quantile(0.75)
        q_bot = ep_payoff.quantile(0.25)

        top_eps = ep_payoff[ep_payoff >= q_top].index
        bot_eps = ep_payoff[ep_payoff <= q_bot].index

        prev = derive_prev(df)
        top_df = prev[prev["episode_id"].isin(top_eps)]
        bot_df = prev[prev["episode_id"].isin(bot_eps)]

        print(f"\n############################")
        print(f"### {name} ###")
        print(f"############################")
        print(
            f"Top quartile: {len(top_eps)} episodes "
            f"(payoff >= {q_top:.2f})"
        )
        print(f"  mean punishment: {top_df['punishment'].mean():.2f}")
        print(
            f"Bottom quartile: {len(bot_eps)} episodes "
            f"(payoff <= {q_bot:.2f})"
        )
        print(f"  mean punishment: {bot_df['punishment'].mean():.2f}")

        top_pol = conditional_policy(top_df)
        bot_pol = conditional_policy(bot_df)
        _print_table(
            f"{name} top quartile — P(prev_p | prev_c) (%)", top_pol
        )
        _print_table(
            f"{name} bottom quartile — P(prev_p | prev_c) (%)", bot_pol
        )
        _print_table(
            f"{name} top − bottom (pp diff)",
            top_pol - bot_pol,
        )


if __name__ == "__main__":
    main()
