"""Simulation plots shared by simulate.py and scripts that redraw them.

Pandas, seaborn and matplotlib only (no PyG), so they also run locally on a
fetched per_round.parquet; simulate.create_plots calls them unchanged.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """In place: make `episode` unique per run and add `payoff_sum`."""
    df["episode"] = df["run"] + "__" + df["episode"].astype(str)

    # `payoff_sum` is the per-group sum of per-agent payoffs. Place the
    # value on the first agent-row of each (episode, round, group) and
    # NaN on the rest, so downstream means over agent-rows (seaborn
    # line plots and aggregates) skip duplicates and reduce to one
    # value per episode -- the correct unweighted per-episode mean.
    _gkeys = ["run", "episode", "round_number", "group_id"]
    df["payoff_sum"] = (
        df.groupby(_gkeys)["payoff"].transform("sum").where(~df.duplicated(_gkeys))
    )
    return df


def plot_pairing_side(
    df: pd.DataFrame, pairings: list, output_dir: str, figure_name: str
):
    """Each pairing's two sides as separate lines (comparison_pairing_side.jpg)."""
    pairings_by_name = {p["name"]: p for p in pairings}

    def _side(name, key):
        return pairings_by_name[name][key] if name in pairings_by_name else None

    df_p = df[df["run"].str.contains(" managed by ", na=False)].copy()
    pairing_name = df_p["run"].str.rsplit(" managed by ", n=1).str[1]
    df_p["pairing"] = pairing_name
    g0 = pairing_name.map(lambda n: _side(n, "group_0"))
    g1 = pairing_name.map(lambda n: _side(n, "group_1"))
    df_p["manager_side"] = g0.where(df_p["group_id"] == 0, g1)
    df_p = df_p[df_p["manager_side"].notna()]

    if len(df_p):
        df_p["label"] = df_p["pairing"] + " / " + df_p["manager_side"]
        # `payoff_sum` already per (run, episode, round, group_id)
        # from the top of create_plots, which is the correct
        # per-side sum here too (group_id distinguishes sides).
        dfp_m = df_p.melt(
            id_vars=[
                "episode",
                "round_number",
                "participant_code",
                "label",
            ],
            value_vars=[
                "punishment",
                "contribution",
                "common_good",
                "payoff",
                "payoff_sum",
            ],
        )
        g = sns.relplot(
            data=dfp_m,
            x="round_number",
            y="value",
            col="variable",
            hue="label",
            kind="line",
            facet_kws={"sharey": False, "sharex": True},
            height=3,
            aspect=1.1,
            col_wrap=2,
        )
        g.fig.suptitle(
            f"Pairing-side Comparison: {figure_name}",
            y=1.02,
        )
        g.set(ylim=(0, None))
        out = os.path.join(output_dir, "comparison_pairing_side.jpg")
        g.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


def plot_group_size(df: pd.DataFrame, output_dir: str, figure_name: str):
    """Group size per round, per group (group_size_evolution_global.jpg)."""
    # an empty group has no rows: fill it in as size 0
    per_episode_sizes = (
        df.groupby(["run", "episode", "round_number", "group_id"])["participant_code"]
        .nunique()
        .unstack("group_id", fill_value=0)
        .stack()
        .rename("group_size")
        .reset_index()
    )
    max_agents = (
        df.groupby(["run", "episode", "round_number"])["participant_code"]
        .nunique()
        .max()
    )

    g = sns.relplot(
        data=per_episode_sizes,
        x="round_number",
        y="group_size",
        col="group_id",
        hue="run",
        kind="line",
        facet_kws={"sharey": True, "sharex": True},
        height=4,
        aspect=1.3,
    )
    g.fig.suptitle(f"Group size per round: {figure_name}", y=1.02)
    g.set(ylim=(0, max_agents))
    g.set_axis_labels("round_number", "group_size")
    global_group_size_path = os.path.join(output_dir, "group_size_evolution_global.jpg")
    g.savefig(global_group_size_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {global_group_size_path}")
