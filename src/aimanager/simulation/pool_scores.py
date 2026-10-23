"""The common pool score of a group against an anchor manager (#226, #227).

Definition 4 of #226 (Levin's criterion): per round, a group's common pool
1.6 * sum(c) - sum(p) over its members; a group with no members that round
scores 0 and stays in the average over the episode's rounds. A manager's
score is its group's mean over episodes, in every pairing with the anchor.

Shared by scripts/plotting/plot_winrates.py and the sweep sims of
simulate.py, so the metric has one implementation. Reads a sim's per-round
frame (per_round.parquet's columns); pandas only.
"""

import pandas as pd

POOL = "pool"  # per-agent 1.6*c - p; its group sum is the common pool


def add_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Add the per-agent pool column, in place; return the frame."""
    df[POOL] = 1.6 * df["contribution"] - df["punishment"]
    return df


def pairing_of(run: str) -> tuple:
    return tuple(run.split("managed by ")[-1].split("_vs_"))


def group_round_scores(sub: pd.DataFrame, metric: str, agg: str) -> pd.DataFrame:
    """Per (episode, round): each group's aggregate, empty group-round = 0."""
    grouped = sub.groupby(["episode", "round_number", "agent_group"])[metric]
    per_round = grouped.sum() if agg == "sum" else grouped.mean()
    piv = per_round.reset_index().pivot_table(
        index=["episode", "round_number"], columns="agent_group", values=metric
    )
    return piv.reindex(columns=[0, 1]).fillna(0.0)  # empty group-round -> 0


def episode_members(sub: pd.DataFrame, group: int) -> pd.Series:
    """Mean group size per episode, empty group-round = 0."""
    counts = sub.groupby(["episode", "round_number", "agent_group"]).size()
    cpiv = counts.reset_index(name="n").pivot_table(
        index=["episode", "round_number"], columns="agent_group", values="n"
    )
    cpiv = cpiv.reindex(columns=[0, 1]).fillna(0.0)
    return cpiv[group].groupby("episode").mean()


def against_anchor(df: pd.DataFrame, anchor: str) -> dict:
    """Manager -> per-episode (pool, members) of its group against the anchor.

    Pools every pairing of the manager with the anchor, in either position.
    The anchor's own entry comes from `anchor_vs_anchor`, both groups.
    """
    scores = {}
    for run in df["run"].unique():
        g0, g1 = pairing_of(run)
        if anchor not in (g0, g1):
            continue
        sub = df[df["run"] == run]
        ep = group_round_scores(sub, POOL, "sum").groupby("episode").mean()
        for group, manager in ((0, g0), (1, g1)):
            other = g1 if group == 0 else g0
            if other != anchor:
                continue
            frame = pd.DataFrame(
                {"pool": ep[group], "members": episode_members(sub, group)}
            )
            scores.setdefault(manager, []).append(frame)
    return {m: pd.concat(f, ignore_index=True) for m, f in scores.items()}
