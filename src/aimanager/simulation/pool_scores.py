"""The common pool score of a group against an anchor manager (#226, #227).

Definition 4 of #226: per round, a group's common pool
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


def seated(df: pd.DataFrame, opponent: str) -> dict:
    """Manager -> per-episode (pool, members) of group 0 in each pairing
    `<manager>_vs_<opponent>`: only the seat a sweep gives its points (the
    opponent in group 1), unlike against_anchor, which pools both seats."""
    scores = {}
    for run in df["run"].unique():
        g0, g1 = pairing_of(run)
        if g1 != opponent:
            continue
        sub = df[df["run"] == run]
        ep = group_round_scores(sub, POOL, "sum").groupby("episode").mean()
        frame = pd.DataFrame({"pool": ep[0], "members": episode_members(sub, 0)})
        scores.setdefault(g0, []).append(frame)
    return {m: pd.concat(f, ignore_index=True) for m, f in scores.items()}


# -- Experiment 1: one group, no anchor ----------------------------------

GROUP_PAYOFF = "group_payoff"


def manager_of(run: str) -> str:
    """The manager of a run without pairings ("ah <h> managed by <m>")."""
    return run.split("managed by ")[-1]


def group_payoff_scores(df: pd.DataFrame) -> dict:
    """Manager -> per-episode (pool, members) for experiment 1's score.

    Experiment 1's objective, the average payoff: per round, the mean payoff
    of the players who did not time out, with payoff 20 - c - p + their share
    of the pool, (1.6 * sum(c) - sum(p)) / n over the n valid players (the
    sim's `payoff` column on valid rows; a timed-out player is out of the
    pool and cannot be punished). A round with no valid player scores 0.
    Summed over the episode's rounds, as experiment 1's payoffs are reported.
    `pool` carries the score, so the sweep's consumers read it unchanged;
    `members` is the mean number of valid players per round.
    """
    valid = df[df["contribution_valid"].astype(bool)]
    per_round = valid.groupby(["run", "episode", "round_number"])["payoff"].agg(
        ["mean", "size"]
    )
    rounds = df[["run", "episode", "round_number"]].drop_duplicates()
    full = rounds.set_index(["run", "episode", "round_number"]).join(per_round)
    full = full.fillna(0.0)
    episodes = full.groupby(["run", "episode"]).agg(
        pool=("mean", "sum"), members=("size", "mean")
    )
    scores = {}
    for run, frame in episodes.groupby(level="run"):
        scores[manager_of(run)] = frame.reset_index(drop=True)
    return scores
