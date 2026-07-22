import random

import numpy as np
import pandera as pa
import torch as th
from pandera.typing import Series


class AgentRoundRaw(pa.DataFrameModel):
    episode_id: Series[int]
    round_number: Series[int]
    player_id: Series[int]
    global_group_id: Series[str]
    group_id: Series[int]
    player_no_input: Series[int]
    contribution: Series[float]
    punishment: Series[float]
    common_good: Series[float]


class AgentRound(pa.DataFrameModel):
    round_number: Series[int]
    is_first: Series[bool]
    player_idx: Series[int]
    group_idx: Series[int]
    contribution: Series[int]
    contribution_valid: Series[bool]
    punishment: Series[int]
    punishment_valid: Series[bool]
    common_good: Series[float]
    recorded: Series[bool]
    agent_group: Series[int]
    does_switch: Series[bool]
    switch_mask: Series[bool]
    switch_valid: Series[bool]


def parse_agent_rounds(df, switch_every=None):
    AgentRoundRaw(df)
    df["contribution_valid"] = df["player_no_input"] == 0
    df["punishment_valid"] = df["manager_no_input"] == 0
    df["is_first"] = df["round_number"] == 0

    # missing replaced by 0 only temporarily
    df["punishment"] = df["punishment"].fillna(0).astype(int)
    df["contribution"] = df["contribution"].fillna(0).astype(int)

    # sub-group membership (node feature for GNN)
    df["agent_group"] = df["group_id"].astype(int)

    # does_switch labelled at the DECISION round s (#123): the group choice
    # made after round s is played, realised as the membership change between
    # s and s+1. Round-s features are legal for this target; arrivals still
    # physically happen at s+1 (derive arrival markers from membership change,
    # not from this label).
    df = df.sort_values(["episode_id", "player_id", "round_number"])
    by_player = df.groupby(["episode_id", "player_id"])
    next_group = by_player["group_id"].shift(-1)
    df["does_switch"] = next_group.notna() & next_group.ne(df["group_id"])

    # Decision rows are {switch_every-1, 2*switch_every-1, ...}: the round
    # played right before each arrival. The episode's last round has no
    # following arrival (next_group is NaN), so it is never a decision row.
    if switch_every is not None:
        is_decision = (
            (df["round_number"] + 1) % switch_every == 0
        ) & next_group.notna()
        df["does_switch"] = df["does_switch"] & is_decision
        df["switch_mask"] = is_decision
    else:
        df["switch_mask"] = next_group.notna()

    # switch_valid: drop decisions whose group choice timed out. The timeout
    # is logged at the arrival round s+1 -> align it back to the decision row.
    if "selection_timeout" in df.columns:
        next_timeout = by_player["selection_timeout"].shift(-1)
        df["switch_valid"] = df["switch_mask"] & (
            next_timeout.fillna(0).astype(int) == 0
        )
    else:
        df["switch_valid"] = df["switch_mask"]

    # own-group average contribution (node feature, #114 / M3): leave-one-out
    # mean of the agent's CURRENT-group members' PREVIOUS-round contribution.
    # Computed directly (current-group membership x t-1 contribution) so a
    # switched agent gets its NEW group's prior mean -- the auto prev_ shift
    # would give the OLD group's mean instead. Invalid (no-input) contributions
    # are excluded; round-0 / no-valid-peer cells fall back to the median valid
    # contribution (c_def), matching prev_contribution's round-0 default.
    c_def = np.rint(df.loc[df["contribution_valid"], "contribution"].median())
    prev_c = df.groupby(["episode_id", "player_id"])["contribution"].shift(1)
    prev_valid = (
        df.groupby(["episode_id", "player_id"])["contribution_valid"]
        .shift(1)
        .fillna(False)
    )
    valid_prev_c = prev_c.where(prev_valid)  # invalid / round-0 -> NaN (skipped)
    grp = [df["episode_id"], df["round_number"], df["group_id"]]
    gsum = valid_prev_c.groupby(grp).transform("sum")
    gcnt = valid_prev_c.groupby(grp).transform("count")
    loo_sum = gsum - valid_prev_c.fillna(0.0)  # subtract self only if it counted
    loo_cnt = gcnt - prev_valid.astype(int)  # drop self from the count if valid
    loo = (loo_sum / loo_cnt).where(loo_cnt > 0)  # NaN if no valid peers / round 0
    df["own_grp_prev_mean_contr"] = loo.fillna(float(c_def)).astype(float)

    # episode-batch index (tensor's first dimension)
    episode_group = df["global_group_id"] + "__" + df["episode_id"].astype(str)
    df["group_idx"] = episode_group.rank(method="dense").astype(int) - 1

    # rescale common good by the number of valid participants in group
    group_player_input = df.groupby(["episode_id", "round_number", "group_id"])[
        "contribution_valid"
    ].transform("sum")
    df["common_good"] = (df["common_good"] / group_player_input).fillna(0)
    df["recorded"] = True

    df.drop(
        columns=["global_group_id", "group_id", "player_no_input"],
        inplace=True,
    )
    df.rename(columns={"player_id": "player_idx"}, inplace=True)
    AgentRound(df)
    return df


def shift(tensor, default):
    tensor = th.roll(tensor, 1, 2)
    tensor[:, :, 0] = default
    return tensor


def get_default_values(df):
    p_def = np.rint(df.loc[df["punishment_valid"], "punishment"].median())
    c_def = np.rint(df.loc[df["contribution_valid"], "contribution"].median())
    cg_def = df.loc[df["contribution_valid"], "common_good"].median()
    default_values = {
        "punishment": p_def,
        "contribution": c_def,
        "contribution_valid": False,
        "recorded": False,
        "punishment_valid": False,
        "common_good": cg_def,
        "agent_group": 0,
        "does_switch": False,
        "switch_mask": False,
        "switch_valid": False,
        # round-0 / absent cells inherit the contribution default (see #114):
        # the own-group prev mean of all-c_def previous contributions is c_def.
        "own_grp_prev_mean_contr": c_def,
    }
    return default_values


def create_torch_data_new(df, default_values=None):
    if default_values is None:
        default_values = get_default_values(df)

    data_names = {
        "round_number": th.int64,
        "is_first": th.bool,
        "contribution": th.int64,
        "punishment": th.int64,
        "common_good": th.float,
        "contribution_valid": th.bool,
        "punishment_valid": th.bool,
        "recorded": th.bool,
        "agent_group": th.int64,
        "does_switch": th.bool,
        "switch_mask": th.bool,
        "switch_valid": th.bool,
        "own_grp_prev_mean_contr": th.float,
    }

    n_groups = df["group_idx"].max() + 1
    n_steps = df["round_number"].max() + 1
    n_agents = df["player_idx"].max() + 1

    data = {
        name: th.full(
            (n_groups, n_agents, n_steps),
            fill_value=default_values.get(name, 0),
            dtype=dtype,
        )
        for name, dtype in data_names.items()
    }
    for idx, row in df.iterrows():
        group, step, agent = row[["group_idx", "round_number", "player_idx"]]
        for name in data_names:
            data[name][group, agent, step] = row[name]

    data = {
        **data,
        **{
            f"prev_{k}": shift(t, default_values[k])
            for k, t in data.items()
            if k in default_values
        },
    }

    # Per-episode pair_id (group_key for fold-aware CV). Falls back to
    # the tensor-row index when the column is absent (legacy datasets).
    if "pair_id" in df.columns:
        pair_id = (
            df.drop_duplicates("group_idx")
            .sort_values("group_idx")["pair_id"]
            .to_numpy()
        )
    else:
        pair_id = np.arange(n_groups)

    return data, default_values, pair_id


def create_torch_data(df, default_values=None, switch_every=None):
    df = parse_agent_rounds(df.copy(), switch_every=switch_every)
    data, default_values, pair_id = create_torch_data_new(df, default_values)
    return data, default_values, pair_id


def get_cross_validations(
    data, n_splits, fraction_training=1.0, holdout_fold=None, group_key=None
):
    """Yield (fold_id, train_data, test_data) tuples.

    If `group_key` is provided (array of shape (n_episodes,)), episodes
    sharing a key are always placed in the same fold. Used for the
    doubled (pair-augmented) dataset so flipped copies can't leak across
    train/test.
    """
    episode_idx = list(range(data["contribution"].shape[0]))
    random.shuffle(episode_idx)

    if n_splits is not None:
        if group_key is not None:
            # Collapse to unique groups (preserving shuffled order of
            # first appearance), round-robin split groups into folds,
            # then expand back to episode indices.
            group_to_indices: dict = {}
            order: list = []
            for idx in episode_idx:
                k = int(group_key[idx])
                if k not in group_to_indices:
                    group_to_indices[k] = []
                    order.append(k)
                group_to_indices[k].append(idx)
            fold_groups = [order[i::n_splits] for i in range(n_splits)]
            groups = [
                [idx for k in fg for idx in group_to_indices[k]] for fg in fold_groups
            ]
        else:
            groups = [episode_idx[i::n_splits] for i in range(n_splits)]

        if holdout_fold is not None:
            assert 0 <= holdout_fold < n_splits, (
                f"holdout_fold={holdout_fold} out of range " f"for n_splits={n_splits}"
            )
            test_idx = groups[holdout_fold]
            train_idx = [idx for idx in episode_idx if idx not in test_idx]
            random.shuffle(train_idx)
            train_idx = train_idx[: int(fraction_training * len(train_idx))]

            assert len(set(train_idx).intersection(set(test_idx))) == 0

            test_data = {k: t[test_idx] for k, t in data.items()}
            train_data = {k: t[train_idx] for k, t in data.items()}
            yield None, train_data, test_data
            return

        for i in range(n_splits):
            test_idx = groups[i]
            train_idx = [idx for idx in episode_idx if idx not in test_idx]

            # get a random fraction of the training groups
            random.shuffle(train_idx)
            train_idx = train_idx[: int(fraction_training * len(train_idx))]

            assert len(set(train_idx).intersection(set(test_idx))) == 0
            assert len(train_idx) == len(set(train_idx))
            assert len(test_idx) == len(set(test_idx))
            if fraction_training == 1.0:
                assert (len(test_idx) + len(train_idx)) == len(episode_idx)

            test_data = {k: t[test_idx] for k, t in data.items()}
            train_data = {k: t[train_idx] for k, t in data.items()}
            yield i, train_data, test_data

    train_data = {k: t[episode_idx] for k, t in data.items()}
    yield None, train_data, None
