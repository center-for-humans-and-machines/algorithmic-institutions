import torch as th
import numpy as np
import random
from pydantic import BaseModel


def shift(tensor, default):
    tensor = th.roll(tensor, 1, 2)
    tensor[:, :, 0] = default
    return tensor


class AgentRoundRaw(Basemodel):
    episode_idx: int
    round_number: int
    player_id: int
    global_group_id: int
    player_no_input: int
    contribution: int
    punishment: int
    common_good: int


class AgentRound(BaseModel):
    episode_idx: int
    round_number: int
    player_idx: int
    group_idx: int
    contribution: int
    contribution_valid: bool
    punishment: int
    punishment_valid: bool
    common_good: int


def parse_agent_round(ar: AgentRoundRaw) -> AgentRound:
    return AgentRound(
        episode_idx=ar.episode_idx,
        round_number=ar.round_number,
        player_idx=ar.player_id,
        group_idx=ar.global_group_id,
        contribution=ar.contribution,
        contribution_valid=not ar.player_no_input,
        punishment=ar.punishment if ar.punishment is not None else 0,
        punishment_valid=ar.punishment is None,
        common_good=ar.common_good,
    )


def get_default_values(df):
    default_values = {
        "punishments": np.rint(df.loc[df["player_input"], "punishment"].mean()),
        "contributions": np.rint(df.loc[df["player_input"], "contribution"].mean()),
        "contribution_valid": False,
        "recorded": False,
        "punishment_valid": False,
        "common_good": df.loc[df["player_input"], "common_good"].mean(),
    }
    return default_values


def create_torch_data_new(df, default_values=None):
    if default_values is None:
        default_values = get_default_values(df)

    data_names = {
        "round_number": th.int,
        "is_first": th.bool,
        "contribution": th.int,
        "punishment": th.int,
        "common_good": th.float,
        "contribution_valid": th.bool,
        "punishment_valid": th.bool,
    }

    n_episodes = df["episode_idx"].max() + 1
    n_steps = df["round_number"].max() + 1
    n_agents = df["player_idx"].max() + 1

    data = {
        th.full(
            (n_episodes, n_agents, n_steps),
            fill_value=default_values[name],
            dtype=dtype,
        )
        for name, dtype in data_names.items()
    }
    for idx, row in df.iterrows():
        eps, step, agent = row[["episode_idx", "round_number", "player_id"]]
        data["round_number"][eps, agent, step] = step
        data["is_first"][eps, agent, step] = step == 0
        for name in data_names[2:]:
            data[name][eps, agent, step] = row[name]

    data = {
        **data,
        **{
            f"prev_{k}": shift(t, default_values[k])
            for k, t in data.items()
            if k in default_values
        },
    }
    return data, default_values


def create_torch_data(df, default_values=None):
    df_parsed = df.apply(parse_agent_round, axis=1)

    data_new = create_torch_data_new(df_parsed, default_values)
    data_old = create_torch_data_old(df, default_values)
    for key in data_new:
        assert (data_new[key] == data_old[key]).all(), f"{key} is not equal"
    return data_new


def create_torch_data_old(df, default_values=None):
    df["episode_idx"] = df["episode_id"].rank(method="dense").astype(int)

    n_episodes = df["episode_idx"].max() + 1
    n_steps = df["round_number"].max() + 1
    n_agents = df["player_id"].max() + 1

    episode_group = df["global_group_id"] + "__" + df["episode_idx"].astype(str)
    df["episode_group_idx"] = episode_group.rank(method="dense").astype(int) - 1

    df["player_input"] = df["player_no_input"] == 0
    df["manager_input"] = df["manager_no_input"] == 0
    df["round_player_input"] = df.groupby(["episode_id", "round_number"])[
        "player_input"
    ].transform("sum")
    df["common_good"] = (df["common_good"] / df["round_player_input"]).fillna(0)

    if not default_values:
        default_values = {
            "punishments": np.rint(df.loc[df["player_input"], "punishment"].mean()),
            "contributions": np.rint(df.loc[df["player_input"], "contribution"].mean()),
            "contribution_valid": False,
            "recorded": False,
            "punishment_valid": False,
            "common_good": df.loc[df["player_input"], "common_good"].mean(),
        }

    punishments = th.full(
        (n_episodes, n_agents, n_steps),
        fill_value=default_values["punishments"],
        dtype=th.int64,
    )
    contributions = th.full(
        (n_episodes, n_agents, n_steps),
        fill_value=default_values["contributions"],
        dtype=th.int64,
    )
    round_number = th.zeros((n_episodes, n_agents, n_steps), dtype=th.int64)
    episode_group_idx = th.zeros((n_episodes, n_agents, n_steps), dtype=th.int64)
    is_first = th.zeros((n_episodes, n_agents, n_steps), dtype=th.bool)
    contribution_valid = th.full(
        (n_episodes, n_agents, n_steps),
        fill_value=default_values["contribution_valid"],
        dtype=th.bool,
    )
    recorded = th.full(
        (n_episodes, n_agents, n_steps),
        fill_value=default_values["recorded"],
        dtype=th.bool,
    )
    punishment_valid = th.full(
        (n_episodes, n_agents, n_steps),
        fill_value=default_values["punishment_valid"],
        dtype=th.bool,
    )
    common_good = th.full(
        (n_episodes, n_agents, n_steps),
        fill_value=default_values["common_good"],
        dtype=th.float,
    )

    for idx, row in df.iterrows():
        eps, step, agent = row[["episode_idx", "round_number", "player_id"]]
        contribution_valid[eps, agent, step] = row["player_input"]
        recorded[eps, agent, step] = True
        punishment_valid[eps, agent, step] = row["manager_input"]
        is_first[eps, agent, step] = step == 0
        round_number[eps, agent, step] = step
        common_good[eps, agent, step] = row["common_good"]
        episode_group_idx[eps, agent, step] = row["episode_group_idx"]
        if row["player_input"]:
            punishments[eps, agent, step] = row["punishment"]
            contributions[eps, agent, step] = row["contribution"]

    data = {
        "punishments": punishments,
        "contributions": contributions,
        "contribution_valid": contribution_valid,
        "recorded": recorded,
        "common_good": common_good,
        "round_number": round_number,
        "is_first": is_first,
        "punishment_valid": punishment_valid,
        "episode_group_idx": episode_group_idx,
    }

    data = {
        **data,
        **{
            f"prev_{k}": shift(t, default_values[k])
            for k, t in data.items()
            if k in default_values
        },
    }
    return data, default_values


def get_cross_validations(data, n_splits, fraction_training=1.0):
    episode_idx = list(range(data["contributions"].shape[0]))
    random.shuffle(episode_idx)

    if n_splits is not None:
        groups = [episode_idx[i::n_splits] for i in range(n_splits)]
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
