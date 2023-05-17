import torch as th
import numpy as np
import random
import pandera as pa
from pandera.typing import Series


class AgentRoundRaw(pa.SchemaModel):
    episode_id: Series[int]
    round_number: Series[int]
    player_id: Series[int]
    global_group_id: Series[str]
    player_no_input: Series[int]
    contribution: Series[float]
    punishment: Series[float]
    common_good: Series[float]


class AgentRound(pa.SchemaModel):
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


def parse_agent_rounds(df):
    AgentRoundRaw(df)
    df["contribution_valid"] = df["player_no_input"] == 0
    df["punishment_valid"] = df["manager_no_input"] == 0
    df["is_first"] = df["round_number"] == 0

    # missing replaced by 0 only temporarily
    df["punishment"] = df["punishment"].fillna(0).astype(int)
    df["contribution"] = df["contribution"].fillna(0).astype(int)

    # create a single group idx for each episode of each group
    episode_group = df["global_group_id"] + "__" + df["episode_id"].astype(str)
    df["group_idx"] = episode_group.rank(method="dense").astype(int) - 1

    # rescale common good by the total number of participants in round
    round_player_input = df.groupby(["episode_id", "round_number"])[
        "contribution_valid"
    ].transform("sum")
    df["common_good"] = (df["common_good"] / round_player_input).fillna(0)
    df["recorded"] = True

    df.drop(columns=["global_group_id", "player_no_input"], inplace=True)
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
    return data, default_values


def create_torch_data(df, default_values=None):
    df = parse_agent_rounds(df.copy())
    data, default_values = create_torch_data_new(df, default_values)
    return data, default_values


def get_cross_validations(data, n_splits, fraction_training=1.0):
    episode_idx = list(range(data["contribution"].shape[0]))
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
