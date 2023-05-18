import torch as th
from typing import Optional, List, Union
from pydantic import BaseModel

from aimanager.generic.graph import GraphNetwork
from aimanager.manager.manager import ArtificalManager
from aimanager.generic.data import shift


class Round(BaseModel):
    round: int
    group: List[Union[str, int]]
    contribution: List[int]
    punishment: List[int]
    contribution_valid: List[bool]
    punishment_valid: List[bool]


class RoundExternal(BaseModel):
    round: int
    groups: List[str]
    contributions: List[int]
    punishments: List[Optional[int]]
    missing_inputs: List[bool]


def parse_round(round) -> Round:
    """Parse round data from external to internal format."""
    round = RoundExternal(**round)
    return Round(
        round=round.round - 1,
        group=round.groups,
        contribution=round.contributions,
        punishment=[p if p is not None else 0 for p in round.punishments],
        contribution_valid=[not m for m in round.missing_inputs],
        punishment_valid=[p is not None for p in round.punishments],
    ).dict()


def create_data(rounds, groups, default_values):
    """Create data object for the algorithmic manager based on round records."""

    def create_tensor(record_key, default_key):
        return th.tensor(
            [
                [
                    [
                        int(value)
                        if (is_valid and g1 == g2)
                        else int(default_values[default_key])
                        for value, is_valid, g1 in zip(
                            r[record_key], r[f"{record_key}_valid"], r["group"]
                        )
                    ]
                    for r in rounds
                ]
                for g2 in groups
            ],
            dtype=th.int64,
        )

    def create_bool_tensor(record_key):
        return th.tensor(
            [
                [
                    [
                        is_valid and g1 == g2
                        for is_valid, g1 in zip(r[f"{record_key}_valid"], r["group"])
                    ]
                    for r in rounds
                ]
                for g2 in groups
            ],
            dtype=th.bool,
        )

    contribution = create_tensor("contribution", "contribution")
    contribution_valid = create_bool_tensor("contribution")

    punishment = create_tensor("punishment", "punishment")
    punishment_valid = create_bool_tensor("punishment")

    group_size = len(rounds[-1]["contribution"])
    round_number = th.tensor(
        [[[r["round"]] * group_size for r in rounds] for _ in range(len(groups))],
        dtype=th.int64,
    )

    data = {
        "contribution": contribution.permute(0, 2, 1),
        "contribution_valid": contribution_valid.permute(0, 2, 1),
        "punishment": punishment.permute(0, 2, 1),
        "punishment_valid": punishment_valid.permute(0, 2, 1),
        "round_number": round_number.permute(0, 2, 1),
        "is_first": round_number.permute(0, 2, 1) == 0,
    }

    calc_prev = ["punishment", "contribution", "punishment_valid", "contribution_valid"]
    data = {
        **data,
        **{f"prev_{k}": shift(data[k], default_values[k]) for k in calc_prev},
    }

    return data


class HumanManager:
    def __init__(self, model_path, **_):
        self.model = GraphNetwork.load(model_path, device=th.device("cpu"))
        self.default_values = self.model.default_values

    def get_punishments(self, data):
        pred = self.model.predict(data, sample=True)[0]
        return pred


class RLManager:
    def __init__(self, model_path, **_):
        self.model = ArtificalManager.load(
            model_path, device=th.device("cpu")
        ).policy_model
        self.default_values = self.model.default_values
        # self.model.u_encoder.refrence = "contribution"

    def get_punishments(self, data):
        # the round number only enteres the bias and hence does not effect the
        # output, set to zero to allow for larger rollout length
        data["round_number"] = th.zeros_like(data["round_number"])
        pred = self.model.predict(data, sample=False)[0]
        return pred


class DummyManager:
    def __init__(self, **_):
        self.model = None
        self.default_values = {
            "contribution": 0,
            "punishment": 0,
            "contribution_valid": False,
            "punishment_valid": False,
        }

    def get_punishments(self, data):
        return data["punishment"]


MANAGER_CLASS = {"human": HumanManager, "rl": RLManager, "dummy": DummyManager}


class MultiManager:
    def __init__(self, managers, n_steps=16):
        self.n_steps = n_steps
        self.managers = {
            k: MANAGER_CLASS[m["type"]](**m, n_steps=n_steps)
            for k, m in managers.items()
        }
        self.groups = list(self.managers.keys())
        self.group_idx = {k: i for i, k in enumerate(managers.keys())}
        self.manager_info = managers

    def get_punishments_external(self, rounds: List[RoundExternal]):
        parse_rounds = [parse_round(r) for r in rounds]
        return self.get_punishments(parse_rounds)

    def get_punishments(self, rounds):
        # we use the batch dimension to seperate the different groups
        # we mask contributions and punishments corresponding to the groups
        # the batch size corresponds to the number of models
        # the data for both models is identical in principal, we compute them
        # seperately as the models might use different default values
        data = {
            k: create_data(rounds, self.groups, m.default_values)
            for k, m in self.managers.items()
        }

        punishment = {k: m.get_punishments(data[k]) for k, m in self.managers.items()}

        # we select from the model responds only those matching the right group
        # this is the same for all models and independent of the actual model of
        # the group
        # we also select the last punishment in the round dimension
        group = rounds[-1]["group"]
        group_idx = [self.group_idx[g] for g in group]

        punishment = {
            k: v[group_idx, th.arange(len(group_idx)), -1].tolist()
            for k, v in punishment.items()
        }

        # we select the punishment where the group matches the model
        matched_punishment = [
            punishment[g][i] if (g in punishment) else None for i, g in enumerate(group)
        ]

        return matched_punishment, punishment

    def get_info(self):
        return self.manager_info
