import torch as th
from typing import Optional
from pydantic import BaseModel

from aimanager.generic.graph import GraphNetwork
from aimanager.manager.manager import ArtificalManager


class Round(BaseModel):
    round: int
    groups: list[int]
    contribution: list[int]
    punishment: list[int]
    contribution_valid: list[bool]
    punishment_valid: list[bool]


class RoundExternal(BaseModel):
    round: int
    groups: list[int]
    contributions: list[int]
    punishments: list[Optional[int]]
    missing_inputs: list[bool]


def parse_round(round: RoundExternal) -> Round:
    """Parse round data from external to internal format."""
    return Round(
        round=round.round,
        groups=round.groups,
        contribution=round.contributions,
        punishment=[p if p is not None else 0 for p in round.punishments],
        contribution_valid=[not m for m in round.missing_inputs],
        punishment_valid=[p is not None for p in round.punishments],
    )


def fill_none(values, fill_value=0):
    return [v if v is not None else fill_value for v in values]


def create_data(rounds, n_groups, default_values):
    """Create data object for the algorithmic manager based on round records."""
    current_round = rounds[-1]

    contribution = th.tensor(
        [
            [
                [
                    c if (cv and g1 == g2) else 0
                    for c, cv, g1 in zip(
                        r["contribution"], r["contribution_valid"], r["group"]
                    )
                ]
                for r in rounds
            ]
            for g2 in range(n_groups)
        ]
    )
    contribution_valid = th.tensor(
        [
            [
                [cv and g1 == g2 for cv, g1 in zip(r["contribution_valid"], r["group"])]
                for r in rounds
            ]
            for g2 in range(n_groups)
        ]
    )
    punishment = th.tensor(
        [
            [
                [
                    p if (pv and g1 == g2) else 0
                    for p, pv, g1 in zip(
                        r["punishment"], r["punishment_valid"], r["group"]
                    )
                ]
                for r in rounds
            ]
            for g2 in range(n_groups)
        ]
    )
    punishment_valid = th.tensor(
        [
            [
                [(pv and g1 == g2) for pv, g1 in zip(r["punishment_valid"], r["group"])]
                for r in rounds
            ]
            for g2 in range(n_groups)
        ]
    )

    group_size = len(current_round["contribution"])
    round_number = th.tensor(
        [[[r["round"]] * group_size for r in rounds] for _ in range(n_groups)]
    )

    data = {
        "contribution": contribution.permute(0, 2, 1),
        "contribution_valid": contribution_valid.permute(0, 2, 1),
        "punishment": punishment.permute(0, 2, 1),
        "punishment_valid": punishment_valid.permute(0, 2, 1),
        "round_number": round_number.permute(0, 2, 1),
    }

    calc_prev = ["punishments", "punishment_valid", "contribution_valid"]
    for k in calc_prev:
        prev = th.full_like(data[k], fill_value=default_values[k])
        prev[:, :, 1:] = data[k][:, :, 1:]
        data[f"prev_{k}"] = prev

    # # transpose data
    # data = {k: v.T for k, v in data.items()}

    return data


class HumanManager:
    def __init__(self, model_path):
        self.model = GraphNetwork.load(model_path, device=th.device("cpu"))

    def get_punishments(self, data, n_groups):
        data = create_data(data, n_groups, self.model.default_values)
        pred = self.model.predict(data, sample=True)
        print(pred)
        return pred


class RLManager:
    def __init__(self, model_path):
        self.model = ArtificalManager.load(
            model_path, device=th.device("cpu")
        ).policy_model
        # self.model.u_encoder.refrence = "contribution"

    def get_punishments(self, data, n_groups):
        data = create_data(data, n_groups, self.model.default_values)
        pred = self.model.predict(data, sample=False)
        print(pred)
        return pred
        # return self.model.predict_pure(encoded, sample=False)[0][:, -1].tolist()


MANAGER_CLASS = {"human": HumanManager, "rl": RLManager}


class MultiManager:
    def __init__(self, managers):
        self.managers = {
            k: MANAGER_CLASS[m["type"]](m["path"]) for k, m in managers.items()
        }
        self.n_groups = len(self.managers)

    def get_punishments(self, rounds):
        group = rounds[-1]["group"]

        punishments = {
            k: m.get_punishments(rounds, self.n_groups)[:, :, -1]
            for k, m in self.managers.items()
        }
        punishments = {k: v[group] for k, v in punishments.items()}
        return [
            punishments[g][i] if g in punishments else None for i, g in enumerate(group)
        ], punishments
