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


def create_data(rounds, default_values):
    """Create data object for the algorithmic manager based on round records."""
    current_round = rounds[-1]

    contributions = th.tensor([r["contributions"] for r in rounds], dtype=th.int64)
    contribution_valid = th.tensor(
        [r["contribution_valid"] for r in rounds], dtype=th.bool
    )
    contributions = th.where(
        contribution_valid, contributions, int(default_values["contributions"])
    )

    punishments = th.tensor([r["punishments"] for r in rounds], dtype=th.int64)
    punishment_valid = th.tensor([r["punishment_valid"] for r in rounds], dtype=th.bool)
    punishments = th.where(
        punishment_valid, punishments, int(default_values["punishments"])
    )

    group_size = len(current_round["contributions"])
    round_number = th.tensor(
        [[r["round"]] * group_size for r in rounds], dtype=th.int64
    )

    # create a edge for each pair within the same group
    edge_index = th.tensor(
        [
            [g1idx, g2idx]
            for g1idx, g1 in enumerate(current_round["groups"])
            for g2idx, g2 in enumerate(current_round["groups"])
            if ((g1idx != g2idx) and (g1 == g2))
        ],
        dtype=th.int64,
    )

    # TODO: this looks wrong; double check
    print(current_round["groups"])
    batch = th.zeros(len(current_round["groups"]), dtype=th.int64)

    data = {
        "contributions": contributions,
        "contribution_valid": contribution_valid,
        "punishment": punishments,
        "punishment_valid": punishment_valid,
        "round_number": round_number,
        "edge_index": edge_index,
    }
    calc_prev = ["punishments", "punishment_valid", "contribution_valid"]
    for k in calc_prev:
        prev = th.full_like(data[k], fill_value=default_values[k])
        prev[1:] = data[k][:-1]
        data[f"prev_{k}"] = prev

    # transpose data
    data = {k: v.T for k, v in data.items()}

    data["batch"] = batch
    return data


class Manager:
    def encode(self, rounds):
        data = create_data(rounds, self.model.default_values)
        encoded = self.model.encode_pure(data, y_encode=False)
        return encoded


class HumanManager(Manager):
    def __init__(self, model_path):
        self.model = GraphNetwork.load(model_path, device=th.device("cpu"))

    def get_punishments_pure(self, rounds):
        encoded = self.encode(rounds)
        return self.model.predict_pure(encoded, sample=True)[0][:, -1].tolist()

    def get_punishments_autoreg(self, rounds):
        raise NotImplementedError()
        group_size = len(rounds[-1]["contributions"])
        punishment = th.zeros(group_size, dtype=th.int64)
        punishment_valid = th.zeros(group_size, dtype=th.bool)
        encoded = self.encode(rounds)
        for i in range(group_size):
            encoded["punishment"] = punishment
            encoded["punishment_valid"] = punishment_valid
            punishment, punishment_valid = self.model.predict_autoreg(
                encoded, sample=True
            )
        return punishment.tolist()

    def get_punishments(self, rounds):
        encoded = self.encode(rounds)
        return self.model.predict_pure(encoded, sample=True)[0][:, -1].tolist()


class RLManager(Manager):
    def __init__(self, model_path):
        self.model = ArtificalManager.load(
            model_path, device=th.device("cpu")
        ).policy_model
        self.model.u_encoder.refrence = "contributions"

    def get_punishments(self, rounds):
        encoded = self.encode(rounds)
        return self.model.predict_pure(encoded, sample=False)[0][:, -1].tolist()


MANAGER_CLASS = {"human": HumanManager, "rl": RLManager}


class MultiManager:
    def __init__(self, managers):
        self.managers = {
            k: MANAGER_CLASS[m["type"]](m["path"]) for k, m in managers.items()
        }

    def get_punishments(self, rounds):
        groups = rounds[-1]["groups"]
        punishments = {k: m.get_punishments(rounds) for k, m in self.managers.items()}
        return [
            punishments[g][i] if g in punishments else None
            for i, g in enumerate(groups)
        ], punishments
