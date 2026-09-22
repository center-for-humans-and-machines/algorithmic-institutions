import torch as th
from typing import Optional, List, Union
from pydantic import BaseModel

from aimanager.generic.graph import GraphNetwork
from aimanager.manager.manager import ArtificalManager
from aimanager.generic.data import MAX_CONTRIBUTION, MISSING_CONTRIBUTION, shift
from aimanager.simulation.linear_ah import LinearAHAdapter
from aimanager.manager.sigmoid_rule import sigmoid_punishment


class Round(BaseModel):
    round: int
    group: List[Union[str, int]]
    agent_group: Optional[List[int]] = None
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
    """Create data object for the algorithmic manager based on round records.

    The last round dict is the one being punished: `contribution[..., -1]`
    holds that round's realised contributions (the punisher's same-round
    input) and `prev_contribution[..., -1]` the round before; its punishments
    are not yet known, so `punishment[..., -1]` is the default placeholder.
    """

    def create_tensor(record_key, default_key, missing=None):
        """Own-group cells read the record; other-group cells are masked out
        and read the model's own default fill. `missing` is what an OWN-GROUP
        cell reads when that player or manager gave no input -- the value the
        game actually used, not the imputed default (see MISSING_CONTRIBUTION)."""
        default = int(default_values[default_key])
        miss = default if missing is None else int(missing)

        def cell(value, is_valid, g1, g2):
            if g1 != g2:
                return default
            return int(value) if is_valid else miss

        return th.tensor(
            [
                [
                    [
                        cell(value, is_valid, g1, g2)
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

    contribution = create_tensor(
        "contribution", "contribution", missing=MISSING_CONTRIBUTION
    )
    contribution_valid = create_bool_tensor("contribution")

    punishment = create_tensor("punishment", "punishment")
    punishment_valid = create_bool_tensor("punishment")

    in_group = th.tensor(
        [[[g1 == g2 for g1 in r["group"]] for r in rounds] for g2 in groups],
        dtype=th.bool,
    )

    group_size = len(rounds[-1]["contribution"])
    round_number = th.tensor(
        [[[r["round"]] * group_size for r in rounds] for _ in range(len(groups))],
        dtype=th.int64,
    )
    agent_group = th.tensor(
        [
            [r.get("agent_group", [0] * group_size) for r in rounds]
            for _ in range(len(groups))
        ],
        dtype=th.int64,
    )

    data = {
        "contribution": contribution.permute(0, 2, 1),
        "contribution_valid": contribution_valid.permute(0, 2, 1),
        "punishment": punishment.permute(0, 2, 1),
        "punishment_valid": punishment_valid.permute(0, 2, 1),
        "round_number": round_number.permute(0, 2, 1),
        "agent_group": agent_group.permute(0, 2, 1),
        "is_first": round_number.permute(0, 2, 1) == 0,
        "in_group": in_group.permute(0, 2, 1),
        # as create_torch_data_new: derived from the filled tensor, so
        # invalid / other-group cells (default contribution) read False
        "contribution_max": contribution.permute(0, 2, 1) == MAX_CONTRIBUTION,
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
    def __init__(self, constant_punishment=0, **_):
        self.constant_punishment = int(constant_punishment)
        self.model = None
        self.default_values = {
            "contribution": 0,
            "punishment": 0,
            "contribution_valid": False,
            "punishment_valid": False,
            "in_group": False,
        }

    def get_punishments(self, data):
        return th.full_like(data["punishment"], self.constant_punishment)


class RuleBasedManager:
    """A small family of hand-written, interpretable punishment rules.

    Every rule reads the CURRENT round's contribution `c` -- the same
    quantity the human manager saw when deciding (see the timing note in
    `create_data`) -- and returns a punishment clamped to
    [0, n_punishments - 1]:

    - `never`: p = 0 for every agent in every round.
    - `threshold`: p = `amount` where c <= `threshold`, else 0.
    - `inv_threshold`: p = `amount` where c >= `threshold`, else 0 -- the
      inverted-targeting mirror of `threshold`, punishing the HIGH
      contributors and sparing the low ones. Carried so the cost of
      punishing the wrong people can be measured against the cost of
      punishing at all.
    - `proportional`: p = round(`rate` * (MAX_CONTRIBUTION - c)).
    - `table`: p = `table[c]`, a 21-entry lookup indexed by contribution.
    - `severity_table`: with probability `prob_table[c]` punish `table[c]`,
      else 0 -- a shape that separates how often from how hard.
    - `decay`: p = (20 - c - round_number) // `k`, the original #99-era
      rule, kept so configs written against it keep their meaning.
    - `sigmoid`: the parametrised family of
      `aimanager.manager.sigmoid_rule`, a logistic in the contribution
      times two horizon multipliers. It nests `threshold` as `tau -> 0`
      and `never` as `p_max = 0`; the five parameters are `p_max`, `c0`,
      `tau`, `gamma_ep`, `gamma_sw`. Carried here so a fitted parameter
      vector can be re-run through the established simulation path beside
      the rules it has to beat.

    `skip_invalid` (default False) zeroes the action wherever the env has
    marked the player as having given no input. It is off by default
    because a contribution-keyed rule hits those cells by construction
    (they are served contribution 0) and how often it does is one of the
    things the sweep measures; turning it on isolates what that costs.
    """

    RULES = (
        "never",
        "threshold",
        "inv_threshold",
        "proportional",
        "table",
        "severity_table",
        "decay",
        "sigmoid",
    )

    def __init__(
        self,
        rule,
        threshold=None,
        amount=None,
        rate=None,
        table=None,
        prob_table=None,
        k=1,
        p_max=None,
        c0=None,
        tau=None,
        gamma_ep=0.0,
        gamma_sw=0.0,
        phase=0.0,
        n_rounds=24,
        switch_every=4,
        skip_invalid=False,
        n_punishments=31,
        **_,
    ):
        if rule not in self.RULES:
            raise ValueError(f"Unknown rule {rule!r}; expected one of {self.RULES}")
        self.rule = rule
        self.threshold = None if threshold is None else int(threshold)
        self.amount = None if amount is None else int(amount)
        self.rate = None if rate is None else float(rate)
        self.table = None if table is None else th.tensor(table, dtype=th.float)
        self.prob_table = (
            None if prob_table is None else th.tensor(prob_table, dtype=th.float)
        )
        self.k = int(k)
        self.p_max = None if p_max is None else float(p_max)
        self.c0 = None if c0 is None else float(c0)
        self.tau = None if tau is None else float(tau)
        self.gamma_ep = float(gamma_ep)
        self.gamma_sw = float(gamma_sw)
        self.phase = float(phase)
        self.n_rounds = int(n_rounds)
        self.switch_every = int(switch_every)
        self.skip_invalid = bool(skip_invalid)
        self.n_punishments = int(n_punishments)
        self.model = None
        self.default_values = {
            "contribution": 0,
            "punishment": 0,
            "contribution_valid": False,
            "punishment_valid": False,
            "in_group": False,
        }

    def _lookup(self, table, contribution):
        idx = contribution.clamp(0, table.shape[0] - 1)
        return table.to(contribution.device)[idx]

    def _raw(self, data):
        contribution = data["contribution"]
        if self.rule == "never":
            return th.zeros_like(contribution, dtype=th.float)
        if self.rule == "threshold":
            assert self.threshold is not None and self.amount is not None
            return th.where(
                contribution <= self.threshold,
                th.full_like(contribution, self.amount, dtype=th.float),
                th.zeros_like(contribution, dtype=th.float),
            )
        if self.rule == "inv_threshold":
            assert self.threshold is not None and self.amount is not None
            return th.where(
                contribution >= self.threshold,
                th.full_like(contribution, self.amount, dtype=th.float),
                th.zeros_like(contribution, dtype=th.float),
            )
        if self.rule == "proportional":
            assert self.rate is not None
            shortfall = (MAX_CONTRIBUTION - contribution).clamp(min=0)
            return (self.rate * shortfall).round()
        if self.rule == "table":
            assert self.table is not None
            return self._lookup(self.table, contribution).round()
        if self.rule == "severity_table":
            assert self.table is not None and self.prob_table is not None
            severity = self._lookup(self.table, contribution).round()
            prob = self._lookup(self.prob_table, contribution)
            fires = th.rand(prob.shape, device=prob.device) < prob
            return th.where(fires, severity, th.zeros_like(severity))
        if self.rule == "sigmoid":
            assert (
                self.p_max is not None
                and self.c0 is not None
                and self.tau is not None
                and self.tau > 0
            )
            return sigmoid_punishment(
                contribution,
                data["round_number"],
                p_max=self.p_max,
                c0=self.c0,
                tau=self.tau,
                gamma_ep=self.gamma_ep,
                gamma_sw=self.gamma_sw,
                phase=self.phase,
                n_rounds=self.n_rounds,
                switch_every=self.switch_every,
                n_punishments=self.n_punishments,
            )
        # decay
        return (
            (20 - contribution - data["round_number"])
            .div(self.k, rounding_mode="floor")
            .to(th.float)
        )

    def get_punishments(self, data):
        raw = self._raw(data).clamp(0, self.n_punishments - 1)
        out = raw.to(data["punishment"].dtype)
        if self.skip_invalid:
            out = th.where(data["contribution_valid"], out, th.zeros_like(out))
        return out


class LinearManager:
    # consumes the raw two-group round history instead of the create_data view
    needs_rounds = True

    def __init__(self, model_path, sample=True, **_):
        import joblib

        self.model = LinearAHAdapter(joblib.load(model_path), sample=sample)
        self.default_values = self.model.default_values

    def get_punishments(self, rounds):
        return self.model.get_punishments(rounds)


MANAGER_CLASS = {
    "human": HumanManager,
    "rl": RLManager,
    "dummy": DummyManager,
    "rule_based": RuleBasedManager,
    "linear": LinearManager,
}


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
            if not getattr(m, "needs_rounds", False)
        }

        punishment = {}
        for k, m in self.managers.items():
            if getattr(m, "needs_rounds", False):
                # raw round history in, per-agent [A] out; expand to the
                # [groups, A, T] shape the selection below slices
                pred = m.get_punishments(rounds)
                punishment[k] = pred.view(1, -1, 1).expand(
                    len(self.groups), -1, len(rounds)
                )
            else:
                punishment[k] = m.get_punishments(data[k])

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
