import ast
import json
import math

import torch as th
import yaml
from typing import Optional, List, Union
from pydantic import BaseModel

from aimanager.generic.graph import GraphNetwork
from aimanager.manager.manager import ArtificalManager
from aimanager.generic.data import MAX_CONTRIBUTION, shift
from aimanager.simulation.linear_ah import LinearAHAdapter


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

    def create_tensor(record_key, default_key, invalid_reads_record=False):
        # invalid_reads_record: a flagged own-group cell keeps its recorded
        # value (for contributions, the env's timeout_contribution)
        default = int(default_values[default_key])

        def cell(value, is_valid, g1, g2):
            if g1 != g2:
                return default
            if is_valid or (invalid_reads_record and value is not None):
                return int(value)
            return default

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
        "contribution", "contribution", invalid_reads_record=True
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

    def batched_punish(self, state):
        """Env state ([B, A, 1] tensors) -> punishment [B, A, 1] int64."""
        return th.full_like(state["punishment"], self.constant_punishment)


#: What the rule code sees besides its params, and the name it must assign.
RULE_INPUTS = ("c", "t", "th")
RULE_OUTPUT = "punishment"
RULE_KEYS = {"params", "constraints", "code"}


def _names(tree):
    """(names read, names bound) anywhere in a parsed snippet."""
    read, bound = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed")
        if isinstance(node, ast.Name):
            (read if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    return read, bound


def load_rule(rule_path, params_path):
    """Read a rule YAML and its params JSON; return (compiled code, params).

    The YAML declares its fittable params by name under `params`; the JSON
    must hold exactly those names. The code is checked against the
    declaration, never used to discover it.

    `code` runs inside RuleBasedManager with these names in scope:
      c         contribution this round, per player (float tensor, 0-20)
      t         round number, per player (float tensor, 0-23)
      th        torch
      each name declared under `params`, from the params JSON
    It must set `punishment`; the manager then clamps it to [0, 30] and casts
    to integers. Python builtins are not available: use `th` for the maths.

    Checked here:
      - the only top-level keys are `params`, `constraints` and `code`
      - `params` declares every fittable param; none may be named c, t, th or
        punishment; the params JSON holds exactly these names, as finite numbers
      - `code` reads only c, t, th, declared params and names it assigns, reads
        every declared param, and assigns `punishment`
      - `constraints` (optional) are comparisons over declared params only;
        each must hold for the params JSON
    """
    with open(rule_path) as f:
        rule = yaml.safe_load(f)
    with open(params_path) as f:
        params = json.load(f)

    if not isinstance(rule, dict):
        raise ValueError(f"{rule_path}: must be a YAML mapping")
    unknown = sorted(set(rule) - RULE_KEYS)
    if unknown:
        raise ValueError(f"{rule_path}: unknown keys {unknown}")

    declared = rule.get("params")
    if not isinstance(declared, dict) or not declared:
        raise ValueError(f"{rule_path}: needs a non-empty `params` mapping")
    reserved = sorted(set(declared) & {*RULE_INPUTS, RULE_OUTPUT})
    if reserved:
        raise ValueError(f"{rule_path}: params use reserved names {reserved}")

    code = rule.get("code")
    if not isinstance(code, str):
        raise ValueError(f"{rule_path}: needs a `code` block")
    try:
        read, bound = _names(ast.parse(code, rule_path, "exec"))
    except ValueError as e:
        raise ValueError(f"{rule_path}: code: {e}") from None
    undefined = sorted(read - bound - set(RULE_INPUTS) - set(declared))
    if undefined:
        raise ValueError(f"{rule_path}: code reads undefined names {undefined}")
    unused = sorted(set(declared) - read)
    if unused:
        raise ValueError(f"{rule_path}: code never reads params {unused}")
    if RULE_OUTPUT not in bound:
        raise ValueError(f"{rule_path}: code never assigns `{RULE_OUTPUT}`")

    if not isinstance(params, dict):
        raise ValueError(f"{params_path}: must be a JSON object")
    missing = sorted(set(declared) - set(params))
    undeclared = sorted(set(params) - set(declared))
    if missing or undeclared:
        raise ValueError(
            f"{params_path} does not match the params declared in {rule_path}:"
            f" missing {missing}, undeclared {undeclared}"
        )
    for name, value in params.items():
        is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not is_number or not math.isfinite(value):
            raise ValueError(
                f"{params_path}: {name} = {value!r} is not a finite number"
            )

    constraints = rule.get("constraints", [])
    if not isinstance(constraints, list):
        raise ValueError(f"{rule_path}: `constraints` must be a list")
    for expr in constraints:
        if not isinstance(expr, str):
            raise ValueError(f"{rule_path}: constraint {expr!r} is not a string")
        tree = ast.parse(expr, rule_path, "eval")
        if not isinstance(tree.body, ast.Compare):
            raise ValueError(f"{rule_path}: constraint `{expr}` is not a comparison")
        read, _ = _names(tree)
        undefined = sorted(read - set(declared))
        if undefined:
            raise ValueError(
                f"{rule_path}: constraint `{expr}` reads undeclared names {undefined}"
            )
        if not eval(compile(tree, rule_path, "eval"), {"__builtins__": {}}, params):
            values = ", ".join(f"{n} = {params[n]}" for n in sorted(read))
            raise ValueError(f"{params_path}: constraint `{expr}` failed: {values}")

    return compile(code, rule_path, "exec"), params


class RuleBasedManager:
    # The rule YAML's code, run on this round's contribution and round number
    # (see load_rule).
    def __init__(self, rule=None, params=None, n_punishments=31, **_):
        if rule is None or params is None:
            raise ValueError("rule_based: `rule` and `params` are required")
        self.code, self.params = load_rule(rule, params)
        self.n_punishments = int(n_punishments)
        self.model = None
        self.default_values = {
            "contribution": 0,
            "punishment": 0,
            "contribution_valid": False,
            "punishment_valid": False,
            "in_group": False,
        }

    def get_punishments(self, data):
        raw = self._run_rule(data["contribution"], data["round_number"])
        return raw.clamp(0, self.n_punishments - 1).to(data["punishment"].dtype)

    def batched_punish(self, state):
        """Env state ([B, A, 1] tensors) -> punishment [B, A, 1] int64.

        The rule reads only the current round's contribution and round number,
        which the env state holds as the round history's last column does.
        """
        return self.get_punishments(state)

    def _run_rule(self, contribution, round_number):
        # c and t are fresh float copies, so in-place ops in the rule cannot
        # touch the history; no builtins, so the maths goes through th
        scope = {
            "__builtins__": {},
            "th": th,
            **self.params,
            "c": contribution.to(th.float),
            "t": round_number.to(th.float),
        }
        exec(self.code, scope)
        if RULE_OUTPUT not in scope:
            raise ValueError("rule_based: the rule code did not set `punishment`")
        # a scalar rule (e.g. a constant) is broadcast to every player and round;
        # the cast in get_punishments floors non-integer values after the clamp
        raw = th.as_tensor(
            scope[RULE_OUTPUT], dtype=th.float, device=contribution.device
        )
        if th.isnan(raw).any():
            raise ValueError("rule_based: the rule code produced NaN punishments")
        return raw.broadcast_to(contribution.shape)


class LinearManager:
    # consumes the raw two-group round history instead of the create_data view
    needs_rounds = True

    def __init__(self, model_path, sample=True, **_):
        import joblib

        self.bundle = joblib.load(model_path)
        self.sample = sample
        self.model = LinearAHAdapter(self.bundle, sample=sample)
        self.default_values = self.model.default_values
        self._batched = None

    def get_punishments(self, rounds):
        return self.model.get_punishments(rounds)

    def batched_punish(self, state):
        """Env state ([B, A, 1] tensors) -> punishment [B, A, 1] int64.

        Served by the RL opponent's batched adapter, built on first use: it
        only supports multinomial bundles with local features, which the
        per-episode path does not need.
        """
        if self._batched is None:
            from aimanager.manager.linear_opponent import LinearPunisherOpponent

            self._batched = LinearPunisherOpponent(self.bundle, sample=self.sample)
        return self._batched.predict(state)[0]


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
