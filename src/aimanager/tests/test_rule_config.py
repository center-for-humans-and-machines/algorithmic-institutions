"""Config-defined rules for RuleBasedManager (#230).

Runs on Raven (api_manager imports GraphNetwork -> torch_scatter).
"""

import json
from pathlib import Path

import pytest
import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[3]
RULES = ROOT / "configs/managers/rule_based"


def sigmoid_punishment(
    contribution,
    round_number,
    *,
    p_max,
    c0,
    tau,
    gamma_ep,
    gamma_sw,
    n_rounds=24,
    switch_every=4,
    n_punishments=31,
):
    """Levin's reference rule (#219, origin/auto/rule-sigmoid-family:
    src/aimanager/manager/sigmoid_rule.py), copied at phase = 0."""
    c = contribution.to(th.float)
    t = round_number.to(th.float)
    f = th.sigmoid(-(c - c0) / tau)
    ep_base = ((n_rounds - t) / n_rounds).clamp(min=0.0)
    s = (switch_every - 1) - th.remainder(t, switch_every)
    sw_base = ((s + 1.0) / switch_every).clamp(min=0.0)
    raw = p_max * f * ep_base**gamma_ep * sw_base**gamma_sw
    return raw.round().clamp(0.0, float(n_punishments - 1))


@pytest.fixture
def grid():
    """Every contribution 0-20 x round 0-23, shaped [groups, agents, rounds]."""
    c = th.arange(21).view(1, 21, 1).expand(2, 21, 24).contiguous()
    t = th.arange(24).view(1, 1, 24).expand(2, 21, 24).contiguous()
    return {"contribution": c, "round_number": t, "punishment": th.zeros_like(c)}


def _rule(tmp_path, rule, params, name="rule"):
    """Write a rule YAML and params JSON; return them as manager kwargs."""
    rule_path = tmp_path / f"{name}.yml"
    params_path = tmp_path / f"{name}.json"
    rule_path.write_text(yaml.safe_dump(rule))
    params_path.write_text(params if isinstance(params, str) else json.dumps(params))
    return {"rule": str(rule_path), "params": str(params_path)}


BASE = {
    "params": {"a": "a scale"},
    "constraints": ["a > 0"],
    "code": "punishment = a * (20 - c)",
}


# -- the committed rules --------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_decay_config_equals_builtin(grid, k):
    from aimanager.manager.api_manager import RuleBasedManager

    config = RuleBasedManager(
        rule=str(RULES / "decay.yml"), params=str(RULES / f"params/decay_k{k}.json")
    )
    builtin = RuleBasedManager(k=k)
    assert th.equal(config.get_punishments(grid), builtin.get_punishments(grid))


@pytest.mark.parametrize("name", ["sigmoid_opt_pool", "sigmoid_best_cap10_pool"])
def test_sigmoid_config_equals_reference(grid, name):
    from aimanager.manager.api_manager import RuleBasedManager

    params_path = RULES / f"params/{name}.json"
    manager = RuleBasedManager(rule=str(RULES / "sigmoid.yml"), params=str(params_path))
    params = json.loads(params_path.read_text())
    expected = sigmoid_punishment(grid["contribution"], grid["round_number"], **params)
    got = manager.get_punishments(grid)
    assert got.dtype == th.int64
    assert th.equal(got, expected.to(th.int64))


def test_legacy_k_unchanged(grid):
    from aimanager.manager.api_manager import RuleBasedManager

    c, t = grid["contribution"], grid["round_number"]
    expected = (20 - c - t).div(4, rounding_mode="floor").clamp(0, 30)
    assert th.equal(RuleBasedManager(k=4).get_punishments(grid), expected)
    assert th.equal(
        RuleBasedManager().get_punishments(grid),
        RuleBasedManager(k=1).get_punishments(grid),
    )


def test_multimanager_rule_side():
    from aimanager.manager.api_manager import MultiManager

    mm = MultiManager(
        {
            "rule": {
                "type": "rule_based",
                "rule": str(RULES / "decay.yml"),
                "params": str(RULES / "params/decay_k1.json"),
            },
            "zero": {"type": "dummy", "constant_punishment": 0},
        }
    )
    contribution = [5, 10, 20, 0, 7, 14, 3, 18]
    rounds = [
        {
            "contribution": contribution,
            "contribution_valid": [True] * 8,
            "punishment": [0] * 8,
            "punishment_valid": [False] * 8,
            "agent_group": [0, 0, 0, 0, 1, 1, 1, 1],
            "group": ["rule"] * 4 + ["zero"] * 4,
            "round": 2,
        }
    ]
    matched, _ = mm.get_punishments(rounds)
    assert matched == [max(20 - c - 2, 0) for c in contribution[:4]] + [0] * 4


# -- running a rule -------------------------------------------------------


def test_out_of_range_is_clamped(grid, tmp_path):
    from aimanager.manager.api_manager import RuleBasedManager

    rule = {**BASE, "code": "punishment = a * (100 - 10 * c)"}
    got = RuleBasedManager(**_rule(tmp_path, rule, {"a": 1})).get_punishments(grid)
    assert got.min() == 0 and got.max() == 30


def test_scalar_is_broadcast_and_floored(grid, tmp_path):
    from aimanager.manager.api_manager import RuleBasedManager

    rule = {**BASE, "code": "punishment = a * 7.9"}
    got = RuleBasedManager(**_rule(tmp_path, rule, {"a": 1})).get_punishments(grid)
    assert got.shape == grid["contribution"].shape
    assert got.unique().tolist() == [7]


def test_history_untouched(grid, tmp_path):
    from aimanager.manager.api_manager import RuleBasedManager

    rule = {**BASE, "code": "c.add_(5)\nt.add_(5)\npunishment = a * c"}
    before = {k: v.clone() for k, v in grid.items()}
    RuleBasedManager(**_rule(tmp_path, rule, {"a": 1})).get_punishments(grid)
    assert all(th.equal(grid[k], before[k]) for k in grid)


@pytest.mark.parametrize(
    "code, match",
    [
        ("if a > 5:\n    punishment = a * c", "did not set `punishment`"),
        ("punishment = (c - c) / (c - c) * a", "NaN"),
    ],
)
def test_run_errors(grid, tmp_path, code, match):
    from aimanager.manager.api_manager import RuleBasedManager

    manager = RuleBasedManager(**_rule(tmp_path, {**BASE, "code": code}, {"a": 1}))
    with pytest.raises(ValueError, match=match):
        manager.get_punishments(grid)


# -- checks at construction -----------------------------------------------


def test_rule_and_params_together(tmp_path):
    from aimanager.manager.api_manager import RuleBasedManager

    paths = _rule(tmp_path, BASE, {"a": 1})
    with pytest.raises(ValueError, match="set together"):
        RuleBasedManager(rule=paths["rule"])
    with pytest.raises(ValueError, match="set together"):
        RuleBasedManager(params=paths["params"])
    with pytest.raises(ValueError, match="cannot be combined"):
        RuleBasedManager(**paths, k=1)


@pytest.mark.parametrize(
    "rule, params, match",
    [
        ({**BASE, "parms": {}}, {"a": 1}, r"unknown keys \['parms'\]"),
        ({**BASE, "params": {}}, {}, "non-empty `params`"),
        (
            {**BASE, "params": {"a": "x", "t": "y"}, "code": "punishment = a * t"},
            {"a": 1, "t": 2},
            r"reserved names \['t'\]",
        ),
        ({**BASE, "code": None}, {"a": 1}, "needs a `code` block"),
        (
            {**BASE, "code": "punishment = a * (20 - cc)"},
            {"a": 1},
            r"undefined names \['cc'\]",
        ),
        ({**BASE, "code": "punishment = abs(a * c)"}, {"a": 1}, r"\['abs'\]"),
        (
            {**BASE, "params": {"a": "x", "b": "y"}},
            {"a": 1, "b": 2},
            r"never reads params \['b'\]",
        ),
        ({**BASE, "code": "p = a * c"}, {"a": 1}, "never assigns `punishment`"),
        (
            {**BASE, "code": "import os\npunishment = a * c"},
            {"a": 1},
            "imports are not allowed",
        ),
        (BASE, {}, r"missing \['a'\]"),
        (BASE, {"a": 1, "z": 2}, r"undeclared \['z'\]"),
        (BASE, "[1]", "must be a JSON object"),
        (BASE, '{"a": NaN}', "not a finite number"),
        (BASE, {"a": True}, "not a finite number"),
        (BASE, {"a": "1"}, "not a finite number"),
        ({**BASE, "constraints": "a > 0"}, {"a": 1}, "must be a list"),
        ({**BASE, "constraints": ["a + 1"]}, {"a": 1}, "not a comparison"),
        (
            {**BASE, "constraints": ["c > 0"]},
            {"a": 1},
            r"reads undeclared names \['c'\]",
        ),
        (BASE, {"a": -0.5}, r"constraint `a > 0` failed: a = -0.5"),
    ],
)
def test_load_errors(tmp_path, rule, params, match):
    from aimanager.manager.api_manager import RuleBasedManager

    with pytest.raises(ValueError, match=match):
        RuleBasedManager(**_rule(tmp_path, rule, params))


def test_load_accepts(tmp_path):
    from aimanager.manager.api_manager import RuleBasedManager

    no_constraints = {k: v for k, v in BASE.items() if k != "constraints"}
    RuleBasedManager(**_rule(tmp_path, no_constraints, {"a": -1}, "a"))
    comprehension = {**BASE, "code": "g = [x * a for x in (1, 2)]\npunishment = g[0]"}
    RuleBasedManager(**_rule(tmp_path, comprehension, {"a": 1}, "b"))


@pytest.mark.parametrize(
    "rule, params",
    [("decay", f"decay_k{k}") for k in (1, 2, 4, 8)]
    + [("sigmoid", "sigmoid_opt_pool"), ("sigmoid", "sigmoid_best_cap10_pool")],
)
def test_committed_rules_load(rule, params):
    from aimanager.manager.api_manager import RuleBasedManager

    RuleBasedManager(
        rule=str(RULES / f"{rule}.yml"), params=str(RULES / f"params/{params}.json")
    )
