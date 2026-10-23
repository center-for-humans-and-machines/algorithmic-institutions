"""Config-defined rules for RuleBasedManager (#230).

Runs on Raven (api_manager imports GraphNetwork -> torch_scatter).
"""

import json

import pytest
import torch as th
import yaml


@pytest.fixture
def grid():
    """An env state: every round 0-23 an episode, every contribution 0-20 a
    player, all in group 0, shaped [episodes, players, 1]."""
    c = th.arange(21).view(1, 21, 1).expand(24, 21, 1).contiguous()
    t = th.arange(24).view(24, 1, 1).expand(24, 21, 1).contiguous()
    return {
        "contribution": c,
        "contribution_valid": th.ones_like(c, dtype=th.bool),
        "prev_contribution": c.flip(1),
        "prev_punishment": th.full_like(c, 3),
        "agent_group": th.zeros_like(c),
        "round_number": t,
        "punishment": th.zeros_like(c),
    }


def _rule(tmp_path, rule, params, name="rule"):
    """Write a rule YAML and params JSON; return them as manager kwargs."""
    rule_path = tmp_path / f"{name}.yml"
    params_path = tmp_path / f"{name}.json"
    rule_path.write_text(yaml.safe_dump(rule))
    params_path.write_text(params if isinstance(params, str) else json.dumps(params))
    return {"rule": str(rule_path), "params": str(params_path)}


A = {"definition": "a scale", "type": "float"}
BASE = {
    "params": {"a": A},
    "sweep_config": {"a": [0.5, 2]},
    "constraints": ["a > 0"],
    "code": "punishment = a * (20 - c)",
}


def test_multimanager_rule_side(tmp_path):
    from aimanager.manager.api_manager import MultiManager

    mm = MultiManager(
        {
            "rule": {
                "type": "rule_based",
                **_rule(tmp_path, BASE, {"a": 1}),
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
    assert matched == [20 - c for c in contribution[:4]] + [0] * 4


def test_episode_path_matches_batched(tmp_path):
    """get_punishments on a round history reads what batched_punish reads in
    the env state: the same inputs, the same punishments."""
    from aimanager.manager.api_manager import RuleBasedManager

    code = (
        "punishment = a * (c_group - c) + (1 - valid) + 0.1 * (c_prev + p_prev)"
        " + n + 2 * n_other + 0.1 * (c_other + p_prev_other) + 0.01 * t"
    )
    manager = RuleBasedManager(**_rule(tmp_path, {**BASE, "code": code}, {"a": 1}))
    groups = [[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1]]
    contributions = [[5, 10, 20, 0, 7, 14, 3, 18], [6, 0, 12, 20, 9, 2, 15, 4]]
    valid = [[True] * 8, [True, False] + [True] * 6]
    charged = [[4, 0, 0, 9, 2, 0, 6, 1], None]
    rounds = [
        {
            "contribution": contributions[r],
            "contribution_valid": valid[r],
            "punishment": charged[r],
            "agent_group": groups[r],
            "round": r,
        }
        for r in range(2)
    ]

    def column(values):
        return th.tensor(values).view(1, 8, 1)

    for r in range(2):
        state = {
            "contribution": column(contributions[r]),
            "contribution_valid": column(valid[r]),
            "prev_contribution": column(contributions[r - 1] if r else [0] * 8),
            "prev_punishment": column(charged[r - 1] if r else [0] * 8),
            "agent_group": column(groups[r]),
            "round_number": column([r] * 8),
        }
        batched = manager.batched_punish(state).view(-1)
        assert th.equal(manager.get_punishments(rounds[: r + 1]), batched)


# -- running a rule -------------------------------------------------------


def test_out_of_range_is_clamped(grid, tmp_path):
    from aimanager.manager.api_manager import RuleBasedManager

    rule = {**BASE, "code": "punishment = a * (100 - 10 * c)"}
    got = RuleBasedManager(**_rule(tmp_path, rule, {"a": 1})).batched_punish(grid)
    assert got.min() == 0 and got.max() == 30


def test_scalar_is_broadcast_and_floored(grid, tmp_path):
    from aimanager.manager.api_manager import RuleBasedManager

    rule = {**BASE, "code": "punishment = a * 7.9"}
    got = RuleBasedManager(**_rule(tmp_path, rule, {"a": 1})).batched_punish(grid)
    assert got.shape == grid["contribution"].shape
    assert got.unique().tolist() == [7]


def test_history_untouched(grid, tmp_path):
    from aimanager.manager.api_manager import RuleBasedManager

    rule = {**BASE, "code": "c.add_(5)\nn.add_(5)\npunishment = a * c"}
    before = {k: v.clone() for k, v in grid.items()}
    RuleBasedManager(**_rule(tmp_path, rule, {"a": 1})).batched_punish(grid)
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
        manager.batched_punish(grid)


# -- checks at construction -----------------------------------------------


def test_rule_and_params_required(tmp_path):
    from aimanager.manager.api_manager import RuleBasedManager

    paths = _rule(tmp_path, BASE, {"a": 1})
    with pytest.raises(ValueError, match="are required"):
        RuleBasedManager()
    with pytest.raises(ValueError, match="are required"):
        RuleBasedManager(rule=paths["rule"])
    with pytest.raises(ValueError, match="are required"):
        RuleBasedManager(params=paths["params"])


@pytest.mark.parametrize(
    "rule, params, match",
    [
        ({**BASE, "parms": {}}, {"a": 1}, r"unknown keys \['parms'\]"),
        ({**BASE, "params": {}}, {}, "non-empty `params`"),
        (
            {**BASE, "params": {"a": A, "t": A}, "code": "punishment = a * t"},
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
            {**BASE, "params": {"a": A, "b": A}, "sweep_config": {"a": 1, "b": 2}},
            {"a": 1, "b": 2},
            r"never reads params \['b'\]",
        ),
        ({**BASE, "params": {"a": "a scale"}}, {"a": 1}, "must declare exactly"),
        (
            {**BASE, "params": {"a": {**A, "range": [0, 1]}}},
            {"a": 1},
            "must declare exactly",
        ),
        ({**BASE, "params": {"a": {**A, "definition": " "}}}, {"a": 1}, "definition"),
        ({**BASE, "params": {"a": {**A, "type": "bool"}}}, {"a": 1}, "type 'bool'"),
        (
            {
                **BASE,
                "params": {"a": {**A, "type": "int"}},
                "sweep_config": {"a": [1, 2]},
            },
            {"a": 1.5},
            "not an integer",
        ),
        ({**BASE, "sweep_config": [1, 2]}, {"a": 1}, "must be a mapping"),
        ({**BASE, "sweep_config": {}}, {"a": 1}, r"missing \['a'\]"),
        (
            {**BASE, "sweep_config": {"a": 1, "z": 2}},
            {"a": 1},
            r"undeclared \['z'\]",
        ),
        ({**BASE, "sweep_config": {"a": [1]}}, {"a": 1}, r"\[low, high\]"),
        ({**BASE, "sweep_config": {"a": [1, "2"]}}, {"a": 1}, r"\[low, high\]"),
        ({**BASE, "sweep_config": {"a": True}}, {"a": 1}, r"\[low, high\]"),
        ({**BASE, "sweep_config": {"a": [2, 2]}}, {"a": 1}, "not below high"),
        ({**BASE, "sweep_config": {"a": [1, 2, "lin"]}}, {"a": 1}, "must be `log`"),
        ({**BASE, "sweep_config": {"a": [0, 2, "log"]}}, {"a": 1}, "low > 0"),
        (
            {
                **BASE,
                "params": {"a": {**A, "type": "int"}},
                "sweep_config": {"a": [0.5, 3]},
            },
            {"a": 1},
            "int range takes integer bounds",
        ),
        (
            {**BASE, "params": {"a": {**A, "type": "int"}}, "sweep_config": {"a": 1.5}},
            {"a": 1},
            "not an integer",
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
        (
            {**BASE, "params": {"a": A, "best": A}, "code": "punishment = a * best"},
            {"a": 1, "best": 2},
            r"reserved names \['best'\]",
        ),
        (BASE, {"best": [1]}, "`best`: must be a JSON object"),
        (BASE, {"best": {"a": -1}, "points": []}, "`best`: constraint `a > 0` failed"),
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
    no_sweep = {k: v for k, v in BASE.items() if k != "sweep_config"}
    RuleBasedManager(**_rule(tmp_path, no_sweep, {"a": 1}, "d"))
    comprehension = {**BASE, "code": "g = [x * a for x in (1, 2)]\npunishment = g[0]"}
    RuleBasedManager(**_rule(tmp_path, comprehension, {"a": 1}, "b"))
    as_int = {**BASE, "params": {"a": {**A, "type": "int"}}}
    for i, sweep in enumerate([2, 2.0, [1, 8], [1, 10, "log"]]):
        rule = {**as_int, "sweep_config": {"a": sweep}}
        RuleBasedManager(**_rule(tmp_path, rule, {"a": 2.0}, f"c{i}"))


# -- where the params come from -------------------------------------------


def test_params_inline_or_best(grid, tmp_path):
    from aimanager.manager.api_manager import RuleBasedManager

    from_file = _rule(tmp_path, BASE, {"a": 0.5})
    expected = RuleBasedManager(**from_file).batched_punish(grid)
    inline = {"a": 0.5}
    sweep = {"best": {"a": 0.5}, "points": [{"name": "s000", "params": {"a": 2}}]}
    sweep_path = _rule(tmp_path, BASE, sweep, "sweep")["params"]
    for params in (inline, sweep_path):
        manager = RuleBasedManager(rule=from_file["rule"], params=params)
        assert th.equal(manager.batched_punish(grid), expected)
    assert inline == {"a": 0.5}


def test_inline_params_errors_name_their_source(tmp_path):
    from aimanager.manager.api_manager import RuleBasedManager

    rule = _rule(tmp_path, BASE, {"a": 1})["rule"]
    with pytest.raises(ValueError, match=r"^inline params: a = 'x'"):
        RuleBasedManager(rule=rule, params={"a": "x"})
    with pytest.raises(ValueError, match=r"^inline params `best` does not match"):
        RuleBasedManager(rule=rule, params={"best": {}})
