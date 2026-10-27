"""Config-defined rules for RuleBasedManager (#230), on the round history (#245).

Runs on Raven (api_manager imports GraphNetwork -> torch_scatter).
"""

import json

import pytest
import torch as th
import yaml


def _state(contribution, valid, groups, t, prev_punishment=None):
    """One round's env state for one episode, [1, players, 1] each."""

    def column(values, dtype=th.long):
        return th.tensor(values, dtype=dtype).view(1, -1, 1)

    n = len(contribution)
    return {
        "contribution": column(contribution),
        "contribution_valid": column(valid, th.bool),
        "prev_punishment": column(prev_punishment or [0] * n),
        "agent_group": column(groups),
        "round_number": column([t] * n),
        "punishment": column([0] * n),
    }


@pytest.fixture
def grid():
    """Round 0's env state: every contribution 0-20 a player, all in group 0."""
    return _state(list(range(21)), [True] * 21, [0] * 21, 0)


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
    "code": "punishment = a * (20 - c[..., -1])",
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
            "round": 0,
        }
    ]
    matched, _ = mm.get_punishments(rounds)
    assert matched == [20 - c for c in contribution[:4]] + [0] * 4


#: Three rounds of eight players: player 1 times out in round 1, players 2-3
#: switch to group 1 before round 2.
GROUPS = [[0, 0, 0, 0, 1, 1, 1, 1]] * 2 + [[0, 0, 1, 1, 1, 1, 1, 1]]
CONTRIBUTIONS = [
    [5, 10, 20, 0, 7, 14, 3, 18],
    [6, 0, 12, 20, 9, 2, 15, 4],
    [8, 11, 3, 16, 0, 20, 5, 9],
]
VALID = [[True] * 8, [True, False] + [True] * 6, [True] * 8]
#: Reads every input: the gap to the own group's mean over the game so far,
#: this round's timeouts, all punishments so far and the round
EVERY_INPUT = (
    "mine = valid * in_group\n"
    "mean = (c * mine).sum((1, 2), keepdim=True) / mine.sum((1, 2), keepdim=True)\n"
    "gap = (mean[..., 0] - c[..., -1]) * in_group[..., -1]\n"
    "punishment = a * gap + (1 - valid[..., -1]) + 0.1 * p.sum(-1) + 0.01 * t"
)


def _play(manager):
    """The three rounds through batched_punish, punishments charged as the
    env does (none on a timed-out player); returns (punishments, rounds)."""
    punished, rounds, charged = [], [], None
    for t in range(3):
        state = _state(CONTRIBUTIONS[t], VALID[t], GROUPS[t], t, charged)
        got = manager.batched_punish(state).view(-1)
        charged = [p if v else 0 for p, v in zip(got.tolist(), VALID[t])]
        punished.append(got)
        rounds.append(
            {
                "contribution": CONTRIBUTIONS[t],
                "contribution_valid": VALID[t],
                "punishment": charged,
                "agent_group": GROUPS[t],
                "round": t,
            }
        )
    return punished, rounds


def test_episode_path_matches_batched(tmp_path):
    """get_punishments on a round history gives what batched_punish gives
    from the env states of the same game."""
    from aimanager.manager.api_manager import RuleBasedManager

    rule = {**BASE, "code": EVERY_INPUT}
    manager = RuleBasedManager(**_rule(tmp_path, rule, {"a": 1}))
    punished, rounds = _play(manager)
    for t in range(3):
        # the round's own punishments are not known yet when it is punished
        now = {**rounds[t], "punishment": [None] * 8}
        assert th.equal(manager.get_punishments(rounds[:t] + [now]), punished[t])


def test_history_matches_the_game(tmp_path):
    """After round 2 the manager holds every round's contributions, timeouts
    and groups, and the punishments charged in rounds 0 and 1."""
    from aimanager.manager.api_manager import RuleBasedManager

    manager = RuleBasedManager(**_rule(tmp_path, BASE, {"a": 1}))
    _, rounds = _play(manager)
    history = {k: v.view(8, -1).T.tolist() for k, v in manager.history.items()}
    assert history["contribution"] == CONTRIBUTIONS
    assert history["valid"] == VALID
    assert history["agent_group"] == GROUPS
    assert history["punishment"] == [r["punishment"] for r in rounds[:2]]


@pytest.mark.parametrize(
    "code, expected",
    [("10 * in_group[..., -1]", 10), ("10 * (1 - in_group[..., -1])", 0)],
)
def test_each_player_sees_its_own_managers_view(tmp_path, code, expected):
    """The rule runs once per group: every player is in the group of the run
    that punishes it, whichever group that is."""
    from aimanager.manager.api_manager import RuleBasedManager

    rule = {**BASE, "code": f"punishment = a * {code}"}
    manager = RuleBasedManager(**_rule(tmp_path, rule, {"a": 1}))
    punished, _ = _play(manager)
    assert all(p.tolist() == [expected] * 8 for p in punished)


def test_batches_restart_and_rounds_cannot_be_skipped(tmp_path):
    from aimanager.manager.api_manager import RuleBasedManager

    manager = RuleBasedManager(**_rule(tmp_path, BASE, {"a": 1}))
    _play(manager)
    manager.batched_punish(_state(CONTRIBUTIONS[0], VALID[0], GROUPS[0], 0))
    assert manager.history["contribution"].shape[-1] == 1
    with pytest.raises(ValueError, match="round 2 without every earlier round"):
        manager.batched_punish(_state(CONTRIBUTIONS[2], VALID[2], GROUPS[2], 2))


# -- running a rule -------------------------------------------------------


def test_out_of_range_is_clamped(grid, tmp_path):
    from aimanager.manager.api_manager import RuleBasedManager

    rule = {**BASE, "code": "punishment = a * (100 - 10 * c[..., -1])"}
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

    rule = {
        **BASE,
        "code": "c.add_(5)\nin_group.add_(5)\npunishment = a * c[..., -1]",
    }
    before = {k: v.clone() for k, v in grid.items()}
    RuleBasedManager(**_rule(tmp_path, rule, {"a": 1})).batched_punish(grid)
    assert all(th.equal(grid[k], before[k]) for k in grid)


@pytest.mark.parametrize(
    "code, match",
    [
        ("if a > 5:\n    punishment = a * c[..., -1]", "did not set `punishment`"),
        ("punishment = (c - c)[..., -1] / (c - c)[..., -1] * a", "NaN"),
        ("punishment = a * c", "one value per player"),
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
        ({**BASE, "code": "q = a * c"}, {"a": 1}, "never assigns `punishment`"),
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
