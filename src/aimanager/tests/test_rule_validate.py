"""Rule validation for sweeps (#227): the Sobol design and validate-rule.

Runs locally: manager/rule.py has no PyG imports.
"""

import sys

import pytest
import torch as th
import yaml

from aimanager.manager.rule import (
    RULE_INPUTS,
    dry_run_inputs,
    load_rule,
    read_rule,
    rule_inputs,
    sobol_design,
    validate_rule,
)

RULE = {
    "params": {
        "c0": {
            "definition": "contribution below which players are punished",
            "type": "int",
        },
        "k": {"definition": "punishment per point of shortfall", "type": "float"},
        "tau": {"definition": "softness of the cutoff", "type": "float"},
    },
    "sweep_config": {"c0": [0, 20], "k": [0, 3], "tau": [0.1, 10, "log"]},
    "constraints": ["k >= 0"],
    "code": "punishment = k * th.clamp(c0 - c[..., -1], min=0) / (1 + tau)",
}


def _write(tmp_path, rule, name="rule"):
    path = tmp_path / f"{name}.yml"
    path.write_text(yaml.safe_dump(rule))
    return str(path)


def _with(**sweep):
    return {**RULE, "sweep_config": {**RULE["sweep_config"], **sweep}}


# -- the design -----------------------------------------------------------


def test_design_ranges_types_and_determinism(tmp_path):
    rule, _ = read_rule(_write(tmp_path, RULE))
    design = sobol_design(rule, 256)
    assert len(design) == 256
    assert design == sobol_design(rule, 256)
    for point in design:
        assert isinstance(point["c0"], int) and 0 <= point["c0"] <= 20
        assert isinstance(point["k"], float) and 0 <= point["k"] <= 3
        assert 0.1 <= point["tau"] <= 10
    # log-uniform: half the points below the geometric mean of the range
    below = sum(point["tau"] < 1 for point in design) / len(design)
    assert below == pytest.approx(0.5, abs=0.05)
    assert {point["c0"] for point in design} == set(range(21))


def test_design_fixed_and_merged(tmp_path):
    rule, _ = read_rule(_write(tmp_path, _with(c0=[1, 8], k=2, tau=1.5)))
    design = sobol_design(rule, 256)
    assert sorted(point["c0"] for point in design) == list(range(1, 9))
    assert all(point["k"] == 2.0 and point["tau"] == 1.5 for point in design)

    rule, _ = read_rule(_write(tmp_path, _with(c0=5, k=2, tau=1.5), "fixed"))
    assert sobol_design(rule, 256) == [{"c0": 5, "k": 2.0, "tau": 1.5}]


def test_int_range_gives_each_integer_an_equal_share(tmp_path):
    rule, _ = read_rule(_write(tmp_path, _with(c0=[0, 3], k=2, tau=1.5)))
    rule["sweep_config"]["k"] = [0, 3]  # keep points distinct, so none merge
    design = sobol_design(rule, 256)
    counts = [sum(point["c0"] == k for point in design) for k in range(4)]
    assert counts == [64, 64, 64, 64]  # the ends too, not half as often


@pytest.mark.parametrize("n", [0, 3, 100])
def test_design_needs_a_power_of_two(tmp_path, n):
    rule, _ = read_rule(_write(tmp_path, RULE))
    with pytest.raises(ValueError, match="power of two"):
        sobol_design(rule, n)


def test_design_points_load_as_params(tmp_path):
    path = _write(tmp_path, RULE)
    rule, _ = read_rule(path)
    for point in sobol_design(rule, 16):
        assert load_rule(path, point)[1] == point


# -- validate_rule --------------------------------------------------------


def test_validate_accepts(tmp_path):
    assert len(validate_rule(_write(tmp_path, RULE))) == 256


@pytest.mark.parametrize(
    "rule, kwargs, match",
    [
        ({k: v for k, v in RULE.items() if k != "sweep_config"}, {}, "sweep_config"),
        (RULE, {"max_params": 2}, "declares 3 params, allowed 1 to 2"),
        (RULE, {"min_params": 4}, "declares 3 params, allowed 4 to 4"),
        (RULE, {"min_params": 4, "max_params": None}, "allowed at least 4"),
        (
            _with(k=[-1, 3]),
            {},
            r"design point s\d{3} .*constraint `k >= 0` failed",
        ),
        (
            {
                **_with(k=[-1, 3]),
                "constraints": [],
                "code": "punishment = th.log(k) * c0 * tau",
            },
            {},
            r"design point s\d{3} .*NaN",
        ),
        (
            {**RULE, "code": "punishment = c[0, c0 + 5] * k * tau"},
            {},
            r"design point s\d{3}",
        ),
        # one value per round, not per player
        (
            {**RULE, "code": "punishment = k * c * c0 * tau"},
            {},
            r"design point s\d{3}.*one value per player",
        ),
        # round 0 has no punishment yet to read
        (
            {**RULE, "code": "punishment = k * p[..., -1] * c0 * tau"},
            {},
            r"design point s\d{3}.*, round 0:",
        ),
        (
            {
                **RULE,
                "params": {
                    **RULE["params"],
                    "in_group": {"definition": "x", "type": "int"},
                },
            },
            {},
            r"reserved names \['in_group'\]",
        ),
        (RULE, {"n_points": 100}, "power of two"),
    ],
)
def test_validate_rejects(tmp_path, rule, kwargs, match):
    if "min_params" in kwargs and "max_params" not in kwargs:
        kwargs = {**kwargs, "max_params": 4}
    with pytest.raises(ValueError, match=match):
        validate_rule(_write(tmp_path, rule), **kwargs)


def test_validate_without_upper_bound(tmp_path):
    """max_params None takes any number of params from min_params (#244)."""
    assert len(validate_rule(_write(tmp_path, RULE), 1, None, 16)) == 16


# -- the CLI --------------------------------------------------------------


def test_cli(tmp_path, monkeypatch, capsys):
    from aimanager.cli import main

    good = _write(tmp_path, RULE, "good")
    monkeypatch.setattr(sys, "argv", ["aimanager", "validate-rule", good])
    main()
    assert "Valid" in capsys.readouterr().out

    bad = _write(tmp_path, RULE, "bad")
    argv = ["aimanager", "validate-rule", bad, "--max-params", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 1
    assert "declares 3 params" in capsys.readouterr().err

    argv = ["aimanager", "validate-rule", bad, "--max-params", "none"]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    assert "Valid" in capsys.readouterr().out


def test_params_are_tensors_in_the_code(tmp_path):
    rule = {
        **RULE,
        "code": "punishment = th.exp(-tau) * th.log1p(k) * (c[..., -1] < c0)",
    }
    assert len(validate_rule(_write(tmp_path, rule))) == 256


# -- what the rule sees ---------------------------------------------------


def test_rule_inputs():
    """Rounds 0..1 of three players; player 2 timed out in round 1 and moved
    to group 1 in it."""
    got = rule_inputs(
        contribution=th.tensor([[[10, 12], [4, 5], [6, 0]]]),
        valid=th.tensor([[[1, 1], [1, 1], [1, 0]]], dtype=th.bool),
        punishment=th.tensor([[[0], [6], [3]]]),
        agent_group=th.tensor([[[0, 0], [0, 0], [0, 1]]]),
        group=0,
    )
    assert set(got) == set(RULE_INPUTS)
    assert all(v.dtype == th.float for v in got.values())
    assert got["c"].tolist() == [[[10, 12], [4, 5], [6, 0]]]
    assert got["valid"].tolist() == [[[1, 1], [1, 1], [1, 0]]]
    assert got["p"].tolist() == [[[0], [6], [3]]]
    assert got["in_group"].tolist() == [[[1, 1], [1, 1], [1, 0]]]
    assert got["t"].shape == () and got["t"] == 1
    # relative to the manager's group
    other = rule_inputs(
        th.zeros(1, 3, 2),
        th.ones(1, 3, 2),
        th.zeros(1, 3, 1),
        th.tensor([[[0, 0], [0, 0], [0, 1]]]),
        group=1,
    )
    assert other["in_group"].tolist() == [[[0, 0], [0, 0], [0, 1]]]


def test_rule_inputs_round_0_and_mismatch():
    got = rule_inputs(
        th.ones(2, 8, 1), th.ones(2, 8, 1), th.zeros(2, 8, 0), th.zeros(2, 8, 1), 0
    )
    assert got["t"] == 0 and got["p"].shape == (2, 8, 0)
    with pytest.raises(ValueError, match="expected one fewer"):
        rule_inputs(
            th.ones(2, 8, 3), th.ones(2, 8, 3), th.zeros(2, 8, 3), th.zeros(2, 8, 3), 0
        )


def test_dry_run_covers_rounds_splits_and_timeouts():
    histories = dry_run_inputs()
    assert [int(h["t"]) for h in histories] == [0, 11, 23]
    for h in histories:
        t = int(h["t"])
        assert h["c"].shape == (2048, 8, t + 1) and h["p"].shape == (2048, 8, t)
        now = h["in_group"][..., -1].sum(1)  # the group's size this round
        assert now.min() == 0 and now.max() == 8
        assert (h["valid"] == 0).any()


def test_validate_reads_every_input(tmp_path):
    code = (
        "now = in_group[..., -1]\n"
        "mean = (c * valid * in_group).sum(-1) / (valid * in_group).sum(-1)"
        ".clamp(min=1)\n"
        "punishment = k * th.clamp(mean - c[..., -1], min=0) * now"
        " + c0 * p.sum(-1) / (1 + t) / tau"
    )
    assert len(validate_rule(_write(tmp_path, {**RULE, "code": code}))) == 256
