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
        "p": {"definition": "punishment per point of shortfall", "type": "float"},
        "tau": {"definition": "softness of the cutoff", "type": "float"},
    },
    "sweep_config": {"c0": [0, 20], "p": [0, 3], "tau": [0.1, 10, "log"]},
    "constraints": ["p >= 0"],
    "code": "punishment = p * th.clamp(c0 - c, min=0) / (1 + tau)",
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
        assert isinstance(point["p"], float) and 0 <= point["p"] <= 3
        assert 0.1 <= point["tau"] <= 10
    # log-uniform: half the points below the geometric mean of the range
    below = sum(point["tau"] < 1 for point in design) / len(design)
    assert below == pytest.approx(0.5, abs=0.05)
    assert {point["c0"] for point in design} == set(range(21))


def test_design_fixed_and_merged(tmp_path):
    rule, _ = read_rule(_write(tmp_path, _with(c0=[1, 8], p=2, tau=1.5)))
    design = sobol_design(rule, 256)
    assert sorted(point["c0"] for point in design) == list(range(1, 9))
    assert all(point["p"] == 2.0 and point["tau"] == 1.5 for point in design)

    rule, _ = read_rule(_write(tmp_path, _with(c0=5, p=2, tau=1.5), "fixed"))
    assert sobol_design(rule, 256) == [{"c0": 5, "p": 2.0, "tau": 1.5}]


def test_int_range_gives_each_integer_an_equal_share(tmp_path):
    rule, _ = read_rule(_write(tmp_path, _with(c0=[0, 3], p=2, tau=1.5)))
    rule["sweep_config"]["p"] = [0, 3]  # keep points distinct, so none merge
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
        (
            _with(p=[-1, 3]),
            {},
            r"design point s\d{3} .*constraint `p >= 0` failed",
        ),
        (
            {
                **_with(p=[-1, 3]),
                "constraints": [],
                "code": "punishment = th.log(p) * c0 * tau",
            },
            {},
            r"design point s\d{3} .*NaN",
        ),
        (
            {**RULE, "code": "punishment = c[0, c0 + 5] * p * tau"},
            {},
            r"design point s\d{3}",
        ),
        (RULE, {"n_points": 100}, "power of two"),
    ],
)
def test_validate_rejects(tmp_path, rule, kwargs, match):
    if "min_params" in kwargs:
        kwargs = {**kwargs, "max_params": 4}
    with pytest.raises(ValueError, match=match):
        validate_rule(_write(tmp_path, rule), **kwargs)


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


def test_params_are_tensors_in_the_code(tmp_path):
    rule = {**RULE, "code": "punishment = th.exp(-tau) * th.log1p(p) * (c < c0)"}
    assert len(validate_rule(_write(tmp_path, rule))) == 256


# -- what the rule sees ---------------------------------------------------


def _state(groups, c, valid, c_prev, p_prev, t):
    def column(values, dtype=th.long):
        return th.tensor(values, dtype=dtype).view(1, -1, 1)

    inputs = rule_inputs(
        column(c),
        column(valid, th.bool),
        column(c_prev),
        column(p_prev),
        column(groups),
        column([t] * len(c)),
    )
    return {k: v.view(-1).tolist() for k, v in inputs.items()}


def test_rule_inputs():
    # player 2 timed out (its recorded contribution 0); group 1 is players 3, 4
    got = _state(
        groups=[0, 0, 0, 1, 1],
        c=[10, 4, 0, 6, 8],
        valid=[1, 1, 0, 1, 1],
        c_prev=[12, 5, 9, 6, 7],
        p_prev=[0, 6, 3, 2, 4],
        t=5,
    )
    assert set(got) == set(RULE_INPUTS)
    assert got["c"] == [10, 4, 0, 6, 8] and got["valid"] == [1, 1, 0, 1, 1]
    assert got["c_prev"] == [12, 5, 9, 6, 7] and got["p_prev"] == [0, 6, 3, 2, 4]
    assert got["t"] == [5] * 5
    assert got["n"] == [3, 3, 3, 2, 2] and got["n_other"] == [2, 2, 2, 3, 3]
    # the rest of the own group, valid players only: player 0 sees player 1,
    # player 2 (timed out) sees players 0 and 1
    assert got["c_group"] == [4, 10, 7, 8, 6]
    assert got["c_other"] == [7, 7, 7, 7, 7]
    assert got["p_prev_other"] == [3, 3, 3, 3, 3]


def test_rule_inputs_round_0_and_empty_groups():
    got = _state(
        groups=[0, 0],
        c=[10, 4],
        valid=[1, 0],
        c_prev=[1, 1],
        p_prev=[9, 9],
        t=0,
    )
    assert got["c_prev"] == [10, 4] and got["p_prev"] == [0, 0]
    assert got["n_other"] == [0, 0] and got["c_other"] == [0, 0]
    assert got["p_prev_other"] == [0, 0]
    # player 0's only group mate timed out: a mean over nobody is 0
    assert got["c_group"] == [0, 10]


def test_dry_run_covers_every_round_and_split():
    inputs = dry_run_inputs()
    assert inputs["t"].unique().tolist() == list(range(24))
    assert inputs["n"].min() == 1 and inputs["n"].max() == 8
    assert (inputs["n_other"] == 0).any() and (inputs["valid"] == 0).any()


def test_validate_reads_every_input(tmp_path):
    code = (
        "punishment = p * th.clamp(c_group - c, min=0) * valid + c0 * (c_prev < 5)"
        " + tau * p_prev / (1 + t) + n / (1 + n_other) + c_other + p_prev_other"
    )
    assert len(validate_rule(_write(tmp_path, {**RULE, "code": code}))) == 256
