"""Rule validation for sweeps (#227): the Sobol design and validate-rule.

Runs locally: manager/rule.py has no PyG imports.
"""

import sys

import pytest
import yaml

from aimanager.manager.rule import (
    load_rule,
    read_rule,
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
            {**RULE, "code": "punishment = c[c0 + 5] * p * tau"},
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
