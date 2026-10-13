"""Tests for the sweep's Gaussian process fit (#238)."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from aimanager.manager.rule import read_rule, sobol_design

SCRIPT = Path(__file__).resolve().parents[1] / "policy_finder/fit_surrogate.py"
SPEC = importlib.util.spec_from_file_location("fit_surrogate", SCRIPT)
fs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fs)

RULE = """\
params:
  x:
    definition: a float that matters, peak at 6
    type: float
  a:
    definition: a log-scaled float that matters, peak at 1
    type: float
  k:
    definition: an int that matters, peak at 4
    type: int
  z:
    definition: a float that does not matter
    type: float
  b:
    definition: a fixed value
    type: float
sweep_config:
  x: [0, 10]
  a: [0.1, 10, log]
  k: [1, 9]
  z: [0, 1]
  b: 2.5
constraints:
{constraints}
code: |
  punishment = x * a * k * z * b * (c < c_group)
"""


def true_pool(p):
    return (
        70
        - 0.3 * (p["x"] - 6) ** 2
        - 3 * math.log(p["a"]) ** 2
        - 0.4 * (p["k"] - 4) ** 2
    )


def _sweep(tmp_path, constraints=("x >= 0",), se=1.5, n_points=256):
    """A sweep.json of the toy rule: the true pool plus noise of `se`."""
    rule_path = tmp_path / "toy.yml"
    rule_path.write_text(
        RULE.format(constraints="\n".join(f"  - {c}" for c in constraints))
    )
    rule, _ = read_rule(rule_path)
    rng = np.random.default_rng(1)
    design = [
        p
        for p in sobol_design(rule, n_points)
        if all(eval(c, {}, dict(p)) for c in constraints)
    ]
    points = [
        {
            "name": f"toy_s{i:03d}",
            "params": p,
            "pool": true_pool(p) + rng.normal(0, se),
            "pool_se": se,
        }
        for i, p in enumerate(design)
    ]
    sim_dir = tmp_path / "toy_sweep"
    sim_dir.mkdir()
    (sim_dir / "sweep.json").write_text(
        json.dumps({"rule": "toy.yml", "points": points})
    )
    return sim_dir, rule_path, rule, points


def test_recovers_the_peak_of_a_noisy_function(tmp_path):
    _, rule_path, rule, points = _sweep(tmp_path)
    out = fs.landscapes({"points": points}, rule, rule_path)
    peak = out["peak"]["params"]
    assert abs(peak["x"] - 6) < 1
    assert 0.7 < peak["a"] < 1.4
    assert peak["k"] in (3, 4, 5) and isinstance(peak["k"], int)
    assert peak["b"] == 2.5  # the fixed param carried along
    assert abs(out["peak"]["mean"] - 70) < 1.5
    # the measured best reads high by luck; the surface at that point does not
    best = max(points, key=lambda p: p["pool"])
    at_best = next(p for p in out["points"] if p["name"] == best["name"])
    assert abs(at_best["mean"] - true_pool(best["params"])) < abs(
        best["pool"] - true_pool(best["params"])
    )


def test_length_scales_say_which_params_matter(tmp_path):
    _, rule_path, rule, points = _sweep(tmp_path)
    out = fs.landscapes({"points": points}, rule, rule_path)
    params = out["params"]
    assert "z" not in out["peak"]["at_edge"]  # flat: its peak value is arbitrary
    assert set(params) == {"x", "a", "k", "z"}  # not the fixed b
    effect = {n: p["effect"] for n, p in params.items()}
    # true moves along each profile: x 10.8, a 15.9, k 9.6, z 0
    assert effect["z"] < 1
    assert min(effect["x"], effect["a"], effect["k"]) > 5
    ls = {n: p["length_scale"] for n, p in params.items()}
    assert ls["z"] > 5 * max(ls["x"], ls["a"], ls["k"])


def test_the_peak_meets_the_constraints(tmp_path):
    _, rule_path, rule, points = _sweep(tmp_path, constraints=("x <= 4",))
    out = fs.landscapes({"points": points}, rule, rule_path)
    assert out["peak"]["params"]["x"] <= 4
    assert "x" not in out["peak"]["at_edge"]  # 4 is not the range's end
    assert out["peak"]["params"]["x"] > 3


def test_main_writes_landscapes_beside_the_sweep(tmp_path, monkeypatch):
    sim_dir, _, _, points = _sweep(tmp_path, n_points=64)
    monkeypatch.chdir(tmp_path)  # sweep.json's rule is relative to here
    monkeypatch.setattr(sys, "argv", ["fit", str(sim_dir)])
    fs.main()
    out = json.loads((sim_dir / "landscapes.json").read_text())
    assert out["rule"] == "toy.yml"
    assert [p["name"] for p in out["points"]] == [p["name"] for p in points]
    assert out["profiles"]["k"]["x"] == list(range(1, 10))
    assert len(out["profiles"]["x"]["x"]) == fs.PROFILE_POINTS
    assert out["profiles"]["a"]["x"][0] == pytest.approx(0.1)
    assert out["profiles"]["a"]["x"][-1] == pytest.approx(10)
