"""Tests for the sweep sim config generator (#227).

The generator's ROOT is pointed at a fake repo under tmp_path holding a copy
of the real TEMPLATE, so rule paths resolve as they do in the repo.
"""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from aimanager.cli import REQUIRED_KEYS
from aimanager.manager.rule import load_rule, validate_rule

SCRIPT = Path(__file__).resolve().parents[1] / "policy_finder/generate_sim_config.py"
SPEC = importlib.util.spec_from_file_location("generate_sim_config", SCRIPT)
gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen)

RULE = {
    "params": {
        "c0": {
            "definition": "contribution below which players are punished",
            "type": "int",
        },
        "p": {"definition": "punishment per point of shortfall", "type": "float"},
    },
    "sweep_config": {"c0": [0, 20], "p": [0, 3]},
    "code": "punishment = p * th.clamp(c0 - c, min=0)",
}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake repo root with the template and a rule config."""
    root = tmp_path / "repo"
    (root / gen.TEMPLATE).parent.mkdir(parents=True)
    shutil.copy(gen.ROOT / gen.TEMPLATE, root / gen.TEMPLATE)
    (root / "configs/managers/rule_based").mkdir(parents=True)
    (root / "configs/managers/rule_based/toy.yml").write_text(yaml.safe_dump(RULE))
    monkeypatch.setattr(gen, "ROOT", root)
    monkeypatch.chdir(root)
    return root


def test_build(repo):
    rule = "configs/managers/rule_based/toy.yml"
    design = validate_rule(rule, n_points=16)
    config = gen.build_sim_config(rule, design, n_episodes=7, seed=3)
    with open(repo / gen.TEMPLATE) as f:
        template = yaml.safe_load(f)

    assert list(config["managers"]) == ["ah"] + [f"toy_s{i:03d}" for i in range(16)]
    assert config["managers"]["ah"] == template["managers"]["ah"]
    for i, point in enumerate(design):
        manager = config["managers"][f"toy_s{i:03d}"]
        assert manager == {"type": "rule_based", "rule": rule, "params": point}
    assert config["pairings"] == [
        {"name": f"toy_s{i:03d}_vs_ah", "group_0": f"toy_s{i:03d}", "group_1": "ah"}
        for i in range(16)
    ]
    assert (config["n_episodes"], config["seed"], config["sweep"]) == (7, 3, True)
    assert config["output_dir"] == "plots/simulation/policy_finder/toy_sweep"
    for key in ("artificial_humans", "episode_batch_size", "timeout_contribution"):
        assert config[key] == template[key]
    assert all(k in config for k in REQUIRED_KEYS["simulate"])


def test_rejects(repo, tmp_path):
    design = [{"c0": 1, "p": 1.0}]
    bad = repo / "configs/managers/rule_based/a_vs_b.yml"
    bad.write_text(yaml.safe_dump(RULE))
    with pytest.raises(ValueError, match="contains `_vs_`"):
        gen.build_sim_config(bad, design)
    outside = tmp_path / "toy.yml"
    outside.write_text(yaml.safe_dump(RULE))
    with pytest.raises(ValueError, match="outside the repo"):
        gen.build_sim_config(outside, design)


def test_main_writes_a_loadable_config(repo, monkeypatch, capsys):
    rule = repo / "configs/managers/rule_based/toy.yml"
    out = repo / "sweep.yml"
    argv = ["gen", "--config", str(rule), "--sobol-points", "8", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    gen.main()
    assert "8 design points x 500 episodes = 4000 episodes" in capsys.readouterr().out

    config = yaml.safe_load(out.read_text())
    assert (
        config["managers"]["toy_s000"]["params"] == validate_rule(rule, n_points=8)[0]
    )
    for name, manager in config["managers"].items():
        if name != "ah":
            load_rule(manager["rule"], manager["params"])


def test_main_rejects_an_invalid_rule(repo, monkeypatch):
    rule = repo / "configs/managers/rule_based/toy.yml"
    monkeypatch.setattr(
        sys, "argv", ["gen", "--config", str(rule), "--max-params", "1"]
    )
    with pytest.raises(SystemExit, match="Invalid: .*declares 2 params"):
        gen.main()


def test_parts_split_the_sweep(repo):
    rule = "configs/managers/rule_based/toy.yml"
    design = validate_rule(rule, n_points=16)
    ((_, whole),) = gen.build_sim_configs(rule, design, n_episodes=300)
    parts = gen.build_sim_configs(rule, design, n_episodes=300, n_parts=3)

    assert [s for s, _ in parts] == ["_p1of3", "_p2of3", "_p3of3"]
    pairings = [p for _, c in parts for p in c["pairings"]]
    assert pairings == whole["pairings"]
    for suffix, config in parts:
        assert (
            config["output_dir"] == f"plots/simulation/policy_finder/toy_sweep{suffix}"
        )
        assert set(config["managers"]) == {"ah"} | {
            p["group_0"] for p in config["pairings"]
        }
    # batch seeds never repeat: part k starts after the parts before it
    batch = whole["episode_batch_size"]
    sizes = [len(c["pairings"]) for _, c in parts]
    assert sizes == [5, 6, 5]
    seeds = [c["seed"] for _, c in parts]
    assert seeds[0] == 42
    for k in (1, 2):
        assert seeds[k] == seeds[k - 1] + -(-sizes[k - 1] * 300 // batch)
    with pytest.raises(ValueError, match="--n-parts"):
        gen.build_sim_configs(rule, design, n_parts=17)


def test_main_writes_parts(repo, monkeypatch):
    rule = repo / "configs/managers/rule_based/toy.yml"
    argv = ["gen", "--config", str(rule), "--sobol-points", "8", "--n-parts", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    gen.main()
    written = sorted(p.name for p in (repo / gen.CONFIG_DIR).iterdir())
    assert written == ["toy_sweep_p1of2.yml", "toy_sweep_p2of2.yml"]
    monkeypatch.setattr(sys, "argv", argv + ["--out", str(repo / "x.yml")])
    with pytest.raises(SystemExit, match="one config"):
        gen.main()
