"""Tests for the sweep sim config generator (#227).

The generator's ROOT is pointed at a fake repo under tmp_path holding a copy
of the real TEMPLATE, so rule paths resolve as they do in the repo.
"""

import importlib.util
import json
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
        "w": {"definition": "punishment per point of shortfall", "type": "float"},
    },
    "sweep_config": {"c0": [0, 20], "w": [0, 3]},
    "code": "punishment = w * th.clamp(c0 - c[..., -1], min=0)",
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
    design = [{"c0": 1, "w": 1.0}]
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


def _sweep(repo):
    """A swept toy rule: sweep.json and landscapes.json in its sweep dir."""
    points = [
        {"name": f"toy_s{i:03d}", "params": {"c0": i, "w": 0.5}, "pool": float(p)}
        for i, p in enumerate([60, 72, 65, 70, 61, 68, 71, 64])
    ]
    sweep_dir = repo / gen.OUTPUT_DIR / "toy_sweep"
    sweep_dir.mkdir(parents=True)
    sweep = {"rule": "configs/managers/rule_based/toy.yml", "points": points}
    (sweep_dir / "sweep.json").write_text(json.dumps(sweep))
    peak = {"c0": 12, "w": 1.25}
    (sweep_dir / "landscapes.json").write_text(json.dumps({"peak": {"params": peak}}))
    return sweep_dir / "sweep.json", sweep, peak


def test_main_writes_the_final_run(repo, monkeypatch, capsys):
    sweep_file, _, peak = _sweep(repo)
    monkeypatch.setattr(sys, "argv", ["gen", "--final", str(sweep_file)])
    gen.main()
    out = repo / gen.CONFIG_DIR / "toy_sweep_final.yml"
    assert "the peak and the 5 best measured points x 2000 episodes" in (
        capsys.readouterr().out
    )
    config = yaml.safe_load(out.read_text())
    # the GP peak, then the 5 best measured points (pools 72, 71, 70, 68, 65)
    assert list(config["managers"]) == [
        "ah", "toy_gp", "toy_s001", "toy_s006", "toy_s003", "toy_s005", "toy_s002"
    ]
    assert config["managers"]["toy_gp"]["params"] == peak
    assert config["managers"]["toy_s006"]["params"] == {"c0": 6, "w": 0.5}
    assert len(config["pairings"]) == 6
    assert config["managers"]["toy_gp"]["rule"] == "configs/managers/rule_based/toy.yml"
    assert config["seed"] == gen.FINAL_SEED
    assert config["n_episodes"] == gen.FINAL_EPISODES
    assert config["output_dir"] == str(gen.OUTPUT_DIR / "toy_sweep/final")
    assert config["sweep"] is True
    # loads like any rule-based manager
    load_rule(config["managers"]["toy_gp"]["rule"], config["managers"]["toy_gp"]["params"])


def test_final_run_top_0_is_the_peak_only(repo, monkeypatch, capsys):
    sweep_file, _, _ = _sweep(repo)
    monkeypatch.setattr(sys, "argv", ["gen", "--final", str(sweep_file), "--top", "0"])
    gen.main()
    assert "the peak x 2000 episodes" in capsys.readouterr().out
    config = yaml.safe_load((repo / gen.CONFIG_DIR / "toy_sweep_final.yml").read_text())
    assert list(config["managers"]) == ["ah", "toy_gp"]


def test_score_final(tmp_path, monkeypatch, capsys):
    """best is the GP peak; best_of_n the best of the final run's points."""
    final = {
        "rule": "configs/managers/rule_based/toy.yml",
        "best": {"c0": 1},
        "best_name": "toy_s001",
        "points": [
            {"name": "toy_gp", "params": {"c0": 9}, "pool": 70.0, "pool_se": 1.1},
            {"name": "toy_s001", "params": {"c0": 1}, "pool": 72.5, "pool_se": 1.0},
            {"name": "toy_s006", "params": {"c0": 6}, "pool": 69.0, "pool_se": 1.2},
        ],
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(final))
    monkeypatch.setattr(sys, "argv", ["gen", "--score-final", str(path)])
    gen.main()
    assert "final toy_gp: 70.0 +- 1.1; best of 3 toy_s001: 72.5" in capsys.readouterr().out
    marked = json.loads(path.read_text())
    assert (marked["best"], marked["best_name"]) == ({"c0": 9}, "toy_gp")
    assert (marked["best_of_n"], marked["best_of_n_name"]) == ({"c0": 1}, "toy_s001")
    assert marked["n_points"] == 3
    # idempotent
    assert gen.score_final(path) == marked


def test_score_final_needs_the_gp_point(tmp_path, monkeypatch):
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps({"rule": "x/toy.yml", "points": [
        {"name": "toy_s001", "params": {}, "pool": 1.0}]}))
    monkeypatch.setattr(sys, "argv", ["gen", "--score-final", str(path)])
    with pytest.raises(SystemExit, match="no point toy_gp"):
        gen.main()


def test_final_run_needs_the_fit(repo, monkeypatch):
    sweep_file, _, _ = _sweep(repo)
    sweep_file.with_name("landscapes.json").unlink()
    monkeypatch.setattr(sys, "argv", ["gen", "--final", str(sweep_file)])
    with pytest.raises(SystemExit, match="fit the sweep first"):
        gen.main()


def test_main_takes_no_upper_param_bound(repo, monkeypatch, capsys):
    """--max-params none (#244): a later version may declare any number."""
    rule = repo / "configs/managers/rule_based/toy.yml"
    out = repo / "sweep.yml"
    argv = [
        "gen",
        "--config",
        str(rule),
        "--sobol-points",
        "8",
        "--out",
        str(out),
        "--max-params",
        "none",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    gen.main()
    assert "8 design points" in capsys.readouterr().out


def test_a_later_version_sweeps_and_runs_its_peak_under_its_own_name(
    repo, monkeypatch, capsys
):
    """Version 2's rule (#244) names its sweep's points, outputs and its
    final run's peak."""
    v2 = repo / "configs/managers/rule_based/toy-v2.yml"
    v2.write_text(yaml.safe_dump(RULE))
    monkeypatch.setattr(
        sys, "argv", ["gen", "--config", str(v2), "--sobol-points", "8"]
    )
    gen.main()
    config = yaml.safe_load((repo / gen.CONFIG_DIR / "toy-v2_sweep.yml").read_text())
    assert "toy-v2_s000" in config["managers"]
    assert config["output_dir"] == str(gen.OUTPUT_DIR / "toy-v2_sweep")

    sweep_dir = repo / gen.OUTPUT_DIR / "toy-v2_sweep"
    sweep_dir.mkdir(parents=True)
    points = [{"name": "toy-v2_s000", "params": {"c0": 3, "w": 0.5}, "pool": 66.0}]
    (sweep_dir / "sweep.json").write_text(
        json.dumps({"rule": "configs/managers/rule_based/toy-v2.yml", "points": points})
    )
    (sweep_dir / "landscapes.json").write_text(
        json.dumps({"peak": {"params": {"c0": 9, "w": 1.0}}})
    )
    monkeypatch.setattr(
        sys, "argv", ["gen", "--final", str(sweep_dir / "sweep.json")]
    )
    gen.main()
    out = repo / gen.CONFIG_DIR / "toy-v2_sweep_final.yml"
    config = yaml.safe_load(out.read_text())
    assert list(config["managers"]) == ["ah", "toy-v2_gp", "toy-v2_s000"]
    assert config["output_dir"] == str(gen.OUTPUT_DIR / "toy-v2_sweep/final")


# -- exp2_multi: every point against each opponent of a file -----------------

OPPONENTS = {
    "opponent_2": {"name": "b", "manager": {"type": "dummy", "constant_punishment": 0}},
    "opponent_1": {"name": "a", "manager": {"type": "dummy", "constant_punishment": 3}},
}


@pytest.fixture
def multi(repo):
    path = repo / gen.SETTINGS["exp2_multi"]["opponents"]
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(OPPONENTS))
    return repo


def test_a_multi_sweep_plays_every_opponent_under_its_key(multi):
    rule = "configs/managers/rule_based/toy.yml"
    design = validate_rule(rule, n_points=4)
    config = gen.build_sim_config(rule, design, n_episodes=5, setting="exp2_multi")

    rivals = ["opponent_1", "opponent_2"]
    assert config["sweep_opponents"] == rivals
    assert config["sweep_metric"] == "pool_vs_opponents"
    for key in rivals:
        assert config["managers"][key] == OPPONENTS[key]["manager"]
    assert "ah" not in config["managers"]
    assert config["pairings"] == [
        pairing
        for i in range(4) for r in rivals
        for pairing in (
            {"name": f"toy_s{i:03d}_vs_{r}", "group_0": f"toy_s{i:03d}", "group_1": r},
            {"name": f"{r}_vs_toy_s{i:03d}", "group_0": r, "group_1": f"toy_s{i:03d}"},
        )
    ]
    # the opponents' real names stay in their file
    text = yaml.safe_dump(config)
    assert "name: a" not in text and "name: b" not in text
    assert gen.pairings_per_point("exp2_multi") == 4 and gen.pairings_per_point("exp2") == 1


def test_multi_parts_seed_past_every_pairing(multi):
    rule = "configs/managers/rule_based/toy.yml"
    design = validate_rule(rule, n_points=16)
    parts = gen.build_sim_configs(rule, design, n_episodes=300, n_parts=2,
                                  setting="exp2_multi")
    (_, first), (_, second) = parts
    points = len({p["group_0"] for p in first["pairings"]} - {"opponent_1", "opponent_2"})
    batch = first["episode_batch_size"]
    assert second["seed"] == first["seed"] + -(-points * 4 * 300 // batch)
