"""Sweep sims (#227): check_sweep, sweep.json's scores and best.

Runs locally: simulation/sweep.py and pool_scores.py are pandas only.
"""

import json

import pandas as pd
import pytest
import yaml

from aimanager.manager.rule import load_rule
from aimanager.simulation.sweep import check_sweep, sweep_result, write_sweep

RULE_PATH = "configs/managers/rule_based/toy.yml"
RULE = {
    "params": {"a": {"definition": "a scale", "type": "float"}},
    "sweep_config": {"a": [0, 2]},
    "code": "punishment = a * (20 - c)",
}


def _config(points):
    managers = {"ah": {"type": "linear", "path": "x.joblib"}}
    pairings = []
    for name, a in points.items():
        managers[name] = {"type": "rule_based", "rule": RULE_PATH, "params": {"a": a}}
        pairings.append({"name": f"{name}_vs_ah", "group_0": name, "group_1": "ah"})
    return {"managers": managers, "pairings": pairings, "seed": 42, "n_episodes": 2}


def _frame(pools):
    """Per-round rows for `{pairing: [(c, p, group) per agent] per episode}`,
    two rounds per episode with the same rows."""
    rows = []
    for pairing, episodes in pools.items():
        for episode, agents in enumerate(episodes):
            for round_number in range(2):
                for c, p, group in agents:
                    rows.append(
                        {
                            "episode": episode,
                            "round_number": round_number,
                            "contribution": c,
                            "punishment": p,
                            "agent_group": group,
                            "run": f"ah group_switching managed by {pairing}",
                        }
                    )
    return pd.DataFrame(rows)


def test_scores_and_best():
    config = _config({"toy_s000": 0.5, "toy_s001": 1.5})
    df = _frame(
        {
            # group 0 pools: episode 0 1.6*10 - 2 = 14, episode 1 1.6*20 = 32
            "toy_s000_vs_ah": [[(10, 2, 0), (5, 0, 1)], [(20, 0, 0), (5, 0, 1)]],
            # group 0 empty in episode 0 (pool 0), 1.6*(10+10) - 4 = 28 in episode 1
            "toy_s001_vs_ah": [[(5, 0, 1)], [(10, 2, 0), (10, 2, 0), (5, 0, 1)]],
        }
    )
    result = sweep_result(df, config)
    s000, s001 = result["points"]
    assert s000["pool"] == pytest.approx(23.0)
    assert s000["pool_se"] == pytest.approx(9.0)  # sd 12.73 over 2 episodes
    assert s000["members"] == pytest.approx(1.0)
    assert s001["pool"] == pytest.approx(14.0) and s001["members"] == 1.0
    assert result["best"] == {"a": 0.5} and result["best_name"] == "toy_s000"
    assert result["rule"] == RULE_PATH and result["anchor"] == "ah"


def test_sweep_json_loads_as_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs/managers/rule_based").mkdir(parents=True)
    (tmp_path / RULE_PATH).write_text(yaml.safe_dump(RULE))
    config = _config({"toy_s000": 0.5})
    df = _frame({"toy_s000_vs_ah": [[(10, 2, 0)], [(20, 0, 0)]]})
    path = write_sweep(df, config, str(tmp_path))
    assert json.load(open(path))["points"][0]["name"] == "toy_s000"
    assert load_rule(RULE_PATH, path)[1] == {"a": 0.5}


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda c: c["managers"].pop("ah"), "anchor manager `ah`"),
        (lambda c: c.pop("pairings"), "needs `pairings`"),
        (lambda c: c["pairings"][0].update(group_1="toy_s001"), "group_1"),
        (lambda c: c["pairings"][0].update(group_0="ah"), "group_0"),
        (lambda c: c["managers"]["toy_s000"].update(type="dummy"), "rule_based"),
        (lambda c: c["managers"]["toy_s000"].update(params="p.json"), "inline"),
        (lambda c: c["managers"]["toy_s000"].update(rule="other.yml"), "one rule"),
    ],
)
def test_check_sweep_rejects(change, match):
    config = _config({"toy_s000": 0.5, "toy_s001": 1.5})
    assert check_sweep(config) == RULE_PATH
    change(config)
    with pytest.raises(ValueError, match=match):
        check_sweep(config)


def _result(points, **extra):
    """A sweep.json-like result: `{name: pool}`."""
    return {
        "rule": RULE_PATH,
        "anchor": "ah",
        "score": "def 4",
        "n_episodes": 500,
        "episode_batch_size": 1000,
        "seed": 42,
        "points": [
            {"name": n, "params": {"a": pool / 100}, "pool": pool}
            for n, pool in points.items()
        ],
        **extra,
    }


def test_merge_sweeps():
    from aimanager.simulation.sweep import merge_sweeps

    merged = merge_sweeps(
        [
            _result({"toy_s1000": 61.0, "toy_s1001": 70.0}, seed=43),
            _result({"toy_s200": 64.0, "toy_s999": 60.0}),
        ]
    )
    assert [p["name"] for p in merged["points"]] == [
        "toy_s200",
        "toy_s999",
        "toy_s1000",
        "toy_s1001",
    ]
    assert merged["best_name"] == "toy_s1001" and merged["best"] == {"a": 0.7}
    assert merged["seeds"] == [43, 42] and "seed" not in merged
    with pytest.raises(ValueError, match="more than one part"):
        merge_sweeps([_result({"toy_s000": 1.0}), _result({"toy_s000": 2.0})])
    with pytest.raises(ValueError, match="differ in `n_episodes`"):
        merge_sweeps([_result({"toy_s000": 1.0}), _result({}, n_episodes=200)])
