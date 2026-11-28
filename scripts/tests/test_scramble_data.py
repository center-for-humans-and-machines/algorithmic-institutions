"""scramble_data.py: the test copies a single-idea agent debugs its script on."""

import json
import random
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy_finder"))
import scramble_data  # noqa: E402


def games(n_runs=5, n_episodes=6, rounds=24, players=8):
    rows = [
        {"run": f"run {r}", "episode": e, "round_number": t, "participant_code": f"{e}_{p}",
         "group_id": p % 2, "contribution": (7 * r + 3 * e + t + p) % 21,
         "punishment": (r + e + 2 * t + p) % 31, "payoff": float(r * 1000 + e * 100 + t + p / 10)}
        for r in range(n_runs) for e in range(n_episodes)
        for t in range(rounds) for p in range(players)
    ]
    return pd.DataFrame(rows)


def test_table_keeps_whole_games_and_shuffles_values():
    df = games()
    out = scramble_data.scramble_table(df, random.Random(1))
    assert list(out.columns) == list(df.columns)
    assert (out.dtypes == df.dtypes).all()
    assert out["run"].nunique() == scramble_data.MAX_RUNS
    # every kept game is complete: 24 rounds x 8 players
    assert set(out.groupby(["run", "episode"]).size()) == {24 * 8}
    assert out.groupby("run")["episode"].nunique().max() <= scramble_data.MAX_EPISODES
    # identifiers stay with their rows; values do not
    merged = out.merge(df, on=["run", "episode", "round_number", "participant_code"],
                       suffixes=("", "_real"))
    assert len(merged) == len(out)
    assert (merged["group_id"] == merged["group_id_real"]).all()
    assert (merged["payoff"] != merged["payoff_real"]).mean() > 0.9
    # a value column is a permutation of the kept rows' values
    assert sorted(out["payoff"]) == sorted(merged["payoff_real"])


def test_table_without_episodes_is_sampled():
    df = pd.DataFrame({"x": range(2000), "y": [i * 0.5 for i in range(2000)]})
    out = scramble_data.scramble_table(df, random.Random(1))
    assert len(out) == 500 and list(out.columns) == ["x", "y"]


def test_json_keeps_structure_and_permutes_floats():
    value = {
        "best_name": "p1", "n_episodes": 500, "ok": True,
        "points": [{"name": f"p{i}", "pool": 60.0 + i, "params": {"k": i, "a": i / 10}}
                   for i in range(40)],
        "params": {"a": {"range": [0.0, 4.0], "profile": [float(i) for i in range(30)]}},
    }
    real = json.loads(json.dumps(value))
    out = scramble_data.scramble_json(value, random.Random(1))
    assert out["best_name"] == "p1" and out["n_episodes"] == 500 and out["ok"] is True
    assert [p["name"] for p in out["points"]] == [p["name"] for p in real["points"]]
    assert [p["params"]["k"] for p in out["points"]] == list(range(40))  # integers stay
    pools = [p["pool"] for p in out["points"]]
    assert sorted(pools) == [p["pool"] for p in real["points"]] and pools != sorted(pools)
    assert out["params"]["a"]["range"] == [0.0, 4.0]  # a pair stays in order
    profile = out["params"]["a"]["profile"]
    assert sorted(profile) == real["params"]["a"]["profile"] and profile != sorted(profile)


def test_build_copies_data_files_at_the_same_paths(tmp_path):
    (tmp_path / "experiments").mkdir()
    sweep = tmp_path / "plots/simulation/policy_finder/x_sweep"
    sweep.mkdir(parents=True)
    games(2, 3).drop(columns="run").to_csv(tmp_path / "experiments/human.csv", index=False)
    games().to_parquet(sweep / "per_round.parquet", index=False)
    (sweep / "sweep.json").write_text(json.dumps({"points": [{"pool": 1.5}, {"pool": 2.5}]}))
    (sweep / "plot.jpg").write_text("not data")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/rule.json").write_text("{}")  # outside the roots
    written = scramble_data.build(str(tmp_path))
    assert sorted(written) == [
        "experiments/human.csv",
        "plots/simulation/policy_finder/x_sweep/per_round.parquet",
        "plots/simulation/policy_finder/x_sweep/sweep.json",
    ]
    out = tmp_path / "test_data"
    assert not (out / "plots/simulation/policy_finder/x_sweep/plot.jpg").exists()
    assert not (out / "configs").exists()
    copy = pd.read_parquet(out / "plots/simulation/policy_finder/x_sweep/per_round.parquet")
    assert set(copy.groupby(["run", "episode"]).size()) == {24 * 8}
    # the same input gives the same copy
    first = (out / "experiments/human.csv").read_text()
    scramble_data.build(str(tmp_path))
    assert (out / "experiments/human.csv").read_text() == first


@pytest.mark.parametrize("column, key", [
    ("episode_id", True), ("global_group_id", True), ("run", True),
    ("contribution", False), ("agent_group", False), ("common_good", False),
])
def test_identifier_columns(column, key):
    assert scramble_data.is_key(column) is key


def test_exp1_pilots_keep_whole_games():
    """Experiment 1's pilots (experiments/pilot_random1_player_round_slim.csv):
    games of 4 over 16 rounds, named by `episode_id` beside a 1-or-2 `episode`."""
    rows = [
        {"session": f"s{g % 3}", "global_group_id": f"s{g % 3} #{g}", "episode": 1 + g % 2,
         "episode_id": 100 + g, "experiment_name": "trail_rounds_2", "round_number": t,
         "participant_code": f"p{g}_{p}", "player_no_input": int((g + t + p) % 9 == 0),
         "manager_no_input": 0.0, "player_id": p, "contribution": float((g + t + p) % 21),
         "punishment": float((g * t + p) % 31), "payoff": float(g * 100 + t + p / 10),
         "common_good": float(g + t), "group_id": 0}
        for g in range(30) for t in range(16) for p in range(4)
    ]
    df = pd.DataFrame(rows)
    out = scramble_data.scramble_table(df, random.Random(3))
    assert list(out.columns) == list(df.columns)
    # whole games by episode_id: 16 rounds x 4 players each, a few of them
    sizes = out.groupby("episode_id").size()
    assert set(sizes) == {16 * 4} and len(sizes) <= scramble_data.MAX_EPISODES
    # the identifiers stay with their rows, the values do not
    merged = out.merge(df, on=["episode_id", "round_number", "participant_code"],
                       suffixes=("", "_real"))
    assert (merged["global_group_id"] == merged["global_group_id_real"]).all()
    assert (merged["payoff"] != merged["payoff_real"]).mean() > 0.9
