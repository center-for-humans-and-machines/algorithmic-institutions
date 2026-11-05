"""Experiment 1's sweep score, pool_scores.group_payoff_scores."""

import pandas as pd
import pytest

from aimanager.simulation.pool_scores import group_payoff_scores


def _rows(run, episode, rnd, payoffs, valid):
    return [
        {"run": run, "episode": episode, "round_number": rnd, "participant_code": i,
         "payoff": pay, "contribution_valid": v}
        for i, (pay, v) in enumerate(zip(payoffs, valid))
    ]


def test_mean_of_valid_players_summed_over_rounds():
    run = "ah exp1 managed by r_s000"
    df = pd.DataFrame(
        _rows(run, 0, 0, [20, 30, 99, 10], [True, True, False, True])
        + _rows(run, 0, 1, [24, 24, 24, 24], [True] * 4)
        + _rows(run, 1, 0, [10, 10, 10, 10], [False] * 4)  # everyone timed out: 0
        + _rows(run, 1, 1, [30, 30, 30, 30], [True] * 4)
    )
    scores = group_payoff_scores(df)
    assert list(scores) == ["r_s000"]
    s = scores["r_s000"]
    assert s["pool"].tolist() == pytest.approx([20 + 24, 0 + 30])
    assert s["members"].tolist() == pytest.approx([3.5, 2.0])
