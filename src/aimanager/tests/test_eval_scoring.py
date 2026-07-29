"""Tests for the evaluation-suite scoring schema (#132)."""

import numpy as np
import pytest

from aimanager.evaluation_suite.scoring import make_repeats, subset

HUMAN_IDS = list(range(0, 100, 2))  # 50 episodes, even ids
SIM_IDS = list(range(100))


def test_repeats_are_deterministic():
    a = make_repeats(HUMAN_IDS, SIM_IDS, n_repeats=5, seed=42)
    b = make_repeats(HUMAN_IDS, SIM_IDS, n_repeats=5, seed=42)
    for (a1, a2, a3), (b1, b2, b3) in zip(a, b):
        assert (a1 == b1).all() and (a2 == b2).all() and (a3 == b3).all()
    # input order must not matter
    c = make_repeats(list(reversed(HUMAN_IDS)), SIM_IDS, n_repeats=5, seed=42)
    assert (a[0][0] == c[0][0]).all()
    # a different seed gives a different plan
    d = make_repeats(HUMAN_IDS, SIM_IDS, n_repeats=5, seed=43)
    assert not (a[0][0] == d[0][0]).all()


def test_repeats_split_properties():
    repeats = make_repeats(HUMAN_IDS, SIM_IDS, n_repeats=20, seed=0)
    assert len(repeats) == 20
    for h_a, h_b, s in repeats:
        assert len(h_a) == len(h_b) == len(s) == 25  # m = 50 // 2
        assert set(h_a).isdisjoint(h_b)
        assert set(h_a) | set(h_b) <= set(HUMAN_IDS)
        assert set(s) <= set(SIM_IDS)
        assert len(set(s)) == 25  # draw without replacement


def test_repeats_vary_across_repeats():
    repeats = make_repeats(HUMAN_IDS, SIM_IDS, n_repeats=2, seed=0)
    assert set(repeats[0][0]) != set(repeats[1][0])
    assert set(repeats[0][2]) != set(repeats[1][2])


def test_sim_pool_too_small_raises():
    with pytest.raises(ValueError, match="need at least m=25"):
        make_repeats(HUMAN_IDS, SIM_IDS[:10], n_repeats=1, seed=0)


def test_subset_filters_episodes():
    import pandas as pd

    df = pd.DataFrame({"episode_id": [0, 0, 2, 4], "x": [1, 2, 3, 4]})
    out = subset(df, np.array([0, 4]))
    assert out["x"].tolist() == [1, 2, 4]
