"""Tests for the evaluation-suite scoring schema (#132)."""

from pathlib import Path

import numpy as np
import pytest

from aimanager.evaluation_suite.convert import HUMAN_DATA_FILE, load_human
from aimanager.evaluation_suite.metrics import (
    ContributionMetrics,
    ResponseMetrics,
    SwitchingMetrics,
)
from aimanager.evaluation_suite.scoring import (
    denominators,
    make_repeats,
    score_row,
    subset,
)

REPO = Path(__file__).resolve().parents[3]
HUMAN_IDS = list(range(0, 100, 2))  # 50 episodes, even ids
SIM_IDS = list(range(100))

C = ContributionMetrics()
S = SwitchingMetrics()
R = ResponseMetrics()


@pytest.fixture(scope="module")
def human():
    return load_human(REPO / HUMAN_DATA_FILE)


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


@pytest.fixture(scope="module")
def human_repeats(human):
    ids = human["episode_id"].unique()
    return make_repeats(ids, ids, n_repeats=40, seed=7)


def test_human_as_sim_scores_near_one(human, human_repeats):
    # drawing the "sim" from the human pool itself must sit at the
    # noise ceiling for every row kind
    for group, name in [(C, "CD"), (C, "CB"), (S, "SA")]:
        r = score_row(group, name, human, human, human_repeats)
        assert 0.7 < r["score"] < 1.3, (name, r)
        assert r["repeats_used"] == r["n_repeats"] == 40


def test_shifted_pool_scores_far_above_one(human, human_repeats):
    shifted = human.assign(contribution=human["contribution"] + 5)
    assert score_row(C, "CD", human, shifted, human_repeats)["score"] > 5


def test_precomputed_denominators_match(human, human_repeats):
    denoms = denominators(C, "CD", human, human_repeats)
    direct = score_row(C, "CD", human, human, human_repeats)
    reused = score_row(C, "CD", human, human, human_repeats, denoms=denoms)
    assert direct["score"] == pytest.approx(reused["score"])


def test_unsupported_repeats_are_dropped(human, human_repeats):
    # a "sim" that never punishes >= 16 empties RSA's 16+ bin in every
    # draw: no repeat survives, score NaN, warning emitted
    capped = human.assign(punishment=human["punishment"].clip(upper=15))
    with pytest.warns(UserWarning, match="RSA: no repeat supports"):
        r = score_row(R, "RSA", human, capped, human_repeats)
    assert np.isnan(r["score"])
    assert r["repeats_used"] == 0


def test_lin_ridge_rsa_partial_support(human):
    # the one real #134 case: 6 of 100 lin_ridge episodes carry a 16+
    # punishment event, so ~17% of draws cannot support the bin -- the
    # score averages over the surviving repeats and reports the count
    from aimanager.evaluation_suite.convert import SIM_EXAMPLE_FILE, load_sim

    sim = load_sim(REPO / SIM_EXAMPLE_FILE)[
        "ah group_switching managed by lin_ridge_self"
    ]
    repeats = make_repeats(
        human["episode_id"].unique(),
        sim["episode_id"].unique(),
        n_repeats=100,
        seed=42,
    )
    r = score_row(R, "RSA", human, sim, repeats)
    assert r["repeats_used"] == 83
    assert not np.isnan(r["score"])
