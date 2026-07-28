"""Extraction tests for the evaluation-suite metrics (#128).

Each metric is pinned twice: exact values on a small hand-computed frame
(covering NaN masking and the empty-group cases), and regression pins on
the converted human reference data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aimanager.evaluation_suite.convert import (
    HUMAN_DATA_FILE,
    SIM_EXAMPLE_FILE,
    load_human,
    load_sim,
)
from aimanager.evaluation_suite.metrics import ContributionMetrics

REPO = Path(__file__).resolve().parents[3]

C = ContributionMetrics()


@pytest.fixture(scope="module")
def human():
    return load_human(REPO / HUMAN_DATA_FILE)


@pytest.fixture()
def frame():
    """One episode, four participants. Round 1: participant a gave no
    input. Round 2: everyone congregates in group 0 (group 1 empty)."""
    rows = [
        (0, "a", 0, 0, 0.0),
        (0, "b", 0, 0, 10.0),
        (0, "c", 0, 1, 20.0),
        (0, "d", 0, 1, 20.0),
        (0, "a", 1, 0, np.nan),
        (0, "b", 1, 0, 10.0),
        (0, "c", 1, 1, 0.0),
        (0, "d", 1, 1, 20.0),
        (0, "a", 2, 0, 5.0),
        (0, "b", 2, 0, 10.0),
        (0, "c", 2, 0, 15.0),
        (0, "d", 2, 0, 20.0),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "episode_id",
            "participant_code",
            "round_number",
            "group_id",
            "contribution",
        ],
    )


def test_ca_participant_means(frame):
    obs = C.ca(frame)
    assert obs.tolist() == pytest.approx([2.5, 10.0, 35 / 3, 20.0])


def test_cb_round_means(frame):
    stat = C.cb(frame)
    assert stat.tolist() == pytest.approx([12.5, 10.0, 12.5])


def test_cc_group_means_keep_surviving_group(frame):
    obs = C.cc(frame)
    assert obs.tolist() == pytest.approx([5.0, 20.0, 10.0, 10.0, 12.5])
    assert (0, 2, 1) not in obs.index  # empty group: no observation


def test_cd_raw_contributions(frame):
    obs = C.cd(frame)
    assert len(obs) == 11  # the NaN row drops


def test_ce_drops_empty_group_rounds(frame):
    obs = C.ce(frame)
    assert obs.tolist() == pytest.approx([-15.0, 0.0])  # rounds 0 and 1
    assert list(obs.index.get_level_values("round_number")) == [0, 1]


def test_cf_boundary_shares(frame):
    stat = C.cf(frame)
    assert stat.loc[(0, "share_at_0")] == pytest.approx(0.25)
    assert stat.loc[(0, "share_at_20")] == pytest.approx(0.5)
    assert stat.loc[(1, "share_at_0")] == pytest.approx(1 / 3)  # of 3 valid
    assert stat.loc[(2, "share_at_20")] == pytest.approx(0.25)


def test_human_reference_pins(human):
    extractions = C.extract_all(human)
    assert {k: len(v) for k, v in extractions.items()} == {
        "CA": 400,  # 50 episodes x 8 participants
        "CB": 24,
        "CC": 2239,  # 2256 cells - 144 empty-group - 17 all-no-input
        "CD": 9320,  # 9600 rows - 280 no-input
        "CE": 1039,  # 1200 game-rounds - 144 empty-group - 17 all-no-input
        "CF": 48,
    }
    assert extractions["CB"].loc[0] == pytest.approx(9.067358, abs=1e-5)
    assert extractions["CE"].mean() == pytest.approx(-0.383599, abs=1e-5)
    assert extractions["CF"].loc[(0, "share_at_0")] == pytest.approx(0.051813, abs=1e-5)


def test_all_metrics_extract_from_sim():
    sims = load_sim(REPO / SIM_EXAMPLE_FILE)
    sim = sims[sorted(sims)[0]]
    for name, e in C.extract_all(sim).items():
        assert len(e) > 0, name
        assert not e.isna().any(), name
