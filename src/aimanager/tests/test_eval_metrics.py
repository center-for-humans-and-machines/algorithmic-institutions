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
from aimanager.evaluation_suite.metrics import (
    ContributionMetrics,
    PunishmentMetrics,
    SwitchingMetrics,
)

REPO = Path(__file__).resolve().parents[3]

C = ContributionMetrics()
S = SwitchingMetrics()
P = PunishmentMetrics()


@pytest.fixture(scope="module")
def human():
    return load_human(REPO / HUMAN_DATA_FILE)


@pytest.fixture()
def frame():
    """One episode, four participants. Round 1: participant a gave no
    input (and their manager no punishment input). Round 2: everyone
    congregates in group 0 (group 1 empty)."""
    rows = [
        (0, "a", 0, 0, 0.0, 0.0),
        (0, "b", 0, 0, 10.0, 5.0),
        (0, "c", 0, 1, 20.0, 0.0),
        (0, "d", 0, 1, 20.0, 10.0),
        (0, "a", 1, 0, np.nan, np.nan),
        (0, "b", 1, 0, 10.0, 0.0),
        (0, "c", 1, 1, 0.0, 30.0),
        (0, "d", 1, 1, 20.0, 0.0),
        (0, "a", 2, 0, 5.0, 0.0),
        (0, "b", 2, 0, 10.0, 0.0),
        (0, "c", 2, 0, 15.0, 0.0),
        (0, "d", 2, 0, 20.0, 0.0),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "episode_id",
            "participant_code",
            "round_number",
            "group_id",
            "contribution",
            "punishment",
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


@pytest.fixture()
def switch_frame():
    """One episode, four participants. Decision rounds 3 and 7: a switches
    at 3 (c's choice timed out), b switches at 7. Round 5: everyone in
    group 1 (group 0 empty)."""
    rows = [
        (0, "a", 3, 0, True, True, True),
        (0, "b", 3, 0, False, True, True),
        (0, "c", 3, 1, False, True, False),
        (0, "d", 3, 1, False, True, True),
        (0, "a", 4, 1, False, False, False),
        (0, "b", 4, 0, False, False, False),
        (0, "c", 4, 1, False, False, False),
        (0, "d", 4, 1, False, False, False),
        (0, "a", 5, 1, False, False, False),
        (0, "b", 5, 1, False, False, False),
        (0, "c", 5, 1, False, False, False),
        (0, "d", 5, 1, False, False, False),
        (0, "a", 7, 1, False, True, True),
        (0, "b", 7, 1, True, True, True),
        (0, "c", 7, 1, False, True, True),
        (0, "d", 7, 1, False, True, True),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "episode_id",
            "participant_code",
            "round_number",
            "group_id",
            "does_switch",
            "switch_mask",
            "switch_valid",
        ],
    )


def test_sa_overall_switch_rate(switch_frame):
    # 7 valid opportunities (c's round-3 timeout drops), 2 switches
    assert S.sa(switch_frame).loc["switch_rate"] == pytest.approx(2 / 7)


def test_sb_rate_per_opportunity(switch_frame):
    stat = S.sb(switch_frame)
    assert stat.loc[3] == pytest.approx(1 / 3)  # a of {a, b, d}
    assert stat.loc[7] == pytest.approx(1 / 4)
    assert list(stat.index) == [3, 7]


def test_sc_larger_group_keeps_empty_rounds(switch_frame):
    obs = S.sc(switch_frame)
    assert obs.loc[(0, 4)] == 3  # groups split 1-3
    assert obs.loc[(0, 5)] == 4  # group 0 empty: kept as max segregation
    assert (0, 3) not in obs.index  # rounds < 4 excluded


def test_pa_raw_punishments(frame):
    obs = P.pa(frame)
    assert len(obs) == 11  # the manager-no-input NaN drops
    assert obs.sum() == pytest.approx(45.0)


def test_pb_round_means(frame):
    stat = P.pb(frame)
    assert stat.tolist() == pytest.approx([3.75, 10.0, 0.0])  # r1: of 3 valid


def test_pc_zero_shares(frame):
    stat = P.pc(frame)
    assert stat.tolist() == pytest.approx([0.5, 2 / 3, 1.0])


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


def test_human_switching_pins(human):
    extractions = S.extract_all(human)
    assert extractions["SA"].loc["switch_rate"] == pytest.approx(0.296668, abs=1e-5)
    assert list(extractions["SB"].index) == [3, 7, 11, 15, 19]
    assert extractions["SB"].loc[3] == pytest.approx(0.44186, abs=1e-5)
    sc = extractions["SC"]
    assert len(sc) == 1000  # 50 episodes x rounds 4..23
    # the 144 empty-group rounds appear as larger-group size 8
    assert sc.value_counts().loc[8] == 144
    assert sc.mean() == pytest.approx(6.088, abs=1e-5)


def test_human_punishment_pins(human):
    extractions = P.extract_all(human)
    pa = extractions["PA"]
    assert len(pa) == 9193  # 9600 rows - 407 manager-no-input
    assert pa.mean() == pytest.approx(1.791254, abs=1e-5)
    assert pa.eq(0).mean() == pytest.approx(0.694224, abs=1e-5)
    assert extractions["PB"].loc[0] == pytest.approx(4.090659, abs=1e-5)
    assert extractions["PC"].loc[0] == pytest.approx(0.445055, abs=1e-5)


def test_all_metrics_extract_from_sim():
    sims = load_sim(REPO / SIM_EXAMPLE_FILE)
    sim = sims[sorted(sims)[0]]
    for group in [C, S, P]:
        for name, e in group.extract_all(sim).items():
            assert len(e) > 0, name
            assert not e.isna().any(), name
