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
    ResponseMetrics,
    SwitchingMetrics,
)

REPO = Path(__file__).resolve().parents[3]

C = ContributionMetrics()
S = SwitchingMetrics()
P = PunishmentMetrics()
R = ResponseMetrics()


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


def test_human_response_pins(human):
    rcb = R.rcb(human)
    assert rcb.loc["(0,0.25]"] == pytest.approx(0.892, abs=1e-3)
    assert rcb.loc[">1"] == pytest.approx(2.014, abs=1e-3)
    assert R.weights("RCB", human).to_dict() == {
        "(0,0.25]": 1238,
        "(0.25,0.5]": 672,
        "(0.5,1]": 473,
        ">1": 277,
    }
    assert R.rcc(human).loc["contrast"] == pytest.approx(-7.035, abs=1e-3)
    assert R.weights("RCA", human).to_dict() == {
        "no_switch_allowed": 6902,
        "stayed_comp_changed": 1102,
        "switched": 538,
        "chose_to_stay": 182,
    }


def test_all_metrics_extract_from_sim():
    sims = load_sim(REPO / SIM_EXAMPLE_FILE)
    sim = sims[sorted(sims)[0]]
    for group in [C, S, P]:
        for name, e in group.extract_all(sim).items():
            assert len(e) > 0, name
            assert not e.isna().any(), name


@pytest.fixture()
def response_frame():
    """Two episodes, decision round 3, arrival round 4. Episode 0: a
    switches g0 -> g1 (b's group shrinks, c/d's group grows); a, c, b, d
    are punished with shortfall rates 0.5, 0.125, 13/12, 9/14. Episode 1:
    nobody switches (pure chose-to-stay), h's choice timed out, e gave
    no input at round 4 (their round-3 punishment drops from RCB), i is
    a punished full contributor and j an unpunished one (RCC)."""
    rows = [
        # episode 0, rounds 2-5
        (0, "a", 2, 0, 10.0, 5.0, False, False, False),
        (0, "b", 2, 0, 8.0, 0.0, False, False, False),
        (0, "c", 2, 1, 4.0, 2.0, False, False, False),
        (0, "d", 2, 1, 6.0, 0.0, False, False, False),
        (0, "a", 3, 0, 12.0, 0.0, True, True, True),
        (0, "b", 3, 0, 8.0, 13.0, False, True, True),
        (0, "c", 3, 1, 4.0, 0.0, False, True, True),
        (0, "d", 3, 1, 6.0, 9.0, False, True, True),
        (0, "a", 4, 1, 5.0, 0.0, False, False, False),
        (0, "b", 4, 0, 9.0, 0.0, False, False, False),
        (0, "c", 4, 1, 3.0, 0.0, False, False, False),
        (0, "d", 4, 1, 7.0, 0.0, False, False, False),
        (0, "a", 5, 1, 5.0, 0.0, False, False, False),
        (0, "b", 5, 0, 9.0, 0.0, False, False, False),
        (0, "c", 5, 1, 3.0, 0.0, False, False, False),
        (0, "d", 5, 1, 7.0, 0.0, False, False, False),
        # episode 1, rounds 3-4
        (1, "e", 3, 0, 11.0, 6.0, False, True, True),
        (1, "f", 3, 0, 12.0, 0.0, False, True, True),
        (1, "g", 3, 1, 13.0, 0.0, False, True, True),
        (1, "h", 3, 1, 14.0, 0.0, False, True, False),
        (1, "i", 3, 0, 20.0, 4.0, False, True, True),
        (1, "j", 3, 1, 20.0, 0.0, False, True, True),
        (1, "e", 4, 0, np.nan, 0.0, False, False, False),
        (1, "f", 4, 0, 12.0, 0.0, False, False, False),
        (1, "g", 4, 1, 13.0, 0.0, False, False, False),
        (1, "h", 4, 1, 14.0, 0.0, False, False, False),
        (1, "i", 4, 0, 14.0, 0.0, False, False, False),
        (1, "j", 4, 1, 18.0, 0.0, False, False, False),
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
            "does_switch",
            "switch_mask",
            "switch_valid",
        ],
    )


def test_dc_values(response_frame):
    df = R._with_dc(response_frame).set_index(
        ["episode_id", "participant_code", "round_number"]
    )
    assert df.loc[(0, "a", 2), "dc"] == 2.0  # 12 - 10
    assert df.loc[(0, "a", 3), "dc"] == -7.0  # 5 - 12, across the switch
    assert pd.isna(df.loc[(1, "e", 3), "dc"])  # next contribution invalid
    assert pd.isna(df.loc[(0, "a", 5), "dc"])  # last round: no next


def test_round_types(response_frame):
    df = R._round_types(response_frame).set_index(
        ["episode_id", "participant_code", "round_number"]
    )
    assert df.loc[(0, "a", 2), "round_type"] == "no_switch_allowed"
    assert df.loc[(0, "a", 3), "round_type"] == "switched"
    assert df.loc[(0, "b", 3), "round_type"] == "stayed_comp_changed"  # a left
    assert df.loc[(0, "c", 3), "round_type"] == "stayed_comp_changed"  # a joined
    assert df.loc[(1, "f", 3), "round_type"] == "chose_to_stay"
    assert pd.isna(df.loc[(1, "h", 3), "round_type"])  # timed out: no choice


def test_rca_observations(response_frame):
    obs = R.rca(response_frame)
    by_type = {k: sorted(v.tolist()) for k, v in obs.groupby(level=0)}
    assert by_type["switched"] == [-7.0]
    assert by_type["stayed_comp_changed"] == [-1.0, 1.0, 1.0]
    # e's NaN dc drops; i and j (full contributors) fall by -6 and -2
    assert by_type["chose_to_stay"] == [-6.0, -2.0, 0.0, 0.0]
    # ep0 rounds 2 and 4 only: ep0 r5 and all ep1 r4 rows have no next round
    assert len(by_type["no_switch_allowed"]) == 8


def test_rca_weights_are_human_frequencies(response_frame):
    w = R.weights("RCA", response_frame)
    assert w.to_dict() == {
        "no_switch_allowed": 8,
        "stayed_comp_changed": 3,
        "chose_to_stay": 4,
        "switched": 1,
    }


def test_rca_d_weighted_stratum_emd(response_frame):
    # bump a's round-5 contribution: only one no_switch_allowed dc moves
    # 0 -> 4, so EMD(nsa) = 4/8 and d = (4/8) * 8/16 = 1/4
    bumped = response_frame.copy()
    bumped.loc[
        (bumped["participant_code"] == "a") & (bumped["round_number"] == 5),
        "contribution",
    ] = 9.0
    assert R.d("RCA", response_frame, bumped) == pytest.approx(1 / 4)
    assert R.d("RCA", response_frame, response_frame) == pytest.approx(0)


def test_rca_d_raises_on_empty_stratum(response_frame):
    # a comparison side that never switches has no "switched" stratum
    no_switch = response_frame.assign(does_switch=False)
    with pytest.raises(ValueError, match="RCA: empty strata \\['switched'\\]"):
        R.d("RCA", response_frame, no_switch)


def test_rcb_stat_and_weights(response_frame):
    # a: rate 5/10=0.5 dc +2 | c: 2/16=0.125 dc 0 | b: 13/12>1 dc +1 |
    # d: 9/14 dc +1; e is punished but dc-invalid, i is full -> both out
    stat = R.rcb(response_frame)
    assert stat.to_dict() == pytest.approx(
        {"(0,0.25]": 0.0, "(0.25,0.5]": 2.0, "(0.5,1]": 1.0, ">1": 1.0}
    )
    assert R.weights("RCB", response_frame).to_dict() == {
        "(0,0.25]": 1,
        "(0.25,0.5]": 1,
        "(0.5,1]": 1,
        ">1": 1,
    }


def test_rcb_d(response_frame):
    # bump b's round-4 contribution 9 -> 11: only the ">1" bin's mean
    # moves (+1 -> +3), so d = 2/4 with equal bin counts
    bumped = response_frame.copy()
    bumped.loc[
        (bumped["participant_code"] == "b") & (bumped["round_number"] == 4),
        "contribution",
    ] = 11.0
    assert R.d("RCB", response_frame, bumped) == pytest.approx(0.5)


def test_rcc_contrast(response_frame):
    # punished full contributor i falls -6, unpunished j falls -2
    assert R.rcc(response_frame).loc["contrast"] == pytest.approx(-4.0)
    bumped = response_frame.copy()
    bumped.loc[
        (bumped["participant_code"] == "j") & (bumped["round_number"] == 4),
        "contribution",
    ] = 20.0
    assert R.d("RCC", response_frame, bumped) == pytest.approx(2.0)


def test_switch_events(response_frame):
    events = R._switch_events(response_frame)
    assert len(events) == 1
    e = events.iloc[0]
    assert e["participant_code"] == "a"
    assert e["contribution"] == 12.0
    assert e["dc"] == -7.0
    assert e["receiving_mean"] == 5.0  # (4 + 6) / 2, roster a saw at round 3


def test_d_self_comparison_is_zero(human):
    half = human[human["episode_id"] < human["episode_id"].median()]
    for group in [C, S, P]:
        for name in group.KINDS:
            assert group.d(name, human, human) == pytest.approx(0), name
            # callable on episode subsets, still zero against itself
            assert group.d(name, half, half) == pytest.approx(0), name


def test_d_distribution_is_emd(frame):
    shifted = frame.assign(contribution=frame["contribution"] + 2)
    assert C.d("CD", frame, shifted) == pytest.approx(2.0)


def test_d_statistic_weighted_mean(frame):
    # +2 only in round 0: per-round |diff| = [2, 0, 0]; explicit uniform
    # weights over the synthetic frame's 3 rounds
    bumped = frame.copy()
    bumped.loc[bumped["round_number"] == 0, "contribution"] += 2
    w = pd.Series({0: 1.0, 1: 1.0, 2: 1.0})
    assert C.d("CB", frame, bumped, weights=w) == pytest.approx(2 / 3)


def test_d_accepts_precomputed_weights(frame):
    # a custom weight vector overrides the row's default scheme
    bumped = frame.copy()
    bumped.loc[bumped["round_number"] == 0, "contribution"] += 2
    w = pd.Series({0: 1.0, 1: 3.0, 2: 4.0})
    assert C.d("CB", frame, bumped, weights=w) == pytest.approx(2 / 8)


def test_d_raises_on_empty_stratum(frame):
    # comparison side lacks round 2 entirely
    other = frame[frame["round_number"] < 2]
    w = pd.Series({0: 1.0, 1: 1.0, 2: 1.0})
    with pytest.raises(ValueError, match="CB: empty strata \\[2\\]"):
        C.d("CB", frame, other, weights=w)
    # default weights span all 24 rounds, so a 3-round frame raises too
    with pytest.raises(ValueError, match="empty strata"):
        C.d("CB", frame, frame)


def test_uniform_precomputed_weights():
    # design-fixed strata: full index regardless of the data passed
    assert C.weights("CB", None).tolist() == [1.0] * 24
    cf = C.weights("CF", None)
    assert len(cf) == 48
    assert cf.loc[(23, "share_at_20")] == 1.0
    assert S.weights("SA", None).loc["switch_rate"] == 1.0
    assert list(S.weights("SB", None).index) == [3, 7, 11, 15, 19]
    assert P.weights("PB", None).equals(P.weights("PC", None))


def test_std_diff_is_signed(frame):
    shifted = frame.assign(contribution=frame["contribution"] + 2)
    doubled = frame.assign(contribution=frame["contribution"] * 2)
    assert C.std_diff("CD", shifted, frame) == pytest.approx(0)  # shift: same std
    raw_std = pd.Series([0, 10, 20, 20, 10, 0, 20, 5, 10, 15, 20], dtype=float).std()
    assert C.std_diff("CD", doubled, frame) == pytest.approx(raw_std)
