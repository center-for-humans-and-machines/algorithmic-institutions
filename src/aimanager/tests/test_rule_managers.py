"""The rule family maps contribution to punishment the way its name says.

These rules are the whole experiment on auto/rule-based-manager-sweep -- if
one of them silently punishes a different cell than its one-sentence
definition, the sweep's ranking is meaningless -- so each rule's mapping is
pinned here.
"""

import pytest
import torch as th

from aimanager.manager.api_manager import RuleBasedManager


def make_data(contribution, valid=None, round_number=0):
    c = th.tensor(contribution, dtype=th.int64).reshape(1, -1, 1)
    if valid is None:
        valid = [True] * len(contribution)
    return {
        "contribution": c,
        "contribution_valid": th.tensor(valid, dtype=th.bool).reshape(1, -1, 1),
        "punishment": th.zeros_like(c),
        "round_number": th.full_like(c, round_number),
    }


def punish(manager, contribution, **kw):
    return manager.get_punishments(make_data(contribution, **kw)).reshape(-1).tolist()


def test_never_punishes_nothing():
    m = RuleBasedManager(rule="never")
    assert punish(m, [0, 5, 10, 20]) == [0, 0, 0, 0]


def test_threshold_is_inclusive_and_flat():
    m = RuleBasedManager(rule="threshold", threshold=9, amount=5)
    assert punish(m, [0, 9, 10, 20]) == [5, 5, 0, 0]


def test_proportional_scales_with_the_shortfall():
    m = RuleBasedManager(rule="proportional", rate=0.5)
    assert punish(m, [0, 10, 16, 20]) == [10, 5, 2, 0]


def test_table_is_indexed_by_contribution():
    table = [float(20 - c) for c in range(21)]
    m = RuleBasedManager(rule="table", table=table)
    assert punish(m, [0, 7, 20]) == [20, 13, 0]


def test_severity_table_fires_at_its_own_rate():
    table = [6.0] * 21
    prob = [0.0] * 21
    prob[3] = 1.0
    m = RuleBasedManager(rule="severity_table", table=table, prob_table=prob)
    assert punish(m, [3, 4, 3]) == [6, 0, 6]


def test_output_is_clamped_to_the_action_space():
    m = RuleBasedManager(rule="threshold", threshold=20, amount=99, n_punishments=31)
    assert punish(m, [0, 20]) == [30, 30]


@pytest.mark.parametrize("skip", [False, True])
def test_skip_invalid_controls_whether_timeouts_are_hit(skip):
    # a timed-out player is served contribution 0, so a contribution-keyed
    # rule hits that cell unless skip_invalid is on (review finding D1)
    m = RuleBasedManager(rule="threshold", threshold=4, amount=7, skip_invalid=skip)
    got = punish(m, [0, 12], valid=[False, True])
    assert got == ([0, 0] if skip else [7, 0])


def test_decay_keeps_its_original_meaning():
    m = RuleBasedManager(rule="decay", k=2)
    assert punish(m, [10, 20], round_number=2) == [4, 0]


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        RuleBasedManager(rule="punish_everyone_always")


def test_inv_threshold_is_inclusive_and_flat():
    m = RuleBasedManager(rule="inv_threshold", threshold=11, amount=10)
    assert punish(m, [0, 9, 10, 11, 20]) == [0, 0, 0, 10, 10]


def test_inv_thr11_is_the_exact_reflection_of_thr9():
    # the inverted arm's level-matched mirror: inv_thr11_p10 fires on c
    # exactly where thr9_p10 fires on 20 - c, so the two punish the same
    # NUMBER of contribution levels (10 of 21) at the same amount, and both
    # spare the midpoint c = 10.
    fwd = RuleBasedManager(rule="threshold", threshold=9, amount=10)
    inv = RuleBasedManager(rule="inv_threshold", threshold=11, amount=10)
    grid = list(range(21))
    assert punish(inv, grid) == punish(fwd, [20 - c for c in grid])
    assert sum(p > 0 for p in punish(inv, grid)) == 10
    assert punish(inv, [10]) == punish(fwd, [10]) == [0]


def test_inv_threshold_never_hits_a_timed_out_cell():
    # The env SERVES a timed-out player contribution 0 (sweep finding D1), so
    # at the rule's input a timeout is indistinguishable from a total
    # free-rider: a low-threshold rule punishes every one of them and an
    # inverted rule can never reach one. This is the single asymmetry
    # between the two directions that is not a design choice, and the
    # inverted arm measures it rather than correcting it.
    inv = RuleBasedManager(rule="inv_threshold", threshold=7, amount=10)
    fwd = RuleBasedManager(rule="threshold", threshold=9, amount=10)
    served = [0, 0]  # two timed-out players, as the env presents them
    assert punish(inv, served, valid=[False, False]) == [0, 0]
    assert punish(fwd, served, valid=[False, False]) == [10, 10]


def test_band16_fires_only_on_the_near_ceiling_band():
    # The near-ceiling rules test a different hypothesis from the broad
    # mirrors: that punishing nearly-full contributors is worthwhile because
    # they are cheap to push to the ceiling. That only holds if the rule
    # leaves the whole withdrawal zone below it untouched, which is what is
    # pinned here -- c = 15 is spared and c = 16 is not.
    mild = RuleBasedManager(rule="inv_threshold", threshold=16, amount=10)
    matched = RuleBasedManager(rule="inv_threshold", threshold=16, amount=20)
    assert punish(mild, [0, 9, 12, 15, 16, 20]) == [0, 0, 0, 0, 10, 10]
    assert punish(matched, [15, 16, 20]) == [0, 20, 20]
    assert sum(p > 0 for p in punish(mild, list(range(21)))) == 5
