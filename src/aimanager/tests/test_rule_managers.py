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
