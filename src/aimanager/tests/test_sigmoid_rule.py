"""The parametrised sigmoid rule.

The load-bearing test is `test_reproduces_thr9_p10`: the family was chosen
because the logistic nests the incumbent continuously, and a family that
cannot reproduce the best manager measured in this project is not a
generalisation of it. Everything else pins the two horizon multipliers and
the clipping.

No torch_geometric here, so these run locally; the one test that compares
against `RuleBasedManager` imports it inside the test and skips without PyG.
"""

import pytest
import torch as th

from aimanager.manager.sigmoid_rule import (
    PARAM_NAMES,
    ConstantManager,
    SigmoidRuleBatch,
    sigmoid_punishment,
)

TAU0 = 1e-6
C = th.arange(21).view(-1, 1)  # every contribution level
T = th.arange(24).view(1, -1)  # every round of an episode
CC, TT = C.expand(21, 24), T.expand(21, 24)


def _p(**kw):
    kw = {"p_max": 10.0, "c0": 9.5, "tau": TAU0, "gamma_ep": 0.0, "gamma_sw": 0.0, **kw}
    return sigmoid_punishment(CC, TT, **kw)


def test_reproduces_thr9_p10_as_a_step():
    """P_max=10, c0=9.5, tau->0, gamma=0 IS `thr9_p10`: punish 10 at c <= 9."""
    want = th.where(CC <= 9, 10.0, 0.0)
    assert th.equal(_p(), want)


def test_reproduces_thr9_p10_against_the_rule_family():
    """The same claim, against the incumbent's own implementation."""
    pytest.importorskip("torch_geometric")
    from aimanager.manager.api_manager import RuleBasedManager

    data = {"contribution": CC, "round_number": TT, "punishment": th.zeros_like(CC)}
    step = RuleBasedManager(rule="threshold", threshold=9, amount=10)
    sig = RuleBasedManager(
        rule="sigmoid", p_max=10, c0=9.5, tau=TAU0, gamma_ep=0, gamma_sw=0
    )
    assert th.equal(sig.get_punishments(dict(data)), step.get_punishments(dict(data)))


@pytest.mark.parametrize("threshold,amount", [(4, 10), (9, 5), (14, 5), (19, 30)])
def test_tau_zero_is_a_hard_threshold_everywhere(threshold, amount):
    want = th.where(CC <= threshold, float(amount), 0.0)
    assert th.equal(_p(p_max=amount, c0=threshold + 0.5), want)


def test_never_is_p_max_zero():
    assert th.equal(_p(p_max=0.0), th.zeros_like(CC, dtype=th.float))


def test_gamma_zero_disables_both_multipliers():
    """With gamma = 0 the punishment cannot depend on the round at all."""
    p = _p(p_max=17.0, c0=12.0, tau=2.0)
    assert th.equal(p, p[:, :1].expand_as(p))


def test_episode_multiplier_is_the_remaining_horizon():
    """gamma_ep = 1 scales by (T - t) / T, which is 1 at t = 0 and 1/24 last."""
    p = _p(p_max=24.0, c0=25.0, tau=TAU0, gamma_ep=1.0)  # f == 1 for every c
    want = (24.0 * (24 - TT.to(th.float)) / 24.0).round()
    assert th.equal(p, want.expand_as(p))
    assert p[0, 0] == 24 and p[0, 23] == 1


def test_switch_multiplier_keys_on_rounds_until_reshuffle():
    """s = (S - 1) - t mod S: 1 right after a reshuffle, smallest before one."""
    p = _p(p_max=20.0, c0=25.0, tau=TAU0, gamma_sw=1.0)
    want = th.tensor([20.0, 15.0, 10.0, 5.0]).repeat(6)
    assert th.equal(p[0], want)


def test_multipliers_compose():
    p = _p(p_max=30.0, c0=25.0, tau=TAU0, gamma_ep=0.5, gamma_sw=2.0)
    t = TT.to(th.float)
    s = 3 - th.remainder(t, 4)
    want = (30.0 * ((24 - t) / 24) ** 0.5 * ((s + 1) / 4) ** 2.0).round()
    assert th.equal(p, want)


def test_clipped_to_the_action_space():
    p = _p(p_max=500.0, c0=25.0, tau=TAU0)
    assert p.max() == 30.0 and p.min() == 30.0


def test_large_tau_is_monotone_and_soft():
    p = _p(p_max=30.0, c0=10.0, tau=5.0)[:, 0]
    assert (p[1:] <= p[:-1]).all(), "punishment must never rise with contribution"
    assert 0 < p[-1] < p[0] < 30, "a soft rule fires everywhere, hardest at c = 0"


def test_batch_carries_one_parameter_vector_per_episode():
    theta = th.tensor(
        [
            [10.0, 9.5, TAU0, 0.0, 0.0],  # thr9_p10
            [0.0, 10.0, 1.0, 0.0, 0.0],  # never
            [30.0, 25.0, TAU0, 1.0, 0.0],  # flat, episode-decayed
        ]
    )
    assert theta.shape[1] == len(PARAM_NAMES)
    mgr = SigmoidRuleBatch(theta)
    state = {
        "contribution": th.tensor([[3, 15], [3, 15], [3, 15]]).unsqueeze(-1),
        "round_number": th.full((3, 2, 1), 12),
    }
    out, _ = mgr.predict(state)
    assert out.dtype == th.int64
    assert out.squeeze(-1).tolist() == [[10, 0], [0, 0], [15, 15]]


def test_batch_rejects_non_positive_tau():
    with pytest.raises(AssertionError):
        SigmoidRuleBatch(th.tensor([[10.0, 9.5, 0.0, 0.0, 0.0]]))


def test_constant_manager_is_the_never_control():
    state = {"contribution": th.zeros(2, 4, 1, dtype=th.int64)}
    out, _ = ConstantManager(0).predict(state)
    assert out.sum() == 0 and out.dtype == th.int64
