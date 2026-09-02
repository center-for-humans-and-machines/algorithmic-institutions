"""Tests for the conditional-Bernoulli WHO-leaves sampler (plan step 3).

Plain pytest, no PyG imports -- the module under test imports nothing from
`graph.py`, so this file needs none of the `torch_scatter` /
`torch_geometric.nn` stand-ins the sibling `test_joint_exodus_*.py` files
install. Runs locally on macOS:

    PYTHONPATH=$PWD/src uv run pytest tests/switch/test_conditional_bernoulli.py -q

Context: notes/autoresearch_log/switch-joint-exodus.md, plan step 3.
"""

import math

import pytest
import torch as th

from aimanager.generic.conditional_bernoulli import (
    MAX_GROUP_SIZE,
    conditional_bernoulli_log_prob,
    sample_conditional_bernoulli,
)

SEED = 20260902
GRID = 1 << MAX_GROUP_SIZE


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def poisson_binomial_pmf(p):
    """pmf of the sum of independent Bernoulli(p_i), by the standard O(k^2)
    DP recursion. Used only to build the test's ground truth for `m`; not
    part of the module under test."""
    p = p.reshape(-1).to(th.float64)
    k = p.shape[0]
    pmf = th.zeros(k + 1, dtype=th.float64)
    pmf[0] = 1.0
    for i in range(k):
        pi = p[i]
        new = pmf * (1.0 - pi)
        new[1:] += pmf[:-1] * pi
        pmf = new
    return pmf


def all_k_m_pairs(max_k=MAX_GROUP_SIZE):
    return [(k, m) for k in range(max_k + 1) for m in range(k + 1)]


def widen(p_row, width=MAX_GROUP_SIZE):
    """Pad a 1-D list of real member probabilities out to `width`, with an
    arbitrary (never-read) fill value on the padded tail."""
    pad = [0.5] * (width - len(p_row))
    return p_row + pad


# --------------------------------------------------------------------------- #
# 1. marginal recovery -- the key correctness property
# --------------------------------------------------------------------------- #
def test_marginal_recovery_from_the_poisson_binomial_of_m():
    """If `m` is drawn from the TRUE sum distribution implied by the p_i
    themselves (the Poisson-binomial), the conditional-Bernoulli draws,
    averaged over many samples, must recover the input p_i: this is exactly
    what "conditioning on the sum" means.

    N_SAMPLES = 20000, tolerance = 5 binomial standard errors per agent
    (matching the convention `tests/switch/test_switch_copula.py`'s own
    marginal test uses), fixed seed 20260902.
    """
    N_SAMPLES = 20000
    p = th.tensor([0.05, 0.12, 0.23, 0.35, 0.5, 0.63, 0.78, 0.92], dtype=th.float64)
    pmf = poisson_binomial_pmf(p)

    th.manual_seed(SEED)
    m = th.multinomial(pmf.expand(N_SAMPLES, -1), 1).reshape(-1)
    p_batch = p.expand(N_SAMPLES, -1)
    selected = sample_conditional_bernoulli(p_batch, m)

    freq = selected.to(th.float64).mean(0)
    se = th.sqrt(p * (1.0 - p) / N_SAMPLES)
    err = (freq - p).abs()
    assert bool((err < 5.0 * se).all()), (err / se).tolist()


# --------------------------------------------------------------------------- #
# 2. propensity ordering is preserved
# --------------------------------------------------------------------------- #
def test_propensity_ordering_is_preserved():
    """p_1 < ... < p_k, m fixed: the empirical selection frequency must be
    non-decreasing in p. 20000 draws of the same (p, m) pair, fixed seed."""
    N_SAMPLES = 20000
    p = th.linspace(0.05, 0.9, MAX_GROUP_SIZE, dtype=th.float64)
    m = th.full((N_SAMPLES,), 3, dtype=th.int64)

    th.manual_seed(SEED)
    selected = sample_conditional_bernoulli(p.expand(N_SAMPLES, -1), m)
    freq = selected.to(th.float64).mean(0)

    diffs = freq[1:] - freq[:-1]
    assert bool((diffs >= -0.01).all()), freq.tolist()
    # and the ordering has teeth: the extremes are clearly separated
    assert freq[-1] - freq[0] > 0.3, freq.tolist()


# --------------------------------------------------------------------------- #
# 3 + 4. exactly m selected, for every (k, m), including the boundaries
# --------------------------------------------------------------------------- #
def test_exactly_m_are_selected_for_every_k_m_pair_including_boundaries():
    """Every (k, m) with 0 <= m <= k <= 8 in ONE batched call -- k = 0 (an
    empty group) and m = k (everybody leaves) fall out of the same code path
    as every other cell, not a special case."""
    pairs = all_k_m_pairs()
    th.manual_seed(SEED)
    rows_p, rows_mask, rows_m = [], [], []
    for k, m in pairs:
        p_real = th.rand(k, dtype=th.float64) * 0.8 + 0.1  # in [0.1, 0.9]
        rows_p.append(widen(p_real.tolist()))
        rows_mask.append([True] * k + [False] * (MAX_GROUP_SIZE - k))
        rows_m.append(m)

    p = th.tensor(rows_p, dtype=th.float64)
    mask = th.tensor(rows_mask, dtype=th.bool)
    m = th.tensor(rows_m, dtype=th.int64)
    selected = sample_conditional_bernoulli(p, m, mask=mask)

    assert th.equal(selected.sum(-1), m)
    assert not bool((selected & ~mask).any()), "a padded slot was selected"

    for row, (k, mm) in enumerate(pairs):
        if mm == 0:
            assert not bool(selected[row].any()), (k, mm)
        if mm == k and k > 0:
            assert th.equal(selected[row], mask[row]), (k, mm)


def test_boundary_k_equals_one():
    """k = 1: the only two cells, m = 0 and m = 1, both degenerate to a
    single valid subset."""
    p = th.tensor([[0.37] + [0.5] * (MAX_GROUP_SIZE - 1)], dtype=th.float64)
    mask = th.tensor([[True] + [False] * (MAX_GROUP_SIZE - 1)])

    th.manual_seed(SEED)
    stay = sample_conditional_bernoulli(p, th.tensor([0]), mask=mask)
    leave = sample_conditional_bernoulli(p, th.tensor([1]), mask=mask)
    assert not bool(stay.any())
    assert leave[0, 0].item() is True


# --------------------------------------------------------------------------- #
# 5. equal probabilities give uniform selection -- a closed form, checked
#    exactly (no Monte Carlo needed: with p_i all equal, every subset of a
#    given size carries the same total log-odds by construction)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k,m,p_value", [(6, 3, 0.37), (8, 4, 1e-3), (5, 0, 0.9)])
def test_equal_probabilities_give_uniform_selection(k, m, p_value):
    from math import comb, log

    p = th.tensor([widen([p_value] * k)], dtype=th.float64)
    mask = th.tensor([[True] * k + [False] * (MAX_GROUP_SIZE - k)])
    log_prob, _, valid = conditional_bernoulli_log_prob(p, th.tensor([m]), mask=mask)

    n_valid = int(valid[0].sum().item())
    assert n_valid == comb(k, m)
    values = log_prob[0][valid[0]]
    expected = -log(comb(k, m))
    assert th.allclose(
        values, th.full_like(values, expected), atol=1e-9
    ), values.tolist()
    assert math.isclose(float(log_prob[0].exp().sum()), 1.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# 6. numerical robustness at extreme probabilities
# --------------------------------------------------------------------------- #
def test_extreme_probabilities_do_not_produce_nan_or_inf():
    p = th.tensor([[1e-9, 1.0 - 1e-9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], dtype=th.float64)
    log_prob, _, valid = conditional_bernoulli_log_prob(p, th.tensor([4]))
    assert not bool(th.isnan(log_prob).any())
    assert th.isfinite(log_prob[valid]).all()
    proba = log_prob.exp()
    assert math.isclose(float(proba.sum()), 1.0, abs_tol=1e-9)

    th.manual_seed(SEED)
    selected = sample_conditional_bernoulli(p, th.tensor([4]))
    assert not bool(th.isnan(selected.to(th.float64)).any())
    assert int(selected.sum()) == 4


def test_all_slots_at_the_extremes():
    """Every slot pinned near 0 or 1 at once -- the odds themselves would
    overflow float64 well before this if computed as a raw product."""
    p = th.tensor([[1e-9] * 4 + [1.0 - 1e-9] * 4], dtype=th.float64)
    log_prob, _, valid = conditional_bernoulli_log_prob(p, th.tensor([4]))
    assert not bool(th.isnan(log_prob).any())
    assert th.isfinite(log_prob[valid]).all()

    th.manual_seed(SEED)
    selected = sample_conditional_bernoulli(p, th.tensor([4]))
    # with such extreme separation the 4 high-p members leave essentially
    # surely -- not asserted as a hard equality (it is still a draw), but a
    # regression on the odds direction would flip this immediately
    assert int(selected[0, 4:].sum()) >= 3


# --------------------------------------------------------------------------- #
# 7. determinism
# --------------------------------------------------------------------------- #
def test_determinism_under_fixed_seed():
    p = th.linspace(0.05, 0.9, MAX_GROUP_SIZE, dtype=th.float64).expand(10, -1)
    m = th.randint(0, MAX_GROUP_SIZE + 1, (10,))
    m = th.minimum(m, th.tensor(MAX_GROUP_SIZE))

    th.manual_seed(SEED)
    first = sample_conditional_bernoulli(p, m)
    th.manual_seed(SEED)
    second = sample_conditional_bernoulli(p, m)
    assert th.equal(first, second)


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_rejects_m_greater_than_k():
    p = th.full((1, MAX_GROUP_SIZE), 0.5, dtype=th.float64)
    mask = th.tensor([[True] * 3 + [False] * (MAX_GROUP_SIZE - 3)])
    with pytest.raises(AssertionError, match="exceed"):
        sample_conditional_bernoulli(p, th.tensor([4]), mask=mask)


def test_rejects_negative_m():
    p = th.full((1, MAX_GROUP_SIZE), 0.5, dtype=th.float64)
    with pytest.raises(AssertionError, match="non-negative"):
        sample_conditional_bernoulli(p, th.tensor([-1]))


def test_rejects_wrong_width():
    p = th.full((1, MAX_GROUP_SIZE - 1), 0.5, dtype=th.float64)
    with pytest.raises(AssertionError, match="max_group_size"):
        sample_conditional_bernoulli(p, th.tensor([1]))


def test_rejects_mismatched_m_length():
    p = th.full((2, MAX_GROUP_SIZE), 0.5, dtype=th.float64)
    with pytest.raises(AssertionError, match="rows of p"):
        sample_conditional_bernoulli(p, th.tensor([1]))
