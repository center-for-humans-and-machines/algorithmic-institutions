"""Unit tests for the conditional-Bernoulli WHO-leaves sampler: the
subset distribution over which ``m`` of a group's ``k`` members leave, and
the single batched draw from it.

Plain pytest, no PyG imports -- the module under test is torch only, so this
file needs none of the ``torch_scatter`` / ``torch_geometric`` stand-ins the
PyG-dependent suites install. Expected values are derived independently here
(a plain Python enumeration over subsets, a Python Poisson-binomial
recursion) rather than restated from the implementation, and the structural
claims are written as invariances -- exactly ``m`` selected, marginals
recovered, ordering preserved -- so a wrong implementation cannot pass by
agreement.

Invariants and rationale:
``notes/autoresearch_log/switch-joint-exodus-gmlp.md``, plan step 3.

Local test (CPU torch, no PyG):
    .venv/bin/python -m pytest tests/switch/test_conditional_bernoulli.py
"""

import math
import sys
from itertools import combinations
from math import comb, log
from pathlib import Path

import pytest
import torch as th

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # tests/switch -> repo root
# this checkout's src must win over any installed/editable aimanager
sys.path.insert(0, str(ROOT / "src"))

from aimanager.generic.conditional_bernoulli import (  # noqa: E402
    MAX_GROUP_SIZE,
    conditional_bernoulli_log_prob,
    sample_conditional_bernoulli,
)

SEED = 20260903
GRID = 1 << MAX_GROUP_SIZE


# --------------------------------------------------------------------------- #
# helpers -- ground truth computed in plain Python, not with the module
# --------------------------------------------------------------------------- #
def poisson_binomial_pmf(p):
    """pmf of the sum of independent Bernoulli(p_i), by the standard O(k^2)
    DP recursion. Used only to build the test's ground truth for ``m``; not
    part of the module under test."""
    p = [float(v) for v in p]
    pmf = [1.0]
    for pi in p:
        nxt = [0.0] * (len(pmf) + 1)
        for j, w in enumerate(pmf):
            nxt[j] += w * (1.0 - pi)
            nxt[j + 1] += w * pi
        pmf = nxt
    return th.tensor(pmf, dtype=th.float64)


def subset_probs_by_python(p_real, m):
    """``{frozenset(indices): probability}`` for every size-``m`` subset of
    ``p_real``, from the definition: weight = product of odds, normalised
    over the subsets of that size. Deliberately a product of odds (not a sum
    of log-odds) so it cannot share a bug with the implementation."""
    weights = {}
    for combo in combinations(range(len(p_real)), m):
        w = 1.0
        for i in combo:
            w *= p_real[i] / (1.0 - p_real[i])
        weights[frozenset(combo)] = w
    total = sum(weights.values())
    return {s: w / total for s, w in weights.items()}


def all_k_m_pairs(max_k=MAX_GROUP_SIZE):
    return [(k, m) for k in range(max_k + 1) for m in range(k + 1)]


def widen(p_row, width=MAX_GROUP_SIZE):
    """Pad a 1-D list of real member probabilities out to ``width``, with an
    arbitrary (never-read) fill value on the padded tail."""
    pad = [0.5] * (width - len(p_row))
    return p_row + pad


def bits_to_set(bit_row):
    return frozenset(i for i, v in enumerate(bit_row.tolist()) if v)


# --------------------------------------------------------------------------- #
# 1. the distribution itself, against a Python enumeration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k,m", [(8, 3), (6, 2), (5, 4), (4, 1)])
def test_subset_distribution_matches_a_python_enumeration(k, m):
    """The exact per-subset probabilities, checked against a product-of-odds
    enumeration written out longhand."""
    p_real = [0.07 + 0.11 * i for i in range(k)]  # spread over (0, 1)
    p = th.tensor([widen(p_real)], dtype=th.float64)
    mask = th.tensor([[True] * k + [False] * (MAX_GROUP_SIZE - k)])

    log_prob, bits, valid = conditional_bernoulli_log_prob(p, th.tensor([m]), mask=mask)
    expected = subset_probs_by_python(p_real, m)

    assert int(valid[0].sum().item()) == len(expected) == comb(k, m)
    got = {
        bits_to_set(bits[code]): float(log_prob[0, code].exp())
        for code in range(GRID)
        if bool(valid[0, code])
    }
    assert set(got) == set(expected)
    for s, want in expected.items():
        assert math.isclose(got[s], want, rel_tol=1e-10, abs_tol=1e-12), (s, got[s])
    # everything off the support is EXACTLY zero, not merely small
    assert float(log_prob[0][~valid[0]].exp().abs().max()) == 0.0
    assert math.isclose(float(log_prob[0].exp().sum()), 1.0, abs_tol=1e-12)


# --------------------------------------------------------------------------- #
# 2. marginal recovery -- the key correctness property
# --------------------------------------------------------------------------- #
def test_marginal_recovery_from_the_poisson_binomial_of_m():
    """If ``m`` is drawn from the TRUE sum distribution implied by the p_i
    themselves (the Poisson-binomial), the conditional-Bernoulli draws,
    averaged over many samples, must recover the input p_i: this is exactly
    what "conditioning on the sum" means.

    N_SAMPLES = 20000, tolerance = 5 binomial standard errors per member,
    fixed seed so it cannot flake.
    """
    N_SAMPLES = 20000
    p = th.tensor([0.05, 0.12, 0.23, 0.35, 0.5, 0.63, 0.78, 0.92], dtype=th.float64)
    pmf = poisson_binomial_pmf(p)

    th.manual_seed(SEED)
    m = th.multinomial(pmf.expand(N_SAMPLES, -1), 1).reshape(-1)
    selected = sample_conditional_bernoulli(p.expand(N_SAMPLES, -1), m)

    freq = selected.to(th.float64).mean(0)
    se = th.sqrt(p * (1.0 - p) / N_SAMPLES)
    err = (freq - p).abs()
    assert bool((err < 5.0 * se).all()), (err / se).tolist()
    # the drawn counts must also average to sum(p) -- a marginal on m itself
    assert math.isclose(
        float(m.to(th.float64).mean()), float(p.sum()), abs_tol=5.0 * 0.02
    )


# --------------------------------------------------------------------------- #
# 3. propensity ordering is preserved
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
# 4. exactly m selected, for every (k, m), including the boundaries
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
    assert int(leave.sum()) == 1


# --------------------------------------------------------------------------- #
# 5. padded slots: a batch of mixed widths must not NaN anywhere
# --------------------------------------------------------------------------- #
def test_mixed_widths_never_produce_nan():
    """A batch mixing k = 8, k = 1 and k = 0 rows. The padded slots sit at a
    neutral finite logit precisely so the `0 * -inf` of an excluded padded
    slot cannot appear; that failure is silent and row-specific, so it is
    checked head on."""
    rows_p, rows_mask, rows_m = [], [], []
    for k, m in [(8, 4), (1, 1), (0, 0), (1, 0), (8, 8), (3, 2)]:
        # extreme probabilities on the real slots, and an EXACT 0/1 pair in
        # the padded tail, which would be an infinite logit unmasked
        p_real = [1e-9, 1.0 - 1e-9, 0.5, 0.25, 0.75, 1e-9, 0.9, 0.1][:k]
        pad = [0.0, 1.0] * MAX_GROUP_SIZE
        rows_p.append(p_real + pad[: MAX_GROUP_SIZE - k])
        rows_mask.append([True] * k + [False] * (MAX_GROUP_SIZE - k))
        rows_m.append(m)

    p = th.tensor(rows_p, dtype=th.float64)
    mask = th.tensor(rows_mask, dtype=th.bool)
    m = th.tensor(rows_m, dtype=th.int64)

    log_prob, _, valid = conditional_bernoulli_log_prob(p, m, mask=mask)
    assert not bool(th.isnan(log_prob).any())
    assert bool(th.isfinite(log_prob[valid]).all())
    assert th.allclose(
        log_prob.exp().sum(-1), th.ones(len(rows_m), dtype=th.float64), atol=1e-12
    )

    th.manual_seed(SEED)
    selected = sample_conditional_bernoulli(p, m, mask=mask)
    assert th.equal(selected.sum(-1), m)
    assert not bool((selected & ~mask).any())


# --------------------------------------------------------------------------- #
# 6. equal probabilities give uniform selection -- a closed form, checked
#    exactly (no Monte Carlo needed: with p_i all equal, every subset of a
#    given size carries the same total log-odds by construction)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k,m,p_value", [(6, 3, 0.37), (8, 4, 1e-3), (5, 0, 0.9)])
def test_equal_probabilities_give_uniform_selection(k, m, p_value):
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
# 7. numerical robustness at extreme probabilities
# --------------------------------------------------------------------------- #
def test_extreme_probabilities_do_not_produce_nan_or_inf():
    p = th.tensor([[1e-9, 1.0 - 1e-9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], dtype=th.float64)
    log_prob, _, valid = conditional_bernoulli_log_prob(p, th.tensor([4]))
    assert not bool(th.isnan(log_prob).any())
    assert bool(th.isfinite(log_prob[valid]).all())
    assert math.isclose(float(log_prob.exp().sum()), 1.0, abs_tol=1e-9)

    th.manual_seed(SEED)
    selected = sample_conditional_bernoulli(p, th.tensor([4]))
    assert int(selected.sum()) == 4


def test_all_slots_at_the_extremes():
    """Every slot pinned near 0 or 1 at once -- the odds themselves would
    overflow float64 well before this if computed as a raw product."""
    p = th.tensor([[1e-9] * 4 + [1.0 - 1e-9] * 4], dtype=th.float64)
    log_prob, _, valid = conditional_bernoulli_log_prob(p, th.tensor([4]))
    assert not bool(th.isnan(log_prob).any())
    assert bool(th.isfinite(log_prob[valid]).all())

    th.manual_seed(SEED)
    selected = sample_conditional_bernoulli(p, th.tensor([4]))
    # with such extreme separation the 4 high-p members leave essentially
    # surely -- not asserted as a hard equality (it is still a draw), but a
    # regression on the odds direction would flip this immediately
    assert int(selected[0, 4:].sum()) >= 3


def test_exact_zero_and_one_probabilities_stay_finite():
    """An exact 0 and an exact 1 on REAL slots: the clamp is what keeps the
    log-odds finite, and it has to bite at both ends."""
    p = th.tensor([[0.0, 1.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], dtype=th.float64)
    log_prob, bits, valid = conditional_bernoulli_log_prob(p, th.tensor([3]))
    assert not bool(th.isnan(log_prob).any())
    assert bool(th.isfinite(log_prob[valid]).all())

    th.manual_seed(SEED)
    selected = sample_conditional_bernoulli(p, th.tensor([3]))
    assert int(selected.sum()) == 3
    # the p = 1 member is overwhelmingly favoured, the p = 0 one shunned
    best = int(log_prob[0].argmax())
    assert bits_to_set(bits[best]) >= frozenset({1})
    assert 0 not in bits_to_set(bits[best])


# --------------------------------------------------------------------------- #
# 8. determinism
# --------------------------------------------------------------------------- #
def test_determinism_under_fixed_seed():
    p = th.linspace(0.05, 0.9, MAX_GROUP_SIZE, dtype=th.float64).expand(10, -1)
    th.manual_seed(SEED)
    m = th.randint(0, MAX_GROUP_SIZE + 1, (10,))

    th.manual_seed(SEED)
    first = sample_conditional_bernoulli(p, m)
    th.manual_seed(SEED)
    second = sample_conditional_bernoulli(p, m)
    assert th.equal(first, second)


def test_one_rng_draw_for_the_whole_batch():
    """The RNG contract, checked by replaying the global generator: after
    sampling a 64-row batch the RNG state is exactly where a SINGLE
    `th.multinomial` over a (64, 256) table leaves it -- one draw for the
    batch, not one per row. A later step relies on the per-round draw count
    being predictable so that a run with the head off is bitwise identical
    to the base model."""
    p = th.full((64, MAX_GROUP_SIZE), 0.5, dtype=th.float64)
    m = th.full((64,), 4, dtype=th.int64)

    th.manual_seed(SEED)
    sample_conditional_bernoulli(p, m)
    after_sampler = th.get_rng_state()

    th.manual_seed(SEED)
    th.multinomial(th.full((64, GRID), 1.0, dtype=th.float64), 1)
    after_one_call = th.get_rng_state()

    assert th.equal(after_sampler, after_one_call)


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
