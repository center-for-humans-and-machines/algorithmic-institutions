"""Tests for the annealed, local behaviour policy.

Two claims are made in notes/autoresearch_log/rl-manager-annealed-local.md and
both are tested here, because both fail silently: a schedule that never
reaches its floor leaves the behaviour policy permanently off the evaluated
one, and a boundary rule that clips rather than truncates piles mass on
actions 0 and 30 and would manufacture exactly the kind of policy-shape
artefact the arm is trying to remove.
"""

import math

import pytest
import torch as th

from aimanager.manager.exploration import Exploration

N_PUNISHMENTS = 31
CPU = th.device("cpu")


def make(**kw):
    kw.setdefault("eps", 0.1)
    kw.setdefault("n_actions", N_PUNISHMENTS)
    kw.setdefault("device", CPU)
    return Exploration(**kw)


# --------------------------------------------------------------------------- #
# the schedule
# --------------------------------------------------------------------------- #
def test_no_schedule_is_constant():
    """Unmodified behaviour must be recoverable from config alone."""
    e = make()
    assert e.epsilon(0) == 0.1
    assert e.epsilon(4000) == 0.1
    assert e.epsilon(None) == 0.1


def test_linear_schedule_endpoints_and_midpoint():
    e = make(eps=0.1, eps_final=0.01, eps_anneal_steps=3000)
    assert e.epsilon(0) == pytest.approx(0.1)
    assert e.epsilon(1500) == pytest.approx(0.055)
    assert e.epsilon(750) == pytest.approx(0.1 - 0.25 * 0.09)


def test_schedule_reaches_floor_exactly_at_anneal_steps_and_holds():
    """The floor must land at the stated step -- not before, not after."""
    e = make(eps=0.1, eps_final=0.01, eps_anneal_steps=3000)
    assert e.epsilon(2999) > 0.01
    assert e.epsilon(3000) == pytest.approx(0.01)
    assert e.epsilon(3001) == pytest.approx(0.01)
    assert e.epsilon(4000) == pytest.approx(0.01)
    assert e.epsilon(10**9) == pytest.approx(0.01)


def test_schedule_is_monotone_decreasing():
    e = make(eps=0.1, eps_final=0.01, eps_anneal_steps=3000)
    vals = [e.epsilon(s) for s in range(0, 4000, 37)]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))


def test_schedule_keys_must_be_set_together():
    with pytest.raises(ValueError):
        make(eps_final=0.01)
    with pytest.raises(ValueError):
        make(eps_anneal_steps=3000)


def test_rejects_degenerate_parameters():
    with pytest.raises(ValueError):
        make(eps_final=0.01, eps_anneal_steps=0)
    with pytest.raises(ValueError):
        make(sigma=0.0)


# --------------------------------------------------------------------------- #
# the local proposal
# --------------------------------------------------------------------------- #
def truncated_gaussian(a0, sigma, n=N_PUNISHMENTS):
    """The claimed distribution, written out independently of the module."""
    w = [math.exp(-((a - a0) ** 2) / (2 * sigma**2)) for a in range(n)]
    z = sum(w)
    return [x / z for x in w]


@pytest.mark.parametrize("a0", [0, 1, 15, 29, 30])
def test_proposal_matches_the_truncated_gaussian_everywhere(a0):
    e = make(sigma=2.0)
    got = e.proposal_probs[a0].tolist()
    want = truncated_gaussian(a0, 2.0)
    assert got == pytest.approx(want, abs=1e-6)
    assert sum(got) == pytest.approx(1.0)


def test_boundary_is_truncated_not_clipped():
    """At a0 = 0 the out-of-range tail is redistributed over the whole
    support in proportion, not dumped on action 0. For sigma = 2 clipping puts
    0.600 of the mass on the single action 0; truncation puts 0.333."""
    sigma = 2.0
    e = make(sigma=sigma)
    p0 = e.proposal_probs[0]

    clipped_p0 = 0.0
    z = sum(
        math.exp(-((a - 0) ** 2) / (2 * sigma**2)) for a in range(-200, N_PUNISHMENTS)
    )
    for a in range(-200, 1):
        clipped_p0 += math.exp(-((a - 0) ** 2) / (2 * sigma**2)) / z

    assert clipped_p0 == pytest.approx(0.600, abs=0.01)
    assert float(p0[0]) == pytest.approx(0.333, abs=0.01)
    assert float(p0[0]) == pytest.approx(truncated_gaussian(0, sigma)[0], abs=1e-9)

    # And the 0.333 is a rescaling, not a spike: the boundary row is the
    # interior row's centre times the ratio of the two normalisers, which is
    # what "no mass piled on the endpoint" means quantitatively.
    interior = float(e.proposal_probs[15][15])
    z_interior = sum(math.exp(-((a - 15) ** 2) / (2 * sigma**2)) for a in range(31))
    z_boundary = sum(math.exp(-(a**2) / (2 * sigma**2)) for a in range(31))
    assert float(p0[0]) == pytest.approx(interior * z_interior / z_boundary, rel=1e-9)


def test_boundary_preserves_the_gaussian_ratios():
    """The sharp statement of "truncated, not clipped": every pair of in-range
    actions keeps the ratio the untruncated kernel gives them, at the boundary
    exactly as in the interior. Clipping breaks this at the endpoint."""
    sigma = 2.0
    e = make(sigma=sigma)
    for a0 in (0, 30):
        p = e.proposal_probs[a0]
        for a, b in ((0, 1), (1, 3), (27, 29), (29, 30)):
            want = math.exp((-((a - a0) ** 2) + (b - a0) ** 2) / (2 * sigma**2))
            assert float(p[a] / p[b]) == pytest.approx(want, rel=1e-5)


def test_boundaries_are_mirror_images():
    e = make(sigma=2.0)
    lo = e.proposal_probs[0].tolist()
    hi = e.proposal_probs[N_PUNISHMENTS - 1].tolist()
    assert lo == pytest.approx(list(reversed(hi)), abs=1e-6)


def test_proposal_is_local():
    """sigma = 2 must keep the draw near the greedy action; the uniform
    proposal it replaces sits about 10 points away on average."""
    e = make(sigma=2.0)
    a = th.arange(N_PUNISHMENTS, dtype=th.float)
    mad_local = float((e.proposal_probs[15] * (a - 15).abs()).sum())
    mad_uniform = float((a - 15).abs().mean())
    assert mad_local < 2.0
    assert mad_uniform > 7.0


@pytest.mark.parametrize("a0", [0, 15, 30])
def test_sampled_frequencies_match_the_claimed_distribution(a0):
    th.manual_seed(0)
    e = make(sigma=2.0)
    draws = e.propose(th.full((200000,), a0, dtype=th.long))
    assert int(draws.min()) >= 0
    assert int(draws.max()) <= N_PUNISHMENTS - 1
    freq = th.bincount(draws, minlength=N_PUNISHMENTS).float() / draws.numel()
    want = th.tensor(truncated_gaussian(a0, 2.0))
    assert float((freq - want).abs().max()) < 0.01


def test_uniform_proposal_when_sigma_unset():
    th.manual_seed(0)
    e = make()
    assert e.proposal_probs is None
    draws = e.propose(th.full((200000,), 15, dtype=th.long))
    freq = th.bincount(draws, minlength=N_PUNISHMENTS).float() / draws.numel()
    assert float((freq - 1.0 / N_PUNISHMENTS).abs().max()) < 0.01


# --------------------------------------------------------------------------- #
# the two together
# --------------------------------------------------------------------------- #
def test_call_leaves_greedy_alone_at_zero_epsilon():
    e = make(eps=0.1, eps_final=0.0, eps_anneal_steps=10, sigma=2.0)
    greedy = th.randint(0, N_PUNISHMENTS, (1000,))
    assert th.equal(e(greedy, update_step=10), greedy)


def test_call_perturbs_about_epsilon_of_the_cells():
    th.manual_seed(0)
    e = make(eps=0.5, sigma=2.0)
    greedy = th.full((100000,), 15, dtype=th.long)
    changed = (e(greedy) != greedy).float().mean()
    # eps of the cells resample; a resample lands back on the greedy action
    # with probability proposal_probs[15, 15], so the observed change rate is
    # eps * (1 - that).
    expected = 0.5 * (1 - float(e.proposal_probs[15, 15]))
    assert float(changed) == pytest.approx(expected, abs=0.01)


def test_off_path_reproduces_the_original_draws_bit_for_bit():
    """The control arm reruns the unmodified behaviour on two new seeds, so
    `Exploration` with both mechanisms off must consume the RNG in the same
    order and produce the same actions as the inlined code it replaced --
    otherwise the five-seed control is not one experiment."""
    n, eps = N_PUNISHMENTS, 0.1
    greedy = th.randint(0, n, (64, 8, 1))

    th.manual_seed(1234)
    random_actions = th.randint(0, n, size=greedy.shape, device=CPU)
    random_numbers = th.rand(size=greedy.shape, device=CPU)
    original = th.where(random_numbers < eps, random_actions, greedy)

    th.manual_seed(1234)
    got = make(eps=eps)(greedy, update_step=17)

    assert th.equal(got, original)


def test_annealed_local_injects_far_less_punishment_than_uniform():
    """The quantity the arm exists to shrink: mean punishment added to the
    greedy action per agent-round, at the start and at the end of training."""
    th.manual_seed(0)
    greedy = th.zeros(200000, dtype=th.long)

    control = make(eps=0.1)
    arm = make(eps=0.1, eps_final=0.01, eps_anneal_steps=3000, sigma=2.0)

    added_control = float((control(greedy, 0) - greedy).float().mean())
    added_start = float((arm(greedy, 0) - greedy).float().mean())
    added_end = float((arm(greedy, 3000) - greedy).float().mean())

    assert added_control == pytest.approx(1.5, abs=0.05)
    assert added_start < 0.25
    assert added_end < 0.03
