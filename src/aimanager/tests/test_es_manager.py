"""Unit tests for the evolution-strategies arm's mechanics.

No artifacts and no environment: everything here is the parts of the method
that can be checked against a closed form -- the perturbation, the fitness
aggregation and the policy-shape binning. The rollout-level guarantees (the
deterministic evaluation of the mean parameters, and reproducibility from a
seed through the real environment) live in test_es_rollout.py.

Raven only: `es_manager` imports the GNN stack.
"""

import pandas as pd
import pytest
import torch as th

from aimanager.es_manager import (
    N_RPA_BINS,
    ShapeAccumulator,
    centered_ranks,
    draw_perturbations,
    flat_params,
    rpa_bin,
    set_flat_params_,
)
from aimanager.evaluation_suite.metrics import RPA_EDGES, RPA_LABELS


class Tiny(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = th.nn.Linear(3, 4)
        self.b = th.nn.Linear(4, 2)


# -- parameter vector ------------------------------------------------


def test_flat_params_roundtrip():
    model = Tiny()
    original = flat_params(model)
    target = th.randn_like(original)
    set_flat_params_(model, target)
    assert th.equal(flat_params(model), target)
    # and the model really changed, i.e. the write was not a no-op
    assert not th.equal(original, target)


def test_set_flat_params_rejects_wrong_size():
    model = Tiny()
    with pytest.raises(AssertionError):
        set_flat_params_(model, th.zeros(flat_params(model).numel() + 1))


# -- perturbation ----------------------------------------------------


def test_mirrored_perturbations_are_exactly_antithetic():
    g = th.Generator().manual_seed(0)
    eps = draw_perturbations(8, 5, g, th.device("cpu"), mirrored=True)
    assert eps.shape == (8, 5)
    for k in range(4):
        assert th.equal(eps[2 * k], -eps[2 * k + 1])
    # The pair structure is the whole point: the mean perturbation is zero to
    # machine precision, so the estimator carries no bias from theta's own
    # fitness level.
    assert eps.mean(dim=0).abs().max() < 1e-6


def test_mirrored_perturbations_need_an_even_population():
    g = th.Generator().manual_seed(0)
    with pytest.raises(AssertionError):
        draw_perturbations(7, 5, g, th.device("cpu"), mirrored=True)


def test_unmirrored_perturbations_are_not_antithetic():
    g = th.Generator().manual_seed(0)
    eps = draw_perturbations(8, 5, g, th.device("cpu"), mirrored=False)
    assert not th.equal(eps[0], -eps[1])


def test_perturbations_reproduce_from_a_seed():
    def draw(seed):
        g = th.Generator().manual_seed(seed)
        return [draw_perturbations(6, 7, g, th.device("cpu")) for _ in range(3)]

    a, b, c = draw(11), draw(11), draw(12)
    assert all(th.equal(x, y) for x, y in zip(a, b))
    assert not any(th.equal(x, y) for x, y in zip(a, c))
    # Successive generations must differ; a generator reset per generation
    # would silently score the same population 4000 times.
    assert not th.equal(a[0], a[1])


# -- fitness aggregation ---------------------------------------------


def test_centered_ranks_are_an_even_grid_in_order():
    f = th.tensor([3.0, 1.0, 2.0, 0.0])
    r = centered_ranks(f)
    assert th.allclose(r, th.tensor([0.5, -1 / 6, 1 / 6, -0.5]))
    assert float(r.sum()) == pytest.approx(0.0)
    assert float(r.min()) == -0.5 and float(r.max()) == 0.5


def test_centered_ranks_ignore_the_scale_of_the_returns():
    f = th.tensor([1.0, 2.0, 3.0, 4.0])
    outlier = th.tensor([1.0, 2.0, 3.0, 4.0e6])
    assert th.equal(centered_ranks(f), centered_ranks(outlier))


def test_fitness_weighted_average_points_up_the_fitness_gradient():
    """The estimator, on a fitness that is linear in the perturbation."""
    d = th.zeros(10)
    d[3] = 1.0
    sigma, n_members = 0.02, 400
    g = th.Generator().manual_seed(3)
    eps = draw_perturbations(n_members, 10, g, th.device("cpu"))
    fitness = (eps * d).sum(dim=1)
    for weights in (centered_ranks(fitness), fitness):
        grad = (weights.unsqueeze(1) * eps).sum(dim=0) / (n_members * sigma)
        cos = th.nn.functional.cosine_similarity(grad, d, dim=0)
        assert float(cos) > 0.9, weights


def test_raw_fitness_weighting_is_captured_by_one_outlier():
    """Why centered ranks are not optional here."""
    d = th.zeros(10)
    d[3] = 1.0
    g = th.Generator().manual_seed(5)
    eps = draw_perturbations(40, 10, g, th.device("cpu"))
    fitness = (eps * d).sum(dim=1)
    # one member is scored on a freak episode -- a group of 8 rather than 0
    fitness[7] += 1e4
    raw = ((fitness.unsqueeze(1) * eps).sum(dim=0)).squeeze()
    ranked = ((centered_ranks(fitness).unsqueeze(1) * eps).sum(dim=0)).squeeze()
    cos = th.nn.functional.cosine_similarity
    assert float(cos(raw, eps[7], dim=0)) > 0.99
    assert float(cos(ranked, d, dim=0)) > float(cos(raw, d, dim=0))


# -- policy shape ----------------------------------------------------


def test_rpa_bin_matches_the_evaluation_suite_cut():
    c = th.arange(0, 21)
    got = [RPA_LABELS[i] for i in rpa_bin(c).tolist()]
    want = (
        pd.cut(pd.Series(c.numpy(), dtype=float), RPA_EDGES, labels=RPA_LABELS)
        .astype(str)
        .tolist()
    )
    assert got == want
    assert got[0] == "{0}" and got[20] == "{20}"


class FakeEnv:
    """The five fields ShapeAccumulator reads, shaped as the env shapes them."""

    def __init__(self, contribution, punishment, groups, valid):
        self.n_agents = len(contribution[0])
        self.contribution = th.tensor(contribution).unsqueeze(-1)
        self.punishment = th.tensor(punishment).unsqueeze(-1)
        self.agent_groups = th.tensor(groups).unsqueeze(-1)
        self.contribution_valid = th.tensor(valid).unsqueeze(-1)


def test_shape_accumulator_bins_by_member_and_keeps_only_the_rl_group():
    env = FakeEnv(
        contribution=[[0, 20], [0, 20]],
        punishment=[[4, 1], [10, 3]],
        groups=[[0, 1], [0, 0]],
        valid=[[True, True], [True, True]],
    )
    acc = ShapeAccumulator(2, th.device("cpu"))
    acc.add(env, rl_group_id=0, member_of_episode=th.tensor([0, 1]))
    rows = {
        (r["member"], r["contribution_bin"]): r
        for r in acc.rows("x", 0, per_member=True)
        if r["subset"] == "all"
    }
    # episode 0: only agent 0 is in group 0 (contribution 0 -> "{0}")
    assert rows[(0, "{0}")]["mean_punishment"] == 4.0
    assert rows[(0, "{0}")]["n"] == 1.0
    assert rows[(0, "{20}")]["n"] == 0.0
    # episode 1: both agents are in group 0
    assert rows[(1, "{0}")]["mean_punishment"] == 10.0
    assert rows[(1, "{20}")]["mean_punishment"] == 3.0
    # pooled over members
    assert rows[(-1, "{0}")]["mean_punishment"] == pytest.approx(7.0)
    assert rows[(-1, "{0}")]["n"] == 2.0


def test_shape_accumulator_valid_subset_drops_timed_out_cells():
    env = FakeEnv(
        contribution=[[9, 9]],
        punishment=[[0, 6]],
        groups=[[0, 0]],
        valid=[[False, True]],
    )
    acc = ShapeAccumulator(1, th.device("cpu"))
    acc.add(env, rl_group_id=0, member_of_episode=th.tensor([0]))
    rows = {(r["subset"], r["contribution_bin"]): r for r in acc.rows("x", 0, False)}
    # contribution 9 is bin "6-10" for both cells; the imputed one drags the
    # unconditional mean down, which is exactly the asymmetry the two subsets
    # are there to show.
    assert rows[("all", "6-10")]["n"] == 2.0
    assert rows[("all", "6-10")]["mean_punishment"] == 3.0
    assert rows[("valid", "6-10")]["n"] == 1.0
    assert rows[("valid", "6-10")]["mean_punishment"] == 6.0


def test_shape_accumulator_emits_every_bin_even_when_empty():
    env = FakeEnv([[0]], [[1]], [[0]], [[True]])
    acc = ShapeAccumulator(1, th.device("cpu"))
    acc.add(env, rl_group_id=0, member_of_episode=th.tensor([0]))
    rows = [r for r in acc.rows("x", 7, per_member=True) if r["subset"] == "all"]
    # pooled + one member, six bins each
    assert len(rows) == 2 * N_RPA_BINS
    assert {r["contribution_bin"] for r in rows} == set(RPA_LABELS)
    assert all(r["update_step"] == 7 and r["sampling"] == "x" for r in rows)
