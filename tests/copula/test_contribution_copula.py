"""Tests for the herding-copula sampler at 21 contribution levels (plan step 3
of contribution-herding-copula-v2).

Plain pytest, torch + numpy only (no PyG, no scipy) -- runs locally:
    uv run pytest tests/copula/ -q

The 2-level switch head's version of these gates lives in
tests/switch/test_switch_copula.py; this file re-checks the ones that depend on
the level count -- marginal preservation over a 21-level categorical, the
induced within-cell dependence, the AR(1) latent, and the inverse-CDF
convention the rho estimator is fitted under -- for the contribution head.
Context: notes/autoresearch_log/contribution-herding-copula-v2.md.
"""

import math

import numpy as np
import pytest
import torch as th

from aimanager.generic.copula import levels_from_u, sample_correlated_levels

N_LEVELS = 21
# PR #149's calibrated within-round residual dependence of the frozen
# contribution GNN -- the dose the simulation will actually run at, and the
# number step 8 has to reproduce exactly.
RHO = 0.06958238086256316
# A deliberately strong persistence, so the lag-1 autocorrelation is
# unambiguous; the calibrated phi lands at step 8.
PHI = 0.9
HIGH_RHO = 0.5
SEED = 20260827


def contribution_proba(n_rows, seed=0):
    """(n_rows, 21) float64 rows shaped like the GNN's contribution marginals:
    a spike at 0, a spike at 20 and a bumpy interior, jittered per row so the
    checks never lean on identical marginals."""
    rng = np.random.default_rng(seed)
    base = np.full(N_LEVELS, 0.02)
    base[0] = 0.25
    base[-1] = 0.15
    base[10] = 0.08
    w = base[None, :] * rng.uniform(0.5, 1.5, (n_rows, N_LEVELS))
    return th.from_numpy(w / w.sum(axis=1, keepdims=True))


def cells_of_size(n_cells, size):
    """Dense cell ids 0..n_cells-1, each repeated `size` times."""
    return th.arange(n_cells, dtype=th.int64).repeat_interleave(size)


def corr(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


# --------------------------------------------------------------------------- #
# 1. marginals preserved at the calibrated dose
# --------------------------------------------------------------------------- #
def test_marginals_preserved_at_calibrated_rho():
    """Each of the 21 per-row level frequencies stays inside its binomial band
    at the rho the contribution slot will run at."""
    n_rows, n_repeats = 16, 4000
    proba = contribution_proba(n_rows, seed=1)
    cell_id = cells_of_size(n_rows // 4, 4)

    th.manual_seed(SEED)
    counts = th.zeros(n_rows, N_LEVELS, dtype=th.float64)
    ones = th.ones(n_rows, 1, dtype=th.float64)
    for _ in range(n_repeats):
        levels, _ = sample_correlated_levels(proba, cell_id, RHO)
        counts.scatter_add_(1, levels.reshape(-1, 1), ones)

    freq = (counts / n_repeats).numpy()
    p = proba.numpy()
    se = np.sqrt(p * (1.0 - p) / n_repeats)
    # 2e-3 of absolute slack keeps a near-zero level from failing on a single
    # extra count; it is far below any distortion worth catching.
    assert np.all(np.abs(freq - p) < 5.0 * se + 2e-3), np.max(np.abs(freq - p) / se)
    # the extremes are the levels a continuous head would erase (PR #155)
    assert freq[:, 0].min() > 0.0 and freq[:, -1].min() > 0.0


# --------------------------------------------------------------------------- #
# 2. dependence is within cells only
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rho,lo,hi", [(HIGH_RHO, 0.3, 1.0), (0.0, -0.02, 0.02)])
def test_within_cell_level_correlation(rho, lo, hi):
    """A high rho correlates cell mates' contribution levels; rho = 0 does
    not. Cross-cell pairs are uncorrelated either way."""
    n_cells = 20000
    proba = contribution_proba(2 * n_cells, seed=2)
    cell_id = cells_of_size(n_cells, 2)

    th.manual_seed(SEED)
    levels, z_cell = sample_correlated_levels(proba, cell_id, rho)
    assert z_cell.shape == (n_cells,)
    a, b = levels[0::2].numpy(), levels[1::2].numpy()
    within = corr(a, b)
    across = corr(a, np.roll(b, 1))  # partners from neighbouring cells
    assert lo < within < hi, within
    assert abs(across) < 0.02, across


def test_calibrated_rho_correlates():
    """At the calibrated rho the within-cell level correlation is small but
    clearly positive -- the effect the CG claim rests on."""
    n_cells = 60000
    proba = contribution_proba(2 * n_cells, seed=3)
    th.manual_seed(SEED)
    levels, _ = sample_correlated_levels(proba, cells_of_size(n_cells, 2), RHO)
    assert corr(levels[0::2].numpy(), levels[1::2].numpy()) > 0.03


# --------------------------------------------------------------------------- #
# 3. AR(1) latent
# --------------------------------------------------------------------------- #
def latent_path(rho, phi, n_cells=40000, n_steps=10, seed=4):
    """The z_cell sequence of `n_steps` consecutive sampler calls, fed back."""
    proba = contribution_proba(n_cells, seed=seed)
    cell_id = th.arange(n_cells, dtype=th.int64)
    th.manual_seed(SEED)
    z, path = None, []
    for _ in range(n_steps):
        _, z = sample_correlated_levels(proba, cell_id, rho, z_prev=z, phi=phi)
        assert z.dtype == th.float64
        path.append(z.numpy())
    return np.stack(path)


def test_ar1_lag1_autocorrelation_tracks_phi():
    """phi carries the cell latent across rounds without changing its scale --
    the persistence the contribution slot needs every round."""
    zs = latent_path(RHO, PHI)
    assert abs(zs.var() - 1.0) < 0.02, zs.var()
    ac = corr(zs[1:].ravel(), zs[:-1].ravel())
    assert abs(ac - PHI) < 0.01, ac


def test_zero_phi_leaves_consecutive_latents_uncorrelated():
    """phi = 0 with z_prev fed back is the fresh-latent-per-round arm: the
    latent still exists, but carries nothing across rounds."""
    zs = latent_path(RHO, 0.0, n_cells=20000, n_steps=6, seed=5)
    assert abs(zs.var() - 1.0) < 0.03, zs.var()
    ac = corr(zs[1:].ravel(), zs[:-1].ravel())
    assert abs(ac) < 0.02, ac


# --------------------------------------------------------------------------- #
# 4. the unit-root boundary phi = 1.0 -- the adopted persistence (step 8b)
# --------------------------------------------------------------------------- #
def test_unit_phi_freezes_the_latent_exactly():
    """At phi = 1.0 the fed-back latent comes out bit-for-bit unchanged: one
    shared standard normal per (episode, group) held for the whole episode,
    the static-latent reading of the calibration (plan revision, step 8b)."""
    n_cells = 64
    proba = contribution_proba(n_cells, seed=8)
    cell_id = th.arange(n_cells, dtype=th.int64)

    th.manual_seed(SEED)
    levels_first, z_first = sample_correlated_levels(proba, cell_id, RHO)
    z, levels = z_first, levels_first
    for r in range(11):
        levels, z_next = sample_correlated_levels(
            proba, cell_id, RHO, z_prev=z, phi=1.0
        )
        assert th.equal(z_next, z), r
        z = z_next
    assert th.equal(z, z_first)
    # only the latent is frozen: the per-row innovation is still fresh, so the
    # levels themselves keep moving from round to round
    assert not th.equal(levels, levels_first)


def test_marginals_preserved_at_unit_phi():
    """The frozen latent does not reweight anything: pooled over episodes, the
    21 per-row level frequencies still sit on the predicted marginals. Rounds
    inside an episode share the latent, so the binomial SE is inflated by that
    within-cell dependence -- bounded by `1 + (n_rounds - 1) * rho`."""
    n_rows, n_episodes, n_rounds = 16, 1200, 4
    proba = contribution_proba(n_rows, seed=9)
    cell_id = cells_of_size(n_rows // 4, 4)

    th.manual_seed(SEED)
    counts = th.zeros(n_rows, N_LEVELS, dtype=th.float64)
    ones = th.ones(n_rows, 1, dtype=th.float64)
    for _ in range(n_episodes):
        z = None
        for _ in range(n_rounds):
            levels, z = sample_correlated_levels(proba, cell_id, RHO, z_prev=z, phi=1.0)
            counts.scatter_add_(1, levels.reshape(-1, 1), ones)

    n_draws = n_episodes * n_rounds
    freq = (counts / n_draws).numpy()
    p = proba.numpy()
    se = np.sqrt(p * (1.0 - p) / n_draws) * math.sqrt(1.0 + (n_rounds - 1) * RHO)
    assert np.all(np.abs(freq - p) < 5.0 * se + 2e-3), np.max(np.abs(freq - p) / se)
    assert freq[:, 0].min() > 0.0 and freq[:, -1].min() > 0.0


def test_rng_contract_unchanged_at_unit_phi():
    """The boundary consumes the global RNG exactly as any other phi --
    `randn(n_cells)` then `randn(N)`, both float64 -- so admitting phi = 1.0
    cannot shift a simulation's draw sequence."""
    n_cells, size = 3, 4
    n = n_cells * size
    proba = contribution_proba(n, seed=10)
    cell_id = cells_of_size(n_cells, size)
    z_prev = th.full((n_cells,), 0.5, dtype=th.float64)

    def probe(**kw):
        th.manual_seed(SEED)
        sample_correlated_levels(proba, cell_id, RHO, **kw)
        return th.randn(4, dtype=th.float64)

    ref = probe(z_prev=z_prev, phi=PHI)
    assert th.equal(probe(z_prev=z_prev, phi=1.0), ref)
    assert th.equal(probe(z_prev=z_prev, phi=0.0), ref)
    assert th.equal(probe(), ref)

    th.manual_seed(SEED)
    th.randn(n_cells, dtype=th.float64)
    th.randn(n, dtype=th.float64)
    assert th.equal(th.randn(4, dtype=th.float64), ref)


# --------------------------------------------------------------------------- #
# 5. the estimator's inverse-CDF convention at 21 levels
# --------------------------------------------------------------------------- #
def test_matches_estimator_inverse_cdf_convention():
    """`level = min{a : F(a) >= u}`, i.e. numpy searchsorted side="left" on the
    row cumsum -- reimplemented here so the check does not lean on the code
    under test. The u grid straddles every one of the 20 bin edges, and an
    exact edge must fall to the LOWER level (the strict inequality the rho
    estimator was fitted under). The boundaries come from the same torch
    cumsum `levels_from_u` searches, so the comparison is not a float64
    cumsum-order artefact.
    """
    row = contribution_proba(1, seed=6)[0]
    cum = row.cumsum(0)
    cum_np = cum.numpy()

    grid = [0.0, 0.5, 1.0]
    for c in cum_np.tolist():
        grid += [max(0.0, math.nextafter(c, 0.0)), c, min(1.0, math.nextafter(c, 1.0))]
    u = th.tensor(grid, dtype=th.float64)
    proba = row.expand(len(u), N_LEVELS)

    expected = np.minimum(np.searchsorted(cum_np, u.numpy(), side="left"), N_LEVELS - 1)
    assert np.array_equal(levels_from_u(proba, u).numpy(), expected)
    # every level is reachable, so the grid really exercises all 21 bins
    assert set(expected.tolist()) == set(range(N_LEVELS))


def test_exact_bin_edge_falls_to_the_lower_level():
    """The one case a right/left mix-up would flip, spelled out per level."""
    row = contribution_proba(1, seed=7)[0]
    cum = row.cumsum(0)
    for level in range(N_LEVELS - 1):
        u = th.tensor([cum[level].item()], dtype=th.float64)
        assert levels_from_u(row.reshape(1, N_LEVELS), u).item() == level
