"""Tests for the herding-copula sampler (plan step 12).

Plain pytest, no PyG imports -- runs locally:
    uv run pytest tests/switch/ -q

The reference numbers rho / phi are the calibrated ones from
notes/autoresearch_log/switch-herding-copula.md (notes 12-13).
"""

import math

import numpy as np
import pytest
import torch as th
from scipy.special import ndtri

from aimanager.generic.copula import (
    levels_from_u,
    ndtr,
    sample_correlated_levels,
)

RHO = 0.116482333585783  # note 12
PHI = 0.70366020589033  # note 13
SEED = 20260812


def binary_proba(p):
    """(N, 2) rows [1 - p, p]; level 1 == switch."""
    p = th.as_tensor(p, dtype=th.float64).reshape(-1)
    return th.stack([1.0 - p, p], dim=1)


def cells_of_size(n_cells, size):
    """Dense cell ids 0..n_cells-1, each repeated `size` times."""
    return th.arange(n_cells, dtype=th.int64).repeat_interleave(size)


def corr(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


# --------------------------------------------------------------------------- #
# 1. inverse CDF is exact at the bin edges
# --------------------------------------------------------------------------- #
def test_levels_from_u_at_bin_edges():
    """`level = min{a : F(a) >= u}`: a cumsum boundary belongs to the LOWER
    level (searchsorted right=False / numpy side='left')."""
    row = th.tensor([0.2, 0.3, 0.5], dtype=th.float64)
    cum = [0.2, 0.5, 1.0]
    up = math.nextafter
    u = th.tensor(
        [
            0.0,
            cum[0],  # exact edge -> lower level
            up(cum[0], 1.0),  # just above -> next level
            cum[1],
            up(cum[1], 1.0),
            1.0,
        ],
        dtype=th.float64,
    )
    expected = [0, 0, 1, 1, 2, 2]
    proba = row.expand(len(u), 3)
    assert levels_from_u(proba, u).tolist() == expected


def test_levels_from_u_binary_orientation():
    """Binary rows: level 1 iff u > 1 - p, matching `w > t = Phi^-1(1 - p)`
    in scripts/baselines/switch_copula_rho.py (switch = upper tail)."""
    p = 0.2937
    edge = 1.0 - p
    u = th.tensor(
        [0.0, math.nextafter(edge, 0.0), edge, math.nextafter(edge, 1.0), 1.0],
        dtype=th.float64,
    )
    proba = binary_proba([p] * len(u))
    assert levels_from_u(proba, u).tolist() == [0, 0, 0, 1, 1]


def test_levels_from_u_saturated_u_stays_in_range():
    """ndtr saturates at 1.0 for large w; the clamp keeps the level valid."""
    proba = binary_proba([0.5, 0.5])
    u = ndtr(th.tensor([40.0, -40.0], dtype=th.float64))
    assert u[0].item() == 1.0
    assert levels_from_u(proba, u).tolist() == [1, 0]


def test_float32_proba_is_promoted():
    """A float32 `proba` must not truncate the float64 latent algebra."""
    proba32 = binary_proba([0.3, 0.7]).to(th.float32)
    u = th.tensor([0.5, 0.5], dtype=th.float64)
    assert levels_from_u(proba32, u).tolist() == [0, 1]


# --------------------------------------------------------------------------- #
# 2. marginals preserved
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_marginals_preserved(rho):
    """Per-row switch frequency stays within 5 binomial SE of its own p,
    with or without the shared latent."""
    n_repeats = 3000
    p = th.linspace(0.05, 0.9, 64, dtype=th.float64)
    proba = binary_proba(p)
    cell_id = cells_of_size(len(p) // 4, 4)

    th.manual_seed(SEED)
    hits = th.zeros(len(p), dtype=th.float64)
    for _ in range(n_repeats):
        levels, _ = sample_correlated_levels(proba, cell_id, rho)
        hits += levels.to(th.float64)
    freq = (hits / n_repeats).numpy()
    p_np = p.numpy()
    se = np.sqrt(p_np * (1.0 - p_np) / n_repeats)
    assert np.all(np.abs(freq - p_np) < 5.0 * se), np.max(np.abs(freq - p_np) / se)


# --------------------------------------------------------------------------- #
# 3. correlation is within cells only
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rho,lo,hi", [(0.5, 0.2, 1.0), (0.0, -0.03, 0.03)])
def test_within_cell_correlation(rho, lo, hi):
    """rho = 0.5 correlates cell mates; rho = 0 does not. Cross-cell pairs are
    uncorrelated either way."""
    n_cells = 20000
    proba = binary_proba([0.3] * (2 * n_cells))
    cell_id = cells_of_size(n_cells, 2)

    th.manual_seed(SEED)
    levels, z_cell = sample_correlated_levels(proba, cell_id, rho)
    assert z_cell.shape == (n_cells,)
    a, b = levels[0::2].numpy(), levels[1::2].numpy()
    within = corr(a, b)
    across = corr(a, np.roll(b, 1))  # partners from neighbouring cells
    assert lo < within < hi, within
    assert abs(across) < 0.03, across


def test_calibrated_rho_correlates():
    """At the calibrated rho the within-cell correlation is small but > 0."""
    n_cells = 40000
    proba = binary_proba([0.3] * (2 * n_cells))
    th.manual_seed(SEED)
    levels, _ = sample_correlated_levels(proba, cells_of_size(n_cells, 2), RHO)
    assert corr(levels[0::2].numpy(), levels[1::2].numpy()) > 0.03


# --------------------------------------------------------------------------- #
# 4. RNG consumption contract
# --------------------------------------------------------------------------- #
def probe_after_call(proba, cell_id, **kw):
    """State of the global RNG after one sampler call, as the next 4 draws."""
    th.manual_seed(SEED)
    sample_correlated_levels(proba, cell_id, **kw)
    return th.randn(4, dtype=th.float64)


def test_draw_count_is_composition_stable():
    """Equal N and n_cells -> identical RNG consumption, whatever the cell
    sizes, rho, phi or z_prev. (Bit-identity against the LEGACY
    th.multinomial path is not asserted here: it consumes the RNG
    differently. That gate lives on graph.py's dispatch, plan steps 13/14.)"""
    proba = binary_proba([0.3] * 6)
    a = th.tensor([0, 0, 0, 1, 1, 2])
    b = th.tensor([0, 1, 1, 2, 2, 2])
    z_prev = th.zeros(3, dtype=th.float64)

    ref = probe_after_call(proba, a, rho=RHO)
    for kw in (
        dict(rho=RHO),
        dict(rho=0.0),
        dict(rho=0.9),
        dict(rho=RHO, z_prev=z_prev, phi=PHI),
    ):
        assert th.equal(probe_after_call(proba, b, **kw), ref), kw


def test_two_draw_calls_exactly():
    """The call consumes randn(n_cells) then randn(N) -- nothing else."""
    proba = binary_proba([0.4] * 12)
    cell_id = cells_of_size(3, 4)
    th.manual_seed(SEED)
    sample_correlated_levels(proba, cell_id, RHO)
    after = th.randn(4, dtype=th.float64)

    th.manual_seed(SEED)
    th.randn(3, dtype=th.float64)
    th.randn(12, dtype=th.float64)
    assert th.equal(th.randn(4, dtype=th.float64), after)


# --------------------------------------------------------------------------- #
# 5. determinism
# --------------------------------------------------------------------------- #
def test_determinism_under_fixed_seed():
    proba = binary_proba(th.linspace(0.05, 0.9, 32, dtype=th.float64))
    cell_id = cells_of_size(8, 4)
    out = []
    for _ in range(2):
        th.manual_seed(SEED)
        out.append(sample_correlated_levels(proba, cell_id, RHO))
    assert th.equal(out[0][0], out[1][0])
    assert th.equal(out[0][1], out[1][1])


# --------------------------------------------------------------------------- #
# 6. AR(1) latent
# --------------------------------------------------------------------------- #
def test_ar1_variance_and_autocorrelation():
    """phi carries the latent across rounds without changing its scale."""
    n_cells, n_steps = 40000, 10
    proba = binary_proba([0.3] * n_cells)
    cell_id = th.arange(n_cells, dtype=th.int64)

    th.manual_seed(SEED)
    z, zs = None, []
    for _ in range(n_steps):
        _, z = sample_correlated_levels(proba, cell_id, RHO, z_prev=z, phi=PHI)
        assert z.dtype == th.float64
        zs.append(z.numpy())
    zs = np.stack(zs)  # (n_steps, n_cells)
    assert abs(zs.var() - 1.0) < 0.02, zs.var()
    ac = corr(zs[1:].ravel(), zs[:-1].ravel())
    assert abs(ac - PHI) < 0.01, ac


def test_ar1_rejects_stale_cell_count():
    proba = binary_proba([0.3] * 6)
    with pytest.raises(AssertionError, match="z_prev"):
        sample_correlated_levels(
            proba, cells_of_size(3, 2), RHO, z_prev=th.zeros(4), phi=PHI
        )


# --------------------------------------------------------------------------- #
# 7. agreement with the calibration-side sampler
# --------------------------------------------------------------------------- #
def reference_levels(p, cell_np, z_np, eps_np, rho):
    """`sample_copula_binary` of scripts/baselines/switch_copula_rho.py,
    inlined: w = sqrt(rho) z_cell + sqrt(1 - rho) eps, y = w > Phi^-1(1 - p).
    The z-side form the estimator was fitted under -- no ndtr/searchsorted."""
    w = math.sqrt(rho) * z_np[cell_np] + math.sqrt(1.0 - rho) * eps_np
    return (w > ndtri(1.0 - p)).astype(np.int64)


@pytest.mark.parametrize("rho", [0.0, RHO, 0.5])
def test_matches_calibration_sampler(rho):
    """Same draws in, same outcomes out.

    The sampler's RNG contract (randn(n_cells) then randn(N), float64) is
    reproduced here by seeding torch identically and issuing the same two
    calls, so the numpy reference sees exactly the sampler's own z and eps.
    """
    n_cells, size = 400, 5
    n = n_cells * size
    cell_id = cells_of_size(n_cells, size)
    rng = np.random.default_rng(0)
    p = rng.uniform(0.03, 0.95, n)
    proba = binary_proba(p)

    th.manual_seed(SEED)
    z = th.randn(n_cells, dtype=th.float64)
    eps = th.randn(n, dtype=th.float64)
    ref = reference_levels(p, cell_id.numpy(), z.numpy(), eps.numpy(), rho)

    th.manual_seed(SEED)
    levels, z_cell = sample_correlated_levels(proba, cell_id, rho)
    assert np.array_equal(levels.numpy(), ref)
    assert np.array_equal(z_cell.numpy(), z.numpy())


def test_matches_calibration_sampler_with_ar1():
    """Same check through the AR(1) mixing z = phi z_prev + sqrt(1-phi^2) z."""
    n_cells, size = 400, 5
    n = n_cells * size
    cell_id = cells_of_size(n_cells, size)
    rng = np.random.default_rng(1)
    p = rng.uniform(0.03, 0.95, n)
    proba = binary_proba(p)
    z_prev = th.from_numpy(rng.standard_normal(n_cells))

    th.manual_seed(SEED)
    z_new = th.randn(n_cells, dtype=th.float64)
    eps = th.randn(n, dtype=th.float64)
    z_ref = PHI * z_prev.numpy() + math.sqrt(1.0 - PHI * PHI) * z_new.numpy()
    ref = reference_levels(p, cell_id.numpy(), z_ref, eps.numpy(), RHO)

    th.manual_seed(SEED)
    levels, z_cell = sample_correlated_levels(
        proba, cell_id, RHO, z_prev=z_prev, phi=PHI
    )
    assert np.array_equal(levels.numpy(), ref)
    assert np.array_equal(z_cell.numpy(), z_ref)


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rho", [-0.01, 1.0, 1.5])
def test_rejects_out_of_range_rho(rho):
    proba = binary_proba([0.3] * 4)
    with pytest.raises(AssertionError, match="rho"):
        sample_correlated_levels(proba, cells_of_size(2, 2), rho)


def test_rejects_mismatched_cell_id():
    proba = binary_proba([0.3] * 4)
    with pytest.raises(AssertionError, match="cell_id"):
        sample_correlated_levels(proba, cells_of_size(2, 3), 0.3)


def test_rejects_negative_cell_id():
    proba = binary_proba([0.3] * 4)
    with pytest.raises(AssertionError, match="cell_id"):
        sample_correlated_levels(proba, th.tensor([-1, 0, 1, 1]), 0.3)
