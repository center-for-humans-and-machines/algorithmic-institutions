"""Gaussian-copula sampling of categorical outcomes with a shared per-cell
latent -- the herding sampler for the GNN switch predictor.

Rows sharing a ``cell_id`` (one (episode, round, group) in the simulation) mix
one shared standard normal at weight ``sqrt(rho)`` with fresh per-row noise at
``sqrt(1 - rho)``, and the resulting uniform is inverted through each row's own
predicted CDF -- so every row keeps its marginal exactly while the outcomes
co-move. ``phi`` carries the cell latent across decision rounds as an AR(1).

Torch only (no torch_geometric / torch_scatter / scipy) so it imports and is
testable on macOS. Conventions, calibration and the estimator that produced
rho / phi: ``notes/autoresearch_log/switch-herding-copula.md``.
"""

import math

import torch as th

_SQRT2 = math.sqrt(2.0)


def ndtr(x):
    """Standard-normal CDF Phi(x), via ``torch.erf`` (no scipy dependency)."""
    return 0.5 * (1.0 + th.erf(x / _SQRT2))


def levels_from_u(proba, u):
    """Inverse-CDF lookup: ``level_i = min{a : F_i(a) >= u_i}``.

    ``th.searchsorted`` with the default ``right=False`` (numpy's
    ``side="left"``) returns exactly that index, which is the convention of
    the calibration sampler: for two levels, ``proba = [1 - p, p]`` gives
    level 1 iff ``u > 1 - p``, i.e. iff ``w > t = Phi^-1(1 - p)`` -- switching
    is the UPPER tail of the latent, and ``u == 1 - p`` exactly falls to
    level 0 (the same strict inequality the estimator was fitted under).

    The row cumsum is promoted to ``u``'s dtype so the float64 latent algebra
    is not truncated by a float32 ``proba``. The clamp only guards the
    ``u == 1.0`` corner (``ndtr`` saturates for ``w`` beyond ~8.3) against a
    cumsum whose last entry rounds below 1.
    """
    n_levels = proba.shape[-1]
    cum = proba.cumsum(-1).to(u.dtype)
    lvl = th.searchsorted(cum.contiguous(), u.reshape(-1, 1).contiguous())
    return lvl.reshape(-1).clamp(0, n_levels - 1).to(th.int64)


def sample_correlated_levels(proba, cell_id, rho, z_prev=None, phi=0.0):
    """Draw one level per row of ``proba`` with a shared latent per cell.

    Args:
        proba: float tensor ``(N, L)``, each row a categorical distribution.
        cell_id: int tensor ``(N,)`` of DENSE cell ids ``0..n_cells - 1``;
            rows sharing an id share one latent.
        rho: latent weight in ``[0, 1)``. ``0.0`` reduces to independent
            inverse-CDF sampling (marginals unchanged either way).
        z_prev: previous cell latents ``(n_cells,)`` or ``None``.
        phi: AR(1) persistence in ``[0, 1]``; ignored when ``z_prev is None``.
            ``1.0`` is the unit-root boundary: ``z_cell = z_prev`` exactly, a
            static latent for the whole span (see the phi ruling in
            ``notes/autoresearch_log/contribution-herding-copula-v2.md``).

    Returns:
        ``(levels, z_cell)`` -- ``levels`` int64 ``(N,)``, ``z_cell`` float64
        ``(n_cells,)``. The caller holds ``z_cell`` and feeds it back as
        ``z_prev`` on the next decision round.

    RNG contract: exactly two draws from torch's global RNG per call, always
    in this order and always these shapes -- ``randn(n_cells)`` then
    ``randn(N)``, both float64, with ``n_cells = int(cell_id.max()) + 1``.
    Neither ``rho``, ``phi``, ``z_prev`` nor the cell composition changes what
    is consumed, so two calls with equal ``N`` and ``n_cells`` leave the
    global RNG in the same state (the simulation seeds torch globally; there
    is no generator argument, mirroring the punisher precedent's
    ``_sample_levels_copula``).
    """
    assert proba.dim() == 2, f"proba must be (N, L), got {tuple(proba.shape)}"
    n, n_levels = proba.shape
    assert n_levels >= 2, f"need at least 2 levels, got {n_levels}"
    cell_id = cell_id.reshape(-1).to(th.int64)
    assert len(cell_id) == n, f"cell_id has {len(cell_id)} entries for {n} rows"
    assert 0.0 <= rho < 1.0, f"rho must lie in [0, 1), got {rho}"
    assert 0.0 <= phi <= 1.0, f"phi must lie in [0, 1], got {phi}"

    # Dense ids: a negative id would wrap the latent gather silently, and a
    # gap would misalign `z_prev` on the next round.
    assert n == 0 or int(cell_id.min().item()) >= 0, "cell_id must be >= 0"
    n_cells = int(cell_id.max().item()) + 1 if n else 0
    z_new = th.randn(n_cells, dtype=th.float64)  # draw 1/2
    eps = th.randn(n, dtype=th.float64)  # draw 2/2

    if z_prev is None:
        z_cell = z_new
    else:
        z_prev = z_prev.reshape(-1).to(th.float64)
        assert len(z_prev) == n_cells, (
            f"z_prev has {len(z_prev)} entries for {n_cells} cells -- the cell "
            f"index must be stable across the rounds an AR(1) latent spans"
        )
        # Var(z_cell) stays 1, so the marginals do not depend on phi. At the
        # boundary phi == 1.0 the innovation weight is sqrt(0) == 0 exactly,
        # so z_cell is z_prev bit-for-bit (the static-latent case).
        z_cell = phi * z_prev + math.sqrt(1.0 - phi * phi) * z_new

    w = math.sqrt(rho) * z_cell[cell_id] + math.sqrt(1.0 - rho) * eps
    return levels_from_u(proba, ndtr(w)), z_cell
