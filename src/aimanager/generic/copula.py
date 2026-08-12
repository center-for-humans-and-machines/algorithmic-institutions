"""Exchangeable Gaussian copula over categorical rows (pure torch, no PyG).

Mirrors ``LinearAHAdapter._sample_levels_copula`` so the GNN and the linear
baselines sample the same way; conventions and rationale:
notes/autoresearch_log/punisher-severity-copula.md (appendix).
"""

import math

import torch as th


def _first_member(cells_flat, n_rows, device):
    """Row index of the first occurrence of each cell id, indexed by the cell's
    position in ``th.unique``. Sort-based because ``scatter_reduce`` with amin
    is unavailable on the pinned torch 1.11 and duplicate-index ``index_copy_``
    is nondeterministic on CUDA."""
    _, inverse = th.unique(cells_flat, return_inverse=True)
    idx = th.arange(n_rows, device=device)
    # unique key sorting lexicographically by (cell, row): no tie to break
    order = th.argsort(inverse * n_rows + idx)
    inv_sorted = inverse[order]
    is_first = th.ones_like(inv_sorted, dtype=th.bool)
    is_first[1:] = inv_sorted[1:] != inv_sorted[:-1]
    first = th.zeros(int(is_first.sum()), dtype=th.int64, device=device)
    first[inv_sorted[is_first]] = order[is_first]
    return first[inverse]


def sample_levels_copula(proba, cells, rho):
    """Sample one level per row, rows of a cell sharing a Gaussian latent:
    ``u_i = Phi(sqrt(rho) z_c(i) + sqrt(1-rho) eps_i)``, inverted through the
    row's own CDF as ``min{a : F_i(a) >= u_i}``. Marginals are therefore the
    caller's ``proba`` exactly, whatever rho -- only the dependence changes.

    ``proba`` is (..., K) with rows summing to 1 (caller-guaranteed), ``cells``
    is int64 of shape ``proba.shape[:-1]``, ``rho`` lies in [0, 1). Returns
    int64 levels in [0, K-1] of shape ``cells.shape``.

    Conventions that the code alone cannot show:
    - exactly 2N draws per call (``zs`` then ``eps``, float64, global torch
      RNG), whatever the cell composition, so the RNG stream is
      composition-stable and rho == 0.0 reduces to independent sampling
      through this same path rather than a shorter one;
    - a cell's latent is the ``zs`` slot of its first member in flattened row
      order, matching the adapter's first-member rule;
    - Phi is ``0.5 (1 + erf(x / sqrt 2))``, bitwise equal to the adapter's
      ``th.special.ndtr`` in float64 on the tested platforms;
    - the final clamp guards u == 1.0 and cumsum overshoot, not a modelling
      choice.
    """
    assert 0.0 <= rho < 1.0, f"rho must lie in [0, 1), got {rho}"
    assert (
        cells.shape == proba.shape[:-1]
    ), f"cells {tuple(cells.shape)} does not match proba {tuple(proba.shape)}"

    device = proba.device
    n_levels = proba.shape[-1]
    flat = proba.reshape(-1, n_levels).double()
    n_rows = flat.shape[0]

    zs = th.randn(n_rows, dtype=th.float64, device=device)
    eps = th.randn(n_rows, dtype=th.float64, device=device)
    pick = _first_member(cells.reshape(-1), n_rows, device)

    x = math.sqrt(rho) * zs[pick] + math.sqrt(1.0 - rho) * eps
    u = 0.5 * (1.0 + th.erf(x / math.sqrt(2.0)))
    cum = flat.cumsum(-1).contiguous()
    lvl = th.searchsorted(cum, u.reshape(-1, 1).contiguous()).reshape(-1)
    return lvl.clamp(0, n_levels - 1).reshape(cells.shape)
