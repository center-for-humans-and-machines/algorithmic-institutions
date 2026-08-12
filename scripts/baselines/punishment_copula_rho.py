"""Calibrate `copula_rho` for the multinomial punisher's severity copula.

Estimates the within-(episode, round, group) latent correlation of human
punishments by pairwise-likelihood MLE of an exchangeable Gaussian copula
(each observation keeps its own fitted discrete marginal), on the bundle's
own train split only, and saves the marginal bundle plus `copula_rho` as
artifacts/baselines/punishment_multinomial_severity_copula.joblib. The
randomized-PIT moment estimator is printed as an attenuated diagnostic only.

Method details, formulas, and the estimator revision history:
notes/autoresearch_log/punisher-severity-copula.md (appendix).

Runs locally (CPU torch, no PyG):
    .venv/bin/python scripts/baselines/punishment_copula_rho.py \
        [--preflight] [--roundtrip]
"""

import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import torch as th  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

from handcrafted_grid import (  # noqa: E402
    build_feature_pool,
    load_config,
    load_episodes,
    validate_feature_legality,
)

BUNDLE = ROOT / "artifacts/baselines/punishment_multinomial_best_with_contr.joblib"
OUT = ROOT / "artifacts/baselines/punishment_multinomial_severity_copula.joblib"
MASK = "punishment_valid"
SEED = 38381  # the marginal model's training seed
N_PIT = 20  # PIT replicates averaged over (diagnostic only)
N_BOOT_PIT = 1000  # cluster bootstrap resamples for the PIT diagnostic
N_BOOT = 200  # cluster bootstrap resamples for the MLE (cost-bounded)
N_PREFLIGHT = 50  # sampling repeats in the pre-flight
N_ROUNDTRIP = 3  # synthetic datasets per rho_true in the recovery table
N_QUAD = 32  # Gauss-Legendre nodes for Phi_2
U_EPS = 1e-12  # CDF clip, keeps Phi^-1 finite (matches the 1e-12 P floor)
P_TINY = 1e-300  # log floor for a rectangle probability
RHO_GRID = np.arange(0.0, 0.9001, 0.05)
RHO_MAX = 0.95  # quadrature stays accurate well inside |rho| = 1

try:  # scipy is present locally; torch 1.11 has the same primitives
    from scipy.special import ndtr, ndtri
except ImportError:  # pragma: no cover - fallback path

    def ndtri(u):
        return th.special.ndtri(th.as_tensor(u, dtype=th.float64)).numpy()

    def ndtr(x):
        return th.special.ndtr(th.as_tensor(x, dtype=th.float64)).numpy()


GL_X, GL_W = np.polynomial.legendre.leggauss(N_QUAD)


def f(x):
    """Unrounded float for the log."""
    return repr(float(x))


# --------------------------------------------------------------------------- #
# data: rebuild the bundle's features, keeping the (episode, agent, round,
# group) index that prepare_data flattens away
# --------------------------------------------------------------------------- #
def build_rows(cfg, features):
    """Feature matrix + target + cell indices for every `punishment_valid`
    observation, built with the SAME utilities the marginal fit used."""
    from aimanager.generic.data import create_torch_data

    validate_feature_legality(cfg)
    th.random.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    df = load_episodes(cfg, ROOT)
    switch_every = cfg["data"].get("switch_every")
    data, _, _ = create_torch_data(df, switch_every=switch_every)
    pool = build_feature_pool(data, switch_every)

    mask = data[MASK].numpy().astype(bool)  # [G, A, T]
    g, a, t = np.nonzero(mask)
    X = np.column_stack([pool[k][mask] for k in features])
    y = data["punishment"].numpy()[mask].astype(np.int64)
    grp = data["agent_group"].numpy()[mask].astype(np.int64)
    n_ep, n_agents, n_rounds = mask.shape
    cell = (g.astype(np.int64) * n_rounds + t) * 2 + grp
    return dict(
        X=X,
        y=y,
        episode=g.astype(np.int64),
        agent=a.astype(np.int64),
        round=t.astype(np.int64),
        group=grp,
        cell=cell,
        shape=(n_ep, n_agents, n_rounds),
    )


def class_probs(bundle, X):
    """Scaled features + class probabilities, built exactly as the adapter's
    LinearAHAdapter._class_probs so calibration and sampling share one P."""
    est, n_levels = bundle["estimator"], int(bundle["n_levels"])
    Xs = bundle["scaler"].transform(X)
    P = np.full((len(Xs), n_levels), 1e-12)
    P[:, est.classes_] = est.predict_proba(Xs)
    temperature = float(bundle.get("temperature", 1.0))
    if temperature != 1.0:
        P = P ** (1.0 / temperature)
    P /= P.sum(1, keepdims=True)
    return Xs, P


# --------------------------------------------------------------------------- #
# cells and within-cell pairs
# --------------------------------------------------------------------------- #
def blocks(cell):
    """Sort rows by cell; return (order, starts, sizes) with each cell a
    contiguous block (what np.add.reduceat needs)."""
    order = np.argsort(cell, kind="stable")
    k = cell[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    sizes = np.diff(np.r_[starts, len(k)])
    return order, starts, sizes


def pair_index(cell):
    """Row indices (i, j), i < j, of every within-cell pair."""
    order, starts, sizes = blocks(cell)
    ii, jj = [], []
    for s, n in zip(starts, sizes):
        if n < 2:
            continue
        idx = order[s : s + n]
        a, b = np.triu_indices(n, k=1)
        ii.append(idx[a])
        jj.append(idx[b])
    if not ii:
        return np.array([], np.int64), np.array([], np.int64)
    return np.concatenate(ii), np.concatenate(jj)


def cdf_bounds(P, y):
    """(Phi^-1(F(y-1)), Phi^-1(F(y))) per observation, clipped off 0/1."""
    cdf = np.cumsum(P, axis=1)
    rows = np.arange(len(y))
    hi = cdf[rows, y]
    lo = hi - P[rows, y]
    return (
        ndtri(np.clip(lo, U_EPS, 1.0 - U_EPS)),
        ndtri(np.clip(hi, U_EPS, 1.0 - U_EPS)),
    )


# --------------------------------------------------------------------------- #
# bivariate normal CDF (Drezner-Wesolowsky, Gauss-Legendre, vectorised)
# --------------------------------------------------------------------------- #
def bvn_cdf(h, k, rho):
    """Phi_2(h, k; rho) elementwise over the arrays h, k for a scalar rho."""
    base = ndtr(h) * ndtr(k)
    if rho == 0.0:
        return base
    r = 0.5 * rho * (GL_X + 1.0)  # [Q] nodes on (0, rho)
    om = 1.0 - r**2
    hh, kk = h[:, None], k[:, None]
    dens = np.exp(-(hh**2 - 2.0 * r * hh * kk + kk**2) / (2.0 * om)) / np.sqrt(om)
    return base + (0.5 * rho / (2.0 * np.pi)) * (dens @ GL_W)


def check_bvn(n=400, seed=SEED):
    """Max abs deviation of bvn_cdf from scipy's mvn CDF on random points."""
    try:
        from scipy.stats import multivariate_normal as mvn
    except ImportError:  # pragma: no cover
        return None
    rng = np.random.default_rng(seed)
    h = rng.uniform(-4.0, 4.0, n)
    k = rng.uniform(-4.0, 4.0, n)
    worst = 0.0
    for rho in (0.05, 0.2, 0.5, 0.8, 0.9):
        mine = bvn_cdf(h, k, rho)
        cov = [[1.0, rho], [rho, 1.0]]
        ref = np.array(
            [mvn.cdf([hi, ki], mean=[0.0, 0.0], cov=cov) for hi, ki in zip(h, k)]
        )
        worst = max(worst, float(np.abs(mine - ref).max()))
    return worst


# --------------------------------------------------------------------------- #
# the estimator: pairwise-likelihood MLE
# --------------------------------------------------------------------------- #
def rect_points(z_lo, z_hi, ii, jj):
    """Stack the four rectangle corners of every pair into one point list."""
    H = np.concatenate([z_hi[ii], z_lo[ii], z_hi[ii], z_lo[ii]])
    K = np.concatenate([z_hi[jj], z_hi[jj], z_lo[jj], z_lo[jj]])
    sgn = np.array([1.0, -1.0, -1.0, 1.0])
    return H, K, sgn


def pair_nll(H, K, sgn, rho):
    """Negative summed pairwise log-likelihood at `rho`."""
    p = (bvn_cdf(H, K, rho).reshape(4, -1) * sgn[:, None]).sum(0)
    return -np.log(np.maximum(p, P_TINY)).sum()


def rho_mle(H, K, sgn, grid=RHO_GRID):
    """Maximise the pairwise log-likelihood: coarse grid, then bounded Brent.
    Returns (rho_hat, nll(rho_hat), nll(0), n_evals)."""
    from scipy.optimize import minimize_scalar

    evals = [0]

    def nll(rho):
        evals[0] += 1
        return pair_nll(H, K, sgn, float(np.clip(rho, 0.0, RHO_MAX)))

    vals = [nll(r) for r in grid]
    b = int(np.argmin(vals))
    lo = float(grid[max(b - 1, 0)])
    hi = float(min(grid[min(b + 1, len(grid) - 1)], RHO_MAX))
    res = minimize_scalar(
        nll, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-6)
    )
    nll0 = vals[0] if grid[0] == 0.0 else nll(0.0)
    return float(res.x), float(res.fun), float(nll0), evals[0]


def mle_on_rows(P, y, cell, sel=None):
    """(rho_hat, n_pairs, n_rows, n_cells>=2) on a row subset."""
    if sel is None:
        sel = np.ones(len(y), bool)
    Pi, yi, ci = P[sel], y[sel], cell[sel]
    z_lo, z_hi = cdf_bounds(Pi, yi)
    ii, jj = pair_index(ci)
    H, K, sgn = rect_points(z_lo, z_hi, ii, jj)
    rho, _, _, _ = rho_mle(H, K, sgn)
    _, _, sizes = blocks(ci)
    return rho, len(ii), int(sel.sum()), int((sizes >= 2).sum())


# --------------------------------------------------------------------------- #
# attenuated diagnostic: randomized-PIT moment estimator
# --------------------------------------------------------------------------- #
def pit_parts(P, y):
    """(F(y-1), p(y)) for the observed levels."""
    cdf = np.cumsum(P, axis=1)
    rows = np.arange(len(y))
    p_y = P[rows, y]
    f_lo = np.where(y > 0, cdf[rows, np.maximum(y - 1, 0)], 0.0)
    return f_lo, p_y


def latents(P, y, n_rep, seed):
    """[n, n_rep] randomized-PIT latents plus the mid-point latent [n]."""
    f_lo, p_y = pit_parts(P, y)
    rng = np.random.default_rng(seed)
    v = rng.random((len(y), n_rep))
    u = np.clip(f_lo[:, None] + v * p_y[:, None], U_EPS, 1.0 - U_EPS)
    u_mid = np.clip(f_lo + 0.5 * p_y, U_EPS, 1.0 - U_EPS)
    return ndtri(u), ndtri(u_mid)


def _cell_moments(Zc, starts, sizes):
    """Per-cell (sum of pair products, pair count) for every column of Zc."""
    S = np.add.reduceat(Zc, starts, axis=0)
    Q = np.add.reduceat(Zc**2, starts, axis=0)
    keep = sizes >= 2
    pair_prod = (S[keep] ** 2 - Q[keep]) / 2.0  # [m_keep, n_rep]
    n_c = sizes[keep]
    return pair_prod, n_c * (n_c - 1) // 2


def rho_pairs(Z, starts, sizes):
    """Exchangeable moment estimator, equal weight per pair; returns
    (rho per replicate, total pairs)."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    var = Zc.var(axis=0, ddof=1)
    pair_prod, pairs = _cell_moments(Zc, starts, sizes)
    return pair_prod.sum(axis=0) / (pairs.sum() * var), int(pairs.sum())


def rho_cells(Z, starts, sizes):
    """Cross-check: equal weight per CELL."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    var = Zc.var(axis=0, ddof=1)
    pair_prod, pairs = _cell_moments(Zc, starts, sizes)
    return (pair_prod / pairs[:, None] / var).mean(axis=0)


def icc_oneway(Z, starts, sizes):
    """Cross-check: one-way random-effects ICC(1), ANOVA form, unequal sizes."""
    n_tot, m = len(Z), len(sizes)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    S = np.add.reduceat(Zc, starts, axis=0)
    cell_mean = S / sizes[:, None]
    ss_b = (sizes[:, None] * cell_mean**2).sum(axis=0)
    ss_t = (Zc**2).sum(axis=0)
    ms_b = ss_b / (m - 1)
    ms_w = (ss_t - ss_b) / (n_tot - m)
    k0 = (n_tot - (sizes.astype(float) ** 2).sum() / n_tot) / (m - 1)
    return (ms_b - ms_w) / (ms_b + (k0 - 1) * ms_w)


def pit_on_subset(Z, cell, sel):
    """(mean rho, n rows, n cells with >=2, n pairs) on a row subset."""
    order, starts, sizes = blocks(cell[sel])
    rho, pairs = rho_pairs(Z[sel][order], starts, sizes)
    return float(rho.mean()), int(sel.sum()), int((sizes >= 2).sum()), pairs


# --------------------------------------------------------------------------- #
# cluster bootstraps over episodes
# --------------------------------------------------------------------------- #
def _per_episode_pairs(episode, ii, jj):
    """{episode: (pair positions,)} -- pairs sit inside one cell, so inside one
    episode; resampling episodes therefore resamples whole pair blocks."""
    ep_of_pair = episode[ii]
    assert np.array_equal(ep_of_pair, episode[jj]), "pair crosses episodes"
    return {int(e): np.flatnonzero(ep_of_pair == e) for e in np.unique(episode)}


def bootstrap_mle(P, y, cell, episode, n_boot, seed, rho_hat):
    """Cluster bootstrap of the pairwise MLE over episodes (percentile CI).
    The refinement grid is narrowed around the full-sample estimate to keep the
    cost bounded; the bracket is still wide enough to contain every resample."""
    z_lo, z_hi = cdf_bounds(P, y)
    ii, jj = pair_index(cell)
    H, K, sgn = rect_points(z_lo, z_hi, ii, jj)
    n_pairs = len(ii)
    per_ep = _per_episode_pairs(episode, ii, jj)
    ep_ids = np.array(sorted(per_ep))
    grid = np.clip(np.arange(rho_hat - 0.2, rho_hat + 0.2001, 0.05), 0.0, RHO_MAX)
    grid = np.unique(np.round(grid, 6))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(ep_ids, size=len(ep_ids), replace=True)
        pos = np.concatenate([per_ep[int(e)] for e in draw])
        idx = np.concatenate([pos + t * n_pairs for t in range(4)])
        out[b] = rho_mle(H[idx], K[idx], sgn, grid=grid)[0]
    return out


def bootstrap_pit(Z, cell, episode, n_boot, seed):
    """Cluster bootstrap of the PIT moment estimator, crossed with the PIT
    replicates: every resample's point estimate is the mean over replicates."""
    order, starts, sizes = blocks(cell)
    Zs, eps = Z[order], episode[order]
    ep_ids = np.unique(eps)
    per_ep, cell_start_set = {}, set(starts.tolist())
    for e in ep_ids:
        idx = np.flatnonzero(eps == e)
        rel = np.array(
            [i for i, r in enumerate(idx) if r in cell_start_set], dtype=np.int64
        )
        per_ep[int(e)] = (idx, rel, np.diff(np.r_[rel, len(idx)]))

    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(ep_ids, size=len(ep_ids), replace=True)
        rows, st, sz, off = [], [], [], 0
        for e in draw:
            idx, rel, siz = per_ep[int(e)]
            rows.append(idx)
            st.append(rel + off)
            sz.append(siz)
            off += len(idx)
        rho, _ = rho_pairs(
            Zs[np.concatenate(rows)], np.concatenate(st), np.concatenate(sz)
        )
        out[b] = rho.mean()
    return out


# --------------------------------------------------------------------------- #
# copula sampling (shared by the round-trip gate and the pre-flight)
# --------------------------------------------------------------------------- #
def sample_copula(P, inv, n_cells, rho):
    """One shared latent per cell, per-agent noise at weight rho, inverted
    through each row's own CDF -- the construction the adapter will use."""
    cum = th.from_numpy(P).cumsum(1).contiguous()
    a, b = float(np.sqrt(rho)), float(np.sqrt(1.0 - rho))
    zs = th.randn(n_cells, dtype=th.float64).numpy()
    eps = th.randn(len(P), dtype=th.float64).numpy()
    u = th.from_numpy(ndtr(a * zs[inv] + b * eps))
    lvl = th.searchsorted(cum, u.reshape(-1, 1)).reshape(-1)
    return lvl.clamp(0, P.shape[1] - 1).numpy()


def roundtrip(P, cell, rho_trues, n_rep, seed):
    """Acceptance gate: generate data at a known rho, re-estimate, report."""
    _, inv = np.unique(cell, return_inverse=True)
    n_cells = int(inv.max()) + 1
    order, starts, sizes = blocks(cell)
    rows = []
    for rho_true in rho_trues:
        th.manual_seed(seed)
        mles, pits = [], []
        for _ in range(n_rep):
            y = sample_copula(P, inv, n_cells, rho_true)
            mles.append(mle_on_rows(P, y, cell)[0])
            Z, _ = latents(P, y, 5, seed)
            pits.append(float(rho_pairs(Z[order], starts, sizes)[0].mean()))
        rows.append((rho_true, float(np.mean(mles)), float(np.mean(pits))))
    return rows


# --------------------------------------------------------------------------- #
# pre-flight: independent vs copula sampling of the human P matrices
# --------------------------------------------------------------------------- #
def spread_ratio(values, cell):
    """std(cell mean) / std(individual), both ddof=1 -- the PD statistic as
    src/aimanager/evaluation_suite/metrics.py._spread_ratio computes it."""
    _, inv = np.unique(cell, return_inverse=True)
    counts = np.bincount(inv)
    means = np.bincount(inv, weights=values.astype(float)) / counts
    return float(np.std(means, ddof=1) / np.std(values.astype(float), ddof=1))


def preflight(P, y, cell, rho, n_rep, seed):
    th.manual_seed(seed)
    Pt = th.from_numpy(P)
    _, inv = np.unique(cell, return_inverse=True)
    n_cells = int(inv.max()) + 1
    ind, cop = [], []
    for _ in range(n_rep):
        lvl = th.multinomial(Pt, 1).reshape(-1).numpy()
        ind.append(spread_ratio(lvl, cell))
        cop.append(spread_ratio(sample_copula(P, inv, n_cells, rho), cell))
    return float(np.mean(ind)), float(np.mean(cop)), spread_ratio(y, cell)


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #
NEW_KEYS = {
    "copula_rho",
    "copula_rho_se",
    "copula_rho_ci",
    "copula_estimator",
    "copula_diag_pit",
    "copula_cell_key",
    "copula_data_file",
    "copula_n_pairs",
}


def save_bundle(bundle, X, rho, se, ci, data_file, n_pairs):
    import joblib

    new = dict(bundle)
    new.update(
        copula_rho=float(rho),
        copula_rho_se=float(se),
        copula_rho_ci=(float(ci[0]), float(ci[1])),
        copula_estimator="pairwise_mle",
        copula_diag_pit="randomized",  # provenance of the diagnostic only
        copula_cell_key="episode_round_group",
        copula_data_file=str(data_file),
        copula_n_pairs=int(n_pairs),
    )
    for k, v in bundle.items():
        assert new[k] is v, f"pre-existing bundle key modified: {k}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(new, OUT)

    ref = bundle["estimator"].predict_proba(bundle["scaler"].transform(X[:100]))
    back = joblib.load(OUT)
    got = back["estimator"].predict_proba(back["scaler"].transform(X[:100]))
    assert np.array_equal(ref, got), "reloaded estimator is not bit-identical"
    assert set(back) - set(bundle) == NEW_KEYS
    print(f"\nsaved {OUT.relative_to(ROOT)}")
    print(f"  copula_rho={f(new['copula_rho'])} estimator=pairwise_mle")
    print("  predict_proba on the first 100 rows: bit-identical after reload")


# --------------------------------------------------------------------------- #
def main():
    import joblib

    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--preflight",
        action="store_true",
        help="replay the human P matrices through the independent and the "
        "copula sampler and print the group-spread ratios (go/no-go only)",
    )
    ap.add_argument(
        "--roundtrip",
        action="store_true",
        help="acceptance gate: recover a known rho from synthetic copula data",
    )
    args = ap.parse_args()

    bundle = joblib.load(BUNDLE)
    features = list(bundle["features"])
    cfg = load_config(ROOT / bundle["config"])
    train_file = cfg["data"]["data_file"]
    print(f"bundle    {BUNDLE.relative_to(ROOT)}")
    print(
        f"  model={bundle['model']} target={bundle['target']} "
        f"n_levels={bundle['n_levels']} features={features}"
    )
    print(
        f"  config={bundle['config']}  temperature="
        f"{bundle.get('temperature', 'absent -> 1.0 (no-op)')}"
    )
    print(
        f"data      {train_file} (exclude_flipped="
        f"{cfg['data'].get('exclude_flipped')})"
    )

    tr = build_rows(cfg, features)
    _, P = class_probs(bundle, tr["X"])
    order, starts, sizes = blocks(tr["cell"])
    n_ep, n_agents, n_rounds = tr["shape"]
    ii, jj = pair_index(tr["cell"])
    print(f"  episodes={n_ep} agents={n_agents} rounds={n_rounds}")
    print(
        f"  rows={len(tr['y'])} cells={len(sizes)} "
        f"cells>=2={(sizes >= 2).sum()} pairs={len(ii)}"
    )
    print(f"  cell size histogram={np.bincount(sizes).tolist()}")
    err = check_bvn()
    print(
        f"  Phi_2 quadrature nodes={N_QUAD}, max abs err vs scipy mvn="
        f"{'unchecked' if err is None else f(err)}"
    )

    # ---------------- the estimator: pairwise-likelihood MLE ---------------- #
    z_lo, z_hi = cdf_bounds(P, tr["y"])
    H, K, sgn = rect_points(z_lo, z_hi, ii, jj)
    t_mle = time.time()
    rho_hat, nll_hat, nll0, n_ev = rho_mle(H, K, sgn)
    t_mle = time.time() - t_mle
    print("\n=== rho: pairwise-likelihood MLE (train split) ===")
    print(f"rho_hat                      {f(rho_hat)}")
    print(f"  pairwise nll(rho_hat)      {f(nll_hat)}")
    print(f"  pairwise nll(0)            {f(nll0)}")
    print(
        f"  2*(nll(0) - nll(rho_hat))  {f(2.0 * (nll0 - nll_hat))}  "
        f"(pairwise LR, not chi2-calibrated)"
    )
    print(f"  n_pairs={len(ii)}  nll evals={n_ev}  fit {t_mle:.2f}s")

    t_bs = time.time()
    boot = bootstrap_mle(P, tr["y"], tr["cell"], tr["episode"], N_BOOT, SEED, rho_hat)
    se = float(boot.std(ddof=1))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    print(f"cluster bootstrap ({N_BOOT} resamples over {n_ep} episodes)")
    print(f"  SE                         {f(se)}")
    print(f"  95% percentile CI          [{f(ci[0])}, {f(ci[1])}]")
    print(
        f"  bootstrap min/max          {f(boot.min())} / {f(boot.max())}"
        f"   [{time.time() - t_bs:.1f}s]"
    )

    # ---------------- acceptance gate ---------------- #
    if args.roundtrip:
        t_rt = time.time()
        rows = roundtrip(P, tr["cell"], (0.1, 0.2, 0.3, 0.4, 0.5), N_ROUNDTRIP, SEED)
        print(
            f"\n=== ACCEPTANCE GATE: round-trip recovery "
            f"({N_ROUNDTRIP} synthetic datasets per rho) ==="
        )
        print(
            "rho_true   rho_hat MLE            bias                   "
            "rho_hat PIT (attenuated)"
        )
        for rho_true, mle, pit in rows:
            print(f"{rho_true:<10.2f} {f(mle):<22} {f(mle - rho_true):<22} {f(pit)}")
        bias = max(abs(m - r) for r, m, _ in rows)
        print(
            f"max |bias| = {f(bias)}  -> "
            f"{'PASS' if bias <= 0.03 else 'FAIL'} (tolerance 0.03)"
            f"   [{time.time() - t_rt:.1f}s]"
        )

    # ---------------- attenuated diagnostic: randomized PIT ---------------- #
    Z, z_mid = latents(P, tr["y"], N_PIT, SEED)
    Zo = Z[order]
    rho_reps, _ = rho_pairs(Zo, starts, sizes)
    mid_rho, _ = rho_pairs(z_mid[order][:, None], starts, sizes)
    print(
        f"\n=== ATTENUATED DIAGNOSTIC: randomized-PIT moment estimator "
        f"(R={N_PIT}) ==="
    )
    print(f"rho_pit (pair-weighted)      {f(rho_reps.mean())}")
    print(f"  per-replicate min/max      {f(rho_reps.min())} / {f(rho_reps.max())}")
    print(f"  replicate sd               {f(rho_reps.std(ddof=1))}")
    print(f"cross-check per-cell weight  {f(rho_cells(Zo, starts, sizes).mean())}")
    print(f"cross-check ICC(1) one-way   {f(icc_oneway(Zo, starts, sizes).mean())}")
    print(f"sensitivity mid-point PIT    {f(mid_rho[0])}")
    boot_pit = bootstrap_pit(
        Zo, tr["cell"][order], tr["episode"][order], N_BOOT_PIT, SEED
    )
    print(
        f"cluster bootstrap ({N_BOOT_PIT} resamples, crossed with the "
        f"{N_PIT} PIT replicates)"
    )
    print(f"  SE                         {f(boot_pit.std(ddof=1))}")
    print(
        f"  95% percentile CI          [{f(np.percentile(boot_pit, 2.5))}, "
        f"{f(np.percentile(boot_pit, 97.5))}]"
    )

    # ---------------- diagnostic splits ---------------- #
    print("\n=== diagnostic splits only (never a selection criterion) ===")
    print("split                        rho MLE                 rho PIT (att.)")
    rounds = tr["round"]

    def split(label, sel):
        mle, n_pr, n_rows, n_cells = mle_on_rows(P, tr["y"], tr["cell"], sel)
        pit = pit_on_subset(Z, tr["cell"], sel)[0]
        print(
            f"{label:<28} {f(mle):<23} {f(pit):<23} "
            f"rows={n_rows} cells>=2={n_cells} pairs={n_pr}"
        )

    split("excluding round 0", rounds > 0)
    for third in np.array_split(np.arange(n_rounds), 3):
        split(f"rounds {third[0]}-{third[-1]}", np.isin(rounds, third))
    size_of_cell = dict(zip(tr["cell"][order][starts], sizes))
    row_size = np.array([size_of_cell[c] for c in tr["cell"]])
    for s in sorted(set(sizes.tolist())):
        if s >= 2:
            split(f"cell size {s}", row_size == s)

    # ---------------- out-of-sample check ---------------- #
    cfg_te = copy.deepcopy(cfg)
    cfg_te["data"]["data_file"] = train_file.replace("_train", "_test")
    te = build_rows(cfg_te, features)
    _, P_te = class_probs(bundle, te["X"])
    mle_te, pr_te, rows_te, cells_te = mle_on_rows(P_te, te["y"], te["cell"])
    Z_te, _ = latents(P_te, te["y"], N_PIT, SEED)
    o_te, s_te, sz_te = blocks(te["cell"])
    pit_te, _ = rho_pairs(Z_te[o_te], s_te, sz_te)
    print(
        f"\nOUT-OF-SAMPLE CHECK ONLY ({cfg_te['data']['data_file']}, "
        f"episodes={te['shape'][0]})"
    )
    print(
        f"  rho MLE={f(mle_te)}  rho PIT={f(pit_te.mean())}  "
        f"rows={rows_te} cells>=2={cells_te} pairs={pr_te}"
    )

    # ---------------- pre-flight ---------------- #
    if args.preflight:
        ind, cop, human = preflight(P, tr["y"], tr["cell"], rho_hat, N_PREFLIGHT, SEED)
        print(
            f"\n=== pre-flight (go/no-go; rho is NEVER tuned to it), "
            f"{N_PREFLIGHT} repeats ==="
        )
        print(f"group-spread ratio independent  {f(ind)}")
        print(f"group-spread ratio copula       {f(cop)}  (rho={f(rho_hat)})")
        print(f"group-spread ratio human        {f(human)}")

    save_bundle(bundle, tr["X"], rho_hat, se, ci, train_file, len(ii))
    print(f"total runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
