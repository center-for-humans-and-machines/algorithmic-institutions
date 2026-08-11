"""Calibrate the severity-copula correlation `rho` for the multinomial
punisher baseline (autoresearch: punisher-severity-copula, steps 2-7).

Why a copula at all
-------------------
A group's punishments in a round are ONE human manager's joint decision, but
the simulation samples every agent's punishment independently, which pins the
group-spread ratio (metric row PD) to the independence floor ~0.58 while humans
sit at ~0.739. A Gaussian copula keeps each agent's fitted marginal exactly and
adds a shared round-level severity latent: one standard normal per
(episode, round, agent_group) cell, mixed with per-agent noise at weight
`rho`, pushed through each agent's own predicted multinomial CDF.

This script estimates `rho` from the human TRAIN split only (the locked
holdout stays closed) and writes it into a copy of the marginal bundle. It
never touches the marginal model.

Randomized PIT (why, not mid-point)
-----------------------------------
`rho` is the correlation of the LATENT normals that the sampler inverts, so it
must be estimated on latents obtained by the exact inverse of that
construction. Punishment is extremely lumpy (a large mass at 0 and at a few
round numbers), so mid-point PIT maps every tied observation to the same u and
collapses the within-cell spread of z, attenuating any correlation estimate
towards 0. The randomized PIT

    u_i = F_i(y_i - 1) + v_i * p_i(y_i),   v_i ~ U(0, 1),   z_i = Phi^-1(u_i)

is, under the fitted marginal, exactly uniform on (0, 1) -- so z is exactly
standard normal and the estimated correlation is on the same scale as the rho
the sampler consumes. The estimate is averaged over R = 20 PIT replicates with
a fixed seed to average out the auxiliary randomisation; the mid-point PIT
value is printed as a sensitivity number only.

The rho estimator
-----------------
Exchangeable-correlation moment estimator over within-cell pairs. Cells are
(episode, round, agent_group) -- one manager decision per cell (2256/2256 rows
carry a constant `manager_no_input` in the human data). With zbar and var(z)
taken over ALL selected observations (every `punishment_valid` cell, including
cells of size 1, which contribute no pair), and var the sample variance
(ddof=1):

    rho_hat = mean over all within-cell pairs (i, j), i < j, of
              (z_i - zbar) (z_j - zbar) / var(z)

i.e. every pair gets equal weight (large cells therefore contribute more
pairs). Two cross-checks are printed: the same quantity averaged with equal
weight PER CELL, and the one-way random-effects ICC(1) (ANOVA form, unequal
cluster sizes). Uncertainty is a cluster bootstrap over the 40 train episodes
(1000 resamples), crossed with the PIT replicates: each resample's point
estimate is itself the mean over the 20 replicates.

Everything else printed (round splits, cell-size splits, the test-split rho) is
DIAGNOSTIC ONLY and never a selection criterion. `--preflight` is a go/no-go
mechanism check: it replays the human P matrices through the independent and
the copula sampler and prints the group-spread ratios against the human one.
rho is NEVER tuned to the pre-flight.

Runs locally (CPU torch, no PyG):
    .venv/bin/python scripts/baselines/punishment_copula_rho.py [--preflight]
"""

import argparse
import copy
import os
import random
import sys
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
N_PIT = 20  # PIT replicates averaged over
N_BOOT = 1000  # cluster bootstrap resamples (over episodes)
N_PREFLIGHT = 50  # sampling repeats in the pre-flight
U_EPS = 1e-12  # PIT clip, keeps Phi^-1 finite

try:  # scipy is present locally; torch 1.11 has the same primitives
    from scipy.special import ndtr, ndtri
except ImportError:  # pragma: no cover - fallback path

    def ndtri(u):
        return th.special.ndtri(th.as_tensor(u, dtype=th.float64)).numpy()

    def ndtr(x):
        return th.special.ndtr(th.as_tensor(x, dtype=th.float64)).numpy()


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
    """The class-probability matrix EXACTLY as LinearAHAdapter._sample_levels
    builds it: 1e-12 floor, predict_proba scattered onto `classes_`, optional
    temperature (this bundle has none -> 1.0, a no-op), row renormalised."""
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
# PIT
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


# --------------------------------------------------------------------------- #
# rho estimators
# --------------------------------------------------------------------------- #
def blocks(cell):
    """Sort rows by cell; return (order, starts, sizes) with each cell a
    contiguous block (what np.add.reduceat needs)."""
    order = np.argsort(cell, kind="stable")
    k = cell[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    sizes = np.diff(np.r_[starts, len(k)])
    return order, starts, sizes


def _cell_moments(Zc, starts, sizes):
    """Per-cell (sum of pair products, pair count) for every column of Zc."""
    S = np.add.reduceat(Zc, starts, axis=0)
    Q = np.add.reduceat(Zc**2, starts, axis=0)
    keep = sizes >= 2
    pair_prod = (S[keep] ** 2 - Q[keep]) / 2.0  # [m_keep, n_rep]
    n_c = sizes[keep]
    return pair_prod, n_c * (n_c - 1) // 2


def rho_pairs(Z, starts, sizes):
    """Exchangeable moment estimator, equal weight per PAIR (the estimator).
    Returns (rho per replicate, total pairs)."""
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


def rho_on_subset(Z, cell, sel):
    """(mean rho, n rows, n cells with >=2, n pairs) on a row subset."""
    order, starts, sizes = blocks(cell[sel])
    Zs = Z[sel][order]
    rho, pairs = rho_pairs(Zs, starts, sizes)
    return float(rho.mean()), int(sel.sum()), int((sizes >= 2).sum()), pairs


# --------------------------------------------------------------------------- #
# cluster bootstrap over episodes, crossed with the PIT replicates
# --------------------------------------------------------------------------- #
def cluster_bootstrap(Z, cell, episode, n_boot, seed):
    order, starts, sizes = blocks(cell)
    Zs, eps = Z[order], episode[order]
    ep_ids = np.unique(eps)
    # episodes are contiguous in `order` (cell ids are episode-major), so both
    # the row block and the relative cell layout can be precomputed per episode
    per_ep = {}
    cell_start_set = set(starts.tolist())
    for e in ep_ids:
        idx = np.flatnonzero(eps == e)
        rel = np.array(
            [i for i, r in enumerate(idx) if r in cell_start_set], dtype=np.int64
        )
        siz = np.diff(np.r_[rel, len(idx)])
        per_ep[int(e)] = (idx, rel, siz)

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
        rows = np.concatenate(rows)
        st = np.concatenate(st)
        sz = np.concatenate(sz)
        rho, _ = rho_pairs(Zs[rows], st, sz)
        out[b] = rho.mean()
    return out


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
    cum = Pt.cumsum(1).contiguous()
    _, inv = np.unique(cell, return_inverse=True)
    inv_t = th.from_numpy(inv.astype(np.int64))
    n_cells = int(inv.max()) + 1
    a, b = float(np.sqrt(rho)), float(np.sqrt(1.0 - rho))
    ind, cop = [], []
    for _ in range(n_rep):
        lvl = th.multinomial(Pt, 1).reshape(-1).numpy()
        ind.append(spread_ratio(lvl, cell))
        zs = th.randn(n_cells, dtype=th.float64)
        eps = th.randn(len(y), dtype=th.float64)
        u = th.from_numpy(ndtr((a * zs[inv_t] + b * eps).numpy()))
        lvl = th.searchsorted(cum, u.reshape(-1, 1)).reshape(-1)
        lvl = lvl.clamp(0, P.shape[1] - 1).numpy()
        cop.append(spread_ratio(lvl, cell))
    return float(np.mean(ind)), float(np.mean(cop)), spread_ratio(y, cell)


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #
def save_bundle(bundle, X, rho, se, ci, data_file, n_pairs):
    import joblib

    new = dict(bundle)
    new.update(
        copula_rho=float(rho),
        copula_rho_se=float(se),
        copula_rho_ci=(float(ci[0]), float(ci[1])),
        copula_pit="randomized",
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
    assert set(back) - set(bundle) == {
        "copula_rho",
        "copula_rho_se",
        "copula_rho_ci",
        "copula_pit",
        "copula_cell_key",
        "copula_data_file",
        "copula_n_pairs",
    }
    print(f"\nsaved {OUT.relative_to(ROOT)}")
    print("  predict_proba on the first 100 rows: bit-identical after reload")


# --------------------------------------------------------------------------- #
def main():
    import joblib

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--preflight",
        action="store_true",
        help="replay the human P matrices through the independent and the "
        "copula sampler and print the group-spread ratios (go/no-go only)",
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
    print(f"  episodes={n_ep} agents={n_agents} rounds={n_rounds}")
    print(
        f"  rows={len(tr['y'])} cells={len(sizes)} "
        f"cells>=2={(sizes >= 2).sum()} "
        f"pairs={int((sizes[sizes >= 2] * (sizes[sizes >= 2] - 1) // 2).sum())}"
    )
    print(f"  cell size histogram={np.bincount(sizes).tolist()}")

    Z, z_mid = latents(P, tr["y"], N_PIT, SEED)
    Zo = Z[order]
    rho_reps, n_pairs = rho_pairs(Zo, starts, sizes)
    rho_hat = float(rho_reps.mean())
    cells_reps = rho_cells(Zo, starts, sizes)
    icc_reps = icc_oneway(Zo, starts, sizes)
    mid_rho, _ = rho_pairs(z_mid[order][:, None], starts, sizes)

    print("\n=== rho (train split, randomized PIT, R=%d) ===" % N_PIT)
    print(f"rho_hat (pair-weighted)      {f(rho_hat)}")
    print(f"  per-replicate min/max      {f(rho_reps.min())} / {f(rho_reps.max())}")
    print(f"  replicate sd               {f(rho_reps.std(ddof=1))}")
    print(f"cross-check per-cell weight  {f(cells_reps.mean())}")
    print(f"cross-check ICC(1) one-way   {f(icc_reps.mean())}")
    print(f"sensitivity mid-point PIT    {f(mid_rho[0])}")
    print(f"n_pairs                      {n_pairs}")

    boot = cluster_bootstrap(Zo, tr["cell"][order], tr["episode"][order], N_BOOT, SEED)
    se = float(boot.std(ddof=1))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    print(
        f"cluster bootstrap ({N_BOOT} resamples over {n_ep} episodes, "
        f"crossed with the {N_PIT} PIT replicates)"
    )
    print(f"  SE                         {f(se)}")
    print(f"  95% percentile CI          [{f(ci[0])}, {f(ci[1])}]")

    print("\n=== diagnostics only (never a selection criterion) ===")
    rounds = tr["round"]
    sub = rho_on_subset(Z, tr["cell"], rounds > 0)
    print(
        f"excluding round 0            rho={f(sub[0])} "
        f"rows={sub[1]} cells>=2={sub[2]} pairs={sub[3]}"
    )
    edges = np.array_split(np.arange(n_rounds), 3)
    for third in edges:
        sel = np.isin(rounds, third)
        sub = rho_on_subset(Z, tr["cell"], sel)
        print(
            f"rounds {third[0]:>2}-{third[-1]:<2}                 "
            f"rho={f(sub[0])} rows={sub[1]} cells>=2={sub[2]} pairs={sub[3]}"
        )
    size_of_cell = dict(zip(tr["cell"][order][starts], sizes))
    row_size = np.array([size_of_cell[c] for c in tr["cell"]])
    for s in sorted(set(sizes.tolist())):
        if s < 2:
            continue
        sub = rho_on_subset(Z, tr["cell"], row_size == s)
        print(
            f"cell size {s}                  rho={f(sub[0])} "
            f"rows={sub[1]} cells>=2={sub[2]} pairs={sub[3]}"
        )

    cfg_te = copy.deepcopy(cfg)
    cfg_te["data"]["data_file"] = train_file.replace("_train", "_test")
    te = build_rows(cfg_te, features)
    _, P_te = class_probs(bundle, te["X"])
    Z_te, _ = latents(P_te, te["y"], N_PIT, SEED)
    o_te, s_te, sz_te = blocks(te["cell"])
    rho_te, pairs_te = rho_pairs(Z_te[o_te], s_te, sz_te)
    print(
        f"\nOUT-OF-SAMPLE CHECK ONLY ({cfg_te['data']['data_file']}, "
        f"episodes={te['shape'][0]})"
    )
    print(
        f"  rho={f(rho_te.mean())} rows={len(te['y'])} "
        f"cells>=2={(sz_te >= 2).sum()} pairs={pairs_te}"
    )

    if args.preflight:
        ind, cop, human = preflight(P, tr["y"], tr["cell"], rho_hat, N_PREFLIGHT, SEED)
        print(
            f"\n=== pre-flight (go/no-go; rho is NEVER tuned to it), "
            f"{N_PREFLIGHT} repeats ==="
        )
        print(f"group-spread ratio independent  {f(ind)}")
        print(f"group-spread ratio copula       {f(cop)}  (rho={f(rho_hat)})")
        print(f"group-spread ratio human        {f(human)}")

    save_bundle(bundle, tr["X"], rho_hat, se, ci, train_file, n_pairs)


if __name__ == "__main__":
    main()
