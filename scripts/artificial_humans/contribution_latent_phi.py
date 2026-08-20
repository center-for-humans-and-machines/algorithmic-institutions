"""Persistence diagnostic for the reference GNN contributor's group residuals.

Autoresearch step 1 of `contribution-group-latent`: how much of the within-group
residual dependence that PR #149 measured *within* a round survives a change of
round? A within-round shock cannot produce cross-round correlation; an
episode-persistent shared group factor must.

Same artifact, same split, same teacher-forced marginals and the same
randomized-PIT normal scores as #149's copula calibration -- the data helpers
come from `contribution_copula_rho.py` and the estimators from
`punishment_copula_rho.py`, both imported unmodified. Three correlations, each
computed on the normal-scores residuals *and* re-estimated by #149's
pairwise-likelihood MLE so the numbers sit on its published scale:

    rho   same round, same group, different agent      (#149's estimand)
    phi   different round, same group, different agent (the persistent share)
    psi   different round, same agent                  (individual persistence)

`phi / rho` is the share of the same-round group dependence that survives a
change of round: the episode-persistent group factor the planned latent carries.

Note: `contribution_copula_rho.main()` is not runnable on this branch (the GNN
copula wiring lives only on `auto/contribution-cg-copula`); only its data
loading and teacher-forcing helpers are used here.

Imports graph.py, so this runs on Raven only:
    .venv/bin/python scripts/artificial_humans/contribution_latent_phi.py \
        --model IN.pt --out OUT.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(ROOT / "scripts" / "artificial_humans"))

# #149's machinery, imported unmodified: both modules' top level is imports,
# constants and defs only, so importing them runs no calibration. Importing
# `contribution_copula_rho` also installs the torch_geometric.nn.meta alias the
# legacy artifact pickle needs and pulls in GraphNetwork (PyG -> Raven only).
import punishment_copula_rho as pc  # noqa: E402
import contribution_copula_rho as cr  # noqa: E402

from aimanager.generic.graph import GraphNetwork  # noqa: E402

f = pc.f
SEED = pc.SEED  # 38381
N_PIT = pc.N_PIT  # 20 randomized-PIT replicates, as #149
N_BOOT = 1000  # episode-level bootstrap resamples (>= 1000 required)
# The pairwise MLE bootstrap costs ~0.3s per likelihood evaluation on the
# ~2.7e5 cross-round pairs, so phi's MLE CI uses #149's own N_BOOT (200-300)
# while the normal-scores CIs -- the ones the step-1 gate reads -- get 1000.
N_BOOT_MLE_BIG = 300
BIG_PAIRS = 50_000  # pair count above which N_BOOT_MLE_BIG applies
LAGS = (1, 2, 3, 4, 5)
N_ROUNDTRIP = 3  # synthetic datasets per acceptance-gate row
# (phi_true, rho_true): the first row is the false-positive check -- a pure
# within-round shock at #149's measured rho, with no persistent component.
ROUNDTRIP_CONFIGS = ((0.0, 0.07), (0.02, 0.07), (0.05, 0.07), (0.07, 0.07))
GATE_PHI = 0.03  # step-1 gate: phi point estimate must reach this
REF_RHO_MLE_149 = 0.06958238086256316  # PR #149 arm A, same artifact/split
REF_RHO_PIT_149 = 0.0457  # #149's attenuated PIT diagnostic (rounded in log)


# --------------------------------------------------------------------------- #
# pair construction: unordered row pairs inside blocks of a key
# --------------------------------------------------------------------------- #
def build_pairs(block, cond=None):
    """Unordered row pairs (i, j) inside every block of equal `block` id.

    `cond(ia, ib)` is an optional boolean mask over the candidate pairs. Blocks
    here are at most one episode-group (~100 rows), so the dense triu per block
    is cheap.
    """
    order, starts, sizes = pc.blocks(block)
    ii, jj = [], []
    for s, n in zip(starts, sizes):
        if n < 2:
            continue
        idx = order[s : s + n]
        a, b = np.triu_indices(n, k=1)
        ia, ib = idx[a], idx[b]
        if cond is not None:
            keep = cond(ia, ib)
            ia, ib = ia[keep], ib[keep]
        if len(ia):
            ii.append(ia)
            jj.append(ib)
    if not ii:
        return np.array([], np.int64), np.array([], np.int64)
    return np.concatenate(ii), np.concatenate(jj)


def pair_sets(tr):
    """The three pair families, plus the per-pair round lag."""
    n_agents = tr["shape"][1]
    ep, ag, rd, gr = tr["episode"], tr["agent"], tr["round"], tr["group"]
    cell = tr["cell"]  # (episode, round, agent_group) -- #149's key
    ep_group = ep * cr.N_GROUPS + gr
    ep_agent = ep * n_agents + ag

    rho_ii, rho_jj = build_pairs(cell)
    phi_ii, phi_jj = build_pairs(
        ep_group, lambda a, b: (ag[a] != ag[b]) & (rd[a] != rd[b])
    )
    psi_ii, psi_jj = build_pairs(ep_agent, lambda a, b: rd[a] != rd[b])

    # #149 built the same-round pairs with pair_index; prove parity so rho here
    # is literally its estimand and not a re-specification.
    ref_ii, ref_jj = pc.pair_index(cell)
    n_rows = len(ep)
    mine = np.sort(rho_ii * n_rows + rho_jj)
    ref = np.sort(ref_ii * n_rows + ref_jj)
    assert np.array_equal(mine, ref), "same-round pairs differ from #149's"

    return dict(
        rho=(rho_ii, rho_jj),
        phi=(phi_ii, phi_jj),
        psi=(psi_ii, psi_jj),
        phi_lag=np.abs(rd[phi_ii] - rd[phi_jj]),
        psi_lag=np.abs(rd[psi_ii] - rd[psi_jj]),
    )


# --------------------------------------------------------------------------- #
# estimator 1: correlation of the normal-scores residuals
# --------------------------------------------------------------------------- #
def corr_pairs(Z, ii, jj):
    """Pair-weighted correlation over an arbitrary pair list, one value per PIT
    replicate. This is `pc.rho_pairs`' convention -- global centring, global
    ddof=1 variance, equal weight per pair -- lifted off the within-cell blocks
    so cross-round pairs can use it."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    var = Zc.var(axis=0, ddof=1)
    return (Zc[ii] * Zc[jj]).sum(axis=0) / (len(ii) * var)


def _pairs_by_episode(episode, ii, jj):
    """{episode: pair positions}. Every pair family here is built inside one
    episode, so resampling episodes resamples whole pair blocks."""
    assert np.array_equal(episode[ii], episode[jj]), "pair crosses episodes"
    ep_of_pair = episode[ii]
    return {int(e): np.flatnonzero(ep_of_pair == e) for e in np.unique(episode)}


def bootstrap_corr(Z, ii, jj, episode, n_boot, seed):
    """Episode-level cluster bootstrap of `corr_pairs`, crossed with the PIT
    replicates as `pc.bootstrap_pit` does: each resample re-centres and
    re-scales on its own rows, and its point estimate is the replicate mean."""
    per_pair = _pairs_by_episode(episode, ii, jj)
    per_row = {int(e): np.flatnonzero(episode == e) for e in np.unique(episode)}
    ep_ids = np.array(sorted(per_pair))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(ep_ids, size=len(ep_ids), replace=True)
        rows = np.concatenate([per_row[int(e)] for e in draw])
        pos = np.concatenate([per_pair[int(e)] for e in draw])
        Zr = Z[rows]
        mu = Zr.mean(axis=0, keepdims=True)
        var = Zr.var(axis=0, ddof=1)
        num = ((Z[ii[pos]] - mu) * (Z[jj[pos]] - mu)).sum(axis=0)
        out[b] = float((num / (len(pos) * var)).mean())
    return out


# --------------------------------------------------------------------------- #
# estimator 2: #149's pairwise-likelihood MLE on an arbitrary pair list
# --------------------------------------------------------------------------- #
def mle_pairs(z_lo, z_hi, ii, jj, grid=pc.RHO_GRID):
    """(rho_hat, nll(rho_hat), nll(0), n_evals) -- `pc.mle_on_rows` with the
    pair list supplied instead of derived from a cell key."""
    H, K, sgn = pc.rect_points(z_lo, z_hi, ii, jj)
    return pc.rho_mle(H, K, sgn, grid=grid)


def bootstrap_mle_pairs(z_lo, z_hi, ii, jj, episode, n_boot, seed, rho_hat):
    """`pc.bootstrap_mle` with the pair list supplied: same episode clustering,
    same narrowed refinement grid around the full-sample estimate."""
    H, K, sgn = pc.rect_points(z_lo, z_hi, ii, jj)
    n_pairs = len(ii)
    per_ep = _pairs_by_episode(episode, ii, jj)
    ep_ids = np.array(sorted(per_ep))
    grid = np.clip(np.arange(rho_hat - 0.2, rho_hat + 0.2001, 0.05), 0.0, pc.RHO_MAX)
    grid = np.unique(np.round(grid, 6))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(ep_ids, size=len(ep_ids), replace=True)
        pos = np.concatenate([per_ep[int(e)] for e in draw])
        idx = np.concatenate([pos + t * n_pairs for t in range(4)])
        out[b] = pc.rho_mle(H[idx], K[idx], sgn, grid=grid)[0]
    assert (
        out.max() < grid[-1] - 1e-9 or grid[-1] >= pc.RHO_MAX
    ), "a resample hit the refinement bracket's upper edge: widen the grid"
    return out


# --------------------------------------------------------------------------- #
# acceptance gate: recover known (phi, rho) from synthetic two-level data
# --------------------------------------------------------------------------- #
def synth_levels(P, cum, inv_group, inv_cell, phi, rho, rng):
    """Levels drawn from each row's OWN predicted marginal under a two-level
    Gaussian latent: a persistent per-(episode, agent_group) component of
    variance phi, a per-(episode, round, agent_group) shock of variance
    rho - phi, and idiosyncratic noise -- so the same-round correlation is rho
    and the cross-round group-mate correlation is exactly phi. Inversion
    matches pc.sample_copula (searchsorted on the cumsum, clamped)."""
    assert 0.0 <= phi <= rho < 1.0, f"need 0 <= phi <= rho < 1, got {phi}, {rho}"
    zg = rng.standard_normal(int(inv_group.max()) + 1)
    zc = rng.standard_normal(int(inv_cell.max()) + 1)
    lat = (
        np.sqrt(phi) * zg[inv_group]
        + np.sqrt(rho - phi) * zc[inv_cell]
        + np.sqrt(1.0 - rho) * rng.standard_normal(len(P))
    )
    u = np.clip(pc.ndtr(lat), pc.U_EPS, 1.0 - pc.U_EPS)
    lvl = np.array(
        [np.searchsorted(cum[i], u[i]) for i in range(len(u))], dtype=np.int64
    )
    return np.clip(lvl, 0, P.shape[1] - 1)


def roundtrip(tr, ps, configs, n_rep, seed):
    """For each (phi_true, rho_true): generate, re-estimate both correlations,
    average over replicates. The phi_true = 0 row is the false-positive check --
    a pure within-round shock must not read as persistence."""
    P = tr["P"]
    cum = np.cumsum(P, axis=1)
    ep_group = tr["episode"] * cr.N_GROUPS + tr["group"]
    _, inv_group = np.unique(ep_group, return_inverse=True)
    _, inv_cell = np.unique(tr["cell"], return_inverse=True)
    rows = []
    for phi_true, rho_true in configs:
        rng = np.random.default_rng(seed)
        acc = {"rho_mle": [], "phi_mle": [], "rho_ns": [], "phi_ns": []}
        for _ in range(n_rep):
            y = synth_levels(P, cum, inv_group, inv_cell, phi_true, rho_true, rng)
            Z, _ = pc.latents(P, y, 5, seed)
            z_lo, z_hi = pc.cdf_bounds(P, y)
            acc["rho_mle"].append(mle_pairs(z_lo, z_hi, *ps["rho"])[0])
            acc["phi_mle"].append(mle_pairs(z_lo, z_hi, *ps["phi"])[0])
            acc["rho_ns"].append(float(corr_pairs(Z, *ps["rho"]).mean()))
            acc["phi_ns"].append(float(corr_pairs(Z, *ps["phi"]).mean()))
        rows.append(
            dict(
                phi_true=phi_true,
                rho_true=rho_true,
                **{k: float(np.mean(v)) for k, v in acc.items()},
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
def ci(boot):
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def point(Z, z_lo, z_hi, ii, jj, label):
    """Both estimators on one pair family, printed and returned."""
    if len(ii) == 0:
        print(f"{label:<28} (no pairs)")
        return dict(n_pairs=0, normal_scores=None, mle=None)
    reps = corr_pairs(Z, ii, jj)
    rho, nll, nll0, _ = mle_pairs(z_lo, z_hi, ii, jj)
    rec = dict(
        n_pairs=int(len(ii)),
        normal_scores=float(reps.mean()),
        normal_scores_rep_sd=float(reps.std(ddof=1)),
        mle=float(rho),
        mle_pairwise_lr=float(2.0 * (nll0 - nll)),
    )
    print(
        f"{label:<28} ns={f(rec['normal_scores']):<23} "
        f"mle={f(rec['mle']):<23} pairs={rec['n_pairs']}"
    )
    return rec


def add_boot(rec, boot, key):
    rec[f"{key}_se"] = float(boot.std(ddof=1))
    rec[f"{key}_ci"] = ci(boot)
    rec[f"{key}_boot_min_max"] = [float(boot.min()), float(boot.max())]
    print(
        f"  {key:<24} SE={f(rec[f'{key}_se'])}  "
        f"95% CI=[{f(rec[f'{key}_ci'][0])}, {f(rec[f'{key}_ci'][1])}]"
    )


# --------------------------------------------------------------------------- #
def analyse(model, data, idx, label, n_boot, split_name, do_roundtrip=False):
    """Every estimate on one split."""
    tr = cr.teacher_forced_rows(model, data, idx)
    P, y = tr["P"], tr["y"]
    n_ep, n_agents, n_rounds = tr["shape"]
    ps = pair_sets(tr)
    print(f"\n=== {label}: episodes={n_ep} agents={n_agents} rounds={n_rounds}")
    print(f"  rows={len(y)}  levels={P.shape[1]}")
    switchers = sum(
        len(set(tr["group"][(tr["episode"] == e) & (tr["agent"] == a)].tolist())) > 1
        for e in range(n_ep)
        for a in range(n_agents)
    )
    print(f"  agent-episodes changing agent_group: {switchers}/{n_ep * n_agents}")

    Z, _ = pc.latents(P, y, N_PIT, SEED)
    z_lo, z_hi = pc.cdf_bounds(P, y)

    print("\n  family                     normal-scores           pairwise MLE")
    out = {}
    out["rho_same_round"] = point(Z, z_lo, z_hi, *ps["rho"], "rho same-round grp-mate")
    out["phi_cross_round"] = point(
        Z, z_lo, z_hi, *ps["phi"], "phi cross-round grp-mate"
    )
    out["psi_same_agent"] = point(
        Z, z_lo, z_hi, *ps["psi"], "psi cross-round same-agent"
    )

    # phi by round lag
    out["phi_by_lag"] = {}
    for lag in LAGS:
        sel = ps["phi_lag"] == lag
        out["phi_by_lag"][str(lag)] = point(
            Z, z_lo, z_hi, ps["phi"][0][sel], ps["phi"][1][sel], f"  phi lag {lag}"
        )
    sel = ps["phi_lag"] > max(LAGS)
    out["phi_by_lag"][f"{max(LAGS) + 1}+"] = point(
        Z, z_lo, z_hi, ps["phi"][0][sel], ps["phi"][1][sel], f"  phi lag >{max(LAGS)}"
    )

    # psi by round lag, for the same-agent reference
    out["psi_by_lag"] = {}
    for lag in LAGS:
        sel = ps["psi_lag"] == lag
        out["psi_by_lag"][str(lag)] = point(
            Z, z_lo, z_hi, ps["psi"][0][sel], ps["psi"][1][sel], f"  psi lag {lag}"
        )

    # round thirds: rows restricted to the third, then pairs -- #149's
    # `split("rounds a-b", isin(rounds, third))` convention
    out["by_third"] = {}
    rd = tr["round"]
    for n, third in enumerate(np.array_split(np.arange(n_rounds), 3)):
        tag = f"rounds {third[0]}-{third[-1]}"
        inside = np.isin(rd, third)
        rec = {}
        for name, key in (("rho", "rho"), ("phi", "phi")):
            ii, jj = ps[key]
            sel = inside[ii] & inside[jj]
            rec[name] = point(Z, z_lo, z_hi, ii[sel], jj[sel], f"  {name} {tag}")
        out["by_third"][f"third_{n + 1}_{tag.replace(' ', '_')}"] = rec

    if n_boot:
        print(f"\n  episode-level cluster bootstrap ({n_boot} resamples, {n_ep} eps)")
        for key, name in (("rho", "rho_same_round"), ("phi", "phi_cross_round")):
            ii, jj = ps[key]
            n_mle = n_boot if len(ii) < BIG_PAIRS else min(n_boot, N_BOOT_MLE_BIG)
            out[name]["n_boot_normal_scores"] = n_boot
            out[name]["n_boot_mle"] = n_mle
            t0 = time.time()
            add_boot(
                out[name],
                bootstrap_corr(Z, ii, jj, tr["episode"], n_boot, SEED),
                "normal_scores",
            )
            add_boot(
                out[name],
                bootstrap_mle_pairs(
                    z_lo, z_hi, ii, jj, tr["episode"], n_mle, SEED, out[name]["mle"]
                ),
                "mle",
            )
            print(f"  [{key}: {time.time() - t0:.1f}s, mle resamples={n_mle}]")
        # cheap extra: the same-agent reference on the normal-scores scale only
        add_boot(
            out["psi_same_agent"],
            bootstrap_corr(Z, *ps["psi"], tr["episode"], n_boot, SEED),
            "normal_scores",
        )

    if do_roundtrip:
        t0 = time.time()
        rt = roundtrip(tr, ps, ROUNDTRIP_CONFIGS, N_ROUNDTRIP, SEED)
        out["roundtrip"] = rt
        print(
            f"\n  === ACCEPTANCE GATE: recovery from synthetic two-level data "
            f"({N_ROUNDTRIP} datasets per row) ==="
        )
        print(
            "  phi_true  rho_true  phi_hat MLE            "
            "rho_hat MLE            phi_hat ns"
        )
        for r in rt:
            print(
                f"  {r['phi_true']:<9.2f} {r['rho_true']:<9.2f} "
                f"{f(r['phi_mle']):<22} {f(r['rho_mle']):<22} {f(r['phi_ns'])}"
            )
        null = [r for r in rt if r["phi_true"] == 0.0]
        bias = max(abs(r["phi_mle"] - r["phi_true"]) for r in rt)
        out["roundtrip_max_abs_bias_phi"] = bias
        out["roundtrip_null_phi_mle"] = null[0]["phi_mle"] if null else None
        print(f"  max |bias| in phi = {f(bias)} (tolerance 0.03)")
        if null:
            print(
                f"  false-positive check: phi_true=0 (pure within-round shock) "
                f"-> phi_hat={f(null[0]['phi_mle'])}"
            )
        print(f"  [{time.time() - t0:.1f}s]")

    r, p = out["rho_same_round"], out["phi_cross_round"]
    out["ratios"] = dict(
        phi_over_rho_normal_scores=p["normal_scores"] / r["normal_scores"],
        phi_over_rho_mle=p["mle"] / r["mle"],
    )
    out["split"] = split_name
    out["n_episodes"] = int(n_ep)
    out["n_rows"] = int(len(y))
    print(
        f"\n  phi/rho  normal-scores={f(out['ratios']['phi_over_rho_normal_scores'])}"
        f"  mle={f(out['ratios']['phi_over_rho_mle'])}"
    )
    return out


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="reference contributor .pt")
    ap.add_argument("--out", required=True, help="output JSON")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument(
        "--holdout",
        action="store_true",
        help="also report the test split as an out-of-sample check (no bootstrap)",
    )
    ap.add_argument(
        "--roundtrip",
        action="store_true",
        help="acceptance gate: recover known (phi, rho) from synthetic data",
    )
    args = ap.parse_args()
    in_path = Path(args.model).resolve()
    out_path = Path(args.out).resolve()

    model = GraphNetwork.load(str(in_path), device="cpu")
    model.eval()
    assert model.y_name == "contribution", f"not a contributor: {model.y_name}"
    print(f"model     {cr.rel(in_path)}")
    print(
        f"  y_name={model.y_name} y_levels={model.y_levels} "
        f"autoregressive={model.autoregressive}"
    )
    print(f"  x_encoding={[e['name'] for e in model.x_encoding]}")
    print(f"  edge_encoding={[e['name'] for e in model.edge_encoding]}")

    data, pair_id, key_to_idx, defaults = cr.load_full()
    tr_idx = cr.select_split(key_to_idx, cr.TRAIN, cr.N_TRAIN_EP)
    te_idx = cr.select_split(key_to_idx, cr.TEST, cr.N_TEST_EP)
    assert not set(tr_idx.tolist()) & set(te_idx.tolist()), "train/test overlap"
    assert not set(pair_id[tr_idx].tolist()) & set(
        pair_id[te_idx].tolist()
    ), "a flip copy of a train game sits in the holdout"
    print(f"data      {cr.rel(cr.FULL)} (episodes={len(key_to_idx)})")
    print(f"  train split {cr.rel(cr.TRAIN)}: {len(tr_idx)} episodes")
    print(f"  contribution default={f(defaults['contribution'])}")
    err = pc.check_bvn()
    print(
        f"  Phi_2 quadrature nodes={pc.N_QUAD}, max abs err vs scipy mvn="
        f"{'unchecked' if err is None else f(err)}"
    )

    res = dict(
        train=analyse(
            model,
            data,
            tr_idx,
            "TRAIN SPLIT",
            args.n_boot,
            cr.rel(cr.TRAIN),
            do_roundtrip=args.roundtrip,
        )
    )
    if args.holdout:
        res["holdout_check_only"] = analyse(
            model, data, te_idx, "HOLDOUT (check only)", 0, cr.rel(cr.TEST)
        )

    # ---------------- sanity check against #149 ---------------- #
    rho = res["train"]["rho_same_round"]
    print("\n=== sanity check vs PR #149 (same artifact, same split) ===")
    print(f"rho MLE here                 {f(rho['mle'])}")
    print(f"rho MLE #149 arm A           {f(REF_RHO_MLE_149)}")
    print(f"  abs difference             {f(abs(rho['mle'] - REF_RHO_MLE_149))}")
    print(f"rho normal-scores here       {f(rho['normal_scores'])}")
    print(f"rho PIT #149 (logged)        {f(REF_RHO_PIT_149)}")
    res["sanity_vs_149"] = dict(
        rho_mle_here=rho["mle"],
        rho_mle_149=REF_RHO_MLE_149,
        rho_mle_abs_diff=abs(rho["mle"] - REF_RHO_MLE_149),
        rho_normal_scores_here=rho["normal_scores"],
        rho_pit_149_logged=REF_RHO_PIT_149,
    )

    phi = res["train"]["phi_cross_round"]
    gate = dict(
        threshold=GATE_PHI,
        phi_mle=phi["mle"],
        phi_mle_ci=phi.get("mle_ci"),
        phi_normal_scores=phi["normal_scores"],
        phi_normal_scores_ci=phi.get("normal_scores_ci"),
    )
    res["gate"] = gate
    print("\n=== STEP-1 GATE READOUT (decision belongs to the orchestrator) ===")
    print(f"phi MLE            {f(phi['mle'])}  CI={phi.get('mle_ci')}")
    print(
        f"phi normal-scores  {f(phi['normal_scores'])}  "
        f"CI={phi.get('normal_scores_ci')}"
    )
    print(f"threshold          point estimate >= {GATE_PHI}, CI clearly above 0")

    res["meta"] = dict(
        model=cr.rel(in_path),
        script=cr.rel(__file__),
        seed=SEED,
        n_pit_replicates=N_PIT,
        n_boot=args.n_boot,
        estimators=(
            "normal_scores = pair-weighted correlation of the randomized-PIT "
            "normal scores (attenuated by the PIT randomisation, identical "
            "convention to #149's PIT diagnostic); mle = #149's "
            "pairwise-likelihood MLE of an exchangeable Gaussian copula on the "
            "same pair list"
        ),
        pair_families=(
            "rho: same (episode, round, agent_group), different agent -- #149's "
            "estimand; phi: same (episode, agent_group), different agent, "
            "different round; psi: same (episode, agent), different round"
        ),
        date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {cr.rel(out_path)}")
    print(f"total runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
