"""Estimate the dose of the contribution group copula (autoresearch step 6,
notes/autoresearch_log/contribution-gmlp-group-copula.md).

ONE number sets this experiment's sampler: `rho_total`, the within-(episode,
round, group) latent correlation of the gaussian_mlp_v2 contribution emission,
fitted ONCE on the training split by the repo's interval-censored pairwise
Gaussian-copula MLE (`punishment_copula_rho.rho_mle` / `pair_nll` / `bvn_cdf`
/ `rect_points` / `cdf_bounds`, imported unmodified) against the 21-bin
marginal the sampler actually realises (`gaussian_mlp_preflight.bin_probs`,
tails folded into levels 0 and 20 -- exactly what `clip(rint(mu + sigma z))`
produces). The dose is used as-is:

    rho_p = rho_total   (one persistent latent per (episode, group))
    rho_t = 0

No grid, no tuning, no arm selection, no score anywhere in this file.

Why the censored MLE and not a moment correlation: 21.6 % of the training rows
sit on a censoring bound (0 or 20) and every row is rounded, which attenuates a
moment correlation of standardised residuals toward zero (log Note 7 measures
3-12 %). The rectangle-probability pairwise likelihood is consistent for the
latent correlation under exactly that censoring, and it is the estimator PRs
#149 / #160 / #165 established.

Also computed and printed prominently, but NEVER stamped:

  * the DECLARED FALSIFIER -- the cross-member lag-1 refit (agent i at round t
    against agent j != i in the same group at round t + 1, self-pairs excluded)
    and the implied two-component reading (rho_p_lag1, rho_total - rho_p_lag1).
    Under the Declaration that reading predicts CG ~5.3 and a FAIL, against
    ~2.8-3.4 for the fully persistent reading. It is reported so the verdict
    cannot be re-read after the fact; it does not set the structure (grounds
    1-4 of the Declaration do).
  * ATTENUATED MOMENT DIAGNOSTICS, round-thirds splits and out-of-sample
    test-split MLEs -- never used to choose anything. Log Note 20: the test
    split's lag-1 moment estimate (+0.0514) exceeds its own within-cell value
    (+0.0330) on 10 episodes, which is impossible under the model, so
    test-split numbers are noise indicators, not estimates.

  * a ROUND-TRIP ACCEPTANCE GATE that drives the adapter's OWN sampler
    (`LinearAHAdapter._sample_levels_gaussian_copula`, step 4) over the real
    feature rows at known (rho_p, rho_t) and re-estimates both quantities.
    max |bias| <= 0.02 on each is PASS; anything else is an implementation bug
    and stops the experiment. The `(0.03, 0)` arm is reported separately as
    the POWER TEST of the falsifier estimator at the fitted dose: a purely
    persistent panel with no transient part, so the lag-1 fit *should* recover
    ~0.03. Either outcome is fine and neither changes the dose.

Runs locally (CPU torch, no PyG):
    uv run python scripts/baselines/contribution_gmlp_copula_rho.py \
        [--write-params]
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import torch as th  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

from aimanager.simulation.linear_ah import LinearAHAdapter  # noqa: E402
from gaussian_mlp_preflight import bin_probs  # noqa: E402
from gmlp_group_copula_diagnostic import (  # noqa: E402
    build_rows,
    cluster_ci,
    encode_cell,
    pairs_cross_group,
    pairs_lag1,
    pairs_same_agent,
    pairs_within_cell,
    pooled_corr,
    thirds_bounds,
)
from handcrafted_grid import load_config  # noqa: E402
from punishment_copula_rho import (  # noqa: E402
    N_BOOT,
    RHO_MAX,
    SEED,
    blocks,
    bootstrap_mle,
    cdf_bounds,
    check_bvn,
    icc_oneway,
    pair_index,
    rect_points,
    rho_mle,
)

BUNDLE_PATH = ROOT / "artifacts/baselines/contribution_gaussian_mlp_v2_best.joblib"
TRAIN_CFG = ROOT / "configs/training/baselines/contribution/gaussian_mlp_v2.yml"
OUT_JSON = (
    ROOT / "artifacts/baselines/contribution_gaussian_mlp_v2_group_copula.params.json"
)

K = 21  # contribution levels 0..20 -- the sampler's grid, not bundle n_levels
ESTIMATOR_TAG = "pairwise_mle_censored_gaussian"
STRUCTURE = "persistent_episode_group"

# data-path agreement checks against PR #165, the orchestrator's probe and
# step 2's diagnostic. The within-cell count is asserted; the lag-1 count is
# reported (it is the falsifier's, not the dose's, data path).
REF_N_PAIRS_WITHIN = 15090
REF_N_PAIRS_LAG1 = 28714

# (rho_p, rho_t) -> the round-trip gate. Truth: within-cell latent correlation
# is rho_p + rho_t (u and v are both shared inside a (round, group) cell);
# cross-member lag-1 is rho_p alone (v is redrawn every round).
ROUNDTRIP_ARMS = ((0.1, 0.0), (0.0, 0.1), (0.05, 0.05), (0.03, 0.0))
POWER_ARM = (0.03, 0.0)
RT_TOL = 0.02
# panels per arm; the arm estimate is their mean, so the gate measures
# BIAS and not one panel's Monte Carlo draw (per-panel sd ~0.012-0.018).
N_ROUNDTRIP = 6


def fmt(x):
    """Unrounded float for the log."""
    return repr(float(x))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return None


# --------------------------------------------------------------------------- #
# marginals: the 21-bin law the sampler realises
# --------------------------------------------------------------------------- #
def score_bundle(bundle, rows):
    """Teacher-forced (mu, sigma) of the bundle on ITS OWN 7 features and
    scaler, plus the scaled design matrix the adapter's sampler consumes."""
    X = np.column_stack([rows["pool"][k][rows["mask"]] for k in bundle["features"]])
    Xs = bundle["scaler"].transform(X)
    est = bundle["estimator"]
    mu = np.asarray(est.predict(Xs), float).reshape(-1)
    sigma = np.asarray(est.predict_std(Xs), float).reshape(-1)
    return Xs, mu, sigma


# --------------------------------------------------------------------------- #
# pair sets
# --------------------------------------------------------------------------- #
def within_cell(rows):
    """(cell id per row, i-index, j-index) for every cross-member pair inside
    one (episode, round, group) cell -- the dose's pair set."""
    cell = encode_cell(rows["episode"], rows["round"], rows["group"])
    ii, jj = pair_index(cell)
    return cell, ii, jj


def lag1_cross_pairs(episode, round_, group, agent):
    """THE FALSIFIER'S pair set: agent i at round t against agent j != i in the
    same group at round t + 1.

    Each side is keyed by its OWN group at its own round (so a switcher pairs
    correctly on both sides), and SELF-PAIRS ARE EXCLUDED -- own persistence is
    a different quantity, measured at +0.0662 in step 2, an order of magnitude
    above the cross-member value. Delegates to step 2's `pairs_lag1` so the
    falsifier's MLE and its moment diagnostic are computed on byte-identical
    pair sets (PR #165's `cross_pairs` at lag 1, minus the diagonal).
    """
    return pairs_lag1(episode, round_, group, agent)


# --------------------------------------------------------------------------- #
# the estimator: interval-censored pairwise Gaussian-copula MLE
# --------------------------------------------------------------------------- #
def mle_on_pairs(P, y, ii, jj):
    """(rho_hat, nll(rho_hat), nll(0), n_evals) on an arbitrary pair list."""
    z_lo, z_hi = cdf_bounds(P, y)
    H, Kc, sgn = rect_points(z_lo, z_hi, ii, jj)
    return rho_mle(H, Kc, sgn)


def bootstrap_pairs_mle(P, y, ii, jj, episode, rho_hat, n_boot=N_BOOT, seed=SEED):
    """Cluster bootstrap of the pairwise MLE over EPISODES on an arbitrary pair
    list -- `punishment_copula_rho.bootstrap_mle` for pair sets it cannot build
    itself (the lag-1 falsifier, and the round-trip panels).

    Episodes, not pairs, are the resampling unit: pairs inside one episode are
    dependent, and every pair here sits inside a single episode (asserted). The
    refinement grid is narrowed around the full-sample estimate exactly as
    `bootstrap_mle` narrows it, to keep the cost bounded."""
    z_lo, z_hi = cdf_bounds(P, y)
    H, Kc, sgn = rect_points(z_lo, z_hi, ii, jj)
    n_pairs = len(ii)
    ep_of_pair = episode[ii]
    assert np.array_equal(ep_of_pair, episode[jj]), "pair crosses episodes"
    ep_ids = np.unique(episode)
    per_ep = {int(e): np.flatnonzero(ep_of_pair == e) for e in ep_ids}
    grid = np.clip(np.arange(rho_hat - 0.2, rho_hat + 0.2001, 0.05), 0.0, RHO_MAX)
    grid = np.unique(np.round(grid, 6))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(ep_ids, size=len(ep_ids), replace=True)
        pos = np.concatenate([per_ep[int(e)] for e in draw])
        idx = np.concatenate([pos + t * n_pairs for t in range(4)])
        out[b] = rho_mle(H[idx], Kc[idx], sgn, grid=grid)[0]
    return out


def ci_of(boot):
    """(2.5th, 97.5th percentile, SE) of a bootstrap draw."""
    return (
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
        float(boot.std(ddof=1)),
    )


# --------------------------------------------------------------------------- #
# round-trip: the ADAPTER'S OWN sampler on the real feature rows
# --------------------------------------------------------------------------- #
def episode_round_blocks(rows):
    """[(episode, [(round, row indices), ...]), ...] in ascending round order --
    the order the adapter sees rounds in a real episode, which is what makes
    the persistent latent's lifetime meaningful."""
    out = []
    for ep in np.unique(rows["episode"]):
        sel = np.flatnonzero(rows["episode"] == ep)
        per_round = [
            (int(t), sel[rows["round"][sel] == t])
            for t in np.unique(rows["round"][sel])
        ]
        out.append((int(ep), per_round))
    return out


def synth_panel(bundle, Xs, rows, ep_blocks, rho_p, rho_t, seed=SEED):
    """Synthetic contribution panel drawn by the adapter's own
    `_sample_levels_gaussian_copula` at (rho_p, rho_t), on the REAL feature
    rows: one call per (episode, round) with that round's post-arrival group
    array, `_reset_history()` between episodes so each episode gets fresh
    persistent latents. No reimplementation of the sampler lives in this file.

    The features are held fixed at the human rows, so this validates the
    ESTIMATOR, not the closed-loop dynamics -- contributions are re-drawn but
    never fed back into the next round's features."""
    adapter = LinearAHAdapter(
        dict(bundle, copula_rho_p=float(rho_p), copula_rho_t=float(rho_t)),
        n_contributions=K,
        sample=True,
    )
    th.manual_seed(seed)
    y = np.full(len(Xs), -1, dtype=np.int64)
    for _, per_round in ep_blocks:
        adapter._reset_history()
        for _, idx in per_round:
            y[idx] = adapter._sample_levels_gaussian_copula(
                Xs[idx], K, rows["group"][idx]
            )
    assert (y >= 0).all(), "some row was never sampled"
    return y


def roundtrip(
    bundle,
    Xs,
    rows,
    P,
    mu,
    sigma,
    cell_pairs,
    lag_pairs,
    arms,
    n_rep=None,
    boot_arms=(),
):
    """Drive the adapter's sampler at each (rho_p, rho_t) and re-estimate both
    quantities with the code paths used on the human data.

    `n_rep` independent panels per arm, and the arm's estimate is the MEAN over
    them, because the gate is about BIAS: one panel's own sampling sd is ~0.012
    to 0.018 at these doses (measured), so a single draw conflates estimator
    bias with Monte Carlo noise and could fail a correct implementation.
    `punishment_copula_rho.roundtrip` averages 3 synthetic datasets per rho for
    the same reason. Panels use consecutive seeds from SEED, so every arm sees
    the same RNG streams (common random numbers).

    The attenuated moment share of each panel is printed alongside as a
    diagnostic only -- it calibrates how much the moment estimator on the human
    residuals is pulled toward zero by the censoring and rounding.
    """
    if n_rep is None:
        n_rep = N_ROUNDTRIP
    ii_w, jj_w = cell_pairs
    ii_l, jj_l = lag_pairs
    ep_blocks = episode_round_blocks(rows)
    rows_out = []
    for rho_p, rho_t in arms:
        t0 = time.time()
        tots, lags, mom_w, mom_l, first = [], [], [], [], None
        for rep in range(n_rep):
            y = synth_panel(bundle, Xs, rows, ep_blocks, rho_p, rho_t, SEED + rep)
            if first is None:
                first = y
            fit_w = mle_on_pairs(P, y, ii_w, jj_w)
            fit_l = mle_on_pairs(P, y, ii_l, jj_l)
            tots.append(fit_w[0])
            lags.append(fit_l[0])
            if rep == 0:  # the LR a WELL-SPECIFIED copula produces at this rho
                lr0 = (
                    2.0 * (fit_w[2] - fit_w[1]) / len(ii_w),
                    2.0 * (fit_l[2] - fit_l[1]) / len(ii_l),
                )
            r_syn = (y - mu) / sigma
            mom_w.append(pooled_corr(r_syn, ii_w, jj_w)[0])
            mom_l.append(pooled_corr(r_syn, ii_l, jj_l)[0])
        tot, lag = float(np.mean(tots)), float(np.mean(lags))
        rec = dict(
            panel0_lr_per_pair_total=float(lr0[0]),
            panel0_lr_per_pair_lag1=float(lr0[1]),
            rho_p=float(rho_p),
            rho_t=float(rho_t),
            true_total=float(rho_p + rho_t),
            true_lag1=float(rho_p),
            n_rep=int(n_rep),
            rho_total_hat=tot,
            rho_lag1_hat=lag,
            rho_total_panels=[float(v) for v in tots],
            rho_lag1_panels=[float(v) for v in lags],
            rho_total_sd=float(np.std(tots, ddof=1)),
            rho_lag1_sd=float(np.std(lags, ddof=1)),
            bias_total=float(tot - (rho_p + rho_t)),
            bias_lag1=float(lag - rho_p),
            moment_total=float(np.mean(mom_w)),
            moment_lag1=float(np.mean(mom_l)),
        )
        if (rho_p, rho_t) in boot_arms:
            # cluster bootstrap on the FIRST panel -- the same machinery, on
            # the same resampling unit, as the human-data CIs
            b_t = bootstrap_pairs_mle(P, first, ii_w, jj_w, rows["episode"], tots[0])
            b_l = bootstrap_pairs_mle(P, first, ii_l, jj_l, rows["episode"], lags[0])
            rec["panel0_rho_total"] = float(tots[0])
            rec["panel0_rho_lag1"] = float(lags[0])
            rec["panel0_rho_total_ci"] = list(ci_of(b_t)[:2])
            rec["panel0_rho_lag1_ci"] = list(ci_of(b_l)[:2])
        rec["seconds"] = round(time.time() - t0, 1)
        rows_out.append(rec)
    return rows_out


# --------------------------------------------------------------------------- #
# diagnostics (printed, never used to choose anything)
# --------------------------------------------------------------------------- #
def misspec_diagnostics(P, y, pair_sets):
    """Where the pairwise likelihood's signal actually sits.

    An exchangeable Gaussian copula is one number; if the human dependence is
    not that shape, the fitted rho is a compromise and the MLE and the moment
    estimator can disagree even though both are calibrated on synthetic data
    from the model (the round-trip proves they are). Two reads that localise
    it, both printed and NEITHER used to choose anything:

      * interior-only refit (y not on a censoring bound): if the fitted rho
        moves a lot, the dependence is concentrated in the censored corners
        rather than spread over the interior;
      * corner clustering: the observed rate of a pair sitting on the SAME
        bound (both 0, or both 20) against the rate the row-wise marginals
        imply under independence. A small Gaussian correlation lifts this
        ratio barely above 1, so a large ratio is dependence of a shape the
        copula cannot represent.
    """
    interior = (y != 0) & (y != K - 1)
    out = {}
    for label, (ii, jj) in pair_sets:
        keep = interior[ii] & interior[jj]
        rho_int = mle_on_pairs(P, y, ii[keep], jj[keep])[0]
        corners = {}
        for lvl in (0, K - 1):
            obs = float(((y[ii] == lvl) & (y[jj] == lvl)).mean())
            exp = float((P[ii, lvl] * P[jj, lvl]).mean())
            corners[lvl] = (obs, exp, obs / exp)
        out[label] = dict(rho_interior=float(rho_int), n_interior=int(keep.sum()))
        out[label].update(
            {f"both_{lvl}": corners[lvl] for lvl in corners}  # (obs, exp, ratio)
        )
        print(
            f"  {label:<12} interior-only rho MLE={fmt(rho_int)}  "
            f"(n={int(keep.sum())})"
        )
        for lvl in (0, K - 1):
            obs, exp, ratio = corners[lvl]
            print(
                f"  {'':<12}   both == {lvl:<2}: observed={obs:.5f}  "
                f"independence={exp:.5f}  ratio={ratio:.3f}"
            )
    return out


def moment_diagnostics(r, rows):
    """The attenuated moment shares of step 2, recomputed here so the MLE and
    the moment sit side by side. Every number is biased toward zero by the
    censoring and rounding -- that bias is the reason the MLE sets the dose."""
    ep = rows["episode"]
    out = {}
    for label, (ii, jj) in (
        (
            "within-cell (episode, round, group)",
            pairs_within_cell(ep, rows["round"], rows["group"]),
        ),
        (
            "cross-member lag-1 (t -> t+1)",
            lag1_cross_pairs(ep, rows["round"], rows["group"], rows["agent"]),
        ),
        (
            "same-agent cross-round (own persistence)",
            pairs_same_agent(ep, rows["agent"]),
        ),
        (
            "cross-group same-round (episode shock)",
            pairs_cross_group(ep, rows["round"], rows["group"]),
        ),
    ):
        rho, vals = pooled_corr(r, ii, jj)
        lo, hi, se = cluster_ci(vals, ep[ii])
        out[label] = (rho, len(ii), lo, hi, se)
        print(
            f"  {label:<44} rho={rho:+.5f}  n={len(ii):>6}  "
            f"SE={se:.5f}  95% CI=[{lo:+.5f}, {hi:+.5f}]"
        )
    cell = encode_cell(ep, rows["round"], rows["group"])
    order, starts, sizes = blocks(cell)
    icc = float(icc_oneway(r[order][:, None], starts, sizes)[0])
    print(f"  {'ICC(1) one-way over the same cells':<44} icc={icc:+.5f}")
    return out, icc


def split_mles(P, rows, r, n_rounds):
    """Round-thirds splits of both the censored MLE and the moment share."""
    print("  third            rho MLE                 rho moment      rows   pairs")
    out = []
    for lo3, hi3 in thirds_bounds(n_rounds):
        sel = (rows["round"] >= lo3) & (rows["round"] <= hi3)
        cell_s = encode_cell(
            rows["episode"][sel], rows["round"][sel], rows["group"][sel]
        )
        ii, jj = pair_index(cell_s)
        idx = np.flatnonzero(sel)
        rho = mle_on_pairs(P[sel], rows["y"][sel].astype(np.int64), ii, jj)[0]
        mom = pooled_corr(r, idx[ii], idx[jj])[0]
        print(
            f"  rounds {lo3:>2}-{hi3:<2}      {fmt(rho):<23} {mom:+.5f}  "
            f"{int(sel.sum()):>8} {len(ii):>7}"
        )
        out.append(dict(rounds=[lo3, hi3], rho_mle=float(rho), rho_moment=float(mom)))
    return out


# --------------------------------------------------------------------------- #
def main():
    import joblib

    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--write-params",
        action="store_true",
        help="write the JSON sidecar step 7 stamps onto the bundle",
    )
    args = ap.parse_args()

    bundle = joblib.load(BUNDLE_PATH)
    cfg = load_config(TRAIN_CFG)
    train_file = cfg["data"]["data_file"]
    print("=" * 78)
    print("STEP 6 -- the dose of the contribution group copula (gaussian_mlp_v2)")
    print("=" * 78)
    print(f"bundle    {BUNDLE_PATH.relative_to(ROOT)}")
    print(f"  sha256={sha256(BUNDLE_PATH)}")
    print(f"  model={bundle['model']} target={bundle['target']}")
    print(f"  features={bundle['features']}")
    print(f"config    {TRAIN_CFG.relative_to(ROOT)}")
    print(
        f"data      {train_file} (mask={cfg['data']['mask']}, "
        f"exclude_flipped={cfg['data'].get('exclude_flipped')})"
    )
    print(f"git       {git_sha()}")

    rows = build_rows(cfg)
    Xs, mu, sigma = score_bundle(bundle, rows)
    y = rows["y"].astype(np.int64)
    P = bin_probs(mu, sigma, K)
    r = (rows["y"] - mu) / sigma
    n_ep, n_agents, n_rounds = rows["shape"]
    cell, ii_w, jj_w = within_cell(rows)
    _, _, sizes = blocks(cell)
    ii_l, jj_l = lag1_cross_pairs(
        rows["episode"], rows["round"], rows["group"], rows["agent"]
    )
    censored = float(((y == 0) | (y == K - 1)).mean())
    print(
        f"\nrows={len(y)} episodes={n_ep} agents={n_agents} rounds={n_rounds}  "
        f"cells={len(sizes)} cells>=2={(sizes >= 2).sum()}"
    )
    print(f"  cell size histogram={np.bincount(sizes).tolist()}")
    print(
        f"  censored rows (y in {{0, {K - 1}}}): share={fmt(censored)}  "
        f"at 0={fmt((y == 0).mean())}  at {K - 1}={fmt((y == K - 1).mean())}"
    )
    print(
        f"  standardised residual r=(c-mu)/sigma  mean={fmt(r.mean())}  "
        f"var={fmt(r.var(ddof=1))}"
    )

    err = check_bvn()
    print(
        f"  Phi_2 quadrature max abs err vs scipy mvn = "
        f"{'unchecked' if err is None else fmt(err)}"
    )
    assert err is None or err < 1e-6, f"bivariate normal quadrature is unsound: {err}"

    print(
        f"\n  within-cell pairs = {len(ii_w)} (expected {REF_N_PAIRS_WITHIN}) -- "
        f"{'MATCH' if len(ii_w) == REF_N_PAIRS_WITHIN else 'MISMATCH'}"
    )
    assert len(ii_w) == REF_N_PAIRS_WITHIN, (
        f"within-cell pair count {len(ii_w)} != {REF_N_PAIRS_WITHIN}: the data "
        "path disagrees with PR #165 and step 2 -- STOP, do not use this dose"
    )
    print(
        f"  lag-1 cross-member pairs = {len(ii_l)} (step 2 saw " f"{REF_N_PAIRS_LAG1})"
    )

    # ---------------- (a) THE DOSE ---------------- #
    print("\n" + "=" * 78)
    print("(a) rho_total -- THE DOSE  (interval-censored pairwise MLE, train)")
    print("=" * 78)
    t_mle = time.time()
    rho_total, nll_hat, nll0, n_ev = mle_on_pairs(P, y, ii_w, jj_w)
    t_mle = time.time() - t_mle
    print(f"rho_total                    {fmt(rho_total)}")
    print(f"  pairwise nll(rho_total)    {fmt(nll_hat)}")
    print(f"  pairwise nll(0)            {fmt(nll0)}")
    print(
        f"  2*(nll(0) - nll(rho))      {fmt(2.0 * (nll0 - nll_hat))}  "
        f"(pairwise LR, not chi2-calibrated)"
    )
    print(f"  n_pairs={len(ii_w)}  nll evals={n_ev}  fit {t_mle:.2f}s")
    t_bs = time.time()
    boot_tot = bootstrap_mle(P, y, cell, rows["episode"], N_BOOT, SEED, rho_total)
    lo_t, hi_t, se_t = ci_of(boot_tot)
    print(f"cluster bootstrap ({N_BOOT} resamples over {n_ep} episodes)")
    print(f"  SE                         {fmt(se_t)}")
    print(f"  95% percentile CI          [{fmt(lo_t)}, {fmt(hi_t)}]")
    print(
        f"  bootstrap min/max          {fmt(boot_tot.min())} / "
        f"{fmt(boot_tot.max())}   [{time.time() - t_bs:.1f}s]"
    )

    # ---------------- (b) THE FALSIFIER ---------------- #
    print("\n" + "=" * 78)
    print("(b) rho_lag1 -- THE DECLARED FALSIFIER  (reported, NEVER stamped)")
    print("=" * 78)
    rho_lag1, nll_hat_l, nll0_l, n_ev_l = mle_on_pairs(P, y, ii_l, jj_l)
    print("rho_lag1 (cross-member t -> t+1, self-pairs excluded)")
    print(f"  rho_lag1                   {fmt(rho_lag1)}")
    print(
        f"  2*(nll(0) - nll(rho))      {fmt(2.0 * (nll0_l - nll_hat_l))}  "
        f"(pairwise LR)"
    )
    print(f"  n_pairs={len(ii_l)}  nll evals={n_ev_l}")
    t_bs = time.time()
    boot_lag = bootstrap_pairs_mle(P, y, ii_l, jj_l, rows["episode"], rho_lag1)
    lo_l, hi_l, se_l = ci_of(boot_lag)
    print(f"cluster bootstrap ({N_BOOT} resamples over {n_ep} episodes)")
    print(f"  SE                         {fmt(se_l)}")
    print(f"  95% percentile CI          [{fmt(lo_l)}, {fmt(hi_l)}]")
    print(
        f"  bootstrap min/max          {fmt(boot_lag.min())} / "
        f"{fmt(boot_lag.max())}   [{time.time() - t_bs:.1f}s]"
    )
    print("\nimplied TWO-COMPONENT READING (the falsifier's reading, NOT the dose):")
    print(f"  (rho_p, rho_t) = ({fmt(rho_lag1)}, " f"{fmt(rho_total - rho_lag1)})")
    print(
        "  Declaration: that reading predicts CG ~5.3 and a FAIL; the fully\n"
        "  persistent reading (rho_p = rho_total, rho_t = 0) predicts CG\n"
        "  ~2.8-3.4 and a band upgrade. The measured CG adjudicates; nothing\n"
        "  here selects between them."
    )

    # ---------------- (c) diagnostics ---------------- #
    print("\n" + "=" * 78)
    print("(c) DIAGNOSTICS -- printed, never used to choose anything")
    print("=" * 78)
    print("attenuated moment shares (biased toward zero by censoring/rounding):")
    moments, icc = moment_diagnostics(r, rows)
    print("\nround-thirds splits (censored MLE and moment, train):")
    thirds = split_mles(P, rows, r, n_rounds)

    print(
        "\nmisspecification diagnostics -- where the pairwise likelihood's "
        "signal sits:"
    )
    misspec = misspec_diagnostics(
        P, y, (("within-cell", (ii_w, jj_w)), ("lag-1", (ii_l, jj_l)))
    )

    cfg_te = copy.deepcopy(cfg)
    cfg_te["data"]["data_file"] = train_file.replace("_train", "_test")
    rows_te = build_rows(cfg_te)
    _, mu_te, sigma_te = score_bundle(bundle, rows_te)
    y_te = rows_te["y"].astype(np.int64)
    P_te = bin_probs(mu_te, sigma_te, K)
    r_te = (rows_te["y"] - mu_te) / sigma_te
    cell_te, ii_wte, jj_wte = within_cell(rows_te)
    ii_lte, jj_lte = lag1_cross_pairs(
        rows_te["episode"], rows_te["round"], rows_te["group"], rows_te["agent"]
    )
    rho_w_te = mle_on_pairs(P_te, y_te, ii_wte, jj_wte)[0]
    rho_l_te = mle_on_pairs(P_te, y_te, ii_lte, jj_lte)[0]
    cens_te = float(((y_te == 0) | (y_te == K - 1)).mean())
    print(
        f"\nOUT-OF-SAMPLE CHECK ONLY ({cfg_te['data']['data_file']}, "
        f"{rows_te['shape'][0]} episodes, {len(y_te)} rows, censored share "
        f"{fmt(cens_te)})"
    )
    print(f"  within-cell rho MLE        {fmt(rho_w_te)}  (n={len(ii_wte)})")
    print(f"  lag-1 rho MLE              {fmt(rho_l_te)}  (n={len(ii_lte)})")
    print(
        f"  within-cell moment         {pooled_corr(r_te, ii_wte, jj_wte)[0]:+.5f}\n"
        f"  lag-1 moment               {pooled_corr(r_te, ii_lte, jj_lte)[0]:+.5f}"
    )
    print(
        "  Note 20: on 10 episodes the test split's lag-1 moment exceeded its\n"
        "  own within-cell value, which is impossible under the model -- these\n"
        "  are noise indicators, not estimates. The dose is the train fit."
    )

    # ---------------- (d) round-trip acceptance gate ---------------- #
    print("\n" + "=" * 78)
    print("(d) ROUND-TRIP ACCEPTANCE GATE -- the adapter's OWN sampler")
    print("=" * 78)
    print(
        "Synthetic panels are drawn by "
        "LinearAHAdapter._sample_levels_gaussian_copula\n"
        "(step 4) on the REAL human feature rows: contributions are re-drawn "
        "but never\nfed back, so this validates the ESTIMATOR, not the "
        "closed-loop dynamics."
    )
    ep_blocks = episode_round_blocks(rows)
    print(
        f"Episode structure: {len(ep_blocks)} episodes x up to "
        f"{max(len(b) for _, b in ep_blocks)} rounds, `_reset_history()` at "
        f"every episode\nboundary, so each (episode, group) draws its OWN "
        f"persistent latent -- the same\nstructure the estimator keys cells "
        f"and lag-1 pairs on. Without that reset one\nfrozen latent per group "
        f"id would span the whole panel and no arm would recover."
    )
    rt = roundtrip(
        bundle,
        Xs,
        rows,
        P,
        mu,
        sigma,
        (ii_w, jj_w),
        (ii_l, jj_l),
        ROUNDTRIP_ARMS,
        boot_arms=(POWER_ARM,),
    )
    print(
        f"Each arm is the mean of {N_ROUNDTRIP} independent panels (per-panel "
        f"sd printed):\nthe gate measures the estimator's BIAS, not one "
        f"panel's Monte Carlo draw."
    )
    print(
        "\n  rho_p rho_t | within-cell: true  rho_hat   sd     bias    "
        "moment | lag-1: true  rho_hat   sd     bias    moment"
    )
    for rec in rt:
        print(
            f"  {rec['rho_p']:<5.2f} {rec['rho_t']:<5.2f} |"
            f"             {rec['true_total']:.2f}  {rec['rho_total_hat']:.5f}"
            f"  {rec['rho_total_sd']:.4f} {rec['bias_total']:+.5f}"
            f" {rec['moment_total']:+.4f} |"
            f"        {rec['true_lag1']:.2f}  {rec['rho_lag1_hat']:.5f}"
            f"  {rec['rho_lag1_sd']:.4f} {rec['bias_lag1']:+.5f}"
            f" {rec['moment_lag1']:+.4f}"
        )
    print(
        "  (moment = the attenuated moment share of the same panels, "
        "diagnostic only)"
    )
    print(
        "\n  pairwise LR per pair, human vs the arm whose fitted rho matches "
        "(shape check):"
    )
    lr_w_human = 2.0 * (nll0 - nll_hat) / len(ii_w)
    lr_l_human = 2.0 * (nll0_l - nll_hat_l) / len(ii_l)
    print(
        f"    human   within-cell {lr_w_human:.5f}   lag-1 {lr_l_human:.5f}\n"
        + "\n".join(
            f"    arm ({x['rho_p']}, {x['rho_t']})   within-cell "
            f"{x['panel0_lr_per_pair_total']:.5f}   lag-1 "
            f"{x['panel0_lr_per_pair_lag1']:.5f}"
            for x in rt
        )
    )
    print(
        "    A well-specified copula's LR per pair is set by rho and the\n"
        "    censoring; a human LR far above the arm at the SAME fitted rho\n"
        "    means the dependence is not exchangeable-Gaussian in shape. It\n"
        "    changes nothing about the dose (this file selects nothing), but\n"
        "    it is what step 11 needs to read the falsifier honestly."
    )

    max_bias = max(max(abs(x["bias_total"]), abs(x["bias_lag1"])) for x in rt)
    verdict = "PASS" if max_bias <= RT_TOL else "FAIL"
    print(f"\n  max |bias| = {fmt(max_bias)}  ->  {verdict} (tolerance {RT_TOL})")
    assert verdict == "PASS", (
        f"round-trip max |bias| {max_bias} > {RT_TOL}: this is an "
        "implementation bug, STOP"
    )

    power = next(r_ for r_ in rt if (r_["rho_p"], r_["rho_t"]) == POWER_ARM)
    print("\n  " + "-" * 74)
    print("  POWER TEST OF THE FALSIFIER ESTIMATOR at the fitted dose")
    print("  " + "-" * 74)
    print(
        f"  arm (rho_p, rho_t) = ({POWER_ARM[0]}, {POWER_ARM[1]}): purely "
        f"persistent, no transient part."
    )
    print(
        f"  the lag-1 fit recovers    {fmt(power['rho_lag1_hat'])}  "
        f"(mean of {power['n_rep']} panels, sd {fmt(power['rho_lag1_sd'])})"
    )
    print(f"    per-panel values        {np.round(power['rho_lag1_panels'], 5)}")
    print(
        f"    panel 0 = {fmt(power['panel0_rho_lag1'])}, cluster-bootstrap 95% "
        f"CI [{fmt(power['panel0_rho_lag1_ci'][0])}, "
        f"{fmt(power['panel0_rho_lag1_ci'][1])}]"
    )
    print(
        f"  (its within-cell fit      {fmt(power['rho_total_hat'])}  "
        f"sd {fmt(power['rho_total_sd'])}; panel 0 CI "
        f"[{fmt(power['panel0_rho_total_ci'][0])}, "
        f"{fmt(power['panel0_rho_total_ci'][1])}])"
    )
    seen = power["rho_lag1_hat"] >= 0.5 * POWER_ARM[0]
    if seen:
        print(
            "  -> the lag-1 estimator CAN see a purely persistent component at\n"
            "     this dose, so the small value it returns on human data is\n"
            "     informative and the FALSIFIER READING IS LIVE."
        )
    else:
        print(
            "  -> the lag-1 estimator CANNOT see the persistence it is meant to\n"
            "     falsify, which confirms Declaration ground 3 EMPIRICALLY\n"
            "     rather than by argument."
        )
    print("  Either outcome is fine; neither changes the dose.")

    # ---------------- final block ---------------- #
    print("\n" + "=" * 78)
    print("FINAL -- what step 7 stamps")
    print("=" * 78)
    print(f"  THE DOSE       rho_p = rho_total = {fmt(rho_total)}")
    print("                 rho_t = 0.0")
    print(f"                 95% CI [{fmt(lo_t)}, {fmt(hi_t)}]  (SE {fmt(se_t)})")
    print(f"                 structure = {STRUCTURE}")
    print(f"                 estimator = {ESTIMATOR_TAG}")
    print(
        f"  THE FALSIFIER  rho_lag1 = {fmt(rho_lag1)}  95% CI "
        f"[{fmt(lo_l)}, {fmt(hi_l)}]"
    )
    print(
        f"                 two-component reading (rho_p, rho_t) = "
        f"({fmt(rho_lag1)}, {fmt(rho_total - rho_lag1)}) -- NOT stamped"
    )
    print(f"  ROUND TRIP     {verdict}  (max |bias| {fmt(max_bias)}, tol {RT_TOL})")
    print(
        f"  POWER TEST     lag-1 recovers {fmt(power['rho_lag1_hat'])} from a "
        f"purely persistent {POWER_ARM[0]}"
    )
    print(
        "  CAVEAT (read the falsifier with it, do not act on it here): the\n"
        "  censored MLE and the moment estimator agree on every synthetic\n"
        "  panel but disagree on the human lag-1 pairs, and the human pairwise\n"
        "  LR per pair sits far above the arm at the same fitted rho -- the\n"
        "  human within-group dependence is concentrated on the censoring\n"
        "  bounds (see the corner ratios above) and is not exchangeable-\n"
        "  Gaussian in shape. The dose is unchanged; step 11 reads the\n"
        "  falsifier knowing its MLE and its moment point opposite ways."
    )

    if args.write_params:
        params = dict(
            rho_total=float(rho_total),
            rho_total_ci=[lo_t, hi_t],
            rho_total_se=se_t,
            rho_total_lr=float(2.0 * (nll0 - nll_hat)),
            rho_p=float(rho_total),
            rho_t=0.0,
            rho_lag1=float(rho_lag1),
            rho_lag1_ci=[lo_l, hi_l],
            rho_lag1_se=se_l,
            two_component_reading=dict(
                rho_p=float(rho_lag1),
                rho_t=float(rho_total - rho_lag1),
                role="declared falsifier -- reported, never stamped as the dose",
            ),
            n_pairs_within=int(len(ii_w)),
            n_pairs_lag1=int(len(ii_l)),
            estimator=ESTIMATOR_TAG,
            structure=STRUCTURE,
            cell_key="episode_round_group",
            data_file=train_file,
            mask=cfg["data"]["mask"],
            n_rows=int(len(y)),
            n_episodes=int(n_ep),
            censored_share=censored,
            base_bundle=str(BUNDLE_PATH.relative_to(ROOT)),
            base_bundle_sha256=sha256(BUNDLE_PATH),
            git_sha=git_sha(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            bvn_max_dev=None if err is None else float(err),
            n_boot=int(N_BOOT),
            boot_seed=int(SEED),
            moment_diagnostics={
                k: dict(rho=v[0], n_pairs=v[1], ci=[v[2], v[3]], se=v[4])
                for k, v in moments.items()
            },
            icc_oneway=icc,
            round_thirds=thirds,
            misspecification=misspec,
            lr_per_pair_human=dict(
                within_cell=float(lr_w_human), lag1=float(lr_l_human)
            ),
            test_split=dict(
                data_file=cfg_te["data"]["data_file"],
                rho_within_mle=float(rho_w_te),
                rho_lag1_mle=float(rho_l_te),
                n_pairs_within=int(len(ii_wte)),
                n_pairs_lag1=int(len(ii_lte)),
                censored_share=cens_te,
                role="out-of-sample noise indicator only (log Note 20)",
            ),
            roundtrip=dict(
                tolerance=RT_TOL,
                verdict=verdict,
                max_abs_bias=float(max_bias),
                arms=rt,
                power_arm=dict(
                    rho_p=POWER_ARM[0],
                    rho_t=POWER_ARM[1],
                    n_rep=power["n_rep"],
                    rho_lag1_hat=power["rho_lag1_hat"],
                    rho_lag1_sd=power["rho_lag1_sd"],
                    rho_lag1_panels=power["rho_lag1_panels"],
                    panel0_rho_lag1=power["panel0_rho_lag1"],
                    panel0_rho_lag1_ci=power["panel0_rho_lag1_ci"],
                    rho_total_hat=power["rho_total_hat"],
                    rho_total_sd=power["rho_total_sd"],
                    panel0_rho_total=power["panel0_rho_total"],
                    panel0_rho_total_ci=power["panel0_rho_total_ci"],
                    falsifier_estimator_has_power=bool(seen),
                    reading=(
                        "a purely persistent panel; if the lag-1 fit recovers "
                        "~rho_p the falsifier reading is live, if it recovers "
                        "~0 Declaration ground 3 is confirmed empirically"
                    ),
                ),
                note=(
                    "features held fixed at the human rows: validates the "
                    "estimator, not the closed-loop dynamics"
                ),
            ),
        )
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {OUT_JSON.relative_to(ROOT)}")

    print(f"\ntotal runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
