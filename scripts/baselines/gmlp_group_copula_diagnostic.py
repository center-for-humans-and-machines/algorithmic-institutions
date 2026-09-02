"""Diagnostic script for the contribution group-copula experiment
(autoresearch step 2, notes/autoresearch_log/contribution-gmlp-group-copula.md).

Answers two questions, printed to stdout only -- this script writes nothing
and trains nothing:

  Part A -- does the parent's CG deficit (group-mean contribution spread
  under-dispersed vs the human data) reproduce from evaluation_suite's own
  canonical frame? Prints SD(group means) / SD(individual) / their ratio for
  the human data and both parent sims, the gap to the human ratio, that gap
  over the CG denominator, and the ratio values at the score-band edges. The
  candidate sim's gap is asserted against its own evaluation/metrics.csv CG
  row (`d`), to within 1e-12.

  Part B -- on the training split, are the incumbent gaussian_mlp_v2
  bundle's teacher-forced standardised residuals r = (c - mu) / sigma
  correlated WITHIN a group (same episode, round, group) more than they
  should be under independent sampling? Every correlation printed here is an
  ATTENUATED MOMENT DIAGNOSTIC (mean-and-variance based, biased toward zero
  by the 0/20 censoring and integer rounding of the target) -- it motivates
  the experiment but is NOT the dose. The estimate that sets the sampler's
  rho is step 6's interval-censored pairwise Gaussian-copula MLE.

Read-only imports of aimanager.evaluation_suite.{convert,metrics} -- that
package is a frozen surface and nothing here writes to it. Reuses
`blocks` / `pair_index` / `icc_oneway` from punishment_copula_rho.py
unmodified.

Runs locally (CPU, no PyG, no simulation):
    .venv/bin/python scripts/baselines/gmlp_group_copula_diagnostic.py
"""

import copy
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

from aimanager.evaluation_suite.convert import load_human, load_sim  # noqa: E402
from aimanager.evaluation_suite.metrics import GROUP_CELL  # noqa: E402

from handcrafted_grid import (  # noqa: E402
    build_feature_pool,
    load_config,
    load_episodes,
    validate_feature_legality,
)
from punishment_copula_rho import blocks, icc_oneway, pair_index  # noqa: E402

HUMAN_CSV = ROOT / "experiments/2group_8agent_50ep.csv"
BASE_SIM_DIR = (
    ROOT / "plots/simulation/23_2g8a_gmlp2_base_self_gaussian_contr_gnn_switch"
)
CAND_SIM_DIR = (
    ROOT / "plots/simulation/23_2g8a_gmlp2_self_gaussian_mlp_v2_contr_gnn_switch"
)
TRAIN_CFG = ROOT / "configs/training/baselines/contribution/gaussian_mlp_v2.yml"
BUNDLE_PATH = ROOT / "artifacts/baselines/contribution_gaussian_mlp_v2_best.joblib"

CG_DENOM = 0.026448679600274437  # scores.csv CG denominator, this stack
SEED = 38381  # the bundle's own cv.seed
N_BOOT = 200  # cluster-bootstrap resamples over episodes
BOOT_SEED = 38381

REF_N_PAIRS_WITHIN = 15090  # PR #165 / orchestrator probe, train, all-valid


# --------------------------------------------------------------------------- #
# Part A -- CG direction, via evaluation_suite's own canonical frame
# --------------------------------------------------------------------------- #
def cg_stats(df):
    """SD(group-mean contribution) / SD(individual contribution) over
    GROUP_CELL, exactly evaluation_suite.metrics.ContributionMetrics.cg /
    _spread_ratio."""
    valid = df.dropna(subset=["contribution"])
    grouped = valid.groupby(GROUP_CELL)["contribution"].mean()
    sd_group = grouped.std()
    sd_indiv = valid["contribution"].std()
    return float(sd_group), float(sd_indiv), float(sd_group / sd_indiv)


def part_a():
    print("=" * 78)
    print("PART A -- CG direction: human vs the two parent sims")
    print("=" * 78)

    human = load_human(HUMAN_CSV)
    sd_g_h, sd_i_h, ratio_h = cg_stats(human)
    print(f"human      sd_group={sd_g_h!r} sd_indiv={sd_i_h!r} ratio={ratio_h!r}")

    gaps = {}
    for label, sim_dir in (("base", BASE_SIM_DIR), ("candidate", CAND_SIM_DIR)):
        sims = load_sim(sim_dir / "per_round.parquet")
        assert len(sims) == 1, f"expected one run in {sim_dir}, got {list(sims)}"
        [(run_name, sim_df)] = sims.items()
        sd_g, sd_i, ratio = cg_stats(sim_df)
        gap = abs(ratio - ratio_h)
        gaps[label] = gap
        print(
            f"{label:<9}  run={run_name!r}\n"
            f"           sd_group={sd_g!r} sd_indiv={sd_i!r} ratio={ratio!r}\n"
            f"           gap=|ratio - human_ratio|={gap!r}"
            f"  gap/CG_DENOM={gap / CG_DENOM!r}"
        )

    print("\nband edges on the sim ratio (score k = gap / CG_DENOM):")
    for k, label in ((1, "<=1"), (2, "1-2"), (5, "2-5")):
        edge = ratio_h - k * CG_DENOM
        print(f"  score={k}  ({label} boundary)  ratio must exceed {edge!r}")

    metrics_csv = CAND_SIM_DIR / "evaluation" / "metrics.csv"
    m = pd.read_csv(metrics_csv)
    d_csv = float(m.loc[m["metric"] == "CG", "d"].iloc[0])
    match = abs(gaps["candidate"] - d_csv) < 1e-12
    print(
        f"\nmetrics.csv CG row d={d_csv!r}\n"
        f"script's candidate gap={gaps['candidate']!r}\n"
        f"match to 1e-12: {'OK' if match else 'MISMATCH'}"
    )
    assert match, (gaps["candidate"], d_csv)


# --------------------------------------------------------------------------- #
# Part B -- residual moment diagnostics, train split (attenuated; step 6's
# censored MLE is the estimate that matters)
# --------------------------------------------------------------------------- #
def build_rows(cfg):
    """Feature pool + target + (episode, agent, round, group) row indices for
    every `contribution_valid` observation, built with create_torch_data +
    build_feature_pool directly. `handcrafted_grid.prepare_data()` flattens
    the (episode, agent, round) indices away and cannot be used -- this
    mirrors punishment_copula_rho.build_rows against `contribution_valid`."""
    import random

    import torch as th

    from aimanager.generic.data import create_torch_data

    validate_feature_legality(cfg)
    th.random.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    df = load_episodes(cfg, ROOT)
    switch_every = cfg["data"].get("switch_every")
    data, _, _ = create_torch_data(df, switch_every=switch_every)
    pool = build_feature_pool(data, switch_every)

    mask = data[cfg["data"]["mask"]].numpy().astype(bool)  # [G, A, T]
    g, a, t = np.nonzero(mask)
    y = data["contribution"].numpy()[mask].astype(float)
    grp = data["agent_group"].numpy()[mask].astype(np.int64)
    n_ep, n_agents, n_rounds = mask.shape
    return dict(
        pool=pool,
        mask=mask,
        y=y,
        episode=g.astype(np.int64),
        agent=a.astype(np.int64),
        round=t.astype(np.int64),
        group=grp,
        shape=(n_ep, n_agents, n_rounds),
    )


def score_bundle(bundle, rows):
    """Teacher-forced (mu, sigma) of the bundle on its own 7 features/scaler,
    and the standardised residual r = (c - mu) / sigma."""
    X = np.column_stack([rows["pool"][k][rows["mask"]] for k in bundle["features"]])
    Xs = bundle["scaler"].transform(X)
    est = bundle["estimator"]
    mu = np.asarray(est.predict(Xs), float).reshape(-1)
    sigma = np.asarray(est.predict_std(Xs), float).reshape(-1)
    return (rows["y"] - mu) / sigma


def encode_cell(*cols):
    """Contiguous integer cell id for an arbitrary tuple of parallel arrays."""
    stacked = np.stack(cols, axis=1)
    _, inv = np.unique(stacked, axis=0, return_inverse=True)
    return inv.reshape(-1).astype(np.int64)


def pairs_within_cell(episode, round_, group):
    """All i<j pairs sharing (episode, round, group)."""
    cell = encode_cell(episode, round_, group)
    return pair_index(cell)


def pairs_same_agent(episode, agent):
    """All i<j pairs sharing (episode, agent) -- different rounds, same
    participant (individual persistence, for contrast)."""
    cell = encode_cell(episode, agent)
    return pair_index(cell)


def pairs_cross_group(episode, round_, group):
    """Pairs sharing (episode, round) but in DIFFERENT groups -- the
    episode-level common-shock check."""
    cell = encode_cell(episode, round_)
    ii, jj = pair_index(cell)
    keep = group[ii] != group[jj]
    return ii[keep], jj[keep]


def pairs_lag1(episode, round_, group, agent):
    """Directional pairs: agent i at round t vs agent j != i at round t + 1,
    matched on each row's OWN group at its own round (so a switcher pairs
    correctly on either side). Self-pairs (same agent) excluded."""
    n = len(episode)
    left = pd.DataFrame(
        {
            "idx_i": np.arange(n),
            "episode": episode,
            "round": round_,
            "group": group,
            "agent_i": agent,
        }
    )
    right = pd.DataFrame(
        {
            "idx_j": np.arange(n),
            "episode": episode,
            "round": round_ - 1,  # so this row's t+1 aligns to left's t
            "group": group,
            "agent_j": agent,
        }
    )
    merged = left.merge(right, on=["episode", "round", "group"], how="inner")
    merged = merged[merged["agent_i"] != merged["agent_j"]]
    return merged["idx_i"].to_numpy(), merged["idx_j"].to_numpy()


def pooled_corr(r, ii, jj):
    """Equal-weight-per-pair moment correlation: mean over pairs of the
    product of centered, variance-normalised residuals -- the same
    exchangeable-moment definition punishment_copula_rho.rho_pairs uses,
    written directly for continuous r (no PIT step needed)."""
    mean, var = r.mean(), r.var(ddof=1)
    vals = (r[ii] - mean) * (r[jj] - mean) / var
    return float(vals.mean()), vals


def cluster_ci(vals, episode_of_pair, n_boot=N_BOOT, seed=BOOT_SEED):
    """Cluster bootstrap over episodes: resample episodes with replacement,
    average `vals` over the pairs whose episode was drawn. Pairs, not
    episodes, are the unit `vals` is indexed by; every pair sits inside one
    episode (asserted by the caller)."""
    ep_ids = np.array(sorted(set(episode_of_pair.tolist())))
    per_ep = {int(e): np.flatnonzero(episode_of_pair == e) for e in ep_ids}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(ep_ids, size=len(ep_ids), replace=True)
        pos = np.concatenate([per_ep[int(e)] for e in draw])
        boot[b] = vals[pos].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi), float(boot.std(ddof=1))


def report(label, r, episode, ii, jj):
    rho, vals = pooled_corr(r, ii, jj)
    ep_pair = episode[ii]
    assert np.array_equal(ep_pair, episode[jj]), f"{label}: a pair crosses episodes"
    lo, hi, se = cluster_ci(vals, ep_pair)
    print(
        f"  {label:<42} rho={rho:+.5f}  n_pairs={len(ii):>6}  "
        f"boot SE={se:.5f}  95% CI=[{lo:+.5f}, {hi:+.5f}]"
    )
    return rho, len(ii)


def diag_within(r, episode, round_, group, label="within-cell moment corr"):
    ii, jj = pairs_within_cell(episode, round_, group)
    return report(label, r, episode, ii, jj)


def diag_lag1(
    r, episode, round_, group, agent, label="cross-member lag-1 corr (t->t+1)"
):
    ii, jj = pairs_lag1(episode, round_, group, agent)
    return report(label, r, episode, ii, jj)


def diag_same_agent(r, episode, agent, label="same-agent cross-round corr (persist.)"):
    ii, jj = pairs_same_agent(episode, agent)
    return report(label, r, episode, ii, jj)


def diag_cross_group(
    r, episode, round_, group, label="cross-group same-round corr (shock)"
):
    ii, jj = pairs_cross_group(episode, round_, group)
    return report(label, r, episode, ii, jj)


def icc(r, episode, round_, group):
    """One-way random-effects ICC(1), cell=(episode, round, group), via
    punishment_copula_rho.icc_oneway -- shape fits directly (r as a single
    [n, 1] column)."""
    cell = encode_cell(episode, round_, group)
    order, starts, sizes = blocks(cell)
    val = icc_oneway(r[order][:, None], starts, sizes)
    return float(val[0])


def thirds_bounds(n_rounds):
    for third in np.array_split(np.arange(n_rounds), 3):
        yield int(third[0]), int(third[-1])


def run_diagnostics(label, r, episode, round_, group, agent, n_rounds):
    n_ep = len(np.unique(episode))
    print(f"\n--- {label}: {len(r)} rows, {n_ep} episodes ---")
    icc_val = icc(r, episode, round_, group)
    print(f"  ICC(1) one-way, cell=(episode, round, group): {icc_val:+.5f}")

    rho_w, n_w = diag_within(r, episode, round_, group)
    rho_l1, n_l1 = diag_lag1(r, episode, round_, group, agent)
    diag_same_agent(r, episode, agent)
    diag_cross_group(r, episode, round_, group)

    print("  within-cell moment corr by round thirds:")
    for lo3, hi3 in thirds_bounds(n_rounds):
        sel = (round_ >= lo3) & (round_ <= hi3)
        diag_within(
            r[sel],
            episode[sel],
            round_[sel],
            group[sel],
            label=f"    rounds {lo3}-{hi3}",
        )
    return dict(icc=icc_val, within=(rho_w, n_w), lag1=(rho_l1, n_l1))


def interior_diagnostics(r, episode, round_, group, agent, y, split_label):
    interior = (y != 0) & (y != 20)
    n_int, n_all = int(interior.sum()), len(y)
    censored_share = 1.0 - n_int / n_all
    print(
        f"\n  {split_label}: censored (0 or 20) share = {censored_share:.4f}  "
        f"({n_all - n_int}/{n_all} rows)"
    )
    print(f"  interior-only ({n_int} rows, {n_int / n_all:.1%}):")
    diag_within(
        r[interior],
        episode[interior],
        round_[interior],
        group[interior],
        label="    interior within-cell moment corr",
    )
    diag_lag1(
        r[interior],
        episode[interior],
        round_[interior],
        group[interior],
        agent[interior],
        label="    interior lag-1 corr (t->t+1)",
    )
    return censored_share


def part_b():
    print("\n" + "=" * 78)
    print("PART B -- residual moment diagnostics, train split (ATTENUATED --")
    print("the estimate that matters is step 6's censored MLE, not these)")
    print("=" * 78)

    import joblib

    bundle = joblib.load(BUNDLE_PATH)
    cfg = load_config(TRAIN_CFG)
    print(f"bundle    {BUNDLE_PATH.relative_to(ROOT)}")
    print(f"  model={bundle['model']} features={bundle['features']}")
    print(f"  config={bundle['config']}")

    # ---------------- train split ---------------- #
    rows_tr = build_rows(cfg)
    r_tr = score_bundle(bundle, rows_tr)
    n_ep, n_agents, n_rounds = rows_tr["shape"]
    print(
        f"\ntrain data {cfg['data']['data_file']}  rows={len(r_tr)}  "
        f"episodes={len(np.unique(rows_tr['episode']))}  agents={n_agents}  "
        f"rounds={n_rounds}"
    )
    print(
        f"  residual r=(c-mu)/sigma  mean={r_tr.mean():+.5f}  "
        f"var={r_tr.var(ddof=1):.5f}"
    )

    res_tr = run_diagnostics(
        "TRAIN, all-valid rows",
        r_tr,
        rows_tr["episode"],
        rows_tr["round"],
        rows_tr["group"],
        rows_tr["agent"],
        n_rounds,
    )
    n_pairs_within_tr = res_tr["within"][1]
    match_ref = n_pairs_within_tr == REF_N_PAIRS_WITHIN
    print(
        f"\n  REFERENCE CHECK (data-path agreement, PR #165 / orchestrator "
        f"probe): n_pairs within-cell = {n_pairs_within_tr}, expected "
        f"{REF_N_PAIRS_WITHIN} -- {'MATCH' if match_ref else 'MISMATCH'}"
    )

    interior_diagnostics(
        r_tr,
        rows_tr["episode"],
        rows_tr["round"],
        rows_tr["group"],
        rows_tr["agent"],
        rows_tr["y"],
        "TRAIN",
    )

    # ---------------- test split (out-of-sample check only) ---------------- #
    cfg_te = copy.deepcopy(cfg)
    cfg_te["data"]["data_file"] = cfg["data"]["data_file"].replace("_train", "_test")
    rows_te = build_rows(cfg_te)
    r_te = score_bundle(bundle, rows_te)
    n_rounds_te = rows_te["shape"][2]
    print(
        f"\nOUT-OF-SAMPLE CHECK ONLY -- test data "
        f"{cfg_te['data']['data_file']}  rows={len(r_te)}  "
        f"episodes={len(np.unique(rows_te['episode']))}"
    )

    run_diagnostics(
        "TEST, all-valid rows (out-of-sample check only)",
        r_te,
        rows_te["episode"],
        rows_te["round"],
        rows_te["group"],
        rows_te["agent"],
        n_rounds_te,
    )
    interior_diagnostics(
        r_te,
        rows_te["episode"],
        rows_te["round"],
        rows_te["group"],
        rows_te["agent"],
        rows_te["y"],
        "TEST",
    )


def main():
    t0 = time.time()
    part_a()
    part_b()
    print(f"\ntotal runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
