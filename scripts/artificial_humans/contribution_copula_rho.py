"""Calibrate `copula_rho` and `copula_phi` for the GNN contributor's copula.

Estimates the within-(episode, round, agent_group) latent correlation of human
contributions by pairwise-likelihood MLE of an exchangeable Gaussian copula
(each observation keeps the model's own predicted discrete marginal), on the
40 single-copy baseline train episodes only, and then the lag-1 persistence of
that shared latent by the SAME estimator over cross-round pairs:

    phi = rho_lag1 / rho

The PRIMARY lag-1 refit uses CROSS-PLAYER pairs only (different participants at
rounds r and r + 1 of the same (episode, group) cell). Within-round rho is a
cross-player quantity by construction, and the AR(1) latent is a group-level
object whose persistence is identified by cross-player lag-1 dependence
(implied correlation rho * phi). Self-pairs -- the same participant at r and
r + 1 -- confound individual serial stickiness that the marginal model already
conditions on through `prev_contribution`, and at lag 1 that confound
dominates. The all-pairs refit is computed and logged as a DIAGNOSTIC only and
is never used downstream (orchestrator amendment, plan section 2).

The randomized-PIT moment estimator is printed as an attenuated diagnostic
only. The estimator, the bootstrap, the round-trip gate and the pre-flight are
imported unmodified from the punisher calibration (PR #146,
scripts/baselines/punishment_copula_rho.py); only the marginals differ -- they
come from a teacher-forced GNN forward pass instead of a linear bundle. Method
details: notes/autoresearch_log/punisher-severity-copula.md (appendix),
notes/autoresearch_log/contribution-cg-copula.md and
notes/autoresearch_log/contribution-herding-copula-v2.md.

Estimation only: this script never writes an artifact (the copula fields are
stamped onto a copy of the base model by
scripts/artificial_humans/make_contribution_copula_artifact.py).

Imports graph.py, so this runs on Raven only:
    .venv/bin/python scripts/artificial_humans/contribution_copula_rho.py \
        --model IN.pt [--preflight] [--roundtrip] [--write-params PATH]
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch as th  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

# the #146 estimator machinery, imported unmodified: that module's top level is
# imports, constants and defs only, so importing it runs no calibration
import punishment_copula_rho as pc  # noqa: E402

# pytorch geometric meta module has changed place since the artifacts were
# saved; the alias lets the legacy pickles load (simulate.py does the same)
import torch_geometric.nn.models.meta as meta_module  # noqa: E402

sys.modules["torch_geometric.nn.meta"] = meta_module

from aimanager.generic.data import create_torch_data  # noqa: E402
from aimanager.generic.graph import GraphNetwork  # noqa: E402

f = pc.f  # unrounded float for the log
FULL = ROOT / "experiments/2group_8agent_50ep.csv"
TRAIN = ROOT / "experiments/baseline/2group_8agent_50ep_bline_train.csv"
TEST = ROOT / "experiments/baseline/2group_8agent_50ep_bline_test.csv"
EXPERIMENTS = ["ah_group_switching"]
MASK = "contribution_valid"
SWITCH_EVERY = None  # the contribution configs carry no switch_every
SEED = pc.SEED  # 38381, also the AH training seed
N_GROUPS = 2
N_TRAIN_EP = 40
N_TEST_EP = 10
STEP = 1  # contributions are decided every round, so lag 1 = the next round
N_BOOT_PHI = pc.N_BOOT  # 200 paired resamples for the phi CI


# --------------------------------------------------------------------------- #
# data: the training loader on the FULL file, then the split's own episodes
# --------------------------------------------------------------------------- #
def rel(path):
    """Repo-relative path when possible, absolute otherwise (logging only)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def episode_key(df):
    """The (global_group_id, episode_id) key parse_agent_rounds dense-ranks
    into `group_idx` -- recomputed here because the column is dropped before
    the tensors are built."""
    return df["global_group_id"] + "__" + df["episode_id"].astype(str)


def load_full():
    """create_torch_data on the FULL human file, so every default and median
    (contribution default, common-good rescaling) is exactly the one the model
    trained with. Returns the tensors, the per-episode pair_id, and the
    key -> episode-row map."""
    th.random.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    df = pd.read_csv(FULL)
    df = df[df["experiment_name"].isin(EXPERIMENTS)]
    data, default_values, pair_id = create_torch_data(df, switch_every=SWITCH_EVERY)
    key = episode_key(df)
    idx = key.rank(method="dense").astype(int) - 1
    key_to_idx = dict(zip(key, idx))
    assert len(key_to_idx) == data["contribution"].shape[0], "key/tensor mismatch"
    return data, np.asarray(pair_id), key_to_idx, default_values


def select_split(key_to_idx, path, n_expect):
    """Episode rows of the split file. The flip-doubling lives in the CSV and
    the model conditions on agent_group, so the two copies of a game carry
    different marginals -- matching on the key takes exactly the copy the
    split file holds (D5)."""
    df = pd.read_csv(path)
    keys = sorted(set(episode_key(df)))
    missing = [k for k in keys if k not in key_to_idx]
    assert not missing, f"{path.name}: {len(missing)} keys absent from {FULL.name}"
    idx = np.array(sorted(key_to_idx[k] for k in keys), dtype=np.int64)
    assert len(idx) == len(set(idx.tolist())) == n_expect, (
        f"{path.name}: matched {len(set(idx.tolist()))} episodes, "
        f"expected {n_expect}"
    )
    return idx


def teacher_forced_rows(model, data, idx):
    """Predicted marginals + observed levels + cell ids for every valid
    contribution of the selected episodes. Teacher-forced: the model sees the
    human history, never its own draws."""
    sel = th.as_tensor(idx)
    sub = {k: v[sel] for k, v in data.items()}
    n_ep, n_agents, n_rounds = sub["contribution"].shape
    with th.no_grad():  # predict_* leaves the graph attached otherwise
        edge_index = model.create_fully_connected(n_agents, n_batch=n_ep)
        _, proba = model.predict_independent(
            sub, sample=False, reset_rnn=True, edge_index=edge_index
        )
    mask = sub[MASK].numpy().astype(bool)  # [G, A, T]
    g, a, t = np.nonzero(mask)
    # float64 of the float32 softmax, unfloored and unrenormalised: exactly
    # the array the sampler in generic/copula.py inverts
    P = proba.double().numpy()[mask]
    assert np.abs(P.sum(1) - 1.0).max() < 1e-6, "predicted rows do not sum to 1"
    y = sub["contribution"].numpy()[mask].astype(np.int64)
    assert y.max() < P.shape[1], "observed level outside the model's support"
    grp = sub["agent_group"].numpy()[mask].astype(np.int64)
    return dict(
        P=P,
        y=y,
        episode=g.astype(np.int64),
        agent=a.astype(np.int64),
        round=t.astype(np.int64),
        group=grp,
        cell=(g.astype(np.int64) * n_rounds + t) * N_GROUPS + grp,
        shape=(n_ep, n_agents, n_rounds),
    )


def describe(tr, label):
    _, _, sizes = pc.blocks(tr["cell"])
    ii, jj = pc.pair_index(tr["cell"])
    n_ep, n_agents, n_rounds = tr["shape"]
    print(f"{label}: episodes={n_ep} agents={n_agents} rounds={n_rounds}")
    print(
        f"  rows={len(tr['y'])} cells={len(sizes)} "
        f"cells>=2={(sizes >= 2).sum()} pairs={len(ii)}"
    )
    print(f"  cell size histogram={np.bincount(sizes).tolist()}")
    return ii, jj, sizes


# --------------------------------------------------------------------------- #
# AR(1) persistence: cross-round pairs of one group's latent
# --------------------------------------------------------------------------- #
def cross_pairs(rows, step=STEP):
    """Row indices (i at round r, j at round r + step) of the same
    (episode, agent_group) -- the lag-1 pairs whose model correlation is
    rho * phi ** step.

    The block key is the GROUP's latent, so each row joins the block of the
    group it belonged to AT ITS OWN ROUND: a player who switched between r and
    r + step sits in the other group's block at r + step. Every ordered
    i x j combination of the two blocks enters once, self-pairs (the same
    participant at both rounds) included -- the caller splits them out, since
    only the cross-player subset identifies the group latent's persistence
    cleanly. Only rounds exactly `step` apart are paired.
    """
    idx = {}
    keys = zip(rows["episode"].tolist(), rows["round"].tolist(), rows["group"].tolist())
    for pos, key in enumerate(keys):
        idx.setdefault(key, []).append(pos)
    ii, jj = [], []
    for (e, r, grp), a in sorted(idx.items()):
        b = idx.get((e, r + step, grp))
        if b is None:
            continue
        a_arr, b_arr = np.array(a, np.int64), np.array(b, np.int64)
        ii.append(np.repeat(a_arr, len(b_arr)))
        jj.append(np.tile(b_arr, len(a_arr)))
    if not ii:
        return np.array([], np.int64), np.array([], np.int64)
    return np.concatenate(ii), np.concatenate(jj)


def lag1_rho(z_lo, z_hi, ic, jc):
    """The within-round rectangle-probability MLE run over a cross-round pair
    list: same estimator, same corner points, only the pairing changes.
    Returns (rho_lag1, nll(rho_lag1), nll(0), n_evals)."""
    H, K, sgn = pc.rect_points(z_lo, z_hi, ic, jc)
    return pc.rho_mle(H, K, sgn)


def bootstrap_phi(P, y, episode, pair_sets, n_boot, seed):
    """Paired episode-cluster bootstrap over `pair_sets` (a list of (ii, jj)).

    Every set is refit on the SAME episode resample, so the returned rho
    arrays are paired and their ratio is a valid phi draw. Resampling episodes
    resamples whole pair blocks: a within-round pair sits inside one cell and a
    cross-round pair inside one episode, so no pair ever crosses the cluster
    boundary. Each resample is refit on the full RHO_GRID -- the #146
    bootstrap narrows the bracket around the full-sample estimate for cost,
    but a narrowed bracket censors the tail of a ratio.
    """
    z_lo, z_hi = pc.cdf_bounds(P, y)
    prepared = []
    for ii, jj in pair_sets:
        H, K, sgn = pc.rect_points(z_lo, z_hi, ii, jj)
        prepared.append((H, K, sgn, len(ii), pc._per_episode_pairs(episode, ii, jj)))
    ep_ids = np.array(sorted(prepared[0][4]))
    rng = np.random.default_rng(seed)
    out = [np.empty(n_boot) for _ in prepared]
    for b in range(n_boot):
        draw = rng.choice(ep_ids, size=len(ep_ids), replace=True)
        for o, (H, K, sgn, n_pairs, per_ep) in zip(out, prepared):
            pos = np.concatenate([per_ep[int(e)] for e in draw])
            idx = np.concatenate([pos + t * n_pairs for t in range(4)])
            o[b] = pc.rho_mle(H[idx], K[idx], sgn)[0]
    return out


def phi_verdict(phi_hat, ci):
    """(kept?, reason) under the plan's stop-gate: phi > 0 with a 95% CI that
    excludes 0."""
    if not np.isfinite(phi_hat) or phi_hat <= 0.0:
        return False, f"phi_hat {f(phi_hat)} <= 0"
    if ci[0] <= 0.0 <= ci[1]:
        return False, f"95% CI [{f(ci[0])}, {f(ci[1])}] includes 0"
    return True, ""


# --------------------------------------------------------------------------- #
# params JSON: estimates plus full provenance (no artifact is written here)
# --------------------------------------------------------------------------- #
def write_params(path, info, source_path):
    """Dump the estimates plus full provenance (data, base artifact, HEAD)."""
    import hashlib
    import subprocess

    out = dict(info)
    out["source_model_sha256"] = hashlib.sha256(
        Path(source_path).read_bytes()
    ).hexdigest()
    try:
        out["git_head"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - provenance is best effort
        out["git_head"] = f"unavailable: {exc}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {rel(path)}")
    for k in sorted(out):
        print(f"  {k}: {out[k]}")


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="input .pt (rho must be 0)")
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
    ap.add_argument(
        "--write-params",
        metavar="PATH",
        default=None,
        help="write rho, phi, their CIs and the full provenance to JSON",
    )
    args = ap.parse_args()
    in_path = Path(args.model).resolve()

    model = GraphNetwork.load(str(in_path), device="cpu")
    model.eval()
    assert model.y_name == "contribution", f"not a contributor: {model.y_name}"
    assert model.copula_rho == 0.0, f"input already carries {model.copula_rho}"
    assert model.copula_phi == 0.0, f"input already carries {model.copula_phi}"
    print(f"model     {rel(in_path)}")
    print(
        f"  y_name={model.y_name} y_levels={model.y_levels} "
        f"autoregressive={model.autoregressive}"
    )
    print(f"  x_encoding={[e['name'] for e in model.x_encoding]}")
    print(f"  edge_encoding={[e['name'] for e in model.edge_encoding]}")

    data, pair_id, key_to_idx, defaults = load_full()
    tr_idx = select_split(key_to_idx, TRAIN, N_TRAIN_EP)
    te_idx = select_split(key_to_idx, TEST, N_TEST_EP)
    assert not set(tr_idx.tolist()) & set(te_idx.tolist()), "train/test overlap"
    assert not set(pair_id[tr_idx].tolist()) & set(
        pair_id[te_idx].tolist()
    ), "a flip copy of a train game sits in the holdout"
    print(f"data      {rel(FULL)} (episodes={len(key_to_idx)})")
    print(
        f"  train split {rel(TRAIN)}: {len(tr_idx)} episodes, "
        f"{len(set(pair_id[tr_idx].tolist()))} distinct games"
    )
    print(f"  contribution default={f(defaults['contribution'])}")

    tr = teacher_forced_rows(model, data, tr_idx)
    P, y, cell = tr["P"], tr["y"], tr["cell"]
    ii, jj, sizes = describe(tr, "  teacher-forced rows")
    order, starts, _ = pc.blocks(cell)
    err = pc.check_bvn()
    print(
        f"  Phi_2 quadrature nodes={pc.N_QUAD}, max abs err vs scipy mvn="
        f"{'unchecked' if err is None else f(err)}"
    )

    # ---------------- the estimator: pairwise-likelihood MLE ---------------- #
    z_lo, z_hi = pc.cdf_bounds(P, y)
    H, K, sgn = pc.rect_points(z_lo, z_hi, ii, jj)
    t_mle = time.time()
    rho_hat, nll_hat, nll0, n_ev = pc.rho_mle(H, K, sgn)
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
    boot = pc.bootstrap_mle(P, y, cell, tr["episode"], pc.N_BOOT, SEED, rho_hat)
    se = float(boot.std(ddof=1))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    print(f"cluster bootstrap ({pc.N_BOOT} resamples over {len(tr_idx)} episodes)")
    print(f"  SE                         {f(se)}")
    print(f"  95% percentile CI          [{f(ci[0])}, {f(ci[1])}]")
    print(
        f"  bootstrap min/max          {f(boot.min())} / {f(boot.max())}"
        f"   [{time.time() - t_bs:.1f}s]"
    )

    # ---------------- acceptance gate ---------------- #
    rt_bias = None
    if args.roundtrip:
        t_rt = time.time()
        rows = pc.roundtrip(P, cell, (0.1, 0.2, 0.3, 0.4, 0.5), pc.N_ROUNDTRIP, SEED)
        print(
            f"\n=== ACCEPTANCE GATE: round-trip recovery "
            f"({pc.N_ROUNDTRIP} synthetic datasets per rho) ==="
        )
        print(
            "rho_true   rho_hat MLE            bias                   "
            "rho_hat PIT (attenuated)"
        )
        for rho_true, mle, pit in rows:
            print(f"{rho_true:<10.2f} {f(mle):<22} {f(mle - rho_true):<22} {f(pit)}")
        rt_bias = max(abs(m - r) for r, m, _ in rows)
        print(
            f"max |bias| = {f(rt_bias)}  -> "
            f"{'PASS' if rt_bias <= 0.03 else 'FAIL'} (tolerance 0.03)"
            f"   [{time.time() - t_rt:.1f}s]"
        )

    # ---------------- AR(1) persistence: rho_lag1 and phi ---------------- #
    ic, jc = cross_pairs(tr, STEP)
    is_self = tr["agent"][ic] == tr["agent"][jc]
    icx, jcx = ic[~is_self], jc[~is_self]
    print(f"\n=== phi: lag-{STEP} persistence of the group latent ===")
    print(f"cross-round pairs (all)      {len(ic)}")
    print(f"  same participant (self)    {int(is_self.sum())}  DIAGNOSTIC only")
    print(f"  different participants     {len(icx)}  PRIMARY")
    print(
        "  PRIMARY = cross-player: within-round rho is cross-player by "
        "construction and the\n  AR(1) latent is a group-level object; "
        "self-pairs confound the individual serial\n  stickiness the "
        "marginal model already conditions on via prev_contribution."
    )
    assert len(icx) > 0, "no cross-player lag-1 pairs"
    t_l1 = time.time()
    rho_l1_x, nll_x, nll0_x, _ = lag1_rho(z_lo, z_hi, icx, jcx)
    rho_l1_a, nll_a, nll0_a, _ = lag1_rho(z_lo, z_hi, ic, jc)
    phi_hat = rho_l1_x / rho_hat if rho_hat > 0 else float("nan")
    phi_all = rho_l1_a / rho_hat if rho_hat > 0 else float("nan")
    print(f"rho_lag1 cross-player        {f(rho_l1_x)}  PRIMARY")
    print(f"  pairwise nll(rho_lag1)     {f(nll_x)}")
    print(f"  2*(nll(0) - nll(rho_lag1)) {f(2.0 * (nll0_x - nll_x))}")
    print(f"rho_lag1 all pairs           {f(rho_l1_a)}  DIAGNOSTIC")
    print(f"  2*(nll(0) - nll(rho_lag1)) {f(2.0 * (nll0_a - nll_a))}")
    print(f"phi = rho_lag1_cross / rho   {f(phi_hat)}  PRIMARY")
    print(f"phi_allpairs (never used)    {f(phi_all)}  DIAGNOSTIC")
    print(
        f"  the MLE grid is non-negative ({f(pc.RHO_GRID[0])} to "
        f"{f(pc.RHO_GRID[-1])}), so rho_lag1 = 0 is a\n  boundary hit, "
        f"read as no measurable persistence   [{time.time() - t_l1:.1f}s]"
    )

    t_ph = time.time()
    b_rho, b_lag = bootstrap_phi(
        P, y, tr["episode"], [(ii, jj), (icx, jcx)], N_BOOT_PHI, SEED
    )
    phi_b = np.where(b_rho > 0, b_lag / np.where(b_rho > 0, b_rho, 1.0), np.nan)
    finite = phi_b[np.isfinite(phi_b)]
    phi_ci = (
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
    )
    keep, why = phi_verdict(phi_hat, phi_ci)
    print(
        f"paired cluster bootstrap ({N_BOOT_PHI} resamples over "
        f"{len(tr_idx)} episodes, rho and rho_lag1 refit on the same draw)"
    )
    print(f"  phi 95% percentile CI      [{f(phi_ci[0])}, {f(phi_ci[1])}]")
    print(f"  phi bootstrap median       {f(np.median(finite))}")
    print(
        f"  phi SE                     {f(finite.std(ddof=1))}  "
        f"(a RATIO: heavy-tailed as the denominator approaches 0, no finite "
        f"moments -- read the percentile CI, not this)"
    )
    print(f"  degenerate draws (rho=0)   {int(len(phi_b) - len(finite))}")
    print(f"  draws with rho < 0.01      {int((b_rho < 0.01).sum())}")
    print(f"  rho draws min/max          {f(b_rho.min())} / {f(b_rho.max())}")
    print(
        f"  rho_lag1 min/mean/max      {f(b_lag.min())} / {f(b_lag.mean())} / "
        f"{f(b_lag.max())}   [{time.time() - t_ph:.1f}s]"
    )
    verdict = "PHI KEPT" if keep else f"STOP-GATE: PHI DROPPED -- {why}"
    print(f"VERDICT: {verdict}")

    # ---------------- attenuated diagnostic: randomized PIT ---------------- #
    Z, z_mid = pc.latents(P, y, pc.N_PIT, SEED)
    Zo = Z[order]
    rho_reps, _ = pc.rho_pairs(Zo, starts, sizes)
    mid_rho, _ = pc.rho_pairs(z_mid[order][:, None], starts, sizes)
    print(
        f"\n=== ATTENUATED DIAGNOSTIC: randomized-PIT moment estimator "
        f"(R={pc.N_PIT}) ==="
    )
    print(f"rho_pit (pair-weighted)      {f(rho_reps.mean())}")
    print(f"  per-replicate min/max      {f(rho_reps.min())} / {f(rho_reps.max())}")
    print(f"  replicate sd               {f(rho_reps.std(ddof=1))}")
    print(f"cross-check per-cell weight  {f(pc.rho_cells(Zo, starts, sizes).mean())}")
    print(f"cross-check ICC(1) one-way   {f(pc.icc_oneway(Zo, starts, sizes).mean())}")
    print(f"sensitivity mid-point PIT    {f(mid_rho[0])}")
    boot_pit = pc.bootstrap_pit(
        Zo, cell[order], tr["episode"][order], pc.N_BOOT_PIT, SEED
    )
    print(
        f"cluster bootstrap ({pc.N_BOOT_PIT} resamples, crossed with the "
        f"{pc.N_PIT} PIT replicates)"
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
        mle, n_pr, n_rows, n_cells = pc.mle_on_rows(P, y, cell, sel)
        pit = pc.pit_on_subset(Z, cell, sel)[0]
        print(
            f"{label:<28} {f(mle):<23} {f(pit):<23} "
            f"rows={n_rows} cells>=2={n_cells} pairs={n_pr}"
        )

    n_rounds = tr["shape"][2]
    split("excluding round 0", rounds > 0)
    for third in np.array_split(np.arange(n_rounds), 3):
        split(f"rounds {third[0]}-{third[-1]}", np.isin(rounds, third))
    size_of_cell = dict(zip(cell[order][starts], sizes))
    row_size = np.array([size_of_cell[c] for c in cell])
    for s in sorted(set(sizes.tolist())):
        if s >= 2:
            split(f"cell size {s}", row_size == s)

    # ---------------- out-of-sample check ---------------- #
    te = teacher_forced_rows(model, data, te_idx)
    mle_te, pr_te, rows_te, cells_te = pc.mle_on_rows(te["P"], te["y"], te["cell"])
    Z_te, _ = pc.latents(te["P"], te["y"], pc.N_PIT, SEED)
    o_te, s_te, sz_te = pc.blocks(te["cell"])
    pit_te, _ = pc.rho_pairs(Z_te[o_te], s_te, sz_te)
    print(f"\nOUT-OF-SAMPLE CHECK ONLY ({rel(TEST)}, episodes={len(te_idx)})")
    print(
        f"  rho MLE={f(mle_te)}  rho PIT={f(pit_te.mean())}  "
        f"rows={rows_te} cells>=2={cells_te} pairs={pr_te}"
    )

    # ---------------- pre-flight ---------------- #
    pf = None
    if args.preflight:
        ind, cop, human = pc.preflight(P, y, cell, rho_hat, pc.N_PREFLIGHT, SEED)
        pf = dict(independent=ind, copula=cop, human=human)
        print(
            f"\n=== pre-flight (go/no-go; rho is NEVER tuned to it), "
            f"{pc.N_PREFLIGHT} repeats ==="
        )
        print(f"group-spread ratio independent  {f(ind)}")
        print(f"group-spread ratio copula       {f(cop)}  (rho={f(rho_hat)})")
        print(f"group-spread ratio human        {f(human)}")
        print(
            "CAVEAT: the AR(1) persistence (phi) cannot appear in a "
            "ONE-STEP redraw; this\npre-flight bounds the within-round part "
            "only. The compounding is what the sim measures."
        )

    meta = dict(
        rho=float(rho_hat),
        rho_se=se,
        rho_ci=[ci[0], ci[1]],
        rho_lag1_cross=float(rho_l1_x),
        rho_lag1_all=float(rho_l1_a),
        phi=float(phi_hat),
        phi_allpairs=float(phi_all),
        phi_ci=[phi_ci[0], phi_ci[1]],
        phi_kept=bool(keep),
        phi_verdict=verdict,
        phi_drop_reason=None if keep else why,
        copula_switch_every=STEP,
        estimator=(
            "pairwise-likelihood MLE of an exchangeable Gaussian copula "
            "(rectangle probabilities of each observation's own predicted "
            "discrete marginal, Phi_2 by Gauss-Legendre quadrature); rho over "
            "within-(episode, round, agent_group) pairs, rho_lag1 over "
            "CROSS-PLAYER pairs of consecutive rounds of the same "
            "(episode, agent_group); phi = rho_lag1_cross / rho; all-pairs "
            "refit is a diagnostic only"
        ),
        diag_pit="randomized",  # provenance of the diagnostic only
        diag_pit_rho=float(rho_reps.mean()),
        cell_key="episode_round_agent_group",
        lag_step=STEP,
        data_file=rel(TRAIN),
        data_file_holdout=rel(TEST),
        data_note=(
            "create_torch_data on the full human file (training defaults), "
            "restricted to the 40 single-copy train episodes; teacher-forced "
            "marginals from the source model; rows where contribution_valid"
        ),
        n_rows=int(len(y)),
        n_cells=int((sizes >= 2).sum()),
        n_pairs=int(len(ii)),
        n_cross_pairs=int(len(ic)),
        n_cross_pairs_self=int(is_self.sum()),
        n_cross_pairs_cross_player=int(len(icx)),
        n_episodes=int(len(tr_idx)),
        n_bootstrap=int(pc.N_BOOT),
        n_bootstrap_phi=int(N_BOOT_PHI),
        roundtrip_max_bias=rt_bias,
        preflight=pf,
        holdout_rho=float(mle_te),
        source_model=rel(in_path),
        script=rel(__file__),
        seed=SEED,
        date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    if args.write_params:
        write_params(args.write_params, meta, in_path)
    else:
        print("\n--write-params not given: nothing written")
    print(f"total runtime {time.time() - t0:.1f}s")

    if np.isfinite(phi_hat) and phi_hat >= 1.0:
        print(
            f"\n!!! STOP-ESCALATE: phi = {f(phi_hat)} >= 1.0 -- the AR(1) "
            f"latent is non-stationary at this estimate. No artifact, no "
            f"simulation. Escalate to the orchestrator. !!!"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
