"""Calibrate `copula_rho` for the GNN contributor's within-round copula.

Estimates the within-(episode, round, agent_group) latent correlation of human
contributions by pairwise-likelihood MLE of an exchangeable Gaussian copula
(each observation keeps the model's own predicted discrete marginal), on the
40 single-copy baseline train episodes only, and writes a copy of the input
artifact with `copula_rho` stamped in plus a JSON provenance sidecar. The
randomized-PIT moment estimator is printed as an attenuated diagnostic only.

The estimator, the bootstrap, the round-trip gate and the pre-flight are
imported unmodified from the punisher calibration (PR #146,
scripts/baselines/punishment_copula_rho.py); only the marginals differ -- they
come from a teacher-forced GNN forward pass instead of a linear bundle. Method
details: notes/autoresearch_log/punisher-severity-copula.md (appendix) and
notes/autoresearch_log/contribution-cg-copula.md.

Imports graph.py, so this runs on Raven only:
    .venv/bin/python scripts/artificial_humans/contribution_copula_rho.py \
        --model IN.pt --out OUT.pt [--preflight] [--roundtrip]
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
    data, default_values, pair_id = create_torch_data(
        df, switch_every=SWITCH_EVERY
    )
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
# artifact write: the .pt carries rho, a JSON sidecar carries the provenance
# (GraphNetwork.save writes a fixed key list, so nothing else may ride along)
# --------------------------------------------------------------------------- #
def compare_saved(before, after):
    """Every saved key except `copula_rho` must survive the round trip
    unchanged -- module weights compared parameter by parameter."""
    assert set(before) - set(after) == set(), "a saved key disappeared"
    assert set(after) - set(before) <= {"copula_rho"}, "an unexpected key appeared"
    for k, va in before.items():
        if k == "copula_rho":
            continue
        vb = after[k]
        if isinstance(va, th.nn.Module):
            sa, sb = va.state_dict(), vb.state_dict()
            assert set(sa) == set(sb), f"{k}: parameter set changed"
            for p in sa:
                assert th.equal(sa[p], sb[p]), f"{k}.{p} changed"
        elif isinstance(va, th.Tensor):
            assert th.equal(va, vb), f"{k} changed"
        else:
            assert va == vb, f"{k} changed: {va!r} -> {vb!r}"


def save_model(model, in_path, out_path, proba_ref, data, idx, meta):
    """Stamp rho into a COPY of the artifact, then reload and prove that only
    rho moved: the teacher-forced probabilities of the calibration rows must
    come back bit-identical."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.copula_rho = float(meta["copula_rho"])
    model.save(out_path)

    loaded = GraphNetwork.load(str(out_path), device="cpu")
    assert loaded.copula_rho == float(meta["copula_rho"]), "rho did not round-trip"
    back = teacher_forced_rows(loaded, data, idx)
    assert np.array_equal(back["P"], proba_ref), "reloaded logits are not identical"
    compare_saved(
        th.load(in_path, map_location="cpu"),
        th.load(out_path, map_location="cpu"),
    )

    side = out_path.with_suffix(".copula.json")
    side.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nsaved {rel(out_path)}")
    print(f"  copula_rho={f(model.copula_rho)} estimator=pairwise_mle")
    print("  teacher-forced probabilities: bit-identical after reload")
    print("  every other saved key: unchanged vs the input artifact")
    print(f"  provenance {rel(side)}")


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="input .pt (rho must be 0)")
    ap.add_argument("--out", required=True, help="output .pt with rho stamped in")
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
    in_path = Path(args.model).resolve()
    out_path = Path(args.out).resolve()
    assert in_path != out_path, "refusing to overwrite the input artifact"

    model = GraphNetwork.load(str(in_path), device="cpu")
    model.eval()
    assert model.y_name == "contribution", f"not a contributor: {model.y_name}"
    assert model.copula_rho == 0.0, f"input already carries {model.copula_rho}"
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
    print(
        f"\nOUT-OF-SAMPLE CHECK ONLY ({rel(TEST)}, "
        f"episodes={len(te_idx)})"
    )
    print(
        f"  rho MLE={f(mle_te)}  rho PIT={f(pit_te.mean())}  "
        f"rows={rows_te} cells>=2={cells_te} pairs={pr_te}"
    )

    # ---------------- pre-flight ---------------- #
    pf = None
    if args.preflight:
        ind, cop, human = pc.preflight(
            P, y, cell, rho_hat, pc.N_PREFLIGHT, SEED
        )
        pf = dict(independent=ind, copula=cop, human=human)
        print(
            f"\n=== pre-flight (go/no-go; rho is NEVER tuned to it), "
            f"{pc.N_PREFLIGHT} repeats ==="
        )
        print(f"group-spread ratio independent  {f(ind)}")
        print(f"group-spread ratio copula       {f(cop)}  (rho={f(rho_hat)})")
        print(f"group-spread ratio human        {f(human)}")

    meta = dict(
        copula_rho=float(rho_hat),
        copula_rho_se=se,
        copula_rho_ci=[ci[0], ci[1]],
        copula_estimator="pairwise_mle",
        copula_diag_pit="randomized",  # provenance of the diagnostic only
        copula_diag_pit_rho=float(rho_reps.mean()),
        copula_cell_key="episode_round_agent_group",
        copula_data_file=rel(TRAIN),
        copula_data_note=(
            "create_torch_data on the full human file (training defaults), "
            "restricted to the 40 single-copy train episodes; teacher-forced "
            "marginals from the source model; rows where contribution_valid"
        ),
        copula_n_rows=int(len(y)),
        copula_n_cells=int((sizes >= 2).sum()),
        copula_n_pairs=int(len(ii)),
        copula_n_episodes=int(len(tr_idx)),
        copula_roundtrip_max_bias=rt_bias,
        copula_preflight=pf,
        copula_holdout_rho=float(mle_te),
        source_model=rel(in_path),
        model=rel(out_path),
        script=rel(__file__),
        seed=SEED,
        date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    save_model(model, in_path, out_path, tr["P"], data, tr_idx, meta)
    print(f"total runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
