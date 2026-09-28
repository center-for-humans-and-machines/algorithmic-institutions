"""Missing state behind the contribution copula.

Players decide independently given the true situation, so any correlation left
in the contribution model's errors within a group means the model is missing
part of the situation. The herding copula (rho ~ 0.04, static per episode)
stands in for that with a random number. This script asks how much of the
residual within-group correlation is explained by observable group state the
skip trunk does not currently see, and how much is genuinely unobservable.

Two stages, because only the first needs PyG:

  predict  (Raven login node)  teacher-force the PR #181 stimulus-skip trunk
           on the 50 canonical human games (one copy per game, the copy the
           evaluation suite keeps) and write one row per valid agent-round:
           ids, the realised contribution, the expected contribution and the
           full 21-level marginal.
  analyse  (local, pandas/numpy/statsmodels)  build the residuals on the level
           scale and on the copula's latent scale, reproduce the baseline
           (raw within-group co-movement, residual correlation, the copula
           script's pairwise MLE), score every candidate missing-state
           variable, forward-select a joint set, and check persistence.

    python scripts/data_analysis/copula_missing_state.py predict [--model PT]
    python scripts/data_analysis/copula_missing_state.py analyse
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src", "scripts/artificial_humans", "scripts/baselines"):
    sys.path.insert(0, str(ROOT / sub))

OUT_DIR = ROOT / "plots/data_analysis/evaluation/copula_missing_state"
TABLE = OUT_DIR / "residual_table.parquet"
FULL = ROOT / "experiments/2group_8agent_50ep.csv"
DEFAULT_MODEL = (
    "artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip"
    "/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
)
N_LEVELS = 21
SEED = 38381


# --------------------------------------------------------------------------- #
# stage 1: teacher-forced marginals on the canonical human frame (Raven)
# --------------------------------------------------------------------------- #
def predict(args):
    import contribution_copula_rho as cc
    import punishment_copula_rho as pc
    from aimanager.generic.graph import GraphNetwork

    model = GraphNetwork.load(str(ROOT / args.model), device="cpu")
    model.eval()
    assert model.y_name == "contribution"
    print(f"model {args.model}")
    print(f"  x_encoding={[e['name'] for e in model.x_encoding]}")
    print(f"  edge_encoding={[e['name'] for e in model.edge_encoding]}")
    print(f"  group_vnode={model.group_vnode} stimulus_skip={model.stimulus_skip}")

    data, pair_id, key_to_idx, defaults = cc.load_full()
    tr = cc.select_split(key_to_idx, cc.TRAIN, cc.N_TRAIN_EP)
    te = cc.select_split(key_to_idx, cc.TEST, cc.N_TEST_EP)
    canon = np.array(sorted(set(tr.tolist()) | set(te.tolist())))
    # the evaluation suite's rule (convert.load_human): the min episode_id copy
    raw = pd.read_csv(FULL)
    keep = raw["episode_id"] == raw.groupby("pair_id")["episode_id"].transform("min")
    eval_keys = set(cc.episode_key(raw[keep]))
    idx_to_key = {v: k for k, v in key_to_idx.items()}
    assert {idx_to_key[i] for i in canon} == eval_keys, "canonical copy mismatch"
    print(f"canonical frame: {len(canon)} episodes (train {len(tr)} + test {len(te)})")

    rows = cc.teacher_forced_rows(model, data, canon)
    P, y = rows["P"], rows["y"]
    assert P.shape[1] == N_LEVELS
    ep = canon[rows["episode"]]  # back to the full tensor's episode index
    keys = [idx_to_key[int(e)] for e in ep]
    out = pd.DataFrame(
        {
            "episode_idx": ep,
            "global_group_id": [k.rsplit("__", 1)[0] for k in keys],
            "episode_id": [int(k.rsplit("__", 1)[1]) for k in keys],
            "pair_id": pair_id[ep],
            "in_train_split": np.isin(ep, tr),
            "player_id": rows["agent"],
            "round": rows["round"],
            "group": rows["group"],
            "y": y,
            "e": P @ np.arange(N_LEVELS, dtype=np.float64),
        }
    )
    for k in range(N_LEVELS):
        out[f"p{k}"] = P[:, k]

    # the copula script's own estimator on the 40 train episodes, for the
    # baseline reproduction: it must land on the stamped rho 0.03949863621805423
    sel = out["in_train_split"].to_numpy()
    rho_tr, n_pairs, n_rows, _ = pc.mle_on_rows(P, y, rows["cell"], sel)
    rho_all, n_pairs_a, n_rows_a, _ = pc.mle_on_rows(P, y, rows["cell"])
    print(
        f"pairwise MLE rho: train split {rho_tr!r} (rows={n_rows} pairs={n_pairs})"
        f" | all 50 {rho_all!r} (rows={n_rows_a} pairs={n_pairs_a})"
    )
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(TABLE, index=False)
    meta = dict(
        model=args.model,
        rho_mle_train40=rho_tr,
        rho_mle_canon50=rho_all,
        n_rows=int(len(out)),
        n_pairs_train40=int(n_pairs),
        n_pairs_canon50=int(n_pairs_a),
        contribution_default=float(defaults["contribution"]),
        punishment_default=float(defaults["punishment"]),
    )
    pd.Series(meta).to_json(OUT_DIR / "predict_meta.json", indent=2)
    print(f"wrote {TABLE.relative_to(ROOT)} ({len(out)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--model", default=DEFAULT_MODEL)
    for name in ("analyse", "persistence"):
        a = sub.add_parser(name)
        a.add_argument("--n-boot", type=int, default=200)
        a.add_argument("--n-boot-mle", type=int, default=50)
    args = ap.parse_args()
    t0 = time.time()
    if args.stage == "predict":
        predict(args)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from copula_missing_state_analysis import analyse, persistence_boot

        (analyse if args.stage == "analyse" else persistence_boot)(args)
    print(f"wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
