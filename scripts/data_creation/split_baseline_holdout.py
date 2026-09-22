"""Physically split the undoubled 50-episode data into a LOCKED holdout test set
and a train set, for the hand-crafted linear baselines (issue #119).

Rationale: keep the test episodes in a separate file so they can't be touched
during feature/model selection -- a stronger guarantee than a code-level holdout
flag. All development runs on the train file; the test file is opened only once,
at the very end.

The test set is one fold (HOLDOUT_FOLD) of a pair-level split (seed SEED,
group_key=pair_id) over the undoubled data (flipped pair-copies dropped). This is
a ONE-TIME carve: no cv_fold column is written -- the train file's CV folds are
decided at run time by run_baseline_cv's cv args (seed + n_folds).

Writes:
  experiments/baseline/2group_8agent_50ep_bline_test.csv   (locked holdout)
  experiments/baseline/2group_8agent_50ep_bline_train.csv  (the rest)

Usage:
    .venv/bin/python scripts/data_creation/split_baseline_holdout.py
"""
import os
import random
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np
import pandas as pd
import torch as th

from aimanager.generic.data import create_torch_data, get_cross_validations

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments/2group_8agent_50ep.csv"
OUTDIR = ROOT / "experiments/baseline"
SEED = 38381
N_FOLDS = 5
HOLDOUT_FOLD = 0
EXPERIMENTS = ["ah_group_switching"]


def main():
    th.random.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    df = pd.read_csv(SRC)
    df = df[df["experiment_name"].isin(EXPERIMENTS)]
    df = df[~df["global_group_id"].str.contains("(flipped)", regex=False)]  # undoubled

    data, _, pair_id = create_torch_data(df)
    G = data["contribution"].shape[0]
    pid = np.asarray(pair_id)

    # canonical pair-level fold assignment (same as prepare_data)
    data["_eid"] = th.arange(G)
    fold_of_ep = np.full(G, -1, int)
    for i, _, te in get_cross_validations(data, N_FOLDS, 1.0, group_key=pair_id):
        if i is None:  # trailing (None, .., None)
            continue
        for e in te["_eid"].tolist():
            fold_of_ep[e] = i
    assert (fold_of_ep >= 0).all(), "some episode was not assigned a fold"

    # split only -- carve the locked test set (fold 0). We do NOT stamp a
    # cv_fold column: the train file's CV folds are decided at run time by
    # run_baseline_cv's cv args (seed + n_folds), not baked into the data.
    fold_of_pair = {int(pid[e]): int(fold_of_ep[e]) for e in range(G)}
    fold_col = df["pair_id"].map(fold_of_pair)
    assert fold_col.notna().all(), "unmapped pair_id"

    test = df[fold_col == HOLDOUT_FOLD]
    train = df[fold_col != HOLDOUT_FOLD]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    test_path = OUTDIR / "2group_8agent_50ep_bline_test.csv"
    train_path = OUTDIR / "2group_8agent_50ep_bline_train.csv"
    test.to_csv(test_path, index=False)
    train.to_csv(train_path, index=False)

    fold_sizes = {f: sum(v == f for v in fold_of_pair.values()) for f in range(N_FOLDS)}
    test_pairs = sorted(p for p, f in fold_of_pair.items() if f == HOLDOUT_FOLD)
    print(f"episodes={G}, pairs={len(fold_of_pair)} (undoubled)")
    print(f"fold sizes (pairs): {fold_sizes}")
    print(f"TEST  = fold {HOLDOUT_FOLD}: {len(test_pairs)} pairs {test_pairs}")
    print(f"        {len(test)} rows -> {test_path.relative_to(ROOT)}")
    print(f"TRAIN = rest: {len(fold_of_pair) - len(test_pairs)} pairs, "
          f"{len(train)} rows -> {train_path.relative_to(ROOT)}  "
          f"(CV folds decided at run time by run_baseline_cv's cv args)")


if __name__ == "__main__":
    main()
