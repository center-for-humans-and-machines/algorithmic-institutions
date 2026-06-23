"""Multinomial logistic-regression baseline for the punishment AH model.

Reproduces the EXACT 5-fold CV the GNN punishment predictor used
(artifacts/artificial_humans/punishment_rnn_edge_50ep_doubled), trains a
simple multinomial logistic regression on each train fold, and reports
test-fold multiclass log loss -- the interpretable baseline the GNN must beat.

Faithfulness to the GNN run (config punishment/rnn_edge_50ep_doubled.yml):
  * same data + filtering (experiment ah_group_switching, doubled = 100 eps)
  * same seed (38381) and fold logic (get_cross_validations, group_key=pair_id)
  * features pulled from the SAME tensors create_torch_data builds
  * same target  : punishment (31 levels)
  * same mask    : punishment_valid
  * same metric  : sklearn.metrics.log_loss(labels=range(31))

Note: like the GNN config, the baseline has NO group feature -- the PR's point
is that a group-relative punishment rule can't be isolated without one.

GNN reference (final-epoch test log loss, this artifact): 1.2030 (mean of 5).

Usage:
    .venv/bin/python scripts/baselines/punishment_baseline.py
"""
import os
import random
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np
import pandas as pd
import torch as th
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from aimanager.generic.data import create_torch_data, get_cross_validations

ROOT = Path(__file__).resolve().parents[2]
SEED = 38381
N_CV = 5
N_LEVELS = 31
SWITCH_EVERY = 4
DATA = ROOT / "experiments/2group_8agent_50ep.csv"
EXPERIMENTS = ["ah_group_switching"]
MASK = "punishment_valid"
TARGET = "punishment"
FEATS = ["prev_contribution", "prev_punishment", "is_first"]
GNN_REF = 1.2030  # final-epoch test log loss, mean over folds (this artifact)


def flatten(d, feats):
    mask = d[MASK].numpy().astype(bool).reshape(-1)
    X = np.stack([d[k].numpy().astype(float).reshape(-1) for k in feats], axis=1)[mask]
    y = d[TARGET].numpy().astype(int).reshape(-1)[mask]
    return X, y


def full_proba(m, X, n_levels):
    """Map predict_proba onto the full [n, n_levels] label grid (classes
    absent from this train fold get a tiny floor), renormalised per row."""
    p = np.full((X.shape[0], n_levels), 1e-12)
    p[:, m.classes_] = m.predict_proba(X)
    return p / p.sum(axis=1, keepdims=True)


def main():
    th.random.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    df = pd.read_csv(DATA)
    df = df[df["experiment_name"].isin(EXPERIMENTS)]
    data, _, pair_id = create_torch_data(df, switch_every=SWITCH_EVERY)
    print(f"episodes={data['contribution'].shape[0]} (doubled), "
          f"pairs={len(set(pair_id.tolist()))}, folds={N_CV}, seed={SEED}")
    print(f"target={TARGET} ({N_LEVELS} levels), mask={MASK}, features={FEATS}\n")

    folds = [(i, tr, te) for i, tr, te in
             get_cross_validations(data, N_CV, 1.0, group_key=pair_id)
             if i is not None]

    floor_lls, lr_lls = [], []
    for i, tr, te in folds:
        Xtr, ytr = flatten(tr, FEATS)
        Xte, yte = flatten(te, FEATS)
        counts = np.bincount(ytr, minlength=N_LEVELS) + 1.0
        marg = counts / counts.sum()
        floor = log_loss(yte, np.tile(marg, (len(yte), 1)), labels=list(range(N_LEVELS)))
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
        ll = log_loss(yte, full_proba(m, sc.transform(Xte), N_LEVELS),
                      labels=list(range(N_LEVELS)))
        floor_lls.append(floor); lr_lls.append(ll)
        print(f"  fold {i}: floor={floor:.4f}  LR={ll:.4f}  (n_test={len(yte)})")

    floor, lr = np.mean(floor_lls), np.mean(lr_lls)
    print("\n" + "=" * 56)
    print(f"{'constant floor':<40} {floor:.4f}")
    print(f"{'LR (GNN features)':<40} {lr:.4f}  (std {np.std(lr_lls):.4f})")
    print(f"{'GNN punishment model (ref)':<40} {GNN_REF:.4f}")
    print("=" * 56)


if __name__ == "__main__":
    main()
