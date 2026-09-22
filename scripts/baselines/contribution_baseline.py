"""Multinomial logistic-regression baseline for the contribution AH model.

Reproduces the EXACT 5-fold CV the GNN contribution predictor used
(artifacts/artificial_humans/group_switching_contribution_50ep), trains a
simple multinomial logistic regression on each train fold, and reports
test-fold multiclass log loss -- the interpretable baseline the GNN must beat.

Faithfulness to the GNN run (config group_switching_contribution_50ep.yml):
  * same data + filtering (experiment ah_group_switching, doubled = 100 eps)
  * same seed (38381) and fold logic (get_cross_validations, group_key=pair_id)
  * features pulled from the SAME tensors create_torch_data builds
  * same target  : contribution (21 levels)
  * same mask    : contribution_valid
  * same metric  : sklearn.metrics.log_loss(labels=range(21))

Two feature sets (mirrors switch_logit_baseline.py):
  A) GNN-matched : prev_contribution, prev_punishment, agent_group
  B) Enriched    : adds the previous-round own-/other-group mean contribution
                   and their gap (the conditional-cooperation signal). Unlike
                   switch, this barely helps -- the GNN's edge on contribution
                   is the RNN's memory of the own trajectory, not a feature.

GNN reference (final-epoch test log loss, this artifact): 1.9897 (mean of 5).

Usage:
    .venv/bin/python scripts/baselines/contribution_baseline.py
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
N_LEVELS = 21
DATA = ROOT / "experiments/2group_8agent_50ep.csv"
EXPERIMENTS = ["ah_group_switching"]
MASK = "contribution_valid"
TARGET = "contribution"
BASIC = ["prev_contribution", "prev_punishment", "agent_group"]
GNN_REF = 1.9897  # final-epoch test log loss, mean over folds (this artifact)


def group_prev_means(d):
    """Previous-round group mean contribution per (episode, agent, round):
    own group (leave-one-out) and the other group, from the same tensors."""
    pc = d["prev_contribution"].numpy().astype(float)   # self's prev contribution
    gp = d["prev_agent_group"].numpy().astype(int)      # prev group membership
    G, A, T = pc.shape
    own = np.zeros_like(pc)
    oth = np.zeros_like(pc)
    for g in range(G):
        for t in range(T):
            grp, c = gp[g, :, t], pc[g, :, t]
            for s in (0, 1):
                m = grp == s
                if not m.any():
                    continue
                n = m.sum()
                own[g, m, t] = (c[m].sum() - c[m]) / (n - 1) if n > 1 else 0.0
                other = grp == (1 - s)
                if other.any():
                    oth[g, other, t] = c[m].mean()
    return own, oth


def build(d, enriched):
    f = {k: d[k].numpy().astype(float) for k in BASIC}
    if enriched:
        own, oth = group_prev_means(d)
        f["own_grp_prev_mean_c"] = own
        f["oth_grp_prev_mean_c"] = oth
        f["gap_prev_mean_c"] = own - oth
    return f


def flatten(d, feat_dict):
    mask = d[MASK].numpy().astype(bool).reshape(-1)
    X = np.stack([v.reshape(-1) for v in feat_dict.values()], axis=1)[mask]
    y = d[TARGET].numpy().astype(int).reshape(-1)[mask]
    return X, y


def full_proba(m, X, n_levels):
    """Map predict_proba onto the full [n, n_levels] label grid (classes
    absent from this train fold get a tiny floor), renormalised per row."""
    p = np.full((X.shape[0], n_levels), 1e-12)
    p[:, m.classes_] = m.predict_proba(X)
    return p / p.sum(axis=1, keepdims=True)


def run_cv(folds, enriched, label):
    print(f"\n=== {label} ===")
    lls = []
    for i, tr, te in folds:
        Xtr, ytr = flatten(tr, build(tr, enriched))
        Xte, yte = flatten(te, build(te, enriched))
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), ytr)
        ll = log_loss(yte, full_proba(m, sc.transform(Xte), N_LEVELS),
                      labels=list(range(N_LEVELS)))
        lls.append(ll)
        print(f"    fold {i}: test log_loss={ll:.4f}  (n_test={len(yte)})")
    print(f"    --> mean TEST log_loss = {np.mean(lls):.4f} (std {np.std(lls):.4f})")
    return np.mean(lls)


def constant_floor(folds):
    lls = []
    for i, tr, te in folds:
        _, ytr = flatten(tr, build(tr, False))
        _, yte = flatten(te, build(te, False))
        counts = np.bincount(ytr, minlength=N_LEVELS) + 1.0
        marg = counts / counts.sum()
        lls.append(log_loss(yte, np.tile(marg, (len(yte), 1)),
                            labels=list(range(N_LEVELS))))
    print(f"\n=== constant floor (train marginal) ===")
    print(f"    --> mean TEST log_loss = {np.mean(lls):.4f}")
    return np.mean(lls)


def main():
    th.random.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    df = pd.read_csv(DATA)
    df = df[df["experiment_name"].isin(EXPERIMENTS)]
    data, _, pair_id = create_torch_data(df)
    print(f"episodes={data['contribution'].shape[0]} (doubled), "
          f"pairs={len(set(pair_id.tolist()))}, folds={N_CV}, seed={SEED}")
    print(f"target={TARGET} ({N_LEVELS} levels), mask={MASK}")

    folds = [(i, tr, te) for i, tr, te in
             get_cross_validations(data, N_CV, 1.0, group_key=pair_id)
             if i is not None]

    floor = constant_floor(folds)
    a = run_cv(folds, False, "A) LR, GNN-matched features")
    b = run_cv(folds, True, "B) LR, enriched (+own/other/gap group mean)")

    print("\n" + "=" * 56)
    print(f"{'constant floor':<42} {floor:.4f}")
    print(f"{'LR  A  (GNN features)':<42} {a:.4f}")
    print(f"{'LR  B  (enriched features)':<42} {b:.4f}")
    print(f"{'GNN contribution model (ref)':<42} {GNN_REF:.4f}")
    print("=" * 56)


if __name__ == "__main__":
    main()
