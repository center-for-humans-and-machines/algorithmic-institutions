"""Logistic-regression baseline for the switch AH model (50ep doubled).

Reproduces the EXACT 5-fold CV the GNN switch predictor used
(artifacts/artificial_humans/switch_pred_opt_50ep_doubled), trains a simple
logistic regression on each train fold, and reports test-fold log loss on the
predicted switch probabilities -- the interpretable baseline the GNN must beat.

Faithfulness to the GNN run (config opt_50ep_doubled.yml):
  * same data + filtering (experiment ah_group_switching, doubled = 100 eps)
  * same seed (38381) and fold logic (get_cross_validations, group_key=pair_id)
  * features pulled from the SAME tensors create_torch_data builds
  * same target  : does_switch
  * same mask    : switch_valid (decision rounds, non-timeout)
  * same metric  : sklearn.metrics.log_loss(labels=[0, 1])

Two feature sets:
  A) GNN-matched : prev_common_good, prev_punishment, prev_agent_group,
                   round_number   -> apples-to-apples (isolates the value of
                   the GNN's RNN + graph edges over a flat linear model).
  B) Enriched    : adds self_prev_c and the own/other common-good gap, the
                   features the PR argues are the high-value fix.

GNN reference (final-epoch test log loss, this artifact): 0.5163 (mean of 5).

Usage:
    .venv/bin/python switch_logit_baseline.py
"""
import os
import random

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np
import pandas as pd
import torch as th
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from aimanager.generic.data import create_torch_data, get_cross_validations

SEED = 38381
N_CV = 5
FRACTION_TRAINING = 1.0
SWITCH_EVERY = 4
DATA = "experiments/2group_8agent_50ep.csv"
EXPERIMENTS = ["ah_group_switching"]
MASK = "switch_valid"
TARGET = "does_switch"
GNN_REF = 0.5163  # final-epoch test log loss, mean over folds (this artifact)


def build_features(d):
    """Return a flat dict of per-(episode,agent,round) feature arrays + mask.

    All arrays share shape [G, A, T] and are pulled straight from the tensors
    create_torch_data produced (same values the GNN saw)."""
    f = {k: d[k].numpy() for k in
         ["prev_common_good", "prev_punishment", "prev_agent_group",
          "round_number", "prev_contribution"]}
    # other-group previous common good (the signal the GNN must extract via the
    # graph): within each episode/round the two sub-groups each have one cg
    # value; pick the one NOT equal to the agent's previous group.
    grp_prev = d["prev_agent_group"].numpy()          # [G,A,T] in {0,1}
    cg_prev = d["prev_common_good"].numpy()           # own group's cg @ t-1
    G, A, T = cg_prev.shape
    oth = np.zeros_like(cg_prev)
    for g in range(G):
        for t in range(T):
            gp, cg = grp_prev[g, :, t], cg_prev[g, :, t]
            m0 = gp == 0
            cg0 = cg[m0].mean() if m0.any() else 0.0
            cg1 = cg[~m0].mean() if (~m0).any() else 0.0
            oth[g, :, t] = np.where(gp == 0, cg1, cg0)
    f["oth_common_good_prev"] = oth
    f["gap_common_good_prev"] = oth - cg_prev
    y = d[TARGET].numpy().astype(int)
    mask = d[MASK].numpy().astype(bool)
    return f, y, mask


def flatten(f, y, mask, feats):
    sel = mask.reshape(-1)
    X = np.stack([f[k].reshape(-1) for k in feats], axis=1)[sel]
    return X, y.reshape(-1)[sel]


def run_cv(folds, feats, label):
    print(f"\n=== {label} ===")
    print(f"    features: {feats}")
    test_lls, train_lls = [], []
    for i, tr, te in folds:
        ftr, ytr, mtr = build_features(tr)
        fte, yte, mte = build_features(te)
        Xtr, ytr = flatten(ftr, ytr, mtr, feats)
        Xte, yte = flatten(fte, yte, mte, feats)
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
        ptr = m.predict_proba(sc.transform(Xtr))
        pte = m.predict_proba(sc.transform(Xte))
        ll_tr = log_loss(ytr, ptr, labels=[0, 1])
        ll_te = log_loss(yte, pte, labels=[0, 1])
        train_lls.append(ll_tr); test_lls.append(ll_te)
        print(f"    fold {i}: test log_loss={ll_te:.4f}  (train {ll_tr:.4f}, "
              f"n_test={len(yte)}, switch_rate={yte.mean():.3f})")
    print(f"    --> mean TEST log_loss = {np.mean(test_lls):.4f} "
          f"(std {np.std(test_lls):.4f})  | train {np.mean(train_lls):.4f}")
    return np.mean(test_lls)


def constant_baseline(folds):
    """Predict the train switch rate for everyone -> the floor any model beats."""
    lls = []
    for i, tr, te in folds:
        _, ytr, mtr = build_features(tr)
        _, yte, mte = build_features(te)
        ytr = ytr.reshape(-1)[mtr.reshape(-1)]
        yte = yte.reshape(-1)[mte.reshape(-1)]
        p = ytr.mean()
        proba = np.column_stack([np.full(len(yte), 1 - p), np.full(len(yte), p)])
        lls.append(log_loss(yte, proba, labels=[0, 1]))
    print(f"\n=== constant switch-rate baseline (floor) ===")
    print(f"    --> mean TEST log_loss = {np.mean(lls):.4f}")
    return np.mean(lls)


def main():
    th.random.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    df = pd.read_csv(DATA)
    df = df[df["experiment_name"].isin(EXPERIMENTS)]
    data, _, pair_id = create_torch_data(df, switch_every=SWITCH_EVERY)
    n_ep = data["contribution"].shape[0]
    print(f"episodes={n_ep} (doubled), pairs={len(set(pair_id.tolist()))}, "
          f"folds={N_CV}, seed={SEED}")

    folds = [(i, tr, te) for i, tr, te in
             get_cross_validations(data, N_CV, FRACTION_TRAINING,
                                   holdout_fold=None, group_key=pair_id)
             if i is not None]

    floor = constant_baseline(folds)
    a = run_cv(folds, ["prev_common_good", "prev_punishment",
                       "prev_agent_group", "round_number"],
               "A) LR, GNN-matched features (apples-to-apples)")
    b = run_cv(folds, ["prev_common_good", "oth_common_good_prev",
                       "gap_common_good_prev", "prev_punishment",
                       "prev_contribution", "round_number"],
               "B) LR, enriched (adds other-group / gap / self contribution)")

    print("\n" + "=" * 60)
    print(f"{'constant floor':<42} {floor:.4f}")
    print(f"{'LR  A  (GNN features)':<42} {a:.4f}")
    print(f"{'LR  B  (enriched features)':<42} {b:.4f}")
    print(f"{'GNN switch model (final-epoch, ref)':<42} {GNN_REF:.4f}")
    print("=" * 60)
    print("lower is better; GNN should beat the same-feature LR (A).")


if __name__ == "__main__":
    main()
