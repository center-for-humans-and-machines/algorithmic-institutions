# Autoresearch log: contribution-xgboost

## 1. Declaration

- **Slot**: contribution
- **Base model**: new `xgb` categorical (gradient-boosted trees, multinomial
  softprob over the 21 levels) — nearest kin is the categorical linear
  (`lin_multinomial` family): same prev-family features, same single-copy
  training data, same CV protocol, same `.joblib` bundle and
  `LinearAHAdapter` predict_proba path. Scored by swapping into the
  reference stack's contribution slot (replacing the GNN contributor).
- **Target rows** (reference cell, `plots/simulation/
  23_2g8a_self_gnn_contr_gnn_switch/evaluation/scores.csv`, pairing
  `lin_multinomial_self`):
  - **CG 9.850260681510413** (> 5; aim 2–5)
  - **RCD 2.7723214938046725** (2–5; aim < 2)
  - **RCA 2.0348663297861047** (2–5; aim < 2)
- **Stack baseline** (same cell): rows <= 1 = **11/21**, mean =
  **1.7595567320354153**.
- **Hypothesis**: human contribution is a sticky, discrete transition
  process with interactions — the punishment response is conditional on the
  previous own level, and conditional cooperation on the group context.
  Gradient-boosted trees over the existing prev-family feature pool learn
  the full 21-way transition structure (which PR #148 had to hand-code as a
  prev-contribution one-hot) *and* its interactions automatically, while
  keeping a calibrated multinomial predictive distribution — whose absence
  (unimodal Gaussian head) is what killed PR #151 via the CG explosion.
  A sharper conditional 21-way distribution should move the response rows
  (RCD, RCA) and, replacing the GNN contributor (CG 9.85), land CG at the
  cat-family level (2–5).
- **Planned change** (one change: a new model class in the contribution
  slot): register `xgb` in `scripts/baselines/baseline_models.py`, add
  `configs/training/baselines/contribution/xgb.yml`, extend
  `src/aimanager/simulation/linear_ah.py` to sample `xgb` bundles through
  the multinomial predict_proba path, add `xgboost` to the project deps.

**Known priors this builds on**

- PR #148 [SUCCESS]: one-hot transition structure, RCA > 5 → 2–5 in the cat
  family; collateral CG worse (stickier individuals), RCB worse.
- PR #151 [FAIL]: nonlinear gaussian_mlp band-upgraded RCA but mean rose —
  CG 3.98 → 6.58; unimodal head could not hold repeat spike + defection
  tail.
- PR #149/#153 [FAIL]: within-round copula / peer attention — CG's root
  cause (between-participant correlation) is not addressable per-agent;
  this experiment does **not** claim to fix CG below the cat-family floor,
  only to recover the cat-family CG level while beating the GNN on the
  response rows.

## 2. Plan

1. [ ] Branch `auto/contribution-xgboost` + worktree; declaration committed.
2. [ ] Add `xgboost` to `pyproject.toml`; `uv sync` in the worktree.
3. [ ] Register `xgb` in `baseline_models.py` (categorical target,
   `model: xgb`; griddable keys `n_estimators`, `max_depth`,
   `learning_rate`, `min_child_weight`, `subsample`, `reg_lambda`;
   metric `log_loss`, scoring shared with multinomial). Guard
   `inspect_best_model.py` coefficient views (trees have no `coef_`;
   `--save-best` path must work).
4. [ ] `configs/training/baselines/contribution/xgb.yml`: same data section
   as `cat.yml` (single-copy train split, `contribution_valid` mask,
   4 folds, seed 38381), prev-family feature blocks, modest hyperparameter
   grid (trees do their own feature selection — fewer feature-set combos,
   more `setting` cells than the linear configs).
5. [ ] CV sweep locally (`run_baseline_cv.py`); pick rank-1;
   `inspect_best_model.py --save-best` → locked-test evaluation →
   `artifacts/baselines/contribution_xgb_best.joblib`. Gate: TEST log loss
   must beat the incumbent categorical (cat_prev_onehot 1.882) — the GNN
   reference is 1.9897.
6. [ ] Teacher-forced pre-flight: sampled repeat rate + transition diagonal
   vs human (0.414 exact repeats); documented go/no-go.
7. [ ] Extend `linear_ah.py` (`model_type == "xgb"` → multinomial sampling
   path); local adapter smoke test (no PyG needed).
8. [ ] Stage 1: `configs/simulation/manager_testing/
   23_2g8a_self_xgb_contr_gnn_switch.yml` (reference config, contribution
   artifact swapped, own output dir); xgboost into the Raven venv; sim on
   Raven; fetch; `python -m aimanager evaluate`; score vs the reference
   cell.
9. [ ] Verdict per §2 (targets drop, 11/21 must not fall, mean 1.76 must
   not rise, band upgrade required). If band upgrade: Stage 2 sweep
   (8-config family + `evaluation_sweep.py`). PR either way.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-18 | (baseline) reference cell, gnn contr unchanged | 1 | CG 9.850260681510413, RCD 2.7723214938046725, RCA 2.0348663297861047 | 11/21 | 1.7595567320354153 | baseline |

## 4. Notes

1. Declared before any code: the xgb candidate is scored against the
   *reference* (GNN) contributor per §3, but its engineering lineage is the
   linear-baseline pipeline; the cat family's known CG-vs-stickiness
   trade-off (PR #148 collateral) is the main risk to the "mean must not
   rise" gate, offset by starting from the GNN's CG 9.85.
