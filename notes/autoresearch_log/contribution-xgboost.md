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

1. [x] Branch `auto/contribution-xgboost` + worktree; declaration committed.
2. [x] Add `xgboost` to `pyproject.toml`; `uv sync` in the worktree.
3. [x] Register `xgb` in `baseline_models.py` (categorical target,
   `model: xgb`; griddable keys `n_estimators`, `max_depth`,
   `learning_rate`, `min_child_weight`, `subsample`, `reg_lambda`;
   metric `log_loss`, scoring shared with multinomial). Guard
   `inspect_best_model.py` coefficient views (trees have no `coef_`;
   `--save-best` path must work).
4. [x] `configs/training/baselines/contribution/xgb.yml`: same data section
   as `cat.yml` (single-copy train split, `contribution_valid` mask,
   4 folds, seed 38381), prev-family feature blocks, modest hyperparameter
   grid (trees do their own feature selection — fewer feature-set combos,
   more `setting` cells than the linear configs).
5. [x] CV sweep locally (`run_baseline_cv.py`); pick rank-1;
   `inspect_best_model.py --save-best` → locked-test evaluation →
   `artifacts/baselines/contribution_xgb_best.joblib`. Gate: TEST log loss
   must beat the incumbent categorical (cat_prev_onehot 1.882) — the GNN
   reference is 1.9897. (Grid-edge finding → one refinement sweep,
   `xgb_refine.yml`; gate re-anchored to the merged incumbent 2.2726, see
   Notes 2 — the 1.882 stretch reference was beaten anyway.)
6. [x] Teacher-forced pre-flight: sampled repeat rate + transition diagonal
   vs human (0.414 exact repeats); documented go/no-go.
7. [x] Extend `linear_ah.py` (`model_type == "xgb"` → multinomial sampling
   path); local adapter smoke test (no PyG needed).
8. [x] Stage 1: `configs/simulation/manager_testing/
   23_2g8a_self_xgb_contr_gnn_switch.yml` (reference config, contribution
   artifact swapped, own output dir); xgboost into the Raven venv; sim on
   Raven; fetch; `python -m aimanager evaluate`; score vs the reference
   cell. (Revised after run 1: second iterate with the 5-feature tie-rule
   cell, `contribution_xgb_lean.joblib`, see Notes 5.)
9. [x] Verdict per §2 (targets drop, 11/21 must not fall, mean 1.76 must
   not rise, band upgrade required). If band upgrade: Stage 2 sweep
   (8-config family + `evaluation_sweep.py`). PR either way. (Not kept —
   rows <= 1 regressed in both runs; no Stage 2; [FAIL] PR.)

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-18 | (baseline) reference cell, gnn contr unchanged | 1 | CG 9.850260681510413, RCD 2.7723214938046725, RCA 2.0348663297861047 | 11/21 | 1.7595567320354153 | baseline |
| 2026-08-18 | xgb 10-feature (600 trees, d3, lr .02, mcw 30, lam 30) in the contribution slot | 1 | CG 4.404848115505977 (band upgrade >5 → 2–5), RCD 1.145432470718043 (band upgrade 2–5 → 1–2), RCA 2.3289630761228737 (regressed) | 7/21 | 1.5113915203948058 | not kept — rows <= 1 fell 11 → 7 (marginal C block: CA 2.07, CB 1.23, CD 1.21, CF 1.01, RSA 1.19) |
| 2026-08-18 | xgb 5-feature lean (same setting, tie-rule cell) in the contribution slot | 1 | CG 4.004652090744679 (band upgrade >5 → 2–5), RCD 1.8096720704193887 (band upgrade 2–5 → 1–2), RCA 2.1954094554624577 (regressed) | 9/21 | 1.4481830207797590 | not kept — rows <= 1 fell 11 → 9 (CA 1.8318, CF 1.0292, RSA 1.0454); **experiment closed as [FAIL]** |

## 4. Notes

1. Declared before any code: the xgb candidate is scored against the
   *reference* (GNN) contributor per §3, but its engineering lineage is the
   linear-baseline pipeline; the cat family's known CG-vs-stickiness
   trade-off (PR #148 collateral) is the main risk to the "mean must not
   rise" gate, offset by starting from the GNN's CG 9.85.
2. First sweep (`data/baselines/contr_xgb.csv`, 192 cells): rank-1 CV
   log_loss 2.1153 (prev_contribution, prev_punishment, is_first,
   prev_contribution_mean_group, prev_contribution_mean_other) — already
   beats the merged incumbent categorical (CV 2.4455, TEST 2.2726). Every
   top-10 cell sits at the grid edge (max_depth 3, lr 0.05,
   min_child_weight 30, reg_lambda 10) → one refinement sweep past the
   edge (`xgb_refine.yml`: depth 2–3, lr 0.02–0.05, mcw 30–120, lambda
   10–30, 600 trees) on the two winning context sets. Gate note: the plan's
   TEST gate cited cat_prev_onehot's 1.882, but that artifact was never
   merged; the binding gate is the merged incumbent (2.2726), with 1.882
   kept as the stretch reference and the teacher-forced repeat rate as the
   behavioral go/no-go.
3. Refinement sweep (`contr_xgb_refine.csv`, 144 cells): rank-1 CV 2.1036 —
   600 trees, max_depth 3, lr 0.02, min_child_weight 30 (now interior),
   subsample 0.8, reg_lambda 30, 10 features (self + group context +
   round_number). Top ~9 cells within one SE; rank-1 kept (also the
   behaviorally richer set: round_number carries the endgame decline).
   Saved via `--save-best`: **TEST log loss 1.8342** (floor 2.7180) — beats
   the merged incumbent cat (2.2726), the GNN (1.9897), and the unmerged
   one-hot stretch reference (1.882). Likelihood gate passed.
4. Teacher-forced pre-flight (locked test split): sampled repeat rate
   0.4006 (human convention 0.414; linear cat 0.193; PR #148 one-hot
   0.401). Transition diagonal matched at the dominant anchors (prev=0:
   0.454 vs human 0.456; prev=20: 0.821 vs 0.851); documented weak spot at
   the focal midpoint (prev=10: model 0.382 vs human 0.704, n=388).
   **Go for Stage 1.**
5. Stage-1 run 1 (10-feature): both CG and RCD band-upgrade and the mean
   drops 1.76 → 1.51, but rows <= 1 falls 11 → 7 — the GNN's marginal
   C-block advantage (CA/CB/CD/CF, plus RSA) does not survive the swap.
   Not kept per §2. Plan revision (validated): one more Stage-1 iterate
   with the refinement sweep's 5-feature CV-tie cell (rank 4, within one
   SE) — §5's tie rule favored it anyway, and dropping round_number /
   sizes / common_good tests whether the extra context features drive the
   marginal drift under sim feedback. No other change.
6. Stage-1 run 2 (lean): the leaner feature set recovers most of the
   marginal damage (CB 0.89, CD 0.90, CE 0.90 new <= 1; CC 1.11) and gives
   the best cell mean recorded in this line (1.4482 vs reference 1.7596),
   but rows <= 1 lands at 9/21: CA 1.83, CF 1.029, RSA 1.045. CF and RSA
   are within noise of the line; CA is not — every discrete per-round
   sampler in the sweep history sits at CA ~1.7–2.3 (slot avg 1.94), so a
   third variant would be chasing the draw, not the model. Closed as
   [FAIL].
7. What the next agent should know: (a) xgb is now the best per-agent
   conditional model of contribution by likelihood (TEST 1.834 vs GNN
   1.9897 / one-hot 1.882) and it halves CG (9.85 → 4.00) and SC (3.27 →
   2.07) while band-upgrading RCD — the response structure is real and
   cheap to get; (b) what it cannot buy is the GNN's marginal C block:
   CA needs the RNN's own-trajectory memory (or a group/trajectory latent),
   not a better per-round conditional; (c) RCB regressed in both runs
   (1.93 → 2.39/2.47), the same punished-change collateral PR #148 saw —
   a punishment-interaction feature stays a candidate; (d) the natural
   follow-up is a hybrid: xgb (or one-hot) emission + a trajectory-level
   latent state, per the PR #151 post-mortem; the xgb machinery
   (XGBLevelClassifier, adapter path, configs) is sound and reusable.
