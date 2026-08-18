# Autoresearch log: contribution-mlp-regressor

## 1. Declaration

- **Slot:** contribution
- **Base model:** reference GNN contributor
  (`architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`), scored in the
  reference cell `23_2g8a_self_gnn_contr_gnn_switch`, run
  `lin_multinomial_self`.
- **Target rows** (starting scores from the reference cell):
  - **CG 9.850260681510413** (band > 5) — group contribution spread ratio
  - **RCD 2.7723214938046725** (band 2–5) — switching pull
  - **RCA 2.0348663297861047** (band 2–5) — contribution change by round type
- **Stack baselines:** rows <= 1 = 11/21, mean = 1.7595567320354153.
- **Hypothesis:** the human contribution response is nonlinear and
  interactive — the punishment response is conditional on the player's own
  previous level, conditional cooperation on the group context. The linear
  ridge mean cannot express this (ridge slot-average RCA ~5.8) while its
  homoscedastic residual sampling keeps the marginal C block scoreable and
  holds CG near 5 instead of the GNN's ~10. A small nonlinear mean function
  under ridge's exact sampling convention should capture the response
  structure (RCA, RCD) and reduce group drift (CG), with none of PR #151's
  heteroscedastic-sigma fragility and none of PR #155's deterministic
  marginal collapse.
- **Planned change (one change):** replace the linear mean of the continuous
  contribution baseline with a 2-layer MLP mean — `Linear(in, hidden) ->
  tanh -> Linear(hidden, 1)`, trained locally with full-batch Adam on plain
  MSE (no sigma head, no NLL, no categorical softmax). Everything else
  follows the ridge convention unchanged: single-copy train split, grouped
  CV seed 38381, feature sets drawn from the proven baseline pools,
  `StandardScaler`, joblib bundle with homoscedastic
  `sigma = sqrt(train MSE)`, sampled at sim time as
  `round(clip(mu + randn * sigma))` through the existing `LinearAHAdapter`
  ridge branch.
- **Prior art consulted (§9.1):** PR #151 (gaussian MLP: RCA upgrade, CG
  exploded via heteroscedastic independent sampling), PR #154 (xgboost: CG +
  RCD upgrades, killed by the marginal C block / CA), PR #155 (deterministic
  continuous GNN head: unscorable, no 0/20 mass — "do not retry
  point-estimate continuous heads"). This candidate differs from all three:
  MSE point estimate for the *mean* only, with ridge's calibrated
  homoscedastic sampling supplying the marginal mass that #155 lacked, and
  no sigma(x) head to shrink as in #151.

## 2. Plan

Validated 2026-08-18 (targets per §2, all steps legal per §5, frozen surface
untouched).

- [x] 1. Add `scripts/baselines/mlp_regressor.py` — `MLPRegressor`
  (`Linear(in,hidden) -> tanh -> Linear(hidden,1)`, full-batch Adam on MSE,
  seed, output-bias warm start to `mean(y)`), sklearn-ish `fit`/`predict`
  only (no `predict_std`, no `nll`).
- [x] 2. Wire `mlp` into `scripts/baselines/baseline_models.py` —
  `_SPEC["mlp"] = {hidden(int), weight_decay, lr, epochs(int)}`,
  `_METRIC["mlp"] = "mse"`, `resolve_model` continuous set, `build_model`
  branch, `predict_scores`/`floor_score` down the plain-MSE path.
- [x] 3. Add `tests/baselines/test_mlp_regressor.py` (determinism, shape,
  beats intercept-only floor on a nonlinear synthetic target, registry
  contract) and run local pytest.
- [x] 4. Add `configs/training/baselines/contribution/mlp.yml` (shortlist:
  ridge rank-1 9-feature set, gaussian rank-1 12-feature set, 1–2 lean
  subsets; seed 38381; output `data/baselines/contribution_mlp_cv.csv`) and
  smoke-train one cell to confirm local trainability + per-fit wall-clock.
- [x] 5. Run the full shortlist CV over the hidden/lr/weight_decay/epochs
  grid; report top rows vs ridge's best CV MSE (test file stays closed).
- [x] 6. Extend `scripts/baselines/inspect_best_model.py` for `mlp`
  (ridge-style homoscedastic path: `sigma = sqrt(train MSE)`, binned 21-way
  test log-loss, int-cast `hidden`) and save
  `artifacts/baselines/contribution_mlp_best.joblib`.
- [x] 7. Adapter verification test: load the bundle through `load_ah_model`,
  assert the leak guard passes, sampling takes the homoscedastic branch,
  integer levels in [0, 20], reproducible under a fixed torch seed.
- [ ] 8. Add
  `configs/simulation/manager_testing/23_2g8a_self_mlp_regressor_contr_gnn_switch.yml`
  — reference config with only the contribution artifact swapped, output
  dir slug `mlp_regressor` (underscored for the sweep parser convention).
- [ ] 9. Submit the Stage-1 simulation on Raven (squeue PENDING check before
  any rsync --delete), fetch
  `plots/simulation/23_2g8a_self_mlp_regressor_contr_gnn_switch`.
- [ ] 10. `python -m aimanager evaluate` locally, log CG/RCD/RCA, rows <= 1,
  mean, band changes in this file; Stage 2 only on a band upgrade.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-18 | (baseline) reference cell, gnn contr unchanged | 1 | CG 9.850260681510413, RCD 2.7723214938046725, RCA 2.0348663297861047 | 11/21 | 1.7595567320354153 | baseline |

## 4. Notes

1. Deficit profile fetched from the reference cell scores (not slot
   averages): CG 9.85 (> 5), RCD 2.77, RCA 2.03 (2–5). SC 3.27 belongs to
   the switch slot, PD 2.93 to the punisher; not declared.
2. Key mechanism check before declaring: the `LinearAHAdapter` continuous
   branch (`linear_ah.py:228-252`) always samples for AH contributions —
   ridge is *not* deterministic in simulation. Its `elif self.sample and
   self.sigma > 0` branch is not gated on model type, so an `mlp` bundle
   carrying a scalar `sigma` needs no adapter sampling changes.
3. Local trainability confirmed on the real train split (step 4 smoke:
   mse 14.96 vs floor 41.68, ~4 s end to end; full 81-setting x 4-set
   grid, 2.5 min wall-clock).
4. Shortlist CV winner (seed 38381, 4 folds, train split only): ridge's
   rank-1 9-feature set with hidden=32, wd=0.001, lr=0.01, epochs=200 —
   CV MSE 14.276944 vs ridge's best 14.62869. Per-set bests: 9-feat
   14.2769, 12-feat 14.4145, lean-7 14.7764, lean-5 14.8484, floor
   41.6804. All four sets prefer lr 0.01 / 200-500 epochs; the nonlinear
   gain over ridge is consistent (~0.35 MSE) but within one fold-SE
   (~0.97) — the behavioral (sim) test is the decider, as expected.
5. Bundle saved (one sanctioned test-file opening): TEST MSE 14.4643 vs
   ridge 14.5162 (floor 38.8808), sigma 3.6899, TEST binned 21-way
   log-loss 2.4231 — worse than ridge 2.4053 and gaussian 2.3514. The
   nonlinear mean buys ~0.36% MSE but not a better predictive
   distribution under homoscedastic sampling; first-layer importances
   are flat (no dominant feature). Pre-flight expectation tempered:
   RCA movement may be small; the sim decides.
