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

(to be filled after validation)

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
