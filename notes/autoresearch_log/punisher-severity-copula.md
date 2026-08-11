# Autoresearch log: punisher-severity-copula

## 1. Declaration

- **Slot:** punisher
- **Base model:** `lin_multinomial`
  (`artifacts/baselines/punishment_multinomial_best_with_contr.joblib`)
- **Target rows:** PD — reference stack 2.93; slot average 2.68; concordant
  across all 8 contexts (2.19–3.24, always >= 2). The only P-family deficit of
  this base model (PA 0.60, PB 0.86, PC 0.80, RPA 1.21, RPB 0.73 slot avg).
- **Hypothesis:** a group's punishments in a round are one human manager's
  joint decision, but the simulation samples every agent's punishment
  independently, pinning the group-spread ratio to the independence floor
  (~0.58) — the root cause named in `notes/evaluation_metric_defs.md` and on
  PR #140. Managers exhibit round-level severity that correlates their
  punishments beyond what the shared observable features explain; capturing it
  should raise the spread of group mean punishments toward the human ratio and
  move PD.
- **Planned change:** Gaussian-copula sampling for the multinomial punisher —
  one shared standard-normal latent per `get_punishments` call (= per
  manager-round), mixed with per-agent noise at weight rho, transformed
  through each agent's own predicted multinomial CDF. Marginals are preserved
  by construction, so PA/PB/PC/RPA/RPB should not move; no retraining of the
  marginal model. rho is estimated from the human training data (latent
  residual correlation within manager-group-round), stored as a bundle field;
  bundles without the field sample independently as before.
- **Stack guards (must not regress):** rows <= 1 baseline 11/21; mean baseline
  1.76 (reference stack `gnn x gnn x lin_multinomial`).

## 2. Plan

*(pending validation)*

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## 4. Notes

1. Deficit profile fetched from
   `plots/data_analysis/evaluation/23_stack_sweep_updated/score_matrix.csv`:
   PD is the multinomial punisher's only slot-attributable row >= 2 and is
   concordant (2.19–3.24 in all 8 contexts). CG/RCA/SC/RCB are high in every
   punisher context and belong to the other slots.
2. No prior punisher experiments: `notes/autoresearch_log/` did not exist;
   the only `[FAIL]` PR (#144) is contribution-slot.
