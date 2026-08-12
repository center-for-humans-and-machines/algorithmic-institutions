# Autoresearch log: contribution-cg-copula

## 1. Declaration

- **Slot:** contribution
- **Base model:** `gnn` (M0 reference,
  `artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`).
- **Target rows:** CG — reference stack 9.850; slot average 9.651; fully
  concordant (9.10–10.14 in all 8 contexts), the stack's worst row. Band
  > 5; success requires CG < 5 (into 2–5) in the reference stack, confirmed
  by Stage 2.
- **Hypothesis:** members of a group co-move because they see the same
  situation and because the model's error about their shared situation is
  common to all four — but the simulation draws every agent's contribution
  independently from its own marginal (`th.multinomial` per agent in
  `encoder.decode`), throwing the correlated component away and pinning the
  group-mean spread near the independence floor (human spread ratio 0.85 vs
  sim 0.59; the motivating analysis on PR #140). PR #147 showed forcing
  peer-conditioning via self-history dropout moves CG (9.85 → 5.96) but
  pays with the individual change statistics (RCA/RCB worse in 8/8) — its
  closing note names this experiment's family: restore the group-level
  co-movement *without noising the self-level*. A within-round exchangeable
  Gaussian copula over the group's four predicted contribution marginals —
  one shared standard-normal latent per (group, round), mixed with
  per-agent noise at weight rho, pushed through each agent's own predicted
  CDF — preserves every individual marginal exactly (so RCA/RCB/CA/CD are
  structurally protected) while restoring the shared-error correlation the
  independent sampler destroys. The mechanism and the honest rho estimator
  (pairwise-likelihood MLE given the model's own predicted probabilities,
  train split only) are proven on the punisher slot (PR #146, PD 2.93 →
  1.53, merged); contribution's 21 classes are less lumpy than
  punishment's, so the estimator operates on friendlier ground.
- **Planned change:** copula sampling at the GNN contributor's free-running
  draw. Legal variant selection by Stage-1 CG under the stack guards (§5):
  (a) copula on the M0 marginals, (b) copula on the M4 peer-feature
  marginals (`own_grp_prev_mean_contr` + `same_group`, no dropout — PR
  #144's p=0 control: CG 7.587, rows 11/21, mean 1.621, guards clean; its
  `.pt` was never committed, so this arm is a retrain of
  `group_switching_contribution_50ep_own_group_same_group`). Each arm's rho
  is estimated against its own marginals. Sampling-layer change only; no
  loss, architecture, or evaluation-suite change.
- **Stack guards (must not regress):** rows <= 1 baseline 11/21; mean
  baseline 1.76 (reference stack `gnn x gnn x lin_multinomial`).
- **Prior art consulted:** PR #144 [FAIL] (dropout regresses RCA), PR #147
  [FAIL] (kept but no band upgrade; RCA/RCB collateral), PR #146 [SUCCESS]
  (copula mechanism + estimator), PR #140 comment (sampler-vs-model
  decomposition: for contribution ~1/6 of the co-movement gap is
  sampler-bound on a *well-conditioned* model — the free-running GNN
  under-uses state, so its residual shared error, which the copula carries,
  is larger; the calibration step measures it before any cluster run).

## 2. Plan

(to be filled by the validated planner step list)

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-12 | (baseline) reference stack, no change | 1 | CG 9.850 | 11/21 | 1.760 | baseline |

## 4. Notes

1. Deficit profile from `23_stack_sweep_updated/score_matrix.csv`: CG is the
   gnn contributor's only row that is both worst-in-stack and fully
   concordant (9.10–10.14 across all 8 contexts). RCA 2.85 / RCD 2.63 /
   RCB 2.08 are concordant secondaries; PA/PC/RPB spike only in
   gaussian/ridge punisher contexts (punisher-attributable).
2. Parallel experiments checked at branch creation: contribution-prev-onehot
   (cat base, RCA — no overlap), switch-herding-copula (switch slot),
   punisher-severity-copula (merged). Slug `cg_copula` chosen so config and
   output-dir names cannot collide; per the #146 incident, check
   `squeue -u certuer` for PENDING jobs before any cluster rsync.
