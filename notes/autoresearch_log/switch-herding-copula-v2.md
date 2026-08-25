# Autoresearch log: switch-herding-copula-v2

Redo of PR #150's winning arm (arm B: herding copula with AR(1) persistence
for the GNN switch predictor) under the current single-stage, two-gate
protocol, stacked on the parent `[SUCCESS]` PR #160
(`auto/punisher-severity-copula-v2`). The idea, estimator, calibrated
parameters, and code are unchanged from the original branch
`auto/switch-herding-copula`; what changes is the base (the parent PR's
branch instead of main) and the evaluation: one simulation in the parent's
stack, one evaluation, verdict straight from the §2 gates against the
parent's confirmed scores.

## 1. Declaration

- **Slot:** switch
- **Base model:** `gnn` switch predictor
  (`artifacts/artificial_humans/switch_pred_opt_50ep_doubled_reanchored/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`)
- **Parent PR (§9):** #160 `[SUCCESS]`, branch
  `auto/punisher-severity-copula-v2`. Evaluation stack: the parent's —
  `gnn` contribution x `gnn` switch x severity-copula `lin_multinomial`
  punisher
  (`configs/simulation/manager_testing/23_2g8a_severity_copula_v2_self_gnn_contr_gnn_switch.yml`).
  Baseline (the parent's confirmed scores, from
  `plots/simulation/23_2g8a_severity_copula_v2_self_gnn_contr_gnn_switch/evaluation/scores.csv`):
  SC 2.816005922026658, rows <= 1 10/21, mean 1.6879978841849728.
- **Target row:** SC (2.816006, band 2-5). Gate 1 requires SC to finish in
  1-2 or better; gate 2 requires the 21-row mean below 1.6879978841849728.
- **Hypothesis:** eligible players in a group see the same situation and
  their switch decisions co-move beyond what shared observable features
  explain (~40% of within-group co-movement survives conditioning on state —
  the motivating comment on PR #140), and the segregation they build
  persists across an episode's decision rounds (first-half/second-half
  segregation correlation 0.38 human vs 0.03 sim). The simulation draws
  every agent's switch independently each round, pinning group-level
  switching to the independence floor. A shared per-(episode, group)
  standard-normal latent, mixed with per-agent noise at weight rho and
  carried across decision rounds by an AR(1) with coefficient phi, pushed
  through each agent's own predicted Bernoulli marginal, captures both the
  within-round herding and the episode memory; marginals are preserved by
  construction (SA/SB/RSA should not move), no retraining of the marginal
  model. PR #150 established: rho alone REGRESSES SC (one-round herding
  mean-reverts between decision rounds); the persistence is essential —
  only arm B is re-run here.
- **Planned change:** Gaussian-copula sampling for the GNN switch
  predictor, arm B of PR #150 — rho = 0.116482333585783 (binary
  pairwise-likelihood MLE against the GNN's own marginals on the 40-game
  train split, cluster-bootstrap 95% CI [0.0350, 0.2150]),
  phi = 0.70366020589033 (rho_lag1/rho, bootstrap 95% CI [0.117, 2.083]
  excludes 0), copula_switch_every = 4 (the AR(1) latent advances only on
  decision rounds). Stored as fields on a copy of the switch artifact;
  artifacts without the fields sample independently, bit-identical to the
  legacy path.

## 2. Plan

Adapted by the orchestrator from the plan validated on
`auto/switch-herding-copula` (2026-08-12), exactly as the parent PR #160
adapted PR #146's: the estimator, implementation, and arm selection are
carried over as settled (arm A's regression is a recorded finding, not
retried), the two-stage evaluation is replaced by the single §3 evaluation
against the parent baseline. Targets per §2, legality per §5, frozen
surface untouched per §8. Slug: `herding_copula_ar1_v2`; switch token
`herdcopar1` (unchanged — the output dir carries the v2).

- [x] 1. Worktree + Claude commit identity from
      `origin/auto/punisher-severity-copula-v2` (done at branch creation);
      declaration + plan committed.
- [x] 2. Restore from `origin/auto/switch-herding-copula`, unchanged:
      `src/aimanager/generic/copula.py` (torch-only sampler),
      `src/aimanager/generic/graph.py` (gated copula dispatch; rho
      absent/0.0 keeps the legacy path bit-identical, RNG stream included —
      main's copy is byte-identical to the old branch's merge-base and the
      parent does not touch it, so the restore is exactly the original
      diff), `scripts/baselines/switch_copula_rho.py` (calibration),
      `scripts/artificial_humans/dump_switch_probs.py`,
      `scripts/artificial_humans/make_switch_copula_artifact.py`,
      `tests/switch/test_switch_copula.py`,
      `src/aimanager/tests/test_switch_copula_graph.py`, and the
      calibration inputs
      `artifacts/artificial_humans/switch_pred_herding_copula/calibration/switch_probs_{train,test}.parquet`
      (LFS; dumped from the frozen base switch artifact on the human data,
      provenance in the original log notes 9-10).
      (`src/aimanager/simulation/linear_ah.py` needs nothing: the parent's
      copy is already byte-identical to the old branch's.)
- [x] 3. Re-run the calibration locally (`switch_copula_rho.py`: acceptance
      gates, pairwise MLE, AR(1) lag-1 step, bootstrap); require
      rho = 0.116482333585783 and phi = 0.70366020589033 to reproduce
      exactly (deterministic, fixed seeds); write
      `artifacts/artificial_humans/switch_pred_herding_copula/calibration/copula_params.json`
      (parameters bit-identical; provenance fields re-stamped).
- [x] 4. Run local suites: `pytest tests/` + the eval-suite tests (frozen
      surface untouched).
- [x] 5. Arm-B artifact
      `artifacts/artificial_humans/switch_pred_herding_copula_ar1/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`:
      restore the old branch's committed copy (LFS) into the worktree and
      verify the copy already on Raven is bit-identical (sha256); rebuild
      on Raven via `make_switch_copula_artifact.py` only on mismatch.
      (Same-path re-creation, per the parent's precedent with the
      severity-copula joblib — content is bit-identical, so no collision.)
- [x] 6. Sim config
      `configs/simulation/manager_testing/23_2g8a_herding_copula_ar1_v2_self_gnn_contr_herdcopar1_switch.yml`:
      copy of the old branch's arm-B config trimmed to the single
      `lin_multinomial_copula_self` pairing (pairing position 1 — RNG
      stream unchanged; the parent's note 3 established single-pairing
      reproduction is bit-identical), output dir/figure name slugged
      `herding_copula_ar1_v2` (slug before `_self_` so the sweep
      DIR_PATTERN parses); protocol byte-identical to the 23 family.
- [ ] 7. `squeue -u certuer` PENDING check before any sync;
      `scripts/simulate_cluster.sh <config>`; confirm `per_round.parquet`;
      `scripts/fetch_cluster.sh`; `python -m aimanager evaluate <config>`.
- [ ] 8. Verdict per §2 against the parent baseline: SC out of 2-5 into
      1-2 or <= 1, AND mean < 1.6879978841849728. Log unrounded; PR
      `[SUCCESS]`/`[FAIL]` with `--base auto/punisher-severity-copula-v2`,
      body Hypothesis / Results / Collateral.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## 4. Notes

1. PR #150 ran this exact configuration (arm B in this exact stack, same
   seed and protocol) as its Stage 1 and scored SC 1.9138838624773855,
   rows <= 1 11/21, mean 1.5863134655 — under the current gates that is a
   success (SC 2-5 -> 1-2, mean down). This run re-executes the protocol
   on the parent branch rather than importing those numbers, exactly as
   the parent PR #160 re-executed PR #146.
2. Arm A (rho only, no persistence) is not re-run: PR #150 recorded it
   REGRESSING SC to 2.905083 — one-round herding without persistence
   mean-reverts between decision rounds. The persistent latent is the
   mechanism.
