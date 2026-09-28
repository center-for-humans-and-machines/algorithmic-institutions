# Autoresearch log: punisher-severity-copula-v2

Redo of PR #146 (merged, then reverted with the copula-era reference stack)
under the current single-stage, two-gate protocol. The idea, estimator, and
code are unchanged from the original branch `auto/punisher-severity-copula`;
what changes is the evaluation: one simulation in the top-ranked stack, one
evaluation, verdict straight from the §2 gates.

## 1. Declaration

- **Slot:** punisher
- **Base model:** `lin_multinomial`
  (`artifacts/baselines/punishment_multinomial_best_with_contr.joblib`)
- **Evaluation stack (§3):** the top-ranked stack itself,
  `gnn x gnn x multinomial`
  (`configs/simulation/manager_testing/23_2g8a_self_gnn_contr_gnn_switch.yml`).
  Baseline: the `lin_multinomial_self` pairing in
  `plots/simulation/23_2g8a_self_gnn_contr_gnn_switch/evaluation/scores.csv`
  — PD 2.934892, rows <= 1 11/21, mean 1.7595567320354153.
- **Target row:** PD (2.934892, band 2-5). Gate 1 requires PD to finish in
  1-2 or better; gate 2 requires the 21-row mean below 1.7595567320354153.
- **Hypothesis:** a group's punishments in a round are one human manager's
  joint decision, but the simulation samples every agent's punishment
  independently, pinning the group-spread ratio to the independence floor —
  the root cause named in `notes/evaluation_metric_defs.md` and on PR #140.
  Managers exhibit round-level severity that correlates their punishments
  beyond what the shared observable features explain; capturing it should
  raise the spread of group mean punishments toward the human ratio and
  move PD.
- **Planned change:** Gaussian-copula sampling for the multinomial punisher —
  one shared standard-normal latent per group per `get_punishments` call,
  mixed with per-agent noise at weight rho, transformed through each agent's
  own predicted multinomial CDF. Marginals are preserved by construction
  (PA/PB/PC/RPA/RPB should not move); no retraining of the marginal model.
  rho = pairwise-likelihood MLE of the exchangeable Gaussian copula on the
  bundle's own locked train split (original estimate 0.3507588625344979,
  cluster-bootstrap 95% CI [0.278, 0.423]); stored as a bundle field, and
  bundles without the field sample independently as before.

## 2. Plan

Adapted by the orchestrator from the plan validated on
`auto/punisher-severity-copula` (2026-08-11); the estimator revision and
implementation steps are carried over as settled, the two-stage evaluation is
replaced by the single §3 evaluation. Targets per §2, legality per §5, frozen
surface untouched per §8. Slug: `severity_copula_v2`.

- [x] 1. Worktree + Claude commit identity (done at branch creation).
- [x] 2. Restore from `origin/auto/punisher-severity-copula`, unchanged:
      `scripts/baselines/punishment_copula_rho.py` (calibration),
      `tests/baselines/test_punishment_copula.py` (31 local tests),
      `src/aimanager/simulation/linear_ah.py` (gated copula sampling;
      rho absent/0.0 keeps the legacy path bit-identical, RNG stream
      included — main's copy is byte-identical to the old branch's parent,
      so the restore is exactly the original diff).
- [x] 3. Re-run the calibration locally; require rho to reproduce
      0.3507588625344979 exactly (deterministic, fixed seeds); save
      `artifacts/baselines/punishment_multinomial_severity_copula.joblib`
      (= base bundle + `copula_rho` + provenance; no pre-existing key
      modified, predict_proba bit-identical on reload).
- [x] 4. Run local suites: `pytest tests/baselines` + the eval-suite tests.
- [x] 5. Sim config
      `configs/simulation/manager_testing/23_2g8a_severity_copula_v2_self_gnn_contr_gnn_switch.yml`:
      copy of the reference-stack config with the punisher slot swapped to
      the copula bundle, single `lin_multinomial_copula` self pairing,
      output dir slugged `severity_copula_v2` (slug before `_self_` so the
      sweep DIR_PATTERN parses); protocol byte-identical to the 23 family.
- [x] 6. Push the joblib to Raven explicitly (`simulate_cluster.sh` excludes
      `artifacts/`); check `squeue` for PENDING jobs before any rsync.
- [x] 7. `scripts/simulate_cluster.sh <config>`; poll; confirm
      `per_round.parquet`; `scripts/fetch_cluster.sh`; `python -m aimanager
      evaluate <config>`.
- [x] 8. Verdict per §2 against the declared baseline: PD out of 2-5 into
      1-2 or <= 1, AND mean < 1.7595567320354153. Log unrounded; PR
      `[SUCCESS]`/`[FAIL]` with Hypothesis / Results / Collateral.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-25 | copula sampling, rho=0.3507588625344979 (pairwise MLE) | single | PD 1.5324969616723312 (baseline 2.9348924694638336) | 10/21 (baseline 11/21) | 1.6879978841849728 (baseline 1.7595567320354153) | SUCCESS — gate 1 (PD 2-5 -> 1-2) and gate 2 (mean down) both pass |

## 4. Notes

1. The original run of this idea (PR #146) scored PD 1.5324969616723312,
   rows <= 1 10/21, mean 1.687998 in this same stack — under the current
   gates that is a success (PD 2-5 -> 1-2, mean down); the 11 -> 10 rows
   <= 1 dip that forced the old escalation no longer gates. This run
   re-executes the protocol rather than importing those numbers.
2. The calibration reproduced the original estimate bit-for-bit
   (rho_hat 0.3507588625344979, bootstrap SE 0.0377, 95% CI
   [0.278, 0.423]); the pre-flight replay closes 62% of the human
   group-spread gap (independent 0.627 -> copula 0.704, human 0.750).
   314 local tests pass, eval suite untouched.
3. The simulation and evaluation reproduced the original Stage-1 scores
   bit-for-bit on all 21 rows (same seed, same code, same artifact) —
   the verdict is a re-execution, not an import.
4. The single rows <= 1 loss is RSA crossing 0.908837 -> 1.0010089590803533
   (0.1% over the ceiling). The original PR's 8-context sweep already
   adjudicated this as noise (RSA falls in the 4 gnn-switch contexts,
   rises in the 4 lin-switch contexts, slot avg +0.0050); under the
   current protocol rows <= 1 does not gate, so it is context only.
5. SC improves 3.270439 -> 2.816006 within the 2-5 band — the shared
   independent-sampling root cause (PR #140) acting on a sibling row,
   consistent with the original finding of SC improving in 7/8 contexts.
