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
- [x] 7. `squeue -u certuer` PENDING check before any sync;
      `scripts/simulate_cluster.sh <config>`; confirm `per_round.parquet`;
      `scripts/fetch_cluster.sh`; `python -m aimanager evaluate <config>`.
- [x] 8. Verdict per §2 against the parent baseline: SC out of 2-5 into
      1-2 or <= 1, AND mean < 1.6879978841849728. Log unrounded; PR
      `[SUCCESS]`/`[FAIL]` with `--base auto/punisher-severity-copula-v2`,
      body Hypothesis / Results / Collateral.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-25 | arm B copula, rho=0.116482333585783, phi=0.70366020589033 (shared checkout; provenance uncertain, note 4) | single | SC 2.187695344629057 (baseline 2.816005922026658) | 12/21 (baseline 10/21) | 1.6373279817979518 (baseline 1.6879978841849728) | discarded as unverifiable (note 4), then bit-confirmed by the isolated re-run (note 6) |
| 2026-08-25 | same, isolated re-execution with in-job provenance (note 5) | single | SC 2.187695344629057 (baseline 2.816005922026658) | 12/21 (baseline 10/21) | 1.6373279817979518 (baseline 1.6879978841849728) | FAIL — gate 1 not met: SC improves within band (2-5), no band upgrade; gate 2 alone passes (mean down) |
| 2026-08-25 | DIAGNOSTIC, not the verdict: PR #150's exact four-pairing arm-B config, isolated + provenance-verified (note 8) | diag | SC 1.9138838624773855 — bit-identical to PR #150 arm B on the full per-round frame | 11/21 | 1.58631346549462 | reproduces #150 exactly; proves the band position is RNG-context-dependent (note 9) |

## 4. Notes

1. PR #150 ran this exact configuration (arm B in this exact stack, same
   seed and protocol) as its Stage 1 and scored SC 1.9138838624773855,
   rows <= 1 11/21, mean 1.5863134655 — under the current gates that is a
   success (SC 2-5 -> 1-2, mean down). This run re-executes the protocol
   on the parent branch rather than importing those numbers, exactly as
   the parent PR #160 re-executed PR #146.
2. Arm A (rho only, no persistence) is not re-run: PR #150 recorded it
   REGRESSING SC to 2.905083 — one-round herding without persistence
3. Sync race with a parallel experiment, for the record: the `squeue`
   check was clean, but a parallel session's AH training job
   (`auto_cg_schedsamp_v2_pilot_ctrl`, 29612646) was submitted 4 s before
   this experiment's sim (29612647) and the two rsyncs overlapped. This
   run is unaffected (this sync landed last: config + copula code
   verified present remotely after both syncs, and the sim job imported
   them cleanly). The training job survives on its `.log/` copy of its
   config (excluded from sync) but imported `src/` from THIS branch —
   if that experiment's change touches training code, its run is
   compromised and needs re-checking by its own session; escalated in
   the session report.
4. The first simulation (job 29612647, 15:17) is DISCARDED AS INVALID —
   not for its number, for its provenance. It diverged from PR #150's
   arm-B pairing-1 run from episode 0, round 0-1, on the contribution
   side (SC 2.187695344629057, mean 1.6373279817979518, rows <= 1
   12/21), where bit-identical reproduction was expected and every
   controllable factor was verified identical: `src/aimanager/` is
   byte-identical to the #150 branch (empty git diff), the switch
   artifact and joblib sha256-match, and pairing count is proven
   irrelevant (the #150 branch's four-pairing re-sim of the
   severity-copula stack matches the parent's single-pairing run on all
   21 rows bit-for-bit, which also proves the platform stable 08-12
   through 08-25). The cause: the remote venv is EDITABLE (jobs import
   `src/` live at import time) and the shared checkout was hit by two
   interleaved `rsync --delete` syncs (note 3) seconds before both jobs
   imported — file-wise interleaving can leave a MIXED tree (checked
   minutes later, the remote `src/` was the other session's:
   `copula.py` deleted, foreign `graph.py`/`linear_ah.py`). The job
   did not crash and both copulas were demonstrably active (PD
   0.998774, SC moved, marginals held), so the copula modules were
   mine at import time, but the contribution-path modules cannot be
   verified — exactly where the divergence lives.
5. Decision, made before knowing the re-run's outcome: re-execute the
   identical config and seed once, in an isolated remote directory
   (`~/iso_herdcopar1v2`, untouched by any other session's sync), code
   md5-verified against the branch before submission and artifact
   sha256s + `aimanager.__file__` + module md5s printed by the job
   itself into its log. Seed, config, episode count untouched — this
   is an execution-validity fix, not a re-draw; the isolated run's
   result is the §3 evaluation whatever it says, and both runs are
   logged.
6. The isolated run (job 29613189; `aimanager.__file__` in the iso tree,
   module md5s and artifact sha256s printed by the job, torch
   1.11.0+cu113, cuda) reproduced the first run BIT-FOR-BIT — the two
   `per_round.parquet` files are md5-identical (9022c802...). So the
   first run was in fact clean, the pipeline is bitwise deterministic,
   and SC 2.187695344629057 is the model's true draw at seed 42 under
   the §3 protocol. What does NOT reproduce is PR #150's arm-B Stage-1
   number (SC 1.9138838624773855): same byte-identical `src/`, same
   sha-identical artifacts, same seed, and a divergence from episode 0
   round 0-1 on the contribution side. Given #150's own note 18 (its
   arm-B submission followed an evening of failed/partial syncs and
   went up with `--no-sync` against an unverified remote tree), the
   likely explanation is that #150's Stage-1 ran on a contaminated
   tree — the same failure mode note 4 describes, on the other side.
   Its Stage-2 16-context confirmation of the SC *direction* is
   unaffected; its Stage-1 *magnitude* is not reproducible.
7. Verdict on the isolated run, per §2: gate 1 FAILS — SC 2.816006 ->
   2.187695 is a within-band improvement (2-5), not a band upgrade;
   gate 2 alone would pass (mean 1.687998 -> 1.637328). [FAIL], per
   the protocol's plain text ("a within-band improvement, however
   large, is a [FAIL] with valuable notes"). For the record, the run
   is otherwise the best stack state yet measured: rows <= 1 reaches
   12/21 (the sweep's best is 11/21), PD band-upgrades 1.532497 ->
   0.998774 (<= 1, a declared-target-grade move had PD been declared),
   RSA re-enters the ceiling (1.001009 -> 0.839574), and the S
   marginals hold (SA 0.744127, SB 0.796346). Only SC was declared;
   only SC gates.
8. Diagnostic (never a selection criterion; its score is not the
   verdict): PR #150's exact four-pairing arm-B config, run in the same
   isolated, provenance-verified directory, reproduces PR #150's arm-B
   run BIT-FOR-BIT (pairing-1 per-round frame `.equals` the branch's
   committed parquet; SC 1.9138838624773855, rows <= 1 11/21, mean
   1.58631346549462). So #150's Stage-1 was clean after all — note 6's
   contamination suspicion against it is WITHDRAWN — and both numbers
   are real: the same candidate draws SC 2.19 in the parent's
   single-pairing config and 1.91 in #150's four-pairing config.
9. Root cause of the context-dependence, located in code:
   `MultiManager.get_punishments` (`manager/api_manager.py`) runs EVERY
   configured manager on EVERY round with batch size = number of
   managers, then discards the unmatched outputs — so the managers
   listed in the config change the RNG stream consumed during run 1
   even though run 1 never uses their punishments. (For the non-copula
   stack this happened to be inert — the maintainer's four-pairing
   severity-copula run matches the parent's single-pairing run
   bit-for-bit on all 21 rows — but with the copula switch in the
   stack the extra consumption interacts with the shared generators
   and shifts every draw from episode 0.) Protocol implication worth
   the maintainer's attention: "the stack" under §3 is currently
   underdetermined — scores depend not only on the three slot models
   but on the CONFIG's manager list; the reference-config convention
   (single pairing, as the parent used) should be pinned explicitly.
10. Verdict adjudication, for the maintainer: per §3/§9 the evaluation
    context is the parent's stack config (single pairing) and its
    confirmed baseline — under that reading, gate 1 fails and this PR
    is [FAIL] as titled. The four-pairing context — the one PR #150
    itself was ruled [SUCCESS] in, against the SAME baseline numbers —
    gives the same candidate a band upgrade plus a mean improvement.
    The candidate's true SC sits astride the 2.0 band edge
    (draw-to-draw range at least [1.91, 2.19]); whether that counts as
    "reproducing human founding-exodus concentration" is a judgment
    the gates were not designed for. Ruled [FAIL] here by the plain
    text; the maintainer may re-adjudicate at review, as with PR
    #150's RSA boundary.
11. The isolated remote directory `~/iso_herdcopar1v2` (code, both
    run.sh scripts, job logs with provenance, both sim outputs) is
    left in place for inspection.
   mean-reverts between decision rounds. The persistent latent is the
   mechanism.
