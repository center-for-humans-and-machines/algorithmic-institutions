# Autoresearch log: punisher-ar-gnn-v2

Redo of `auto/punisher-ar-gnn` (PR #152) under the current single-stage,
two-gate protocol (`notes/autoresearch.md`). The idea, code, training
config, and trained artifact are unchanged from the original branch; only
the evaluation protocol differs — one simulation in the §3-selected stack,
one evaluation, verdict from the §2 gates.

## 1. Declaration

- **Slot:** punisher
- **Base model:** GNN punisher (`gnn_self`, `architecture node+edge+rnn`,
  `artifacts/artificial_humans/punishment_rnn_edge_50ep_doubled/model/architecture_node+edge+rnn__dataset_50ep_doubled.pt`).
- **Evaluation stack (§3):** ranking `score_matrix.csv`
  (`23_stack_sweep_updated`) by rows <= 1 desc, mean asc, filtered to
  punisher = gnn, the top is `gaussian x gnn x gnn` (7/21, mean 1.7410 —
  beats `gnn x lin x gnn`, 7/21, mean 1.7686, on the tie-break). Candidate
  is swapped into the punisher slot of
  `configs/simulation/manager_testing/23_2g8a_self_gaussian_contr_gnn_switch.yml`.
- **Baseline (full precision, from
  `plots/simulation/23_2g8a_self_gaussian_contr_gnn_switch/evaluation/scores.csv`,
  run = `ah group_switching managed by gnn_self`):**
  PD 2.6128746154448903 (band 2-5); rows <= 1: 7/21;
  mean 1.7407445494371563.
- **Target rows (§6):** PD is the GNN punisher's only slot row with
  slot-average score >= 2 (avg 2.634 over its 8 contexts, concordant —
  2.01-2.88 in 8/8 contexts per the original log, note 1). Gate 1 needs
  PD in a better band: <= 2. Gate 2 needs mean < 1.7407445494371563.
- **Hypothesis:** unchanged from PR #152 — a group's punishments in a round
  are one human manager's joint decision; an autoregressive factorization
  (each agent's punishment conditions on round t-1 observables AND on
  groupmates' already-decided same-round punishments, never current-round
  contributions) can represent non-exchangeable within-round dependence and
  move the group-spread row PD.
- **Planned change:** one change — swap the punisher slot to the AR-GNN
  punisher of `auto/punisher-ar-gnn`: the `ar_punishment` gated edge
  feature (`src/aimanager/generic/graph.py`), trained with the in-repo
  any-order reveal-mask scheme, sampled agent-by-agent via
  `predict_autoreg`. The trained 2750-epoch artifact
  (`artifacts/artificial_humans/punishment_ar_gnn_50ep_doubled/model/architecture_node+edge+rnn+ar__dataset_50ep_doubled__epochs_2750.pt`,
  epoch selected by CV test log-loss on the original branch, note 7) is
  reused as-is — same code, same config, same seed; retraining would
  reproduce it at multi-GPU-hour cost (PR #160 precedent: re-execute the
  protocol, not the estimator).
- **Known prior evidence:** in the `gnn x gnn` context the AR candidate
  moved PD 2.8228 -> 2.5751 (within-band) and improved every other
  P-family row of its base family (original log, note 9). The
  `gaussian x gnn` context is untested — Stage 2 was cancelled.
- **Slug:** `ar_gnn_v2` (sim config + output dir; artifact and training
  config keep their original `ar_gnn` names since they are byte-identical
  imports).

## 2. Plan

Validated by the orchestrator 2026-08-25 (targets per §2, legality per §5,
frozen surface per §8). Rulings: (R1) import via
`git restore --source=origin/auto/punisher-ar-gnn -- <paths>`, never
cherry-pick — the original commits touch that branch's own log file
(frozen, §8) and sit on the pre-revert copula base. (R2) import
`model/` + `metrics/` only (4 files); the ~2 GB `confusion_matrix/`
parquets stay retrievable on the original branch. (R3) no retraining —
same code, config, and seed would reproduce the artifact at GPU-hour cost
(PR #160 precedent). (R4) no `autoregressive` key in the sim config; the
checkpoint flag drives `predict_autoreg` dispatch
(`simulate.py:151-154`) and is asserted on Raven before anything runs.
(R5) `squeue -u certuer` PENDING check before every rsync.

- [x] 1. Worktree preconditions (branch, identity, clean tree).
- [x] 2. Re-verify the declared baseline from
      `plots/simulation/23_2g8a_self_gaussian_contr_gnn_switch/evaluation/scores.csv`
      (gnn_self run: 21 rows, PD 2.6128746154448903, rows<=1 7,
      mean 1.7407445494371563).
- [x] 3. Record this step list; commit.
- [x] 4. Import `src/aimanager/generic/graph.py` and
      `src/aimanager/tests/test_ar_punisher.py` from
      `origin/auto/punisher-ar-gnn`; verify byte-identity to the source
      branch and `graph.py | 81 +-` vs main.
- [x] 5. Import
      `configs/training/artificial_humans/punishment/ar_gnn_50ep_doubled.yml`
      (branch tip = 2750-epoch version); verify byte-identity.
- [x] 6. Import
      `artifacts/artificial_humans/punishment_ar_gnn_50ep_doubled/{model,metrics}`
      (4 files; both epochs for provenance, 2750 is the candidate).
- [x] 7. Verify real LFS content, not pointers: `.pt` md5
      4774e934f08a96da01da875851ad7a2c (2750) /
      f789ab0a17ec870d3e53507db9de34f6 (5000).
- [x] 8. Local batched gate, once, before staging: eval-suite tests +
      `scripts/tests` + `tests/baselines`; black + flake8 on the two
      imported source files. (`test_ar_punisher.py` is PyG — Raven only.)
- [x] 9. PENDING check, then `scripts/remote_test.sh` — full PyG suite
      green incl. the 9 AR tests.
- [x] 10. On Raven: assert the 2750 checkpoint's `autoregressive` flag,
      `edge_encoding == [{name: ar_punishment, n_levels: 31}]`, and
      `x_encoding` parity with the base punisher checkpoint.
- [x] 11. Commit the code + tests; re-verify hooks did not mutate bytes.
- [x] 12. Commit the training config + 4 artifact files (LFS).
- [x] 13. Write
      `configs/simulation/manager_testing/23_2g8a_ar_gnn_v2_self_gaussian_contr_gnn_switch.yml`:
      copy of the baseline config, single manager `ar_gnn` -> the 2750
      `.pt`, single pairing `ar_gnn_self`, slugged
      output dir/figure name; protocol byte-identical.
- [x] 14. Mechanical config check (yaml parse, DIR_PATTERN yields
      contr=gaussian/switch=gnn, artifact paths exist, output dir fresh);
      commit.
- [x] 15. PENDING check, then `scripts/simulate_cluster.sh <config>`.
- [x] 16. Poll to completion; confirm `per_round.parquet` on Raven, exit 0.
- [x] 17. `scripts/fetch_cluster.sh plots/simulation/23_2g8a_ar_gnn_v2_self_gaussian_contr_gnn_switch`.
- [x] 18. `python -m aimanager evaluate <config>`; 21 rows for the single
      `ar_gnn_self` run.
- [x] 19. §2 verdict, unrounded: gate 1 PD <= 2 (baseline
      2.6128746154448903, band 2-5); gate 2 mean < 1.7407445494371563;
      rows<=1 (baseline 7/21) reported as context.
- [x] 20. Fill §3 Results + §4 Notes; commit log + sim outputs (LFS).
- [x] 21. Push; open the PR (`[SUCCESS]`/`[FAIL]`), body Hypothesis /
      Results / Collateral.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-25 | AR-GNN punisher (gated ar_punishment edge feature, epochs 2750) in gaussian x gnn x gnn | single | PD 2.0572788911625852 (baseline 2.6128746154448903 — improved 0.556 but within band 2-5, boundary missed by 0.0573) | 9/21 (baseline 7/21) | 1.6312543616564725 (baseline 1.7407445494371563 — gate 2 passes) | FAIL — gate 1 not met (no band upgrade; §2 rules a within-band improvement a FAIL) |

## 4. Notes

1. Import provenance: all code, config, and artifact files are byte-identical
   to `origin/auto/punisher-ar-gnn` (blob-hash verified; `.pt` md5s
   4774e934f08a96da01da875851ad7a2c / f789ab0a17ec870d3e53507db9de34f6,
   confirmed again on Raven before simulating). Pre-commit hooks mutated
   nothing. Full Raven suite green (99 passed, incl. the 9 AR tests); local
   eval-suite gate 283 passed, 3 skipped; black/flake8 clean.
2. Cluster contention: a concurrent autoresearch session
   (`auto_cg_schedsamp_v2` pilots) was syncing and running jobs in the
   shared Raven directory throughout. Ruling: the documented gate is
   PENDING jobs only — a RUNNING job has already imported its code and
   writes only to rsync-excluded paths (`artifacts/`, `.log/`); a mid-run
   overwrite of our tree fails loudly (their tree lacks the
   `ar_punishment` encoder — KeyError, never silent). Verified after the
   test run that remote `graph.py`/`test_ar_punisher.py` md5s still
   matched ours; sim job 29612881 COMPLETED 2m17s exit 0:0.
3. Checkpoint dispatch verified before simulating (plan R4): the 2750
   checkpoint carries `autoregressive: True` and
   `edge_encoding: [{name: ar_punishment, n_levels: 31}]`, with
   `x_encoding` identical to the base punisher's — the sim config has no
   `autoregressive` key, so the flag alone routes `predict` ->
   `predict_autoreg`, and the independent path would hard-assert.
4. Verdict detail: PD landed at 2.0572788911625852 — 0.0573 above the
   band-2 boundary. §2 and §10 are explicit (a within-band improvement,
   however large, is a FAIL; no rounding), so this is logged as FAIL
   despite gate 2 passing with room (mean 1.63 vs 1.74) and rows <= 1
   rising 7 -> 9 (PA 0.5967 and RPB 0.7201 crossed under the ceiling).
5. Context matters for the mechanism: in the `gnn x gnn` context the same
   artifact managed PD 2.8228 -> 2.5751 (-0.25 vs its base family); in
   this `gaussian x gnn` context it manages 2.6129 -> 2.0573 (-0.556).
   The AR channel plus the 2750-epoch budget nearly closes the band gap
   here — but "nearly" is the recurring shape of this mechanism: observed
   conditioning recovers only part of the within-round dependence that
   the copula's explicit severity latent supplies outright (PR #152 note
   9's diagnosis stands).
6. Collateral, positive: the entire P family improves substantially
   (PA 1.0798 -> 0.5967, PB 0.9600 -> 0.7855, PC 0.9786 -> 0.8094,
   RPA 1.4315 -> 1.2689, RPB 1.1671 -> 0.7201) and the C family mostly
   rides along (CA 2.2647 -> 2.0562, CB 0.9777 -> 0.8347,
   CC 1.5886 -> 1.4747, CD 1.4803 -> 1.3390, RSA 0.9886 -> 0.8237).
   Negative: the two shared-variance rows tick up — CG 3.7264 -> 3.9413,
   SC 2.7144 -> 3.0669 — consistent with the known root cause living in
   the other slots, not the punisher.
7. Follow-up seed, sharpened by this run: severity-copula sampling on top
   of the AR-GNN marginals (seed (a) of PR #152). The AR punisher's
   marginals are now the best in its family and PD sits 0.06 from the
   band edge; a marginal-preserving shared latent is exactly the missing
   piece, and the copula machinery exists on
   `auto/punisher-severity-copula-v2` (PR #160).
