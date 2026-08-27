# Autoresearch log: punisher-ar-copula

Combination experiment seeded by PR #161 (note 7) and PR #152 (seed (a)):
severity-copula sampling on top of the AR-GNN punisher's conditionals. The
AR-GNN alone (PR #161, `[FAIL]`) moved PD 2.6129 -> 2.0573 in its best stack —
0.0573 above the band-2 edge — while making the P-family marginals the best in
the GNN punisher family; the severity copula (PR #160, `[SUCCESS]`) is the
marginal-preserving shared latent that carried PD through the same boundary on
the multinomial punisher. This experiment supplies the copula's shared latent
to the AR model's conditional distributions.

## 1. Declaration

- **Slot:** punisher
- **Base model:** GNN punisher (`gnn_self`, `architecture node+edge+rnn`,
  `artifacts/artificial_humans/punishment_rnn_edge_50ep_doubled/model/architecture_node+edge+rnn__dataset_50ep_doubled.pt`).
- **Evaluation stack (§3):** unchanged from PR #161 — the score matrix
  (`23_stack_sweep_updated`, maintainer-frozen since) ranked by rows <= 1
  desc, mean asc, filtered to punisher = gnn, selects
  `gaussian x gnn x gnn`
  (`configs/simulation/manager_testing/23_2g8a_self_gaussian_contr_gnn_switch.yml`).
  Candidate is swapped into that config's punisher slot.
- **Baseline (re-verified bit-exact from
  `plots/simulation/23_2g8a_self_gaussian_contr_gnn_switch/evaluation/scores.csv`,
  run = `ah group_switching managed by gnn_self`):**
  PD 2.6128746154448903 (band 2-5); rows <= 1: 7/21;
  mean 1.7407445494371563.
- **Target row (§6):** PD — the GNN punisher's only slot row with
  slot-average score >= 2 (concordant, 8/8 contexts; adjudicated in the
  PR #152 and #161 logs). Gate 1 needs PD <= 2 (band 2-5 -> 1-2 or better);
  gate 2 needs the 21-row mean < 1.7407445494371563.
- **Hypothesis:** a group's punishments in a round are one human manager's
  joint decision. Autoregressive conditioning on already-decided groupmate
  punishments recovers the observable part of that dependence (PD
  2.6129 -> 2.0573, PR #161) but not the manager's round-level severity
  mood — the shared latent that the copula supplied outright on the
  multinomial punisher (PD 2.9349 -> 1.5325, PR #160). Sampling the AR
  model's conditionals through a Gaussian copula — one shared standard-normal
  latent per group per round, mixed with per-agent noise at weight rho,
  inverted through each agent's own conditional CDF — preserves the AR
  marginals by construction and adds exactly the residual within-round
  dependence the AR channel cannot see, pushing the group punishment spread
  row PD across the band-2 edge.
- **Planned change (one change: the punisher slot's sampler):** the AR-GNN
  punisher of PR #161 (byte-identical import: `ar_punishment` gated edge
  encoder, 2750-epoch artifact, training config, tests) sampled via a
  copula-correlated `predict_autoreg`: the per-agent categorical draw at
  each AR step is replaced by u_i = Phi(sqrt(rho) z_g + sqrt(1-rho) eps_i)
  inverted through that agent's conditional CDF, with z_g drawn once per
  group per round and held fixed across the round's AR steps. rho is NOT
  reused from PR #160 (the AR channel already explains part of the
  within-round correlation; 0.3508 would double-count) — it is recalibrated
  against the AR model's own conditional CDFs on the training data, method
  fixed at planning time. Gated so that rho absent/0.0 reproduces the
  PR #161 sampling path bit-identically. No retraining; the marginal model
  is the PR #161 artifact as-is.
- **Known prior evidence:** PR #161 (AR alone, this stack): PD 2.0573,
  mean 1.6313, rows <= 1 9/21 — gate 2 passed with room, gate 1 missed by
  0.0573. PR #160 (copula alone, multinomial, `gnn x gnn x multinomial`):
  PD 2.9349 -> 1.5325 with marginals untouched. PR #152 note 9 diagnosed
  observed conditioning as recovering only part of the within-round
  dependence, naming this combination as seed (a).
- **Slug:** `ar_copula` (branch `auto/punisher-ar-copula`; sim config
  `23_2g8a_ar_copula_self_gaussian_contr_gnn_switch.yml`; imported AR
  artifact keeps its original `ar_gnn` name — byte-identical import).

## 2. Plan

Validated by the orchestrator 2026-08-27 (targets per §2, legality per §5,
frozen surface per §8). Rulings: (R1) AR files import via
`git restore --source=origin/auto/punisher-ar-gnn-v2 --`, never cherry-pick
(PR #161 R1); imported files stay byte-identical — copula code goes in new
files or in `graph.py` on top of the verified import. (R2) `copula_rho`
lives in a **copy** of the 2750-epoch checkpoint under a new artifact name;
`GraphNetwork` gets a real `copula_rho` constructor param + `to_save` entry;
the existing `autoregressive` flag keeps driving dispatch — zero
`simulate.py` changes; PR #161's artifact untouched. (R3) rho = exact
cell-level MLE (one shared-latent 1-D integral per group-round cell,
Gauss–Hermite; a pairwise MLE would be biased under AR conditionals) on the
full flip-doubled training data — the shipped checkpoint is the
`i is None` full-data fit, so that IS its training split; SEED 38381 (the
model's training seed); bootstrap clusters on the 50 `pair_id`s; round-trip
recovery (tol 0.03) is an acceptance gate; all other diagnostics are
printed, never used to select rho. (R4) rho absent/0.0 and `sample=False`
reproduce the PR #161 sampling path bit-identically, RNG stream included —
pinned by a dedicated Raven test. (R5) `squeue -u certuer` PENDING check
before every rsync; source `.pt` md5 asserted on Raven before and after
every cluster run (shared-checkout race, PR #161 note 2); `artifacts/` is
rsync-excluded, so the copula `.pt` moves via explicit push/fetch. (R6) if
the `train_cluster.sh ah` job template cannot carry the calibration script,
revise the plan through validation again — no improvised login-node runs.

- [x] 1. Worktree preconditions: branch `auto/punisher-ar-copula` off
      `main`, Claude commit identity, clean tree; re-verify the declared
      baseline bit-exact from
      `plots/simulation/23_2g8a_self_gaussian_contr_gnn_switch/evaluation/scores.csv`,
      run `ah group_switching managed by gnn_self` (21 rows,
      PD 2.6128746154448903, rows<=1 7, mean 1.7407445494371563).
- [x] 2. Record this step list in §2; commit declaration + plan.
- [x] 3. Import `src/aimanager/generic/graph.py` and
      `src/aimanager/tests/test_ar_punisher.py` from
      `origin/auto/punisher-ar-gnn-v2` (R1); verify blob-hash byte-identity
      to the source branch and `graph.py | 81 +-` vs `main`; commit.
- [x] 4. Import
      `configs/training/artificial_humans/punishment/ar_gnn_50ep_doubled.yml`
      and `artifacts/artificial_humans/punishment_ar_gnn_50ep_doubled/{model,metrics}`
      (4 files only); verify real LFS content not pointers, `.pt` md5
      `4774e934f08a96da01da875851ad7a2c` (2750) /
      `f789ab0a17ec870d3e53507db9de34f6` (5000); commit.
- [x] 5. `graph.py`: add `copula_rho=None` to `GraphNetwork.__init__`
      (store on `self`, gate `None` or `0.0 <= rho < 1.0`) and to
      `to_save`; gated copula branch in `predict_autoreg` — `z` of shape
      `(n_batch, n_nodes, n_rounds)` drawn once per call before the AR
      loop, per-step `eps` of shape `(n_batch, n_rounds)`,
      `u = Phi(sqrt(rho)*z[b, agent_group[b,i,r], r] + sqrt(1-rho)*eps)`,
      level via `searchsorted` on the float64 conditional CDF
      (`min{a : F(a) >= u}`), clamped. The `else` branch keeps the original
      statements verbatim and no torch draw precedes the branch — rho
      absent/0.0 or `sample=False` consumes the PR #161 RNG stream exactly.
- [x] 6. New Raven-only test file `src/aimanager/tests/test_ar_copula.py`:
      rho absent / 0.0 / rho>0-with-`sample=False` bit-identical to the
      legacy path including post-call RNG state; inverse-CDF bin-edge
      convention; first-revealed agent's empirical marginal matches its
      conditional row and the legacy sampler (binomial-SE tolerance);
      within-sub-group correlation induced at rho=0.95, absent across
      sub-groups; `z` fresh across calls, deterministic under re-seed; RNG
      consumption independent of group composition; `save`/`load`
      round-trips `copula_rho`, legacy checkpoints load as `None`;
      constructor gate rejects rho outside [0, 1).
- [x] 7. Local batched gate, once, before staging: eval-suite tests +
      `scripts/tests` + `tests/baselines`; single black + flake8 pass on
      `src/`. Commit steps 5-6; re-verify hooks mutated nothing.
- [x] 8. PENDING check, then `scripts/remote_test.sh` — full PyG suite
      green including the 9 AR tests and the new copula tests.
- [x] 9. Write `scripts/artificial_humans/punishment_ar_copula_rho.py`:
      unpickle shim; load the 2750 checkpoint (assert md5,
      `autoregressive`, `edge_encoding == [{ar_punishment, 31}]`); rebuild
      training data (`create_torch_data`, `switch_every=4`) on the full
      flip-doubled `experiments/2group_8agent_50ep.csv`; R=8 seeded
      per-episode reveal permutations x 8 forward passes via
      `apply_mask_pattern`, harvesting each agent's conditional row at its
      own step; `l_i, u_i = Phi^-1(F_i(y_i-1)), Phi^-1(F_i(y_i))` (clip
      1e-12); maximise the exact shared-latent cell log-likelihood
      (64-node Gauss–Hermite, grid 0-0.9 step 0.05 then bounded Brent,
      clip 0.95); cells with <2 valid agents excluded; SEED 38381.
- [x] 10. Same script — cluster bootstrap (200 resamples over the 50
      `pair_id`s); `--roundtrip` acceptance gate (rho_true in
      {0.1, 0.3, 0.5}, 2 synthetic datasets each, generated through the
      shipped sampler, max |bias| <= 0.03); diagnostics printed, never
      selecting rho (per-replicate spread, single-copy vs doubled,
      randomized-PIT moment estimator labelled ATTENUATED, round-third and
      cell-size splits, per-CV-fold estimates); `--preflight` group-spread
      ratio human vs independent-AR vs copula-AR (go/no-go only); write
      `artifacts/artificial_humans/punishment_ar_gnn_copula_50ep_doubled/model/architecture_node+edge+rnn+ar+copula__dataset_50ep_doubled__epochs_2750.pt`
      (source dict + `copula_rho`, nothing else changed),
      `metrics/copula_rho.json` + `metrics/copula_rho_calibration.log`;
      reload self-checks: every parameter tensor bit-identical to the
      source, `autoregressive is True`, new keys == `{"copula_rho"}`.
- [x] 11. Job config
      `configs/training/artificial_humans/punishment/ar_copula_rho.yml`
      carrying the calibration script through the documented
      `train_cluster.sh ah` path (device cpu for determinism); local
      black/flake8 + yaml parse; commit steps 9-11.
- [x] 12. PENDING check; assert remote source `.pt` md5; submit via
      `scripts/train_cluster.sh ah <config>`; poll to COMPLETED, exit 0:0.
- [x] 13. `scripts/fetch_cluster.sh artifacts/artificial_humans/punishment_ar_gnn_copula_50ep_doubled`;
      confirm round-trip gate PASS + reload self-checks; verify local LFS
      content + md5 parity with remote; record rho_hat / SE / CI / spread /
      pre-flight in §4 Notes; commit the artifact files (LFS).
- [x] 14. Write
      `configs/simulation/manager_testing/23_2g8a_ar_copula_self_gaussian_contr_gnn_switch.yml`:
      copy of the baseline stack config, single manager `ar_copula` -> the
      copula `.pt`, single pairing `ar_copula_self`, slugged output
      dir/figure name, rho in a header comment, no `copula_rho`/
      `autoregressive` config keys (checkpoint drives both); protocol
      byte-identical to the 23 family.
- [x] 15. Mechanical config check (yaml parse, DIR_PATTERN yields
      contr=gaussian / switch=gnn, artifact paths exist, output dir fresh,
      `save_per_round: true`, seed 42 / 100 episodes / 24 rounds
      unchanged); commit.
- [x] 16. PENDING check; confirm the copula `.pt` on Raven with matching
      md5 (push explicitly if absent); `scripts/simulate_cluster.sh
      <config>`; poll; confirm `per_round.parquet`, exit 0; re-verify
      remote `graph.py` / `.pt` md5s after the run.
- [x] 17. `scripts/fetch_cluster.sh plots/simulation/23_2g8a_ar_copula_self_gaussian_contr_gnn_switch`;
      `python -m aimanager evaluate <config>`; 21 rows for the single
      `ar_copula_self` run.
- [x] 18. §2 verdict, unrounded: gate 1 PD <= 2 (baseline
      2.6128746154448903, band 2-5); gate 2 mean < 1.7407445494371563;
      rows<=1 (baseline 7/21) as context. Guard: P-family marginals should
      sit near PR #161's (PA 0.5967, PB 0.7855, PC 0.8094, RPA 1.2689,
      RPB 0.7201) — a large P-family move signals a sampler bug, not a
      finding. Fill §3 Results + §4 Notes; commit log + sim outputs (LFS).
- [x] 19. Push; open the PR against `main` titled `[SUCCESS]`/`[FAIL]`,
      body Hypothesis / Results / Collateral.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-27 | AR-GNN punisher + severity copula (rho 0.24182866393507993, exact cell-level MLE on the AR conditionals) in gaussian x gnn x gnn | single | PD 1.275132493610912 (baseline 2.6128746154448903 — band 2-5 -> 1-2) | 9/21 (baseline 7/21) | 1.634398962774845 (baseline 1.7407445494371563 — gate 2 passes) | SUCCESS — gate 1 (PD band upgrade) and gate 2 (mean down) both pass |

## 4. Notes

1. Baseline re-verified bit-exact before planning: PD 2.6128746154448903,
   rows <= 1 7/21, mean 1.7407445494371563 (gnn_self run in the
   `23_2g8a_self_gaussian_contr_gnn_switch` evaluation on main).
2. Calibration (job 29666118, COMPLETED 0:0, 20m46s):
   rho_hat 0.24182866393507993, SE 0.029018191641679864, 95% CI
   [0.18567594377208105, 0.2976370227961655] — well below the multinomial
   punisher's 0.3508, as hypothesized (the AR channel already explains part
   of the within-round correlation). Round-trip gate PASS (max |bias|
   0.0151 <= 0.03); per-replicate rho spread 0.2397-0.2440; single-copy
   0.2421 agrees; LL gain over rho=0: 271.3 nats over 3860 cells; PIT
   moment estimator 0.075 (ATTENUATED, diagnostic only). Reload
   self-checks passed (10 tensors bit-identical, new keys ==
   {'copula_rho'}); local/remote md5 parity
   1f55ebbbd66749a5156efaaaf7d0c7b8.
3. Preflight (go/no-go context only): group-spread ratio human 0.7388,
   AR independent 0.6497, AR copula 0.6688 (predict_autoreg) / 0.6969
   (free sweep) — a go, not a guaranteed band crossing; the verdict comes
   from the single §3 evaluation.
4. Verdict detail (sim job 29666760, COMPLETED 0:0, 2m59s; evaluation 21
   rows, 500 repeats, seed 42): PD 1.275132493610912 — through the band-2
   edge the AR model alone missed by 0.0573 (PR #161: 2.0573) and past the
   copula-on-multinomial's 1.5325 (PR #160). The mechanism decomposition
   now reads: AR conditioning -0.56, shared severity latent a further
   -0.78; the two dependence channels are complementary, as PR #152 note 9
   predicted.
5. Marginal guard (step 18): the P family stays at PR #161's level under
   the copula — PA 0.7082 (#161 0.5967), PB 0.8489 (0.7855), PC 0.8175
   (0.8094), RPA 1.2324 (1.2689), RPB 0.8321 (0.7201). Differences are
   draw noise (the copula consumes a different RNG stream), not a marginal
   shift — marginal preservation held in the full stack.
6. Collateral, positive: the whole C family improves on baseline
   (CA 2.2647 -> 2.0212, CB 0.9777 -> 0.7856, CC 1.5886 -> 1.5035,
   CD 1.4803 -> 1.3433, RSA 0.9886 -> 0.8542) alongside the P family.
   Negative: CG 3.7264 -> 4.0755 and SC 2.7144 -> 3.0271 tick up
   within-band — the same shape as PR #161's run; the shared-variance
   root cause lives in the contribution and switch slots, not the
   punisher.
7. The calibration tee log (`metrics/copula_rho_calibration.log`) is
   untracked (`*.log` is gitignored repo-wide); `copula_rho.json` carries
   the full provenance and is committed.
8. Maintainer-requested rebase (2026-08-27, post-verdict): the branch was
   rebased onto `origin/auto/punisher-ar-gnn-v2` (PR #161) so the PR diff
   shows only the copula work — the step 3-4 import commits dropped as
   already-upstream (the imports were blob-hash-identical by
   construction). Tree-identity verified against the pre-rebase tip
   (`backup/pre-rebase-ar-copula`): only pure additions from the parent
   branch, every experiment file bit-identical, scores.csv blob unchanged.
   No re-run of anything; the verdict and all numbers stand as logged.
   PR #164 base changed to `auto/punisher-ar-gnn-v2` (stacked on #161).
