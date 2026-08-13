# Autoresearch log: punisher-ar-gnn

## 1. Declaration

- **Slot:** punisher
- **Base model:** GNN punisher (`architecture node+edge+rnn`,
  `artifacts/artificial_humans/punishment_rnn_edge_50ep_doubled/model/architecture_node+edge+rnn__dataset_50ep_doubled.pt`),
  candidate judged against the reference punisher
  (`lin_multinomial` + severity copula, PR #146) inside the reference stack.
- **Target rows:** PD — reference stack 1.5324969616723312 (band 1-2; success
  needs <= 1). Secondary target RPA — reference stack 1.1845929635267691
  (band 1-2; success needs <= 1). Deficit evidence: the GNN punisher's own
  slot-attributable deficit is PD (slot avg 2.634, concordant 2.01-2.88 in
  8/8 contexts of `23_stack_sweep_severity_copula`); the severity copula
  closed only part of the same gap (PD 2.93 -> 1.53 in the reference stack,
  48% above the ceiling).
- **Hypothesis:** a group's punishments in a round are one human manager's
  joint decision. The severity copula proved round-level dependence is real
  but also that an exchangeable Gaussian latent is not the whole story: the
  likelihood-implied rho (0.3508, CI [0.278, 0.423]) and the spread-implied
  rho (~0.52) disagree (punisher-severity-copula log, note 8), leaving PD at
  1.53. An autoregressive factorization of the joint punishment distribution
  — each agent's punishment conditions on round t-1 observables AND on
  groupmates' already-decided punishments of the same round (within-decision
  information the manager trivially has; never current-round contributions)
  — can represent non-exchangeable, non-Gaussian dependence (severity plus
  targeting concentration), moving the group-spread row PD to the ceiling
  and plausibly the punishment-response row RPA with it.
- **Planned change:** one change — a GNN punisher trained (flip-doubled data,
  per convention) with teacher-forced autoregressive conditioning on
  same-round groupmate punishments under a random decision order, and
  sequential (agent-by-agent) sampling in simulation. Marginal-quality risk
  is explicit: the current GNN punisher's marginals trail the multinomial's
  (PA slot avg 1.20 vs 0.65), so the P-family guards below gate the keep.
- **Stack guards (must not regress):** rows <= 1 baseline 10/21; mean
  baseline 1.687998 (reference stack, sim
  `23_2g8a_severity_copula_self_gnn_contr_gnn_switch`). P-family marginal
  guards: PA 0.5924374705499833, PB 0.9512984216948207,
  PC 0.9051112565676434, RPB 0.7705194340881973; RSA sits razor-edge at
  1.0010089590803533.
- **Slug:** `ar_gnn` (configs, artifacts, sim output dirs; slug before
  `_self_` so `evaluation_sweep.py`'s DIR_PATTERN parses).

## 2. Plan

Validated by the orchestrator 2026-08-13 (targets per §2, legality per §5,
frozen surface per §8). Slug: `ar_gnn`. Design: one new gated edge feature
(`ar_punishment`, 2 channels per edge-round: same-group AND decided gate,
gated normalised source punishment) on top of the base punisher's exact
node features; training reuses the in-repo any-order reveal-mask scheme
(`apply_mask_pattern`); sampling reuses `predict_autoreg` with the RNN
assert relaxed (the sim re-feeds full history per round, so the GRU
advances once per round by construction).

- [x] 1. Worktree + Claude commit identity + declaration (done at branch
      creation).
- [x] 2. Record the validated step list and rulings; commit.
- [x] 3. `src/aimanager/generic/graph.py`: add `ARPunishmentEdgeEncoder`
      (config `name: ar_punishment`, `n_levels: 31`, size 2: same-group AND
      decided gate; gated normalised source punishment from
      `punishment_masked`/`autoreg_mask` at `edge_index[0]`), register in
      `EDGE_ENCODERS`; docstring states the legality contract (own group,
      same round, manager's own already-decided punishments only).
- [x] 4. `graph.py` `encode()`: forward `f"{y_name}_masked"` and
      `autoreg_mask` (flattened `(N, T)`) into `edge_state` when present;
      empty-`edge_encoding` path byte-unchanged.
- [x] 5. `graph.py` `predict_autoreg()`: drop the no-RNN assert; keep
      `reset_rnn=True` per AR step; comment the contract (order from the
      sim's seeded RNG; only round -1 consumed by `MultiManager`).
- [x] 6. Raven test file `src/aimanager/tests/test_ar_punisher.py`:
      gate/value correctness (incl. mid-episode group switch); cross-group
      invariance; undecided-agent invariance; no self-leak;
      current-round-contribution invariance; train<->sim parity of the AR
      edge construction; `predict_autoreg` with RNN (shapes, determinism,
      later agents depend on earlier draws); save/load round-trip.
- [x] 7. Local: eval-suite tests + `scripts/tests` + `tests/baselines`
      (frozen-surface proof), black + flake8 batched pass.
- [x] 8. Raven: `squeue -u certuer` PENDING check, then
      `scripts/remote_test.sh` — full PyG suite green.
- [x] 9. Training config
      `configs/training/artificial_humans/punishment/ar_gnn_50ep_doubled.yml`:
      copy of `rnn_edge_50ep_doubled.yml` + `autoregression: true`,
      `min_predicted: 1`, `max_predicted: 8`,
      `edge_encoding: [{name: ar_punishment, n_levels: 31}]`,
      `labels.architecture: node+edge+rnn+ar`, epochs 5000,
      `eval_period: 250`,
      `output_dir: artifacts/artificial_humans/punishment_ar_gnn_50ep_doubled`.
- [x] 10. Train on Raven: PENDING check, `scripts/train_cluster.sh ah
      <config>`; poll; confirm 5 CV folds + final full-data fit + artifact.
- [x] 11. `scripts/fetch_cluster.sh artifacts/.../punishment_ar_gnn_50ep_doubled`;
      assert checkpoint `autoregressive` flag, `edge_encoding`, and base
      `x_encoding` parity; checksum vs Raven; commit (LFS).
- [x] 12. Convergence/selection analysis from the metrics parquet: mean CV
      test log_loss per epoch -> epoch E; report the n_pred==8 marginal
      curve vs base best 1.2029941249670393 and n_pred<8 curves (evidence
      the AR channel is used). Log in Notes.
- [x] 13. (conditional, triggered) Optimum clearly earlier than 5000:
      retrain at epochs 2750 (job 29318893), fetch, commit.
- [x] 14. Stage-1 config
      `23_2g8a_ar_gnn_self_gnn_contr_gnn_switch.yml`: copy of the reference
      config, single manager `ar_gnn` (type `human`) -> new `.pt`, single
      `ar_gnn_self` pairing, slugged output dir; protocol byte-identical.
- [x] 15. Simulate on Raven: PENDING check, `scripts/simulate_cluster.sh
      <config>`; poll; confirm `per_round.parquet`. (Job 29319185, 2m14s —
      no walltime issue; D13 not needed.)
- [x] 16. `scripts/fetch_cluster.sh` + `python -m aimanager evaluate
      <config>`.
- [x] 17. Keep gate (unrounded): PD < 1.5324969616723312, band upgrade =
      PD <= 1 (or RPA <= 1, ref 1.1845929635267691); rows<=1 >= 10/21;
      mean <= 1.687998. P-family reported as diagnostics vs both the
      copula reference row and the GNN punisher's own `gnn_self` row
      (ruling D11). Log all 21 rows unrounded. **GATE FAILED** (note 8).
- [x] 18. Gate fails or no band upgrade -> Notes + `[FAIL]` PR, stop.
- [ ] ~~19. Stage-2 configs: the 7 remaining~~ (n/a — Stage 1 failed)
      `23_2g8a_ar_gnn_self_<contr>_contr_<switch>_switch.yml`, copies of
      the matching severity_copula configs with only
      managers/pairings/output_dir/figure_name swapped.
- [ ] 20. Simulate (PENDING check per sync) + fetch + evaluate the 7.
- [ ] 21. Add `ar_gnn` to PUNISHER_ORDER/PUNISHER_COLORS in
      `evaluation_sweep.py` (analysis layer); sweep 24 dirs (8 base +
      8 severity_copula + 8 ar_gnn) into `23_stack_sweep_ar_gnn`.
- [ ] 22. Confirm the slot claim: PD bands + wins across the 8 contexts,
      concordance panel, stack metrics net over contexts.
- [ ] 23. Complete the log; PR `[SUCCESS]`/`[FAIL]`, body Hypothesis /
      Results / Collateral; commits map to steps.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-13 | AR GNN punisher (gated ar_punishment edge feature, epochs 2750 by CV) | 1 | PD 2.5750800730012529 (ref 1.5324969616723312 — regressed); RPA 1.1831865954389955 (ref 1.1845929635267691 — within-band) | 10/21 (ref 10/21) | 1.7058335780300315 (ref 1.6879978841849728 — regressed) | FAIL — no band upgrade possible, no Stage 2 |

## 4. Notes

1. Deficit profile fetched from
   `plots/data_analysis/evaluation/23_stack_sweep_severity_copula/score_matrix.csv`:
   for punisher = gnn, PD is the only P-family row >= 2 (slot avg 2.634,
   concordant 2.01-2.88 across all 8 contexts); CG/SC/RCA/RCB are high in
   every punisher context and belong to the other slots. For the reference
   punisher (multinomial_copula) PD is already 1.07 slot avg but 1.53 in the
   reference stack — the residual is the non-exchangeable dependence this
   experiment targets.
2. Prior work read: `punisher-severity-copula.md` (merged log; note 8 is the
   motivating finding), PR #150 (switch herding copula — persistence of the
   latent was essential there), PR #151 (contribution gaussian_mlp — better
   marginals bought with independent sampling explode CG; a warning that
   joint structure and marginal fit must be judged together).
3. Planner finding: the repo already carries the AR machinery from an
   abandoned legacy experiment — `train.py`'s `apply_mask_pattern` (any-order
   reveal masks, 255 patterns for 8 agents), `GraphNetwork.predict_autoreg`,
   and the checkpoint `autoregressive` flag. The legacy node-feature form
   (`punishment_masked` as `x_encoding`, recoverable via
   `git show b0695b7^:configs/.../autoregressive_50ep.yml`) would leak the
   OTHER group's same-round punishments — the graph is fully connected
   across sub-groups and one `get_punishments` call serves both groups —
   so the AR conditioning must be a same-group-gated EDGE feature (the AR
   analogue of copula ruling D1). The gate also fixes a train/inference
   skew in `predict_autoreg`: undecided agents carry stale past punishments
   in `y_masked` at sim time but are blanked in training; gated by the
   decided mask, both are identically zero.
4. Planner discrepancies, orchestrator rulings: (D1/D2) same-group-gated
   edge feature adopted; gate does NOT require `punishment_valid`
   (`manager_no_input` is constant within all 4512 cells — moot, simpler).
   (D3) in-repo any-order subset-reveal scheme adopted unchanged — unbiased
   over orders, zero training-code change, matches what sampling sees; no
   per-round masks, no reweighting (ties to the simpler model). (D4) RNN
   kept (`add_rnn: True`, base parity); `predict_autoreg`'s no-RNN assert
   relaxed — safe because the sim re-feeds the full history with
   `reset_rnn=True`, so the GRU advances once per round by construction.
   (D5) epochs 5000, `eval_period: 250` (255-pattern eval dominates
   runtime); conditional retrain at the CV-test-log-loss-optimal epoch E —
   legal §5 hyperparameter search, nothing keyed to a metric. (D6) epochs
   label kept in the artifact name; Stage-1 config written after training.
   (D7) the `graph.py` edits count as this punisher experiment, not a
   separate bug fix: additive, opt-in, inert for every existing artifact
   (`SameGroupEdgeEncoder` precedent); the Raven suite must stay green as
   proof. (D8) PENDING-job check before every rsync; fetched artifact
   checksummed against Raven before commit. (D9) metrics parquet committed
   as convention (LFS). (D10) no `autoregressive` key in the sim config;
   the step-11 checkpoint assertion suffices. (D11) hard gates are only
   the §2 stack metrics; P-family rows are diagnostics (the marginal model
   itself changes — the copula guards assumed preserved marginals), read
   against both the copula reference row and the GNN punisher's own
   `gnn_self` row (`plots/simulation/23_2g8a_self_gnn_contr_gnn_switch`).
   (D12) unrounded reporting; boundary calls escalated, copula precedent.
   (D13) `run_simulation.sh` `--time` may be raised to 2h only on an
   actual walltime failure (infrastructure, not frozen protocol); logged
   if it happens.
5. Steps 3-8 done and verified. Full Raven suite green (99 passed, incl.
   the 9 new AR tests: cross-group / undecided / self / current-round
   contribution leak invariants, train<->sim parity of the encoder inputs,
   RNN-enabled `predict_autoreg` determinism and conditioning, save/load).
   Facts worth keeping: `autoreg_mask` polarity is True = to-be-predicted
   (decided = ~mask); `predict_autoreg` reveals a decided agent's WHOLE
   round history (parity with training holds because the gate zeroes
   everything outside same-group-and-decided); `predict_independent` on an
   `ar_punishment` model requires `punishment_masked`/`autoreg_mask` in the
   data (hard assert, no silent fallback) — the sim path always goes
   through `predict_autoreg`, which builds them. Out of scope, untouched:
   `rl_manager.py` and `train.py` are not black-clean on main (pre-existing
   drift).
6. Wall-clock correction: the planner's 4-5 h training estimate was ~7x
   too pessimistic — the 255-pattern AR eval costs ~6 s (not 30-40 s), and
   a 5000-epoch fit runs ~6 min at ~21 it/s. The 6-fit job (29318459)
   completed in 40m06s, exit 0:0.
7. Step-12 analysis (5000-epoch probe, metrics parquet): mean CV test
   log_loss optimum 1.1930113224996335 at epoch 2750 vs base punisher best
   1.2029941249670393 (at its budget end, 1240) — the AR model wins on
   held-out likelihood by ~0.010. Overfit past ~3000 (test 1.207264 at
   4999, train still falling). Conditioning gradient is monotone in the
   number of decided groupmates: test log_loss at best epoch 1.1912
   (n_pred=1, 7 decided) -> 1.1989 (n_pred=8, none decided) across all 8
   levels — direct evidence the AR channel carries signal. Even the
   zero-conditioning marginal (1.1989) beats the base (1.2030). Ruling:
   retrain at E=2750 (step 13 condition met; job 29318893). The 5000-epoch
   artifact stays committed for provenance; the 2750 one becomes the
   candidate.
8. Stage 1 (sim job 29319185, 2m14s): **gate failed.** PD
   2.5750800730012529 vs reference 1.5324969616723312 — the primary
   target regressed badly against the copula punisher; RPA
   1.1831865954389955 vs 1.1845929635267691 — a 0.0014 within-band move;
   mean 1.7058335780300315 vs 1.6879978841849728 (regressed); rows<=1
   10/21 (held). No band upgrade is reachable, so per §9.6 no Stage 2.
   AR dispatch verified: the checkpoint flag drives `predict` (the sim
   config carries no `autoregressive` override), and the independent path
   would have hit the encoder's hard assert.
9. The instructive contrast (D11 diagnostics, vs the GNN punisher's own
   `gnn_self` row): the AR candidate improves EVERY P-family row of its
   base family — PA 1.2666 -> 0.5948178628121934, PB 1.1138 ->
   0.7529474373521686, PC 1.1602 -> 0.7470295004161448, PD 2.8228 ->
   2.5750800730012529, RPA 1.5552 -> 1.1831865954389955, RPB 1.3435 ->
   0.7335300047644394 — the GNN punisher family is now
   marginal-competitive with the multinomial (likely mostly the 2750
   epoch budget: the base's 1250 was under-trained). But the AR channel
   recovers only a sliver of the within-round dependence: PD moves 2.82
   -> 2.58 where the copula's explicit latent (rho = 0.3508) reaches
   1.53. Teacher-forced conditioning on observed groupmate punishments
   learns the small observable signal (the 0.008-nat CV gradient of note
   7) but cannot represent the manager's round-level severity latent —
   at sampling time the first-drawn agents carry no information about
   the round's severity, so the joint spread stays near the independence
   floor. The mechanism that closed PD is a *shared latent*, not
   *observed conditioning*; the two are not substitutes.
10. Follow-up seeds for the next agent: (a) severity-copula sampling ON
    TOP of the AR-GNN marginals (the copula is marginal-preserving, so
    the improved P-family of note 9 would be kept while the latent
    supplies the spread); (b) a plain 2750-epoch retrain of the non-AR
    base punisher to isolate how much of note 9 is budget rather than
    mechanism; (c) a round-level latent severity input to the GNN
    (learned, sampled at sim time) — the direct generative analogue of
    the copula.
