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

Validated by the orchestrator 2026-08-11 (targets per §2, legality per §5,
frozen surface per §8). Slug: `severity_copula`.

- [x] 1. Worktree + Claude commit identity (done at branch creation).
- [x] 2. Calibration script `scripts/baselines/punishment_copula_rho.py`
      (local): load `punishment_multinomial_best_with_contr.joblib`, rebuild
      its features on its own training data (`bundle["config"]` ->
      `experiments/baseline/2group_8agent_50ep_bline_train.csv`, single-copy,
      40 episodes) via `handcrafted_grid` utilities, keep (episode, round,
      group) indices, build the class-probability matrix exactly as the
      adapter does (1e-12 floor, classes_ scatter, renormalise).
- [x] 3. Randomized PIT (punishment is lumpy; mid-point PIT collapses ties
      and attenuates rho): u_i = F_i(y_i - 1) + v_i * p_i(y_i), z = ndtri(u);
      average rho over R=20 PIT replicates, fixed seed; mid-point PIT printed
      as sensitivity only.
- [x] 4. ~~rho = exchangeable moment estimator over within-(episode, round,
      group) pairs~~ **REVISED after step 7** (randomized-PIT moment
      estimator proven ~2.3x attenuated by round-trip at this
      discretisation — 69% mass at level 0): rho = pairwise-likelihood MLE
      of the exchangeable Gaussian copula over within-cell pairs (rectangle
      probabilities from the bivariate normal CDF with each observation's
      own discrete marginal; grid + refine over rho), validated by
      round-trip recovery of known rho_true; PIT moment estimator kept as
      printed diagnostic. Cluster bootstrap over 40 episodes for SE/CI;
      diagnostics-only splits; out-of-sample test-split check (never a
      selection criterion).
- [x] 5. Pre-flight (--preflight): replay human P matrices through
      independent vs copula sampler, print group-spread ratios vs human
      (~0.739 human, ~0.578 independence floor). Go/no-go only; rho is never
      tuned to it.
- [x] 6. Save `artifacts/baselines/punishment_multinomial_severity_copula.joblib`
      = old bundle + `copula_rho` (+ provenance fields); assert no
      pre-existing key modified and predict_proba bit-identical on reload.
- [x] 7. Run calibration; record rho +/- SE, CI, pre-flight ratios in Notes.
      Escalate and stop if rho not clearly > 0 or pre-flight barely moves.
- [ ] 8. Adapter gate in `linear_ah.py.__init__`: `copula_rho` from bundle,
      assert 0 <= rho < 1 and multinomial-punishment-only. `_sample_levels`
      untouched (existing bundles keep exact RNG consumption).
- [ ] 9. `_sample_levels_copula(Xs, n_levels, groups)`: P as in
      `_sample_levels`; fixed 2A torch draws per call (zs, eps);
      **per-group z** (D1: one manager call serves both groups in self
      pairings; human data has one manager decision per group-round);
      u = ndtr(sqrt(rho) z_g + sqrt(1-rho) eps); searchsorted on cumsum(P).
- [ ] 10. Wire into `get_punishments`, groups from `rounds[-1]["agent_group"]`
      (same source as the features); dispatch only when sample and rho > 0.
- [ ] 11. Local unit tests `tests/baselines/test_punishment_copula.py`:
      inverse-CDF correctness; marginals preserved vs independent sampler;
      within-group correlation induced; cross-group ~0; rho-absent/rho=0
      bit-identical to today's path under same seed; determinism; gate
      assertion raises.
- [ ] 12. Run local suites: `pytest tests/baselines` + the eval-suite tests
      (frozen surface untouched proof).
- [ ] 13. Stage-1 config
      `23_2g8a_severity_copula_self_gnn_contr_gnn_switch.yml`: copy of the
      reference config, single manager `lin_multinomial_copula` -> new
      joblib, single self pairing, slugged output dir (slug BEFORE `_self_`
      so `evaluation_sweep.py`'s DIR_PATTERN still parses); protocol
      byte-identical.
- [ ] 14. Push the joblib to Raven explicitly (simulate_cluster.sh excludes
      `artifacts/`): rsync to `raven:~/algorithmic-institutions/artifacts/baselines/`.
- [ ] 15. `scripts/simulate_cluster.sh <config>`; poll; confirm
      per_round.parquet.
- [ ] 16. `scripts/fetch_cluster.sh` + `python -m aimanager evaluate <config>`.
- [ ] 17. Keep gate vs reference `lin_multinomial_self` row: PD < 2.934892;
      rows <= 1 >= 11; mean <= 1.759557; P-guards ~unmoved (PA 0.634134,
      PB 0.878167, PC 0.778075, RPA 1.268305, RPB 0.813541 — a P-family move
      signals a sampler bug). Log unrounded.
- [ ] 18. Gate fails -> Notes + `[FAIL]` PR, stop.
- [ ] 19. Stage 2: 7 more slugged configs (the Stage-1 config is the
      gnn x gnn cell), same single-pairing edits, no protocol change;
      simulate, fetch, evaluate each.
- [ ] 20. Add `multinomial_copula` to PUNISHER_ORDER / PUNISHER_COLORS in
      `evaluation_sweep.py` (analysis script, not frozen; D5), then sweep
      old 8 + new 8 dirs into `23_stack_sweep_severity_copula`.
- [ ] 21. Confirm slot claim: copula beats multinomial on PD in (nearly) all
      8 contexts, P-guards hold, check PD concordance panel.
- [ ] 22. Complete log; PR `[SUCCESS]`/`[FAIL]`, body Hypothesis / Results /
      Collateral; commits map to steps.

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
3. Planner discrepancies, orchestrator rulings: (D1) one `get_punishments`
   call serves BOTH groups in a self pairing -> latent keyed per
   `agent_group`, not per call; human data confirms one manager decision per
   (episode, round, group) cell (2256/2256 constant `manager_no_input`).
   (D3) rho estimated on the locked train split
   (`2group_8agent_50ep_bline_train.csv`, single-copy) instead of the full
   human file — keeps the 10 holdout games closed, parity with the marginal
   fit; test-split rho printed as out-of-sample confirmation only.
   (D4) `simulate_cluster.sh` rsyncs with `--exclude='artifacts/'` — the new
   joblib must be pushed to Raven explicitly. (D5) `evaluation_sweep.py`
   needs the `multinomial_copula` label in PUNISHER_ORDER/PUNISHER_COLORS or
   Stage 2 KeyErrors; analysis layer, not frozen surface. (D2) the bundle
   carries no `temperature` field -> adapter default 1.0, no-op.
4. Feasibility (planner, read-only): unconditional within-cell punishment
   ICC ~0.35; human group-spread ratio ~0.739 vs sim ~0.578 (independence
   floor), so the latent rho must land around/above ~0.3 after
   discretisation attenuation — the step-5 pre-flight checks this before
   any cluster run.
5. Step-7 first run (moment estimator): rho_hat = 0.13829097756981062,
   bootstrap SE 0.018509224874473314, 95% CI [0.10219321599647306,
   0.17397141462131743]; per-cell 0.1897, ICC(1) 0.1633, mid-point PIT
   0.2154; out-of-sample test split 0.09482827476218403. Pre-flight
   (train-split P, 50 repeats): independent 0.6271, copula@rho_hat 0.6514,
   human 0.7503 — closes only 24% of the gap. Feature rebuild parity
   proven: reconstructed train log loss 1.3007198112962215 == stored
   train_metric, bit-identical. (Pre-flight baseline sits above the 1/sqrt(n)
   floor because the human P matrices already carry feature-driven
   correlation and 186+208 cells have size 1-2; human ratio 0.7503 not
   ~0.739 because train split only.)
6. Round-trip diagnostic (generate at known rho, re-estimate): the
   randomized-PIT moment estimator is ~2.3-2.4x attenuated at this
   discretisation (69% of mass at punishment 0 — the auxiliary uniform
   carries most of z's variance). Human rho_hat 0.138 => latent rho ~0.32.
   Plan step 4 revised to the pairwise-likelihood MLE (polychoric-style),
   acceptance = round-trip recovery of rho_true. Rejected alternative:
   moment-matching the group-spread ratio (~0.52 implied) — that is
   calibration at PD's definition, illegal per §5. If the honest MLE closes
   only part of the gap, that is the honest result; the exchangeable
   Gaussian copula is evidently not the whole dependence story
   (correlation-implied 0.32 vs spread-implied 0.52 disagree).
7. Round decay observed: within-cell correlation 0.217 in rounds 0-7 vs
   ~0.10 later — constant rho kept (ties go to the simpler model, §5);
   a round-dependent severity latent is a candidate follow-up experiment.
   (Superseded by the MLE splits in note 8: the decay was mostly an
   attenuation artifact — MLE per round third is 0.36/0.33/0.35, stable.)
8. Step 7 rerun with the pairwise MLE: **rho_hat = 0.3507588625344979**,
   cluster-bootstrap SE 0.037686140204276034 (200 resamples), 95% CI
   [0.2779642544867047, 0.42319106056672023]. Acceptance gate passed:
   round-trip recovery at rho_true in {0.1..0.5} has max |bias|
   0.006148711994136524 (tolerance 0.03); Phi_2 (Drezner-Wesolowsky, 32
   nodes) agrees with scipy mvn to 3.3e-16 and pair_nll(0) matches the
   closed-form independence value to 1e-10. Out-of-sample test split (10
   episodes, check only): 0.24274527759925443. Pre-flight at the MLE rho:
   independent 0.6270942235359207, copula 0.7039800602398409, human
   0.7503476329282582 — 62% of the spread gap closed (vs 20% at the
   attenuated rho). Not pushed further: matching the human ratio would
   need rho ~0.52, outside the likelihood CI — per §5 the metric is not a
   calibration target. Bundle re-saved with copula_rho = MLE,
   copula_estimator='pairwise_mle', PIT fields kept as diagnostics
   (copula_diag_pit). GO for the adapter step.
