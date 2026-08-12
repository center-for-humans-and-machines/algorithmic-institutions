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
  **Re-baselined 2026-08-12 (maintainer ruling, Notes 5):** the reference
  punisher is now the severity-copula multinomial bundle
  (`artifacts/baselines/punishment_multinomial_severity_copula.joblib`,
  PR #146, same `lin_multinomial` slot). New reference cell = #146's
  Stage-1 run (`23_2g8a_severity_copula_self_gnn_contr_gnn_switch`):
  **CG 9.808514; rows <= 1 = 10/21; mean 1.687998** — these replace
  9.850 / 11 / 1.759557 everywhere in the plan's gates.
- **Prior art consulted:** PR #144 [FAIL] (dropout regresses RCA), PR #147
  [FAIL] (kept but no band upgrade; RCA/RCB collateral), PR #146 [SUCCESS]
  (copula mechanism + estimator), PR #140 comment (sampler-vs-model
  decomposition: for contribution ~1/6 of the co-movement gap is
  sampler-bound on a *well-conditioned* model — the free-running GNN
  under-uses state, so its residual shared error, which the copula carries,
  is larger; the calibration step measures it before any cluster run).

## 2. Plan

Validated by the orchestrator 2026-08-12 (targets per §2, legality per §5,
frozen surface per §8; planner discrepancies D1–D9 recorded in Notes).
Slug: `cg_copula`.

- [x] 1. (Raven) Cluster preflight: `squeue -u certuer` for parallel
      PENDING jobs; confirm M0 `.pt` exists remotely; check whether the M4
      `.pt` from PR #144 still exists on Raven (metrics were fetched, model
      never was — D6); confirm scipy importable in the remote `.venv`.
- [x] 2. (Raven) ~~Start the M4 retrain (long pole)~~ D6 confirmed: the M4
      `.pt` exists on Raven (mtime Jul 1 — the seed-pinned product of the
      unchanged config); fetched and committed instead of retraining.
      Original step:
      `scripts/train_cluster.sh ah configs/training/artificial_humans/contribution/group_switching_contribution_50ep_own_group_same_group.yml`
      (unchanged config, seed-pinned). Poll while later steps proceed.
- [x] 3. New pure-torch module `src/aimanager/generic/copula.py` (no PyG
      import, locally testable): `sample_levels_copula(proba, cells, rho)`
      with the #146 sampler conventions — exactly 2N float64 draws per call
      (`zs` then `eps`), latent from each cell's first member,
      `u = ndtr(sqrt(rho) z + sqrt(1-rho) eps)`, discrete-quantile
      inversion via searchsorted on the cumsum, clamp to [0, K-1].
- [x] 4. Wire into `GraphNetwork` (`src/aimanager/generic/graph.py`),
      contribution draw only: `copula_rho=0.0` in `__init__` (assert
      0 <= rho < 1; rho > 0 only with `y_name == "contribution"`), append
      to `to_save`; optional `cells=None` through `predict_encoded`;
      `predict_independent` builds cells from `data["agent_group"]`.
      `predict_autoreg`, `encode`, `forward`, `encoder.decode` byte-
      identical; legacy `.pt` loads default to 0.0 (exact pre-change RNG).
- [ ] 5. Local unit tests `tests/copula/test_contribution_copula.py` (pure
      torch): bin-edge inversion; marginal preservation vs independent
      draws; within-cell corr > 0, cross-cell ~ 0; determinism;
      composition-stable RNG consumption; parity vs
      `LinearAHAdapter._sample_levels_copula` on the same P/groups/seed.
- [ ] 6. (Raven) GNN wiring tests
      `src/aimanager/tests/test_contribution_copula_gnn.py` via
      `remote_test.sh`: toy GraphNetwork induces within-group corr with
      marginals intact; rho absent/0.0 bit-identical to legacy decode
      incl. RNG; save/load round-trips `copula_rho`; committed M0 loads
      at 0.0; init gate raises (protects switch + valid models).
- [ ] 7. Calibration script
      `scripts/artificial_humans/contribution_copula_rho.py`
      (`--model IN.pt --out OUT.pt [--preflight] [--roundtrip]`), importing
      the #146 estimator machinery from
      `scripts/baselines/punishment_copula_rho.py` unmodified; installs the
      `torch_geometric.nn.meta` alias before `GraphNetwork.load` (D7).
- [ ] 8. Calibration data: `create_torch_data` on the FULL human file (so
      defaults match training), then keep the 40 single-copy episodes of
      `experiments/baseline/2group_8agent_50ep_bline_train.csv`, taking the
      copy present in the train file (copy choice is not neutral — D5).
      Teacher-forced `predict_independent(sample=False)` probabilities,
      rows where `contribution_valid`, cells = (episode, round, agent_group).
- [ ] 9. Estimator + gates: pairwise MLE (grid + Brent), cluster bootstrap
      SE/CI over 40 episodes, `--roundtrip` acceptance (max |bias| <= 0.03),
      randomized-PIT printed as attenuated diagnostic, test-split rho as
      out-of-sample check only, `--preflight` group-spread ratios
      (independent / copula / human) as go/no-go only (D9).
- [ ] 10. Stamp rho into a copy of the artifact: `model.copula_rho = rho;
      model.save(out)`; reload-assert logits bit-identical, rho
      round-trips, no other key changed. Arm A out:
      `artifacts/artificial_humans/group_switching_contribution_50ep_cg_copula/model/..._cg_copula.pt`.
- [ ] 11. (Raven) Run arm-A calibration (+preflight +roundtrip). Escalate
      and stop before any sim if rho CI touches 0, round-trip fails, or
      preflight barely moves the ratio. Fetch + commit the `.pt`; log rho,
      SE, CI, round-trip, preflight unrounded.
- [ ] 12. (Raven) M4 arrived: fetch + commit; run arm-B calibration the
      same way (own rho, never reuse arm A's); same gates; log.
- [ ] 13. Stage-1 configs (single `lin_multinomial_self` pairing pointing
      at the severity-copula joblib — the re-baselined reference punisher,
      Notes 5 — protocol byte-identical):
      `23_2g8a_cg_copula_m0_self_gnn_copula_contr_gnn_switch.yml` and
      `23_2g8a_cg_copula_m4_self_gnnm4_copula_contr_gnn_switch.yml`.
- [ ] 14. (Raven) squeue guard, then `simulate_cluster.sh` both arms;
      verify remote artifact paths before submitting; confirm
      `per_round.parquet`.
- [ ] 15. `fetch_cluster.sh` + `python -m aimanager evaluate` both arms;
      read unrounded scores.
- [ ] 16. Arm selection by Stage-1 CG subject to guards. Keep iff
      CG < 9.808514 AND rows<=1 >= 10 AND mean <= 1.687998 (re-baselined,
      Notes 5); RC/CA/CD rows
      materially moving on arm A signals a sampler bug (arm B compares
      against #144's p=0 control: CG 7.587, 11/21, 1.621).
- [ ] 17. Band decision: upgrade requires CG < 5 in the reference stack.
      Kept-but-within-band or gate-fail -> Notes + `[FAIL]` PR (skip
      Stage 2, it cannot become a success).
- [ ] 18. (If Stage 2) add the winning contr token to `CONTR_ORDER` and
      `CONTR_MARKERS` in `scripts/data_analysis/evaluation_sweep.py`
      (analysis layer; D2 — there is no CONTR_COLORS).
- [ ] 19. Stage-2 configs: TWO configs (D1), full four-punisher pairing
      block, gnn-switch and lin-switch variants:
      `23_2g8a_cg_copula_self_<label>_contr_{gnn,lin}_switch.yml`
      = 8 contexts.
- [ ] 20. (Raven) Simulate, fetch, evaluate both; sweep old 8 reference
      dirs + 2 new into `23_stack_sweep_cg_copula`; confirm slot claim on
      CG across all 8 contexts, guards hold, band upgrade survives
      (unrounded scores.csv for boundary calls).
- [ ] 21. Complete this log; PR `[SUCCESS]`/`[FAIL]`, body Hypothesis /
      Results / Collateral; commits map to steps.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|
| 2026-08-12 | (baseline) reference stack, old multinomial punisher | 1 | CG 9.850 | 11/21 | 1.760 | superseded baseline |
| 2026-08-12 | (baseline) reference stack, severity-copula punisher (#146 Stage-1 run) | 1 | CG 9.808514 | 10/21 | 1.687998 | baseline (Notes 5) |

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
3. Planner discrepancies, orchestrator rulings: (D1) contribution Stage 2
   is 2 configs x 4 punisher pairings = 8 contexts, not 8 configs — plan
   step 19 reflects it. (D2) `evaluation_sweep.py` has CONTR_ORDER +
   CONTR_MARKERS, no CONTR_COLORS. (D3) encoder/graph import PyG at module
   level: calibration, rho stamping, and all GNN-level tests run on Raven;
   only the standalone sampler module tests locally. (D4) the GNN trained
   on all 100 flip-doubled episodes, so rho on the 40-episode baseline
   train split is holdout-closing convention, not out-of-sample w.r.t. the
   marginals — no out-of-fold predictions exist to do better; disclosed
   here. (D5) flip-doubling lives in the CSV and M0 conditions on
   agent_group, so the two copies of a game give different marginals —
   calibration takes exactly the copy present in the train file. (D6) the
   M4 `.pt` may still exist on Raven from PR #144 — check before
   retraining. (D7) `copula_rho` rides in the `.pt` via `__init__` +
   `to_save`; legacy artifacts splat to the 0.0 default; re-saving needs
   the `torch_geometric.nn.meta` alias to load legacy pickles. (D8) latent
   keyed per `agent_group` within one predict call (one call covers both
   groups), same resolution as #146's D1. (D9) the preflight group-spread
   ratio is CG's own statistic: go/no-go readout only, rho never tuned to
   it.
4. Planner risk assessment, accepted: the copula is a within-round,
   round-independent shock while CG rewards persistent between-group
   divergence, and marginals mean-revert through prev_contribution and the
   RNN — closing 0.59 -> 0.85 at group size 4 would imply latent rho ~ 0.6,
   likely outside the honest MLE's CI. Realistic failure mode is
   kept-but-within-band; the step-11 preflight is the cheap early read.
5. Maintainer ruling 2026-08-12 (mid-experiment, before any Stage-1 config
   existed): the reference punisher is now the severity-copula multinomial
   bundle from PR #146 — same `lin_multinomial` slot, not a separate
   option; the maintainer updates `notes/autoresearch.md` themselves.
   Experiment re-baselined to #146's Stage-1 cell
   (`23_2g8a_severity_copula_self_gnn_contr_gnn_switch`): CG 9.808514,
   rows <= 1 = 10/21 (RSA 1.001009 — the razor-edge row #146's Stage 2
   adjudicated as noise), mean 1.687998. Target unaffected (CG 9.85 →
   9.81); Stage-1 configs will point `lin_multinomial` at the copula
   joblib.
