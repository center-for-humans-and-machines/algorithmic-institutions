# Autoresearch log: switch-herding-copula

## 1. Declaration

- **Slot:** switch
- **Base model:** `gnn` switch predictor
  (`artifacts/artificial_humans/switch_pred_opt_50ep_doubled_reanchored/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`)
- **Target rows:** SC — reference stack 3.270439 (band 2-5); slot average
  2.65 over the 16 gnn-switch contexts, >= 2 in 15/16 (range 1.77-3.45),
  concordant. The only S-family deficit: SA 0.721349, SB 0.753990,
  RSA 0.908837 in the reference stack — rates match, the group aggregate
  does not. Success requires SC < 2 in the reference stack (band upgrade
  2-5 -> 1-2), Stage-2 confirmed.
- **Hypothesis:** eligible players in a group see the same situation and
  their switch decisions co-move beyond what shared observable features
  explain (~40% of within-group co-movement survives conditioning on state —
  the motivating comment on PR #140). The simulation draws every agent's
  switch independently, which pins group-level switching to the independence
  floor: an independent null handed the *exact* human switch rates
  reproduces our simulated SC almost exactly (mean larger-group size 5.28
  null / 5.46 sim / 6.09 human; P(fully segregated) 0.018 / 0.030 / 0.144),
  human switching is ~1.8x more variable than independent draws allow, and
  episode memory is absent (first-half/second-half segregation correlation
  0.38 human vs 0.03 sim). Capturing the shared component should widen the
  larger-group-size distribution toward the human one and move SC.
- **Planned change:** Gaussian-copula sampling for the GNN switch predictor —
  a shared standard-normal latent per (episode, switch-round, group), mixed
  with per-agent noise at weight rho, pushed through each agent's own
  predicted Bernoulli marginal; marginals preserved by construction, so
  SA/SB/RSA should not move; no retraining of the marginal model. Two arms,
  selected by Stage-1 score per §5: (A) rho only; (B) rho plus an AR(1)
  persistence phi on the group latent across an episode's switch rounds, to
  carry the episode-memory effect. rho (and phi) are estimated from the
  human train split only, by pairwise-likelihood MLE against the GNN's own
  predicted marginals (the PR #146 estimator, adapted to binary outcomes);
  stored as a field on a copy of the switch artifact; artifacts without the
  field sample independently, bit-identical to today.
- **Stack guards (must not regress):** rows <= 1 baseline 11/21; mean
  baseline 1.759557 (reference stack `gnn x gnn x lin_multinomial`);
  marginal guards SA 0.721349, SB 0.753990, RSA 0.908837 — an S-rate move
  signals a sampler bug.

## 2. Plan

Validated by the orchestrator 2026-08-12 (targets per §2, legality per §5,
frozen surface per §8). Slug: `herding_copula`; switch label: `herdcopula`.

- [x] 1. Worktree + Claude commit identity; record the validated plan in
      this file; commit.
- [ ] 2. Local calibration script `scripts/baselines/switch_copula_rho.py`
      (pandas/numpy/scipy, no PyG) with a `--raw-stats` mode reproducing the
      feasibility numbers (note 5) from
      `experiments/baseline/2group_8agent_50ep_bline_train.csv` via
      `parse_agent_rounds(df, switch_every=4)`, `switch_valid` rows only.
      Gate: escalate and stop if within-cell pairs < 1500 or unconditional
      tetrachoric rho <= 0.10.
- [ ] 3. Raven dump script `scripts/artificial_humans/dump_switch_probs.py`:
      load the base switch artifact, `create_torch_data` on the human data,
      full-episode forward (mask `switch_valid`, GRU over rounds = the
      simulation's warm-RNN semantics), write
      `artifacts/artificial_humans/switch_pred_herding_copula/calibration/switch_probs_train.parquet`
      (episode_id, global_group_id, round_number, player_idx, agent_group,
      does_switch, switch_valid, p_switch) + `_test.parquet` for the 10
      holdout games. Assert mean predicted p over eligible rows ~ 0.2937 and
      the per-round profile tracks step 2.
- [ ] 4. `squeue -u certuer` PENDING check (precedent note 11), sync, run the
      dump on the Raven login node, `scripts/fetch_cluster.sh` both parquets,
      commit.
- [ ] 5. Binary pairwise-likelihood MLE in `switch_copula_rho.py`:
      t_i = Phi^-1(1 - p_i), bivariate-normal rectangle probabilities,
      reusing the Drezner-Wesolowsky `bvn_cdf` + grid/Brent structure of
      `punishment_copula_rho.py` adapted to 2 levels.
- [ ] 6. Acceptance gate: round-trip recovery at rho_true in {0.05..0.5}
      (max |bias| <= 0.03) and `pair_nll(0)` equal to the closed-form
      independence value to 1e-10. Abort and escalate on failure.
- [ ] 7. rho_hat on the 40-game train split; cluster bootstrap over episodes
      (200 resamples) for SE / 95% CI; 10-game holdout printed as
      out-of-sample check only; per-round and per-cell-size splits as
      diagnostics (constant rho kept — ties to the simpler model).
- [ ] 8. Arm B: rho_lag1 by the same pairwise MLE over cross-round
      same-(episode, group) pairs at lag 1 (model implies corr = rho * phi);
      phi_hat = rho_lag1 / rho_hat with bootstrap CI. If the CI includes 0
      or phi_hat <= 0: arm B dropped with a Notes entry, never hand-tuned.
- [ ] 9. Pre-flight go/no-go (`--preflight`): redraw switch indicators from
      the dumped marginals per human decision cell, independent vs copula at
      rho_hat; compare larger-group-size distribution and P(fully segregated)
      at s+1 against the realized human one, next to the PR #140 reference
      numbers (5.28 null / 5.46 sim / 6.09 human; 0.018 / 0.030 / 0.144).
      Escalate if the copula barely moves; rho is never tuned to this.
- [ ] 10. Run steps 5-9; write
      `.../switch_pred_herding_copula/calibration/copula_params.json`
      (rho, se, ci, phi, estimator, base-artifact sha256, data file,
      n_pairs, n_cells); record every number unrounded in §4; commit.
- [ ] 11. PyG-free sampler `src/aimanager/generic/copula.py` (torch only):
      `sample_correlated_levels(proba, cell_id, rho, z_prev=None, phi=0.0)`
      — exactly 2N draws per call (z then eps, float64,
      composition-stable), per-cell latent from the cell's first member,
      u = ndtr(sqrt(rho) z_cell + sqrt(1-rho) eps),
      level = searchsorted(cumsum(proba, -1), u); returns levels + new
      per-cell latents for the AR(1) arm.
- [ ] 12. Local tests `tests/switch/test_switch_copula.py` (no PyG imports,
      plain pytest): inverse-CDF exact at bin edges; marginals preserved
      within binomial-SE multiples; within-cell correlation > 0, cross-cell
      ~ 0; 2N draws regardless of composition; determinism; AR(1) keeps
      Var(z) = 1 and corr(z_t, z_{t-1}) = phi.
- [ ] 13. Wire into `src/aimanager/generic/graph.py`: keyword-only
      `copula_rho=0.0, copula_phi=0.0, copula_switch_every=None` on
      `__init__` (assert 0 <= rho < 1, 0 <= phi < 1, rho > 0 only for
      `y_name == "does_switch"`); all three in `to_save` (old artifacts
      load as 0.0); dispatch inside `predict_independent` only when
      `sample and copula_rho > 0`, cell key (batch, round,
      `data["agent_group"]`); AR latent held on the instance, cleared when
      `reset_rnn`, advanced only on decision rounds
      `(round_number + 1) % copula_switch_every == 0`. The copula draw
      happens on EVERY predictor call (fixed 2N draws — the environment
      calls `_run_switch_predictor` every round; only decision-round
      outputs are consumed, only decision rounds advance the AR state).
      Absent/zero field leaves the legacy `th.multinomial` path untouched.
- [ ] 14. Raven test `src/aimanager/tests/test_switch_copula_graph.py`:
      no-rho artifact bit-identical to the pre-change path (values and
      torch RNG consumption, legacy decode reimplemented inside the test);
      copula model preserves marginals, correlates within group not across;
      save/load round-trips the fields; old artifact loads with rho 0.0.
      Run `scripts/remote_test.sh` + local `pytest` (frozen-surface proof).
- [ ] 15. `scripts/artificial_humans/make_switch_copula_artifact.py` (runs
      on Raven — the .pt unpickles torch_geometric modules): copy the base
      dict, assert no pre-existing key modified, insert `copula_*` fields
      from the params JSON, save to
      `artifacts/artificial_humans/switch_pred_herding_copula/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`;
      verify state_dict tensors bit-identical on reload. Fetch and commit.
- [ ] 16. Stage-1 config
      `configs/simulation/manager_testing/23_2g8a_herding_copula_self_gnn_contr_herdcopula_switch.yml`:
      copy of `23_2g8a_self_gnn_contr_gnn_switch.yml`, only `switch_model`
      swapped + `output_dir`/`figure_name` slugged (slug before `_self_`);
      all four pairings kept in reference order (lin_multinomial_self stays
      RNG-identical to the reference; the dir doubles as the Stage-2
      gnn-contribution cell); protocol untouched.
- [ ] 17. `squeue` check, `scripts/simulate_cluster.sh <config>`, confirm
      `per_round.parquet`, fetch, `python -m aimanager evaluate <config>`.
- [ ] 18. Keep gate on the `lin_multinomial_self` row: SC < 3.270439 with
      band upgrade requiring SC < 2; rows <= 1 >= 11; mean <= 1.759557;
      SA 0.721349 / SB 0.753990 / RSA 0.908837 essentially unmoved (a
      marginal move signals a sampler bug). Log unrounded.
- [ ] 19. Arm B only if phi survived step 8: second artifact + config, slug
      `herding_copula_ar1` (switch token `herdcopar1`), identical protocol;
      Stage-1 evaluate; select the better SC under the same guards (§5
      variant selection). Log both rows.
- [ ] 20. Selected arm fails the gate or improves within band only: Notes +
      `[FAIL]` PR, stop (no sweep).
- [ ] 21. Stage 2 (only after a Stage-1 band upgrade): 3 more configs
      `23_2g8a_herding_copula_self_{cat,gaussian,ridge}_contr_herdcopula_switch.yml`
      (copies of the reference configs, switch model + output dir only),
      simulate, fetch, evaluate — with the Stage-1 dir: 4 dirs x 4
      punishers = the 16 contexts mirroring the 16 gnn-switch contexts.
- [ ] 22. Add `herdcopula` to `SWITCH_ORDER` in
      `scripts/data_analysis/evaluation_sweep.py` (analysis layer;
      precedent D5 — no SWITCH_COLORS exists, `fig_switch_unrolled`
      tolerates a third column), sweep the 8 existing dirs + 4 new into
      `23_stack_sweep_herding_copula`.
- [ ] 23. Confirm the slot claim: SC beats the gnn switch in (nearly) all
      16 contexts; check the SC concordance panel and switch slot report;
      verify SA/SB/RSA guards stack-wide; log with unrounded scores.
- [ ] 24. Complete the log; PR `[SUCCESS]`/`[FAIL]`, body Hypothesis /
      Results / Collateral; commits map to steps.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## 4. Notes

1. Deficit profile fetched from
   `plots/data_analysis/evaluation/23_stack_sweep_updated/score_matrix.csv`:
   SC is the gnn switch's only slot-attributable row >= 2 and is concordant
   (>= 2 in 15/16 gnn-switch contexts). CG/RCA/PC/PA are high in every
   switch context and belong to the other slots.
2. Prior switch experiment PR #145 `[FAIL]` (same_group edge feature):
   SC 3.2704 -> 2.8003 Stage 1, slot mean 2.651 -> 2.307 in Stage 2 —
   real but within-band. Its key lesson: contribution-based input features
   made SC *worse* (3.35-3.37) despite better CV loss; do not retry feature
   routes toward SC.
3. Precedent PR #146 `[SUCCESS]` (punisher severity copula): the identical
   mechanism (shared per-round group latent, pairwise-likelihood MLE rho,
   marginals preserved, artifact-gated dispatch) took PD 2.934892 ->
   1.532497 and won 8/8 contexts; its collateral moved SC -0.454 in the
   reference stack — direct evidence the shared-latent mechanism reaches SC.
4. The base switch model has an RNN (`mlp+rnn+edge`), and `predict_autoreg`
   asserts RNN and autoregression cannot combine — the PR #140 comment's
   Option 2 (autoregressive cross-agent sampling) is structurally
   unavailable for this artifact; Option-1-style correlated sampling at the
   sampler is the remaining route, and the copula is its parameter-free-at-
   simulation-time form.
5. Feasibility (planner, read-only, 40-game single-copy train split,
   `switch_valid` rows): 1515 eligible decisions, switch rate 0.29372937
   (per decision round 3/7/11/15/19: 0.4290/0.3026/0.2601/0.2292/0.2434);
   374 (episode, decision-round, group) cells, 330 with >= 2 eligible
   (sizes 1:44, 2:46, 3:51, 4:95, 5:47, 6:41, 7:35, 8:15); 3009 within-cell
   pairs. Co-switching: n11 = 341 vs 215.6 expected under independence,
   odds ratio 2.748, pairwise phi 0.2125, unconditional tetrachoric
   rho = 0.3554. Arm-B proxy: lag-1 autocorrelation of the group
   switch-rate residual 0.1043 (n = 294) — weak; expect conditional latent
   rho ~ 0.15-0.25 after the GNN marginals absorb the feature-driven part,
   and phi near the noise floor (step 8 drop rule may fire).
6. Planner discrepancies, orchestrator rulings: (D1) no GNN holdout exists —
   `fraction_training: 1.0` trains the saved artifact on all 50 games, the
   CV folds are diagnostics only; ruled: estimate rho/phi on the 40-game
   baseline train split, the 10 baseline-holdout games as out-of-sample
   check only (parity with precedent D3), with the caveat that the marginal
   model has seen all 50 games recorded here. (D2) the .pt unpickles
   torch_geometric modules — both the probability dump and the artifact
   copy must run on Raven; only the MLE and pre-flight run locally; the new
   artifact is created on Raven and fetched (reverse of precedent D4).
   (D3) the Stage-2 family for a switch-slot candidate is 4 configs (switch
   varies per directory; punishers are pairings within a directory) = 16
   contexts; the 4 lin-switch dirs are the rival option, not re-simulated.
   (D4) switching is not rare (rate 0.294) — power, not rarity, is the
   constraint; pairwise MLE still the right estimator (PIT attenuation is
   worst for binary outcomes). (D5) `evaluation_sweep.py` needs only
   `SWITCH_ORDER` extended; single-token label `herdcopula` keeps
   DIR_PATTERN unambiguous. (D6) the artifact carries
   `copula_switch_every: 4` so the AR(1) latent advances only on decision
   rounds — per-call advancement would silently decay realized persistence
   to phi^4. (D7) cell key confirmed: the environment's
   `state["agent_group"]` at decision round s is the pre-switch membership,
   identical to `parse_agent_rounds`' `group_id` at row s — simulation and
   calibration cells are the same object.
