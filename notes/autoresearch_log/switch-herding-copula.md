# Autoresearch log: switch-herding-copula

## 1. Declaration

- **Slot:** switch
- **Base model:** `gnn` switch predictor
  (`artifacts/artificial_humans/switch_pred_opt_50ep_doubled_reanchored/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt`)
- **Target rows:** SC — reference stack 2.816006 (band 2-5; 3.270439 under
  the pre-copula punisher); slot average 2.65 over the 16 gnn-switch
  contexts of the original sweep, >= 2 in 15/16 (range 1.77-3.45),
  concordant. The only S-family deficit: SA/SB at or near the ceiling in
  the reference stack — rates match, the group aggregate does not. Success
  requires SC < 2 in the reference stack (band upgrade 2-5 -> 1-2),
  Stage-2 confirmed.
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
- **Stack guards (must not regress):** rows <= 1 baseline 10/21; mean
  baseline 1.6879978841849728 (reference stack `gnn x gnn x
  lin_multinomial` where the multinomial bundle is now the severity-copula
  one, `punishment_multinomial_severity_copula.joblib` — maintainer update
  of 2026-08-12, note 8); marginal guards SA 0.773715, SB 0.848892,
  RSA 1.001009 — an S-rate move signals a sampler bug.

## 2. Plan

Validated by the orchestrator 2026-08-12 (targets per §2, legality per §5,
frozen surface per §8). Slug: `herding_copula`; switch label: `herdcopula`.

- [x] 1. Worktree + Claude commit identity; record the validated plan in
      this file; commit.
- [x] 2. Local calibration script `scripts/baselines/switch_copula_rho.py`
      (pandas/numpy/scipy, no PyG) with a `--raw-stats` mode reproducing the
      feasibility numbers (note 5) from
      `experiments/baseline/2group_8agent_50ep_bline_train.csv` via
      `parse_agent_rounds(df, switch_every=4)`, `switch_valid` rows only.
      Gate: escalate and stop if within-cell pairs < 1500 or unconditional
      tetrachoric rho <= 0.10.
- [x] 3. Raven dump script `scripts/artificial_humans/dump_switch_probs.py`:
      load the base switch artifact, `create_torch_data` on the human data,
      full-episode forward (mask `switch_valid`, GRU over rounds = the
      simulation's warm-RNN semantics), write
      `artifacts/artificial_humans/switch_pred_herding_copula/calibration/switch_probs_train.parquet`
      (episode_id, global_group_id, round_number, player_idx, agent_group,
      does_switch, switch_valid, p_switch) + `_test.parquet` for the 10
      holdout games. Assert mean predicted p over eligible rows ~ 0.2937 and
      the per-round profile tracks step 2.
- [x] 4. `squeue -u certuer` PENDING check (precedent note 11), sync, run the
      dump on the Raven login node, `scripts/fetch_cluster.sh` both parquets,
      commit.
- [x] 5. Binary pairwise-likelihood MLE in `switch_copula_rho.py`:
      t_i = Phi^-1(1 - p_i), bivariate-normal rectangle probabilities,
      reusing the Drezner-Wesolowsky `bvn_cdf` + grid/Brent structure of
      `punishment_copula_rho.py` adapted to 2 levels.
- [x] 6. Acceptance gate: round-trip recovery at rho_true in {0.05..0.5}
      (max |bias| <= 0.03) and `pair_nll(0)` equal to the closed-form
      independence value to 1e-10. Abort and escalate on failure.
- [x] 7. rho_hat on the 40-game train split; cluster bootstrap over episodes
      (200 resamples) for SE / 95% CI; 10-game holdout printed as
      out-of-sample check only; per-round and per-cell-size splits as
      diagnostics (constant rho kept — ties to the simpler model).
- [x] 8. Arm B: rho_lag1 by the same pairwise MLE over cross-round
      same-(episode, group) pairs at lag 1 (model implies corr = rho * phi);
      phi_hat = rho_lag1 / rho_hat with bootstrap CI. If the CI includes 0
      or phi_hat <= 0: arm B dropped with a Notes entry, never hand-tuned.
- [x] 9. Pre-flight go/no-go (`--preflight`): redraw switch indicators from
      the dumped marginals per human decision cell, independent vs copula at
      rho_hat; compare larger-group-size distribution and P(fully segregated)
      at s+1 against the realized human one, next to the PR #140 reference
      numbers (5.28 null / 5.46 sim / 6.09 human; 0.018 / 0.030 / 0.144).
      Escalate if the copula barely moves; rho is never tuned to this.
- [x] 10. Run steps 5-9; write
      `.../switch_pred_herding_copula/calibration/copula_params.json`
      (rho, se, ci, phi, estimator, base-artifact sha256, data file,
      n_pairs, n_cells); record every number unrounded in §4; commit.
- [x] 11. PyG-free sampler `src/aimanager/generic/copula.py` (torch only):
      `sample_correlated_levels(proba, cell_id, rho, z_prev=None, phi=0.0)`
      — exactly 2N draws per call (z then eps, float64,
      composition-stable), per-cell latent from the cell's first member,
      u = ndtr(sqrt(rho) z_cell + sqrt(1-rho) eps),
      level = searchsorted(cumsum(proba, -1), u); returns levels + new
      per-cell latents for the AR(1) arm.
- [x] 12. Local tests `tests/switch/test_switch_copula.py` (no PyG imports,
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
      copy of `23_2g8a_self_gnn_contr_gnn_switch.yml` with (a) `switch_model`
      swapped to the copula switch artifact, (b) the `lin_multinomial`
      manager replaced by `lin_multinomial_copula` ->
      `artifacts/baselines/punishment_multinomial_severity_copula.joblib`
      (the updated reference punisher, note 8; pairing name kept as
      `lin_multinomial_copula_self` to match the severity-copula rival
      dirs in the sweep), (c) `output_dir`/`figure_name` slugged (slug
      before `_self_`); four pairings in reference order (copula
      multinomial first — same RNG position as the severity-copula
      baseline run; ridge/gaussian/gnn at positions 2-4 as in the old
      reference config); protocol untouched.
- [ ] 17. `squeue` check, `scripts/simulate_cluster.sh <config>`, confirm
      `per_round.parquet`, fetch, `python -m aimanager evaluate <config>`.
- [ ] 18. Keep gate on the `lin_multinomial_copula_self` row vs the updated
      reference baseline
      (`23_2g8a_severity_copula_self_gnn_contr_gnn_switch`): SC < 2.816006
      with band upgrade requiring SC < 2; rows <= 1 >= 10; mean <=
      1.6879978841849728; SA 0.773715 / SB 0.848892 / RSA 1.001009
      essentially unmoved (a marginal move signals a sampler bug). Log
      unrounded.
- [ ] 19. Arm B only if phi survived step 8: second artifact + config, slug
      `herding_copula_ar1` (switch token `herdcopar1`), identical protocol;
      Stage-1 evaluate; select the better SC under the same guards (§5
      variant selection). Log both rows.
- [ ] 20. Selected arm fails the gate or improves within band only: Notes +
      `[FAIL]` PR, stop (no sweep).
- [ ] 21. Stage 2 (only after a Stage-1 band upgrade): 3 more configs
      `23_2g8a_herding_copula_self_{cat,gaussian,ridge}_contr_herdcopula_switch.yml`
      (same edits as step 16: copula switch artifact, copula multinomial
      punisher, slugged output dir), simulate, fetch, evaluate — with the
      Stage-1 dir: 4 dirs x 4 punishers = 16 contexts. Rival gnn-switch
      values: the 4 severity-copula gnn-switch dirs for the
      multinomial_copula contexts, the 4 original gnn-switch dirs for
      ridge/gaussian/gnn.
- [ ] 22. Add `herdcopula` to `SWITCH_ORDER` in
      `scripts/data_analysis/evaluation_sweep.py` (analysis layer;
      precedent D5 — no SWITCH_COLORS exists, `fig_switch_unrolled`
      tolerates a third column), sweep the 8 original dirs + the 8
      severity-copula dirs + the 4 new dirs into
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
7. Step 2 ran: every feasibility number reproduces exactly; gate PASS
   (pairs 3009 >= 1500, tetrachoric rho 0.35537564291376417 > 0.1,
   likelihood-ratio 2*delta-nll = 128.86 against independence). Convention
   pinned: within-cell pairs are unordered, so the 2x2 pair table is
   symmetrised (n10 = n01 = 464.5); expected n11 under independence uses
   the pooled pair-slot rate 0.2677 (not the row rate 0.2937 — big cells
   carry most pairs and switch less; both printed side by side). The
   tetrachoric is a true 2x2 bivariate-normal MLE at marginal-MLE
   thresholds, built on the precedent's Drezner-Wesolowsky `bvn_cdf`
   (max abs err vs scipy 2.2e-16), reused verbatim for step 5.
8. Maintainer update (2026-08-12, mid-experiment, before any Stage-1
   config existed): the reference stack's punisher is now the
   severity-copula multinomial bundle
   (`artifacts/baselines/punishment_multinomial_severity_copula.joblib`) —
   same `lin_multinomial` slot option, not a separate class; autoresearch.md
   text to be updated by the maintainer. New reference baselines from
   `plots/simulation/23_2g8a_severity_copula_self_gnn_contr_gnn_switch`:
   SC 2.816006 (still band 2-5, success gate SC < 2 unchanged),
   rows <= 1 = 10/21 (RSA 1.001009 sits 0.1% over the ceiling — the
   Stage-2-adjudicated noise crossing of PR #146), mean
   1.6879978841849728, guards SA 0.773715 / SB 0.848892 / RSA 1.001009.
   Declaration and plan steps 16/18/21/22 updated accordingly. Pairing
   label stays `lin_multinomial_copula` in the new configs so Stage-2
   contexts match the existing severity-copula rival dirs; the naming
   unification is cosmetic and the maintainer's to make. Calibration
   steps 2-10 are unaffected (they involve only the switch GNN and human
   data).
9. Step 3 written and desk-checked: all rounds dumped (7680 train / 1920
   test rows; `switch_valid` marks the 1515 / 376 usable), game key is
   `(global_group_id, episode_id)` (`episode_id` alone is NOT unique
   across sessions — step 5 must group on both), per-game forwards (block
   diagonal anyway; removes cross-game-leakage doubt), `default_values`
   recomputed from the doubled training file (warn-not-assert on mismatch
   with the artifact's stored values; moot here — every agent-round cell
   is populated so padding never fires). Verified locally with PyG stubbed
   and a fake model: zero index round-trip mismatches, every note-5 number
   reproduced. Residual risk: the one-hot class index (p[..., 1] =
   does_switch True) is inferred, not executed — the script's own
   mean-p assert fires at ~0.706 instead of ~0.294 if it is wrong.
   Artifact pickle inspected: `same_group` absent, `edge_encoding == []` —
   the "edge" in the artifact name is `add_edge_model`, not an edge input
   feature (so PR #145's SC-relevant feature is NOT in the production
   artifact).
10. Step 4 ran (Raven login node, 16.4s; queue empty before sync — one
    rsync hiccup: the full `remote_test.sh --sync-only` did not deliver
    the new script, pushed explicitly and md5-verified). Sanity checks
    PASS: recomputed default_values identical to the artifact's stored
    ones; train mean p_switch 0.29519684 vs observed 0.29372937; per-round
    mean p 0.4401/0.2914/0.2466/0.2511/0.2422 tracks the observed
    0.4290/0.3026/0.2601/0.2292/0.2434; the p[..., 1] class-index
    assumption was correct. Log loss 0.51241862 train / 0.44504137 test
    (10 holdout games, eligible 376, rate 0.30851064). Parquets fetched
    and committed (LFS).
11. Steps 5-6 gates PASS: pair_nll(0) equals the closed-form independence
    value exactly (diff 0.0, real — verified element-wise, accumulated
    error below representable spacing at 2973); round-trip recovery max
    |bias| 0.006690795337206679 (tolerance 0.03) — the binary MLE, unlike
    the randomized-PIT moment estimator, is unattenuated by construction.
12. Step 7: **rho_hat = 0.116482333585783**, cluster-bootstrap SE
    0.049666098931839604 (200 resamples), 95% CI [0.035037571771319845,
    0.21502476987895425]; pairwise LR 2*delta-nll = 10.04; out-of-sample
    holdout (10 games, 741 pairs) rho = 0.1108348204265086 — agrees.
    Conditional-on-GNN rho is ~33% of the unconditional 0.3554: the GNN
    marginals absorb ~2/3 of within-group co-movement, matching the PR
    #140 comment's ~40%-survives estimate (linear-fit lower bound).
    Diagnostic splits (never selection criteria): decision round 3 rho
    0.3507 vs 0.021-0.115 at rounds 7-19 — the founding-exodus round
    carries the strongest residual herding, exactly where the SC deficit
    lives (candidate follow-up: round-dependent rho; constant rho kept
    per §5); cell size 2-3 rho 0.455 vs 4+ 0.075.
13. Step 8: rho_lag1 = 0.08196398283355816 over 5406 consecutive
    decision-epoch pairs (same-group blocks; 810 same-player pairs
    included — eps redraws each round so every cross-round pair carries
    rho*phi); **phi_hat = 0.70366020589033**, bootstrap 95% CI
    [0.11702191363043744, 2.0829306438837505] excludes 0 -> **ARM B
    KEPT**. phi SE unusable (ratio statistic, heavy tails) — the CI is
    the drop-rule quantity. Signed-grid refit confirms the lag-1 optimum
    is interior (0.081964), not boundary-truncated. copula_params.json
    written with provenance (base artifact sha256 184f7f5c..., git head
    at estimator commit). Ordering note: the JSON was written before the
    step-9 preflight (it contains only rho/phi parameters, which the
    preflight never feeds — the preflight is go/no-go only).
14. Step 9 preflight (500 one-step redraws of the human decision cells
    from the GNN marginals, rho read from the params JSON): **GO**. The
    copula closes 79.7% of the per-cell switch-count sd gap (human 1.2657,
    independent 1.1838, copula 1.2490) — the direct within-round herding
    measure. Larger-group mean at s+1: 6.0200 / 5.6955 / 5.7691 (22.7%
    closed); P(larger = 8): 0.1200 / 0.0684 / 0.0851. Round-3 slice
    (founding exodus): mean |size diff| 4.7500 / 3.4192 / 3.5672 — only
    ~11% closed one-step, consistent with the round-3 local rho of 0.35
    vs the constant 0.116; arm B's persistence and cumulative multi-round
    dynamics (invisible to a one-step check) are what Stage 1 adds.
    Escalate line (<10% of the sd gap) not hit. Process note: this step
    was implemented by the orchestrator directly — the subagent assigned
    to it stalled on a transient permission-classifier outage, was
    stopped, and its task re-done inline.
15. Steps 11-12: `sample_correlated_levels` (torch-only, general L) with
    conventions pinned to the calibration sampler and the punisher
    precedent: searchsorted left side (u == 1-p falls to level 0, exactly
    the strict `w > t` of the fitted model), unconditionally 2 randn calls
    (n_cells then N, float64) so RNG consumption is composition-, rho- and
    phi-invariant; AR(1) z = phi*z_prev + sqrt(1-phi^2)*z_new keeps
    Var(z) = 1 so marginals are phi-invariant; cumsum promoted UP to the
    latent's float64 (a float32 GNN softmax cannot truncate the latent);
    ndtr via torch.erf (no scipy, imports on macOS). 23 local tests pass,
    including exact array_equal agreement with a numpy re-derivation of
    the calibration sampler at the calibrated rho/phi, fed the same draws
    via the documented RNG contract. Bit-identity of the LEGACY path is
    deliberately not asserted here — it belongs to graph.py's dispatch
    gate (steps 13-14).
