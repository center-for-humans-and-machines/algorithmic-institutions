# Autoresearch log: contribution-conformity-mixture

## 1. Declaration

- **Slot:** contribution
- **Base model:** `gnn`
  (`artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`)
- **Target rows** (reference stack `23_2g8a_self_gnn_contr_gnn_switch`,
  `lin_multinomial_self` run):
  - **CG 9.850261** (band > 5) — worst row of the entire score matrix,
    slot mean 9.65 over the 8 GNN-contribution contexts, concordant 8/8.
  - **RCD 2.772321** (band 2-5) — switching pull: sim switchers do not
    adapt toward their new group's level.
  - **RCA 2.034866** (band 2-5) — contribution change by round type,
    including the composition-changed and switched types.
- **Stack guards (must not regress):** rows <= 1 baseline **11/21**; mean
  baseline **1.759** (autoresearch.md §2, verified against the reference
  stack's `evaluation/scores.csv`).
- **Hypothesis:** human contribution decisions mix two processes: acting
  from one's own disposition and history, and *matching the group's current
  level* (conditional cooperation / norm conformity). New diagnostic on the
  human vs sim data (this experiment, script in Notes):
  - regressing contribution[t] on own[t-1] and group-others-mean[t-1] gives
    a human peer slope of **0.227** vs sim **0.038** — the GNN's realized
    peer response is ~6x too flat, compensated by extra self-persistence
    (own slope 0.764 vs human 0.706). Human own+peer total ≈ 0.93 (near
    unit root: group divergence sustains); sim ≈ 0.80 (mean-reverting:
    spread collapses to the independence floor).
  - the peer slope is robust to own-history controls (three own lags +
    cumulative own mean: 0.184), so it is learnable signal, not
    collinearity artifact.
  - the pull toward the current group's others-mean is ~0.25-0.27 for both
    stayers and switchers — the same mechanism RCD and RCA measure.
  - human spread ratio *grows* round over round (0.47 at round 0 to ~0.90
    from round 6 on) while the sim is flat at ~0.58-0.60: divergence is
    amplified by within-group coupling, not seeded by initial correlation —
    consistent with PR #149's finding that within-round residual
    correlation is only rho ≈ 0.07 (common-shock/copula approaches cannot
    produce this) and with the [ABANDONED] schedsamp log's decomposition
    (teacher-forced ratio 0.784: on human histories the group signal rides
    in each agent's *own* history; free-running there is no cross-agent
    force restoring coherence).
  - why training misses it: under teacher forcing, NLL lets the RNN explain
    the peer-correlated variance through own history (human group members
    have already converged), so the peer channel gets ~no weight — the
    reason feature-only additions (#116 own-group mean, #153 peer
    attention) and self-history dropout (#144/#147) did not fix behavior.
- **Planned change (one change):** restructure the GNN contribution head as
  a two-component **conformity mixture**. Component A is the existing
  categorical readout. Component B ("conform") is a discretized
  distribution over the 21 levels centered on the agent's own-group
  leave-one-out previous-round mean contribution
  (`own_grp_prev_mean_contr`, the #116 feature) with a learned width. A
  learned gate w (function of the node hidden state) mixes them:
  P(c_t) = w * Conform + (1-w) * Categorical, trained end-to-end by NLL.
  The structural channel cannot be bypassed by own-history shortcuts: any
  probability mass routed through B is anchored at the group level by
  construction, restoring the peer response at rollout — group members get
  pulled toward their common mean (CG amplification), switchers toward
  their new group (RCD), composition-change rounds react (RCA).
- **Behavioral sentence (§5):** a share of human contribution decisions are
  norm-matching — the player sets their contribution at their group's
  current level rather than re-deriving it from their own disposition —
  which is what CG (group spread), RCD (switching pull), and RCA
  (round-type reactions) measure.
- **Guardrail watch (not targets):** CA 0.772 / CD 0.650 / CF 0.814 — the
  marginal C block that killed #154/#155/#156; the mixture must not blur
  the 0/20 extremes. Component A keeps the full categorical expressiveness,
  and B's width is learned, so mass concentrates only where the data puts
  it.

## 2. Plan

Validated 2026-08-19 (orchestrator). Stage 2 decided after Stage-1 results.

- [ ] 1. Preserve the diagnostic script as
  `scripts/data_analysis/cg_conformity_diagnostic.py` (flake8-clean; rerun
  reproduces human peer slope ~0.227 vs sim ~0.038).
- [ ] 2. Implement the conformity-mixture head in
  `src/aimanager/generic/graph.py`: flag `conform_mixture` (default off,
  off-path bit-identical), gate `Lin(hidden_size, 1)` on the post-RNN node
  state (bias init -2.0), scalar `conform_log_sigma` (init log 2), component
  B logits `-(k - m_t)^2 / (2 sigma^2)`, `self.bias` added to component A
  before mixing, forward returns normalized mixture log-probs via logsumexp
  (softmax/CrossEntropyLoss consumers unchanged); new attrs serialized in
  `save()`, old checkpoints load with the flag defaulted off.
- [ ] 3. Plumb `m_t` into the encoded dict in `encode()` when the flag is
  on (training path: `create_torch_data` already carries the column;
  sim path: `environment.update_own_grp_prev_mean_contr` already provides
  it). `own_grp_prev_mean_contr` is NOT added to `x_encoding`.
- [ ] 4. Add `src/aimanager/tests/test_conform_mixture.py`: off-flag
  bit-identity, on-flag normalization (log_softmax(out) == out), gradient
  flow to gate + log_sigma, save/load round trip incl. legacy checkpoints,
  shift-sensitivity toward round(m). Run `scripts/remote_test.sh` (Raven).
- [ ] 5. Training config
  `configs/training/artificial_humans/contribution/group_switching_contribution_50ep_conformity_mixture.yml`:
  copy of the reference config + `model_args.conform_mixture: true`, label,
  new output_dir; same seed/CV/epochs/features.
- [ ] 6. Submit training on Raven via `scripts/train_cluster.sh ah <config>`
  (check `squeue` for PENDING jobs before sync). Est. 20-60 min wall clock.
- [ ] 7. Monitor to completion; sanity-read learned sigma and gate level
  from the artifact on Raven; record final-fold test log_loss vs reference
  1.9892.
- [ ] 8. Fetch artifact via `scripts/fetch_cluster.sh`; verify LFS attrs.
- [ ] 9. Sim config
  `configs/simulation/manager_testing/23_2g8a_conformity_mixture_self_gnn_contr_gnn_switch.yml`:
  reference sim config with only the contribution artifact path, output_dir,
  figure_name changed; `save_per_round: true` kept.
- [ ] 10. Run sim on Raven, fetch `per_round.parquet` (all four pairings).
- [ ] 11. Evaluate locally; read CG/RCD/RCA vs 9.850261/2.772321/2.034866,
  rows <= 1 vs 11/21, mean vs 1.759; rerun the diagnostic on the new sim to
  measure the realized peer slope (mechanism check).
- [ ] 12. Update this log: §2 checkboxes, §3 results row, §4 notes (sigma,
  gate, peer-slope delta, CA/CD/CF guardrails); verdict per §2.

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## 4. Notes

1. Diagnostic script: regression of contribution[t] on own[t-1] and
   group-others-mean[t-1], spread-ratio-by-round trajectory, boundary
   stickiness, participant ICC, run on `experiments/2group_8agent_50ep.csv`
   vs the reference sim's `per_round.parquet` (lin_multinomial run).
   Numbers recorded in the Declaration; script preserved in this experiment
   as `scripts/data_analysis/cg_conformity_diagnostic.py`.
2. Prior art checked: #144/#147 (self-history dropout — blinding the
   self-anchor fails RCA/RCB), #149 (within-round copula — rho too small,
   preflight-gated), #151/#154/#155/#156 (head/emission changes — marginal
   C block is the recurring killer), #153 (peer attention — routing exists,
   behavior unchanged), abandoned schedsamp branch (exposure bias — training
   curriculum, complementary non-overlapping mechanism).
