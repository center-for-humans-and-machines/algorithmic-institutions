# Autoresearch log: switch-contrib-features

## 1. Declaration

- **Slot:** switch
- **Base model:** GNN switch predictor
  (`artifacts/artificial_humans/switch_pred_opt_50ep_doubled_reanchored`,
  trained by `configs/training/artificial_humans/switch_predictor/opt_50ep_doubled_reanchored.yml`;
  node features: `common_good`, `punishment`, `agent_group`, `round_number`)
- **Target row:** SC (size of the larger group, EMD). Reference stack
  (gnn contribution x gnn switch x lin_multinomial punisher): **SC = 3.27**.
  Slot-concordant: SC >= 2 in 15/16 gnn-switch contexts, slot mean 2.65.
- **Stack guardrails (baseline):** rows <= 1 = **11/21** (must not fall),
  mean score = **1.76** (must not rise).
- **Hypothesis:** The switch model cannot tell which group is the worse
  one — its inputs carry no contribution information and no cross-group
  comparison, so simulated switches at the first decision round are weakly
  directional and per-agent independent draws produce net flows of ~1.5
  players where humans produce 2.42 (the founding exodus). Giving the model
  contribution information — features the real players observably had at the
  decision (own contribution and both groups' contribution levels, the same
  observables the linear switch baselines use per
  `notes/baseline_feature_defs.md`) — should make first-opportunity leave
  probabilities sharp and directional out of the low-cooperation group,
  producing the correlated co-departures that raise larger-group sizes and
  move SC down.
- **Planned change:** add contribution-based input features to the GNN
  switch predictor (one experiment: feature-set variants selected by
  Stage-1 score, per the legal hyperparameter-search clause). No changes to
  architecture class, training protocol, or any other slot.

## 2. Plan

Four feature variants, all config-only (every named feature already exists in
the training tensor dict and the simulation state; no shared-code change):

| variant | added node features | added edge feature | rationale |
|---|---|---|---|
| A `_a_same_group` | — | `same_group` | control arm: group-aware routing of existing features; per-node mean of the bit also encodes own-group size |
| B `_b_contrib` | `contribution` | — | free-riders leave, high contributors stay: self-level cooperation signal |
| C `_c_contrib_same_group` | `contribution` | `same_group` | central arm: group-aware routing of contributions lets the agent tell which group is worse -> directional founding exodus |
| D `_d_contrib_owngrp_same_group` | `contribution`, `own_grp_prev_mean_contr` | `same_group` | as C, but own-group level given exactly so the other group's level is linearly recoverable (insurance against the single Lin+Tanh edge MLP) |

Steps:

- [x] 1. Write the four variant training configs in
  `configs/training/artificial_humans/switch_predictor/switch_contrib_features_<variant>.yml`
  — byte-copies of `opt_50ep_doubled_reanchored.yml` except: description,
  `labels.features`, `output_dir`, the feature additions per the table, and
  new node features appended to `shuffle_features`. Architecture, optimiser,
  epochs, seed, data file unchanged.
- [ ] 2. Train all four on Raven: `scripts/train_cluster.sh ah <config>`
  (needs `ssh raven` ControlMaster). Report final CV test log-loss per arm.
- [ ] 3. Fetch artifacts (`scripts/fetch_cluster.sh artifacts/artificial_humans/switch_contrib_features_<variant>`)
  and verify actual `.pt` filenames on disk.
- [ ] 4. Write four Stage-1 sim configs
  `configs/simulation/manager_testing/switch_contrib_features_<variant>_2g8a_self_gnn_contr_gnnscf_switch.yml`
  — copies of `23_2g8a_self_gnn_contr_gnn_switch.yml` changing only
  `switch_model`, `output_dir`, `figure_name`.
- [ ] 5. Simulate on Raven (`scripts/simulate_cluster.sh <config>`), fetch
  each sim dir, confirm `per_round.parquet`.
- [ ] 6. Evaluate all four locally (`python -m aimanager evaluate <config>`);
  judge on the `lin_multinomial_self` run's 21 rows: SC, rows <= 1, mean,
  plus SA/SB/RSA (must not degrade from 0.72/0.75/0.91).
- [ ] 7. Gate: winner = lowest SC with rows <= 1 >= 11/21 and mean <= 1.7596;
  ties go to the simpler arm. No arm below SC 3.2704 with guardrails held ->
  [FAIL], skip to step 11.
- [ ] 8. Stage 2: three more sim configs for the winner (cat / gaussian /
  ridge contribution), switch_model swapped, reusing the Stage-1 gnn-contr run.
- [ ] 9. Simulate, fetch, evaluate the three.
- [ ] 10. Confirmation sweep: `evaluation_sweep.py switch_contrib_features_sweep`
  over 12 dirs (8 existing 23-family + 4 candidate) giving a 3-option switch
  axis; confirm only if the candidate beats `gnn` on SC in a clear majority
  of the 16 contexts (slot mean < 2.65) without losing SA/SB/RSA.
- [ ] 11. Fill Results/Notes, batched pre-commit run, open the PR
  (`[SUCCESS]`/`[FAIL]`; body: Hypothesis, Results, Collateral).

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## 4. Notes

1. Deficit profile fetched from `23_stack_sweep_updated/score_matrix.csv`:
   SC is the only switch-family row that is both failing and concordant
   (SA/SB/RSA sit at or near ceiling in multinomial/gnn punisher contexts;
   SA elevation appears only under ridge/gaussian punishers, i.e. it is the
   punisher's noise, not the switch model's).
2. Prior work checked: no merged autoresearch logs; the only [FAIL] PR
   (#144) is in the contribution slot. PR #140 already unrolled the switch
   slot and picked the GNN; PR #143's known constraint places the SC deficit
   in the founding exodus (first decision round net flow, human 2.42 vs sim
   ~1.5), with rates (SA/SB) and post-exodus stickiness already matching.
