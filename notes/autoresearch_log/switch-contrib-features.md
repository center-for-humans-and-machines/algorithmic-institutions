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

(to be filled by the validated step list)

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
