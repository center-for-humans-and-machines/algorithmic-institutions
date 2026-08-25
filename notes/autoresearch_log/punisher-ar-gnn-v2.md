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

(to be filled by the validated step list)

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## 4. Notes
