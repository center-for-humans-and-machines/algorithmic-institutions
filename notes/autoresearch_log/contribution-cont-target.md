# Autoresearch log: auto/contribution-cont-target

## 1. Declaration

- **Slot:** contribution
- **Base model:** GNN (`gnn`), reference artifact
  `artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`
- **Target rows (starting scores, reference stack
  `23_2g8a_self_gnn_contr_gnn_switch`, run `lin_multinomial_self`):**
  - CG 9.850 (band > 5)
  - RCD 2.772 (band 2-5)
  - RCA 2.035 (band 2-5)
- **Stack baselines:** rows <= 1: 11/21; mean score: 1.76.
- **Hypothesis:** Human contributions are ordinal quantities, not 21 unrelated
  categories. The categorical head spreads probability mass across distant
  levels and the independent per-agent multinomial sampling injects
  idiosyncratic round-to-round noise; replacing the 21-logit readout with a
  single continuous output (MSE-trained, rounded to the 0-20 grid at
  prediction time) enforces ordinal structure and removes the sampling noise,
  which should tighten group-mean trajectories (CG) and round-to-round
  contribution-change behavior (RCA, RCD).
- **Planned change:** one new `model_args` parameter, `y_encoding: numeric`
  (default `onehot` preserves every existing artifact and config). For the
  numeric setting the GNN readout is a single scalar, trained with MSE against
  the numeric-encoded (value / 20) target; prediction clamps to [0, 1] and
  decodes by rounding to the 21-level grid. Same architecture, features,
  training data, and 575-epoch budget as the reference contribution config.
  New training config + new sim config with the retrained artifact swapped
  into the reference stack (Stage 1), sweep on band upgrade (Stage 2).

## 2. Plan

(to be filled after validation)

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## 4. Notes

1. Declared before any implementation; deficit profile read from
   `plots/simulation/23_2g8a_self_gnn_contr_gnn_switch/evaluation/scores.csv`
   (run `lin_multinomial_self`) and the 8-stack GNN-contributor average in
   `plots/data_analysis/evaluation/23_stack_sweep_updated/score_matrix.csv`
   (CG 9.65, RCA 2.85, RCD 2.62 averaged over other slots).
