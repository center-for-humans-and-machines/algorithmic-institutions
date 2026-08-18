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

Validated by the orchestrator against §2 (targets CG/RCD/RCA), §5 (single
`y_encoding` parameter, behavioral rationale in the declaration), §8 (no
frozen-surface file touched; sim protocol copied unchanged).

- [x] 1. Add `y_encoding` model_arg to `GraphNetwork`
      (`src/aimanager/generic/graph.py`): kwarg default `"onehot"`, assert in
      `{"onehot", "numeric"}`, pass to the y `IntEncoder`, add to `save()`.
- [x] 2. Numeric prediction path in `predict_encoded`: skip softmax, clamp
      scalar to [0,1], decode + squeeze to int64 (N, R) in 0-20, `sample`
      no-op, degenerate one-hot `y_pred_proba`; assert onehot in
      `predict_autoreg`.
- [x] 3. MSE training branch in `src/aimanager/artificial_humans/train.py`
      keyed on the model's `y_encoding`; no entropy term for numeric; onehot
      path byte-identical.
- [ ] 4. Unit test `src/aimanager/tests/test_numeric_head.py` (Raven);
      run `scripts/remote_test.sh` + local eval-suite pytest.
- [ ] 5. Training config
      `configs/training/artificial_humans/contribution/group_switching_contribution_50ep_cont_target.yml`
      (copy + `y_encoding: numeric`, new output_dir, new architecture label).
- [ ] 6. Stage-1 sim config
      `configs/simulation/manager_testing/23_cont_target_2g8a_self_gnn_contr_gnn_switch.yml`
      (contr token stays `gnn`; slug in prefix — sweep parser hard-codes
      contr tokens).
- [ ] 7. Train on Raven (`train_cluster.sh ah`), fetch artifact, sanity-check
      readout width 1.
- [ ] 8. Stage-1 simulate + fetch + `python -m aimanager evaluate`.
- [ ] 9. Log Stage-1 row; gate on kept-per-§2 + band upgrade.
- [ ] 10. Stage 2 only on band upgrade: `..._gnn_contr_lin_switch` variant,
      sweep `23_stack_sweep_cont_target` over 2 new + 6 unchanged dirs;
      PR either way (`[SUCCESS]` / `[FAIL]`).

## 3. Results

| date | change (one line) | stage | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|---|

## 4. Notes

1. Declared before any implementation; deficit profile read from
   `plots/simulation/23_2g8a_self_gnn_contr_gnn_switch/evaluation/scores.csv`
   (run `lin_multinomial_self`) and the 8-stack GNN-contributor average in
   `plots/data_analysis/evaluation/23_stack_sweep_updated/score_matrix.csv`
   (CG 9.65, RCA 2.85, RCD 2.62 averaged over other slots).
