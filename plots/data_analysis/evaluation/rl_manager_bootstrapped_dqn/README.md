# Bootstrapped-DQN arm — evidence directory

Outputs of `scripts/rl_bootstrapped/guard.py`, plus the reference columns this arm's policy-shape table is read against.

## Reference columns (not measured here)

`reference_policy_shape.csv` and `reference_policy_shape_n.csv` are copied verbatim from `plots/data_analysis/evaluation/rl_manager_two_worlds/policy_shape{,_n}.csv` on `auto/rl-manager-two-worlds`, where they were produced by `scripts/rl_two_worlds/measure.py` from the cross-evaluation simulation `24_rl_new_clones_cross_eval`. They are copied rather than recomputed so this arm's table carries its `human managers` and `lin_punisher` columns without that branch being checked out, and so every arm reads the same reference numbers.

Columns:

- `human managers` — the human reference data, `experiments/2group_8agent_50ep.csv`.
- `lin_punisher` — this project's clone of a human manager, the multinomial punisher that is also the opponent every RL manager trains against.
- `rl_s42`, `rl_s43`, `rl_s44` — the three finished epsilon-greedy seeds. Two of the three are monotone in the *wrong* direction.
- `never`, `prop10`, `thr9_p10` — rule-based managers from the same sweep.

The binning is the evaluation suite's own RPA bins (`RPA_EDGES` / `RPA_LABELS` in `src/aimanager/evaluation_suite/metrics.py`): `{0}`, `1-5`, `6-10`, `11-15`, `16-19`, `{20}`, cut on the contributor's own contribution in that round. `src/aimanager/manager/head_probe.py` imports those same constants, and `test_rpa_bins_match_pandas_cut` asserts the tensor path and `pd.cut` agree cell for cell.

## Measured here

Per job, under `<job_id>/`:

- `gap.csv` — behaviour vs evaluated mean punishment per evaluation point, and their ratio. The quantity this arm is meant to shrink.
- `policy_shape.csv` — mean punishment per contribution bin under the evaluated (consensus) policy, count-weighted over the last evaluation points.
- `head_shape.csv` — per-head slope and the ensemble-diversity diagnostics per evaluation point.

And across jobs: `gap_summary.csv`, `policy_shape_all.csv`.
