# Bootstrapped-DQN arm — evidence directory

Outputs of `scripts/rl_bootstrapped/guard.py`, plus the reference columns this arm's policy-shape table is read against.

## Reference columns (not measured here)

`reference_policy_shape.csv` and `reference_policy_shape_n.csv` are copied verbatim from `plots/data_analysis/evaluation/rl_manager_two_worlds/policy_shape{,_n}.csv` on `auto/rl-manager-two-worlds`, where they were produced by `scripts/rl_two_worlds/measure.py` from the cross-evaluation simulation `24_rl_new_clones_cross_eval`. They are copied rather than recomputed so this arm's table carries its `human managers` and `lin_punisher` columns without that branch being checked out, and so every arm reads the same reference numbers.

Columns:

- `human managers` — the human reference data, `experiments/2group_8agent_50ep.csv`.
- `lin_punisher` — this project's clone of a human manager, the multinomial punisher that is also the opponent every RL manager trains against.
- `rl_s42`, `rl_s43`, `rl_s44` — the three finished epsilon-greedy seeds. Two of the three are monotone in the *wrong* direction.
- `never`, `prop10`, `thr9_p10` — rule-based managers from the same sweep.

**Trust these named-rule columns only because they were checked.** At `0ff44a9`, where this branch starts, `RuleBasedManager.__init__` is `(self, k=1, n_punishments=31, **_)` — one fixed formula. The named rules arrived later on a different branch, so on this branch a simulation config line reading `rule: never` is swallowed by `**_` and silently ignored: the run completes clean and the output carries the label on the default formula's behaviour. `guard.py::validate_reference` hard-fails on that signature — `never` must punish exactly 0.000 with maximum 0, and no two named rules may agree to two decimal places — and it runs automatically whenever this file is loaded. These columns pass: `never` is 0.0000 in every bin, `prop10` peaks at 20.0, `thr9_p10` at 10.0. They were produced on `auto/rl-manager-two-worlds` after it merged `auto/rule-based-manager-sweep`, so the dispatcher existed there. Do not regenerate them from this branch.

The binning is the evaluation suite's own RPA bins (`RPA_EDGES` / `RPA_LABELS` in `src/aimanager/evaluation_suite/metrics.py`): `{0}`, `1-5`, `6-10`, `11-15`, `16-19`, `{20}`, cut on the contributor's own contribution in that round. `src/aimanager/manager/head_probe.py` imports those same constants, and `test_rpa_bins_match_pandas_cut` asserts the tensor path and `pd.cut` agree cell for cell.

## Measured here

Per job, under `<job_id>/`:

- `gap.csv` — behaviour vs evaluated mean punishment per evaluation point, and their ratio. The quantity this arm is meant to shrink.
- `policy_shape.csv` — mean punishment per contribution bin under the evaluated (consensus) policy, count-weighted over the last evaluation points.
- `head_shape.csv` — per-head slope and the ensemble-diversity diagnostics per evaluation point.

And across jobs: `gap_summary.csv`, `policy_shape_all.csv`.
