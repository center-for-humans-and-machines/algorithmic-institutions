# [DRAFT] Train group-switching predictor using artificial human GNN

Issue: #43 | Dependencies: #22 (group ID data format), #23 (pseudo group matching for old data)

## Summary

| # | Section | Change | Priority |
|---|---------|--------|----------|
| 1 | Data pipeline | Add `does_switch` column derivation | High |
| 2 | Training config | New YAML config for switching predictor | High |
| 3 | Default values / data encoding | Register `does_switch` in data layer | High |
| 4 | Simulation integration | Use switching predictor in environment | Medium |
| 5 | Evaluation | Verify binary prediction quality | Medium |

## 1. Data pipeline -- derive `does_switch` target

- **What**: Compute a boolean `does_switch` column indicating whether a participant changed groups between the current round and the next round. The last round of each episode should have `does_switch = False` (no opportunity to switch).
- **Where**: `src/aimanager/generic/data.py`, inside or after `parse_agent_rounds`
- **Why**: The existing pipeline already parses group information (via `global_group_id` / group column from #22). The switching target is a simple shift-and-compare derived from the group column, so it belongs in the same data preparation stage.
- **Note**: For old data processed via #23 (pseudo group matching with no switching), `does_switch` will be `False` for all rows, which is correct and allows joint training.

## 2. Training config for switching predictor

- **What**: Create a new YAML config file analogous to `script_22.yml` but targeting `does_switch` as the prediction variable.
- **Where**: `configs/training/artificial_humans/` (new file, e.g. `switch_predictor.yml`)
- **Why**: The existing pipeline is config-driven; a new config is the standard way to define a new prediction task. Key parameters: `y_name: does_switch`, `y_levels: 2`, appropriate `x_encoding` (should include group ID, previous contribution, previous punishment, previous common good), `autoregression: false`.
- **Open question**: Which input features should be included beyond the ones used for contribution prediction? Group ID will be a new node feature from #22. Whether to include `prev_contribution_valid` or other features needs experimentation.

## 3. Default values and data encoding

- **What**: Register `does_switch` in `get_default_values` and `create_torch_data_new` so the column is converted to a tensor and has a default value (default: `False`). Also generate the `prev_does_switch` shifted version.
- **Where**: `src/aimanager/generic/data.py` -- `data_names` dict in `create_torch_data_new`, `get_default_values`
- **Why**: The training loop in `train.py` expects all target and feature columns to be present in the data dict with proper defaults. Without this registration, the `does_switch` column would be silently dropped during tensor conversion.
- **Note**: The `does_switch` column is boolean (dtype `th.bool`), consistent with how `contribution_valid` is handled. The `IntEncoder` with `onehot` encoding already supports bool-to-int64 casting (see `encoder.py` line 37).

## 4. Simulation integration

- **What**: Load the trained switching predictor model alongside the contribution and validity models in `ArtificialHumanEnv`. After each round, use it to predict which agents switch groups, then call `update_groups` with the new assignments.
- **Where**: `src/aimanager/manager/environment.py` (constructor and `step` method), `src/aimanager/simulation/simulate.py` (config loading and model path), `src/aimanager/manager/artificial_human_group.py`
- **Why**: The simulation currently assumes static group assignments. To study two-manager competition, agents must be able to switch groups dynamically based on the trained predictor, replacing the stub logic in `ArtificialHumanGroup.do_group_selection`.
- **Open question**: When a switch is predicted, which group does the agent move to? In the two-group case this is unambiguous (the other group), but the mechanism should be confirmed against the design in #17.

## 5. Evaluation

- **What**: Verify that the standard evaluation metrics (accuracy, log loss, confusion matrix) are meaningful for the binary `does_switch` target. Consider adding class-balance reporting since switching events may be rare.
- **Where**: `src/aimanager/artificial_humans/evaluation.py`, evaluation notebooks in `notebooks/artificial_humans/`
- **Why**: Binary targets with class imbalance can produce misleadingly high accuracy. The evaluation should surface precision/recall or balanced accuracy to ensure the model is learning genuine switching behavior rather than always predicting the majority class.

## Open Questions

1. **Feature selection for the switching predictor**: Beyond group ID, previous punishment, and previous contribution, should additional features (e.g., cumulative payoff, round number) be included? This likely requires experimentation.
2. **Switching destination**: In the two-group setting, a predicted switch implies moving to the other group. Should the architecture generalize to more than two groups, or is two-group sufficient for now?
3. **Class imbalance handling**: If switching is rare, should the training use weighted cross-entropy loss or oversampling? Or is the standard pipeline sufficient as a first pass?
4. **Temporal scope of `does_switch`**: Should the target reflect switching after the punishment phase of the current round (i.e., the group assignment changes before the next round's contribution)? This depends on the experimental protocol defined in #17.

## Next Actions

- [ ] Resolve open questions with domain experts (especially items 2 and 4)
- [ ] Implement section 1 (data pipeline) -- depends on #22 being merged
- [ ] Implement section 3 (default values / encoding) -- can be done alongside section 1
- [ ] Create training config (section 2) -- can be done independently
- [ ] Train and evaluate initial model (section 5) -- depends on new dataset availability
- [ ] Implement simulation integration (section 4) -- depends on trained model and #17 design
