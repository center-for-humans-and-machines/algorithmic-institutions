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

## Design Decisions

- **Two groups only**: Binary switch prediction (switch or stay). No need to generalize to more groups for now.
- **Minimal features**: Start with group ID, previous punishment, and previous contribution. Rather than adding cumulative features (e.g. cumulative payoff), rely on the RNN to capture temporal patterns from the sequence of per-round inputs. The edge model may also contribute useful inter-agent information.
- **Class imbalance**: Use the standard pipeline with unweighted cross-entropy as the first pass. Only introduce weighted loss or resampling if initial results show the model is not learning switching behavior.
- **Switching timing**: Switching happens between rounds (after a full round completes). The exact moment within the round sequence (after punishment? after common good?) still needs to be verified against the data -- see open questions.

## 1. Data pipeline -- derive `does_switch` target

- **What**: Compute a boolean `does_switch` column indicating whether a participant changed groups between the current round and the next round. The last round of each episode should have `does_switch = False` (no opportunity to switch).
- **Where**: `src/aimanager/generic/data.py`, inside or after `parse_agent_rounds`
- **Why**: The existing pipeline already parses group information (via `global_group_id` / group column from #22). The switching target is a simple shift-and-compare derived from the group column, so it belongs in the same data preparation stage.
- **Note**: For old data processed via #23 (pseudo group matching with no switching), `does_switch` will be `False` for all rows, which is correct and allows joint training.

## 2. Training config for switching predictor

- **What**: Create a new YAML config file targeting `does_switch` as the prediction variable with minimal input features.
- **Where**: `configs/training/artificial_humans/` (new file, e.g. `switch_predictor.yml`)
- **Why**: The existing pipeline is config-driven; a new config is the standard way to define a new prediction task. Key parameters: `y_name: does_switch`, `y_levels: 2`, `x_encoding` limited to group ID, previous punishment, and previous contribution. `autoregression: false`.
- **Note**: The feature set intentionally omits cumulative features. The RNN hidden state should capture temporal dynamics without explicit cumulative inputs. The edge model should also be included to leverage inter-agent information.

## 3. Default values and data encoding

- **What**: Register `does_switch` in `get_default_values` and `create_torch_data_new` so the column is converted to a tensor and has a default value (default: `False`). Also generate the `prev_does_switch` shifted version.
- **Where**: `src/aimanager/generic/data.py` -- `data_names` dict in `create_torch_data_new`, `get_default_values`
- **Why**: The training loop in `train.py` expects all target and feature columns to be present in the data dict with proper defaults. Without this registration, the `does_switch` column would be silently dropped during tensor conversion.
- **Note**: The `does_switch` column is boolean (dtype `th.bool`), consistent with how `contribution_valid` is handled. The `IntEncoder` with `onehot` encoding already supports bool-to-int64 casting.

## 4. Simulation integration

- **What**: Load the trained switching predictor model alongside the contribution and validity models in `ArtificialHumanEnv`. After each round, use it to predict which agents switch groups, then move them to the other group.
- **Where**: `src/aimanager/manager/environment.py` (constructor and `step` method), `src/aimanager/simulation/simulate.py` (config loading and model path), `src/aimanager/manager/artificial_human_group.py`
- **Why**: The simulation currently assumes static group assignments. To study two-manager competition, agents must be able to switch groups dynamically based on the trained predictor, replacing the stub logic in `ArtificialHumanGroup.do_group_selection`.
- **Note**: With two groups, a predicted switch unambiguously means moving to the other group. No destination selection logic needed.

## 5. Evaluation

- **What**: Verify that the standard evaluation metrics (accuracy, log loss, confusion matrix) are meaningful for the binary `does_switch` target. Report class balance alongside accuracy so that a degenerate "always predict stay" model is easy to spot.
- **Where**: `src/aimanager/artificial_humans/evaluation.py`, evaluation notebooks in `notebooks/artificial_humans/`
- **Why**: If switching is rare, raw accuracy may be misleading. Reporting precision/recall or balanced accuracy helps judge whether the model learned genuine switching patterns. Only pursue mitigation (weighted loss, resampling) if the first-pass model fails to predict switches meaningfully.

## Open Questions

1. **Previous vs. current round features**: Switching happens between rounds, but it is unclear whether the model should use features from the round just completed (current punishment, current contribution) or from the prior round (previous punishment, previous contribution). This depends on the exact timing of the switching decision relative to the round phases. Must be verified against the actual dataset schema once available.
2. **Exact timing within the round**: We know switching happens after a full round, but the precise moment (after punishment is applied? after common good is computed?) affects which values are "known" to the agent at decision time. Needs confirmation from the experimental protocol in #17 and the collected data.

## Next Actions

- [ ] Implement section 1 (data pipeline) -- depends on #22 being merged
- [ ] Implement section 3 (default values / encoding) -- can be done alongside section 1
- [ ] Create training config (section 2) -- can be done independently
- [ ] Once data is available, verify timing of switching relative to round phases (resolves open questions 1 and 2)
- [ ] Train and evaluate initial model (section 5) -- depends on new dataset availability
- [ ] Implement simulation integration (section 4) -- depends on trained model and #17 design
