# [DONE] Retrain artificial humans on new group-switching data

## Goal

Train a contribution prediction model exclusively on the new group-switching experiment data (`ah_group_switching`, 8 agents, 2 groups, ~2500 rows). The switch predictor and validity model already exist. Once trained and evaluated, update the simulation config to use this new contribution model instead of the old one.

## Plan

| # | Section | Change | Optional |
|---|---------|--------|----------|
| 1 | Training config | Create new YAML config for contribution model | No |
| 2 | Train on cluster | Run training via `train-ah` CLI | No |
| 3 | Evaluate | Generate confusion matrix using holdout fold | No |
| 4 | Simulation config | Point `04_group_switching.yml` to new model | No |

### 1. Training config

- Create `configs/training/artificial_humans/group_switching_contribution.yml`
- Base on `pseudo_group_combined.yml` with these changes:
  - `experiment_names: [ah_group_switching]` (single experiment, not combined)
  - `data_file: experiments/group_switching_human_human_group_switching_8_agents.csv`
  - `n_groups: 2`
  - `n_cross_val: 10` and `holdout_fold: 0` (for evaluation, matching switch_predictor pattern)
  - `output_dir: artifacts/artificial_humans/group_switching_contribution`
  - `mask_name: contribution_valid` (unchanged)
  - `shuffle_features`: keep `prev_contribution` and `prev_punishment`; include `contribution_valid` only if the data has a `player_no_input` column (which derives it)
  - `batch_size: 4` (smaller dataset than combined, match switch_predictor)
- Keep all model architecture params (`y_levels: 21`, `y_name: contribution`, `hidden_size: 5`, `add_rnn: True`, `add_edge_model: True`, `x_encoding` with `agent_group`)

### 2. Train on cluster

- Run `python -m aimanager train-ah configs/training/artificial_humans/group_switching_contribution.yml` on Raven
- Output lands in `artifacts/artificial_humans/group_switching_contribution/`

### 3. Evaluate

- Run `python scripts/plotting/plot_confusion_matrix.py artifacts/artificial_humans/group_switching_contribution` to generate confusion matrix plot
- Review accuracy; compare with the combined model if needed

### 4. Simulation config

- Created `configs/simulation/ah_testing/group_switching.yml` comparing `pseudo_group`, `group_switching` (with switch model), and `pre_group` AH models against `latest_manager`
- Fixed bug: `agent_groups` from config was not passed through to `ArtificialHumanEnv`
- Verified group switching only occurs when switch model is present; groups stay constant otherwise

## Implementation notes

- The `prev_contribution_valid` feature in `x_encoding` (as `etype: bool`) should remain -- it tells the model which prior contributions are real vs. masked.

## Results

- Test MAE (greedy): **1.94** on holdout fold (10-fold, fold 0)
- Model without `agent_group` feature scored MAE 1.83, but `agent_group` is structural for the multi-group pipeline
- AH comparison simulation (`configs/simulation/ah_testing/group_switching.yml`) shows group_switching AH behaves distinctly from pseudo_group and pre_group models
- Bug fix: `agent_groups` from config was not being passed to `ArtificialHumanEnv` — now wired through

## Next Actions

- [x] Check CSV header for `player_no_input` / `contribution_valid` column presence
- [x] Create `configs/training/artificial_humans/group_switching_contribution.yml`
- [x] Train on Raven cluster
- [x] Run confusion matrix evaluation
- [x] Create AH comparison simulation config and verify
- [x] Create AH comparison simulation config (`configs/simulation/ah_testing/group_switching.yml`)
- [x] Fix `agent_groups` passthrough bug in `simulate.py`
