# [ACTIVE] Simplify switch predictor architecture for baseline and feature analysis

Issue: #64 | Dependencies: #43 (original switch predictor)

## Goal

Train the simplest possible switch predictor (per-node MLP, no RNN, no edge model) and assess whether it produces respectable binary predictions. Separately, analyze feature importance and direction using single-feature models and statistical metrics (e.g. Sharpe ratio of feature-target correlation).

If the simple MLP is insufficient, extend the plan with additional architecture components.

## Plan

| # | Section | Change |
|---|---------|--------|
| 1 | Baseline | MLP-only config (no RNN, no edge model) |
| 2 | Feature ablations | Single-feature configs for each input |
| 3 | Feature statistics | Compute correlation and Sharpe ratio between features and target |
| 4 | Train | Submit all configs on Raven |
| 5 | Evaluate | Compare metrics and assess if baseline is respectable |

### 1. Baseline -- MLP-only

Create `configs/training/artificial_humans/switch_pred_mlp.yml` — identical to `switch_predictor.yml` but with `add_rnn: False`, `add_edge_model: False`. Reduces to `Encode → Linear(3→5, Tanh) → Linear(5→2) → Softmax`. Output: `artifacts/artificial_humans/switch_pred_mlp/`.

### 2. Feature ablations

Single-feature configs using MLP-only architecture:

| Config | `x_encoding` | Output dir suffix |
|--------|-------------|---------------------|
| `switch_pred_feat_contrib.yml` | `prev_contribution` only | `switch_pred_feat_contrib` |
| `switch_pred_feat_punish.yml` | `prev_punishment` only | `switch_pred_feat_punish` |
| `switch_pred_feat_group.yml` | `agent_group` only | `switch_pred_feat_group` |

Set `shuffle_features: []` since there is only one feature.

### 3. Feature statistics

Compute directly from the raw data (no model needed):
- Per-feature correlation with `does_switch` at decision rounds
- Sharpe ratio of the feature-target relationship (signal-to-noise)
- Class-conditional distributions (feature values when switching vs not)

### 4-5. Train and evaluate

- Submit baseline + 3 ablation configs on Raven (4 runs, independent)
- Compare test accuracy and log loss against the current full model (~60% test accuracy)
- If MLP baseline is respectable: done, document findings
- If not: extend this plan with architecture additions (edge model, RNN)

## Implementation notes

- No code changes needed — purely config-driven for training
- Feature statistics can be computed locally from the CSV with a script or notebook

## Next Actions

- [ ] Create baseline config (`switch_pred_mlp.yml`)
- [ ] Create 3 single-feature ablation configs
- [ ] Compute feature statistics from raw data
- [ ] Submit training runs on Raven
- [ ] Evaluate results and document findings
