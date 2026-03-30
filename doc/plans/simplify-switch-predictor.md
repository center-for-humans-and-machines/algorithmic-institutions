# [ACTIVE] Simplify switch predictor architecture for baseline and feature analysis

Issue: #64 | Dependencies: #43 (original switch predictor)

## Goal

Train the simplest possible switch predictor (per-node MLP, no RNN, no edge model) and assess whether it produces respectable binary predictions. Separately, analyze feature importance and direction using single-feature models and statistical metrics (e.g. Sharpe ratio of feature-target correlation).

If the simple MLP is insufficient, extend the plan with additional architecture components.

## Plan

| # | Section | Change |
|---|---------|--------|
| 1 | Baseline | MLP-only config (no RNN, no edge model) |
| 2 | MLP + edge model | Add neighbor info (simplest social comparison) |
| 3 | Feature ablations | Single-feature configs for each input |
| 4 | Feature statistics | Compute correlation and Sharpe ratio between features and target |
| 5 | Train and evaluate | Submit configs, compare metrics |

### 1. Baseline -- MLP-only (done)

Created `configs/training/artificial_humans/switch_predictor/mlp.yml` — `add_rnn: False`, `add_edge_model: False`. Reduces to per-node 2-layer MLP (`4 → h → 2`).

**Without `prev_common_good`**: predicts majority class only regardless of hidden size (5, 20, 100). No signal in per-node features alone.

**With `prev_common_good`** (float-encoded, norm=20): the model starts learning. Best result at h100, lr 3e-3, batch 11 (full-batch), 20k epochs: 84.5% train, 60.4% test, 40% switch recall. Overfits significantly (loss curve shows test loss diverging from ~epoch 500). The `prev_common_good` feature carries implicit group-level signal that enables the MLP to learn without the edge model.

### 2. MLP + edge model

Create `configs/training/artificial_humans/switch_predictor/mlp_edge.yml` — adds `add_edge_model: True` while keeping `add_rnn: False`. The edge model lets each node see the mean of its neighbors' features via `scatter_mean`, enabling social comparison (e.g. "my contribution vs group average"). This is the minimal addition needed for a switching decision that is inherently relational.

### 3. Feature ablations

Single-feature configs using MLP + edge architecture:

| Config | `x_encoding` | Output dir suffix |
|--------|-------------|---------------------|
| `switch_predictor/feat_contrib.yml` | `prev_contribution` only | `switch_pred_feat_contrib` |
| `switch_predictor/feat_punish.yml` | `prev_punishment` only | `switch_pred_feat_punish` |
| `switch_predictor/feat_group.yml` | `agent_group` only | `switch_pred_feat_group` |

Set `shuffle_features: []` since there is only one feature.

### 4. Feature statistics

Compute directly from the raw data (no model needed):
- Per-feature correlation with `does_switch` at decision rounds
- Sharpe ratio of the feature-target relationship (signal-to-noise)
- Class-conditional distributions (feature values when switching vs not)

### 5. Train and evaluate

- Submit MLP+edge + 3 ablation configs on Raven
- Compare test accuracy — must beat 74% (majority class baseline) to be useful
- Check confusion matrix for actual switch-class recall
- If MLP+edge is insufficient: add RNN next

## Experiments

Majority class baseline: 74% (71 stay / 25 switch in test set).

| Model | Features | h | lr | batch | epochs | Train | Test | Gap | Switch recall |
|---|---|---|---|---|---|---|---|---|---|
| Full RNN+edge | 3 orig | 5 | 3e-4 | 4 | 10k | 84.8% | 54.2% | 30.7% | - |
| MLP | 3 orig | 5 | 3e-4 | 4 | 10k | 72.2% | 74.0% | -1.8% | 0% |
| MLP | 3 orig | 20 | 3e-4 | 4 | 10k | 72.2% | 74.0% | -1.8% | 0% |
| MLP | 3 orig | 100 | 3e-4 | 4 | 10k | 72.7% | 74.0% | -1.2% | 0% |
| MLP+edge | 3 orig | 5 | 3e-4 | 4 | 10k | 73.3% | 69.8% | 3.5% | 8% |
| MLP+edge | 3 orig | 5 | 6e-4 | 4 | 10k | 75.0% | 70.8% | 4.2% | 12% |
| MLP+edge | 3 orig | 20 | 3e-3 | 4 | 10k | 96.0% | 53.1% | 42.9% | 24% |
| MLP+cg | +cg | 50 | 3e-3 | 11 | 20k | 84.5% | 60.4% | 24.1% | 40% |
| MLP+cg | +cg | 100 | 5e-3 | 11 | 30k | 81.6% | 63.5% | 18.1% | 20% |
| MLP+cg | +cg | 100 | 3e-3 | 11 | 50k | 88.1% | 52.1% | 36.0% | 16% |
| **MLP+cg** | **+cg** | **100** | **3e-3** | **11** | **20k** | **84.5%** | **60.4%** | **24.1%** | **40%** |

"3 orig" = prev_contribution, prev_punishment, agent_group. "+cg" = adds prev_common_good.

## Implementation notes

- One code change: `clamp_grad` in `train.py` changed from `train_args["clamp_grad"]` to `train_args.get("clamp_grad")` to make it optional
- Feature statistics can be computed locally from the CSV with a script or notebook

## Next Actions

- [x] Create baseline config (`switch_predictor/mlp.yml`) — predicts majority class only
- [x] Create MLP+edge config (`switch_predictor/mlp_edge.yml`) — starts learning switches
- [ ] Create 3 single-feature ablation configs
- [ ] Compute feature statistics from raw data
- [ ] Submit training runs on Raven
- [ ] Evaluate results and document findings
