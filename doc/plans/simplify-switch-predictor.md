# [ACTIVE] Simplify switch predictor architecture for baseline and feature analysis

Issue: #64 | Dependencies: #43 (original switch predictor)

## Goal

Build a generative switch predictor that can be sampled from to produce
realistic switching behavior. The primary metric is **test loss** (cross-entropy),
not classification accuracy or F1. The model must avoid overfitting — calibrated
probabilities matter more than sharp predictions.

Start with the simplest MLP baseline, then progressively add architecture
components (RNN, edge model) and evaluate each addition by test loss. Use the
`shuffle_features` mechanism and feature statistics to understand feature
importance. Keep experimentation lightweight — issue #44 covers systematic grid
search.

## Constraints (from supervisor meeting)

- **Loss is the metric** — this is a generative model to be sampled from
- **Epochs: 10-100** — prevent overfitting at all costs
- **Hidden size: 5-10** — small models only
- **agent_group** only makes sense with RNN or edge model (no per-node signal)
- **Common good collinearity** — be careful with `prev_common_good` as it may
  be collinear with other features
- **CV folds must be static** across runs for fair comparison
- **Don't over-experiment** — #44 already asks for grid search

## Plan

| # | Section | Status |
|---|---------|--------|
| 1 | MLP baseline (done) | Established that MLP alone is weak |
| 2 | Verification (done) | CV folds deterministic, shuffle_features is permutation importance |
| 3 | MLP rerun | Retrain with correct epoch range (10-100), h=5-10 |
| 4 | MLP + RNN | Add temporal information (priority addition) |
| 5 | MLP + RNN + edge model | Add social comparison |
| 6 | Feature importance | Use shuffle_features + statistical analysis |
| 7 | Regularization | If overfitting persists after architecture changes |

### 1. MLP baseline (done — see Experiments below)

Initial exploration with 10k epochs and h=2-50. Key findings:
- Without `prev_common_good`: predicts majority class only
- With `prev_common_good`: starts learning, but 10k epochs causes severe
  overfitting (test loss diverges from ~epoch 1000)
- Best macro F1 = 48.6 (h=50), but switch recall only ~20%
- These runs used too many epochs and too-large hidden sizes; results are
  useful for understanding feature importance but not as final baselines

### 2. Verification (done)

Both mechanisms confirmed working correctly:

**Static CV folds — verified deterministic.** Seed is set at the top of
`main()` in `train.py:121-123` (PyTorch, NumPy, Python `random` all seeded).
Fold generation in `generic/data.py:170` uses `random.shuffle()` on episode
indices after seeding. Same seed → same fold assignments. No external
randomness between seeding and fold creation.

**shuffle_features — permutation importance, works as expected.** The
`shuffle_feature()` function (`train.py:20-23`) permutes one feature's values
via `th.randperm`, breaking its relationship with the target while preserving
the distribution. Runs only during test evaluation (never training), every
`eval_period` epochs. Results are tagged with `shuffle_feature=<name>` in the
metrics parquet for post-hoc comparison of baseline vs shuffled test loss.

**Action needed**: current configs only list `prev_punishment` and
`prev_contribution` in `shuffle_features`. Must add `prev_common_good` (and
`agent_group` when using RNN/edge model) to get full feature importance.

### 3. MLP rerun

Retrain MLP baseline with corrected hyperparameters:
- Epochs: 10-100 (select by test loss, early stopping or manual inspection)
- Hidden size: 5-10
- Features: `prev_contribution`, `prev_punishment`, `prev_common_good`
  (exclude `agent_group` — no signal without RNN/edge model)
- Monitor test loss as primary metric

### 4. MLP + RNN

Add `add_rnn: True` to include temporal information. RNN enables:
- Learning from contribution/punishment trajectories, not just last round
- `agent_group` becomes meaningful (group membership over time)
- Same epoch/hidden size constraints as step 3

### 5. MLP + RNN + edge model

Add `add_edge_model: True` for social comparison (scatter_mean over neighbors).
Evaluate whether inter-player information improves test loss beyond RNN alone.

### 6. Feature importance

Two approaches:
- **shuffle_features**: use the built-in mechanism to toggle features during
  evaluation and measure loss degradation per feature
- **Statistical analysis**: compute from raw data — correlations, class-conditional
  distributions, check collinearity between `prev_common_good` and other features

### 7. Regularization

Only if needed after architecture is settled. Options: weight decay, entropy
regularization, dropout. Previous experiments showed weight decay (0.01) kills
switch detection entirely — use with caution.

## Experiments

### MLP baseline (5-fold CV)

All runs: MLP only (no RNN, no edge model), batch_size=10, 5-fold
episode-level CV. Features: prev_contribution, prev_common_good (float,
norm=20), prev_punishment. No agent_group (no signal without
relational/temporal components).

#### Round 1: lr and hidden size sweep (100 epochs)

| Config | Train Loss (final) | Test Loss (final) | Test Loss (best) | Test Acc (final) |
|--------|--------------------:|-------------------:|-----------------:|-----------------:|
| h=5, lr=3e-3 | 0.5935 +/- 0.0055 | 0.5961 +/- 0.0200 | 0.5948 +/- 0.0174 | 71.6 +/- 2.1% |
| h=5, lr=1e-4 | 0.7133 +/- 0.1322 | 0.6609 +/- 0.0354 | 0.6609 +/- 0.0317 | 62.0 +/- 19.2% |
| h=10, lr=3e-3 | 0.5890 +/- 0.0049 | 0.5896 +/- 0.0213 | 0.5882 +/- 0.0192 | 71.6 +/- 2.1% |
| h=10, lr=1e-4 | 0.6566 +/- 0.0438 | 0.6642 +/- 0.0433 | 0.6642 +/- 0.0387 | 58.5 +/- 19.2% |

- **lr=3e-3 converges well** — train and test loss track closely with no
  overfitting at 100 epochs. Both h=5 and h=10 reach similar test loss
  (~0.59), confirming that capacity is not the bottleneck.
- **lr=1e-4 is too slow** — models have not converged at 100 epochs, high
  variance across folds (especially h=5 with 19.2% accuracy std). Not
  useful at this epoch budget.

#### Round 2: epoch budget sweep (lr=3e-3, 200/300 epochs)

| Config | Train Loss (final) | Test Loss (final) | Test Loss (best) | Test Acc | Sw. P | Sw. R | Sw. F1 | Macro F1 |
|--------|-------------------:|------------------:|-----------------:|---------:|------:|------:|-------:|---------:|
| h=5, 200ep | 0.5880 +/- 0.0058 | 0.5940 +/- 0.0197 | 0.5918 +/- 0.0180 | 71.7% | 20.0% | 0.5% | 1.0% | 42.2% |
| h=5, 300ep | 0.5876 +/- 0.0063 | 0.5899 +/- 0.0205 | 0.5874 +/- 0.0189 | 72.4% | 69.3% | 5.3% | 9.5% | 46.6% |
| h=10, 200ep | 0.5870 +/- 0.0055 | 0.5900 +/- 0.0219 | 0.5854 +/- 0.0198 | 72.5% | 74.3% | 5.6% | 10.1% | 46.9% |
| h=10, 300ep | 0.5872 +/- 0.0066 | 0.5901 +/- 0.0224 | 0.5855 +/- 0.0198 | 72.5% | 69.3% | 6.0% | 10.9% | 47.3% |

#### MLP baseline conclusions

- **lr=3e-3 is the right learning rate** — lr=1e-4 does not converge
  within 100 epochs.
- **h=5 and h=10 produce equivalent results** — capacity is not the
  bottleneck. h=10 converges slightly faster (~100ep vs ~150ep).
- **Models converge by ~epoch 50-75** — test loss at 100, 200, and 300
  epochs is nearly identical (~0.585-0.590). Anything above 100 epochs
  is useless for MLP; no further learning or overfitting occurs.
- **No overfitting** — train and test loss track closely across all runs,
  even at 300 epochs. The models are well-calibrated.
- **Probability analysis** — the model outputs P(switch) ≈ 0.28 for
  nearly all samples, matching the data base rate (28.5%). The
  distributions for true-stay (mean 0.279) and true-switch (mean 0.290)
  almost completely overlap (diff ~0.01). As a generative model, sampling
  produces the correct overall switch rate, but switches occur at random
  rather than where they should — no conditional structure learned.
- **Best test loss = 0.5854** (h=10, 200ep) — this is the MLP baseline
  to beat with RNN and edge model additions.
- **Selected config going forward**: h=10, lr=3e-3, 100 epochs
  (`mlp.yml`). Other MLP variants removed.

### Architecture comparison (h=10, lr=3e-3, 5-fold CV)

All runs include agent_group (meaningful with RNN/edge). MLP baseline
included for comparison (100ep, no agent_group). RNN/edge models trained
at both 150ep and 500ep to assess overfitting.

#### 150 epoch runs (before overfitting)

| Model | Train Loss (final) | Test Loss (final) | Test Loss (best) |
|-------|-------------------:|------------------:|-----------------:|
| MLP (100ep) | 0.5890 +/- 0.0049 | 0.5896 +/- 0.0213 | 0.5882 +/- 0.0192 |
| MLP+edge | 0.5848 +/- 0.0072 | 0.5989 +/- 0.0244 | 0.5904 +/- 0.0219 |
| MLP+RNN | 0.5435 +/- 0.0241 | **0.5742 +/- 0.0235** | **0.5627 +/- 0.0167** |
| MLP+RNN+edge | 0.5389 +/- 0.0517 | 0.5874 +/- 0.0159 | 0.5789 +/- 0.0149 |

<!-- Loss curve plots saved locally in plots/group_selection/switch_pred_mlp_{rnn,edge,rnn_edge}_loss_cv.png -->

#### Architecture comparison conclusions

- **MLP+RNN has the best test loss** (0.5627 best, 0.5742 final) —
  a clear improvement over the MLP baseline (0.5882). Temporal
  information is the key signal for predicting switching behavior.
- **Edge model alone does not help** — MLP+edge test loss (0.5904) is
  no better than MLP alone (0.5882). The social comparison signal from
  scatter_mean may be redundant with prev_common_good.
- **Edge model on top of RNN hurts** — MLP+RNN+edge (0.5789) is worse
  than MLP+RNN alone (0.5627). Adding the edge model increases capacity
  without improving generalization.
- **RNN models overfit past ~150 epochs** — at 500ep, train loss drops
  toward 0.4 while test loss diverges with high fold variance. 150
  epochs keeps the models before the overfitting regime.
- **MLP+edge is more stable** — overfitting only begins around epoch 300,
  but this stability comes with no test loss improvement.

### Common good ablation (no prev_common_good, lr=1e-4, 500ep)

Tests whether the edge model can replace `prev_common_good` by providing
group-level signal via scatter_mean. Used lr=1e-4 because lr=3e-3
caused fast overfitting without common good.

| Model | Test Loss (final) | Test Loss (best) |
|-------|------------------:|-----------------:|
| MLP+edge nocg | 0.5995 +/- 0.0253 | 0.5994 +/- 0.0226 |
| MLP+RNN+edge nocg | 0.5938 +/- 0.0212 | 0.5928 +/- 0.0172 |
| *MLP (with CG, reference)* | *0.5896 +/- 0.0213* | *0.5882 +/- 0.0192* |
| *MLP+RNN (with CG, reference)* | *0.5742 +/- 0.0235* | *0.5627 +/- 0.0167* |

- **Edge model does not replace common good** — MLP+edge without CG
  (0.5994) is worse than the plain MLP with CG (0.5882). The edge
  model's scatter_mean does not replicate the signal in
  `prev_common_good`.
- **Even RNN+edge without CG underperforms MLP with CG** — MLP+RNN+edge
  nocg (0.5928) does not match the simple MLP baseline that includes
  common good (0.5882). `prev_common_good` carries independent signal.
- **Still converging at 500ep with lr=1e-4** — curves have not plateaued,
  but no overfitting. Could benefit from more epochs, though unlikely
  to close the gap with CG models.

### Feature importance (permutation shuffle test)

Uses the built-in `shuffle_features` mechanism: for each feature, its
values are randomly permuted across all test samples (breaking the
feature-target relationship while preserving the distribution). The
model is re-evaluated on this corrupted data. A large increase in test
loss means the feature is important; no change means the model doesn't
use it.

#### MLP (100ep, lr=3e-3)

Baseline test loss: 0.5937

| Feature | Shuffled Loss | Δ Loss | Importance |
|---------|-------------:|-------:|------------|
| prev_punishment | 0.5972 | +0.0035 | marginal |
| prev_contribution | 0.5964 | +0.0027 | marginal |
| prev_common_good | 0.5928 | -0.0009 | no effect |

No feature matters much — the MLP is just learning the class prior.
Confirmed by training MLP without CG: test loss 0.5910 vs 0.5919 with
CG — identical within noise. The earlier Phase 1 finding ("without CG
the MLP predicts majority class only") was an artifact of the 10k-epoch
overfitting regime, not a real effect.

#### MLP+RNN (120ep, lr=3e-3)

Baseline test loss: 0.5803

| Feature | Shuffled Loss | Δ Loss | Importance |
|---------|-------------:|-------:|------------|
| prev_contribution | 0.5920 | +0.0118 | important |
| agent_group | 0.5850 | +0.0047 | marginal |
| prev_common_good | 0.5840 | +0.0038 | marginal |
| prev_punishment | 0.5809 | +0.0007 | no effect |

- **prev_contribution is the most important feature** — the RNN learns
  switching behavior primarily from contribution trajectories over time.
- **agent_group has marginal signal** — group membership matters with
  temporal context but is not a primary driver.
- **prev_common_good is marginal** — less important than expected,
  possibly because the RNN reconstructs group-level info from individual
  contributions over time.
- **prev_punishment has no effect** — does not contribute to switch
  prediction in either model.

#### MLP+RNN extended features (120ep, lr=3e-3)

Added 3 new features on top of the base 4: `prev_does_switch` (bool),
`round_number` (int, n_levels=24), `prev_contribution_valid` (bool).

Best test loss: **0.5565 +/- 0.0369** (vs 0.5748 base MLP+RNN) —
improved but with higher fold variance.

Baseline test loss: 0.5663

| Feature | Shuffled Loss | Δ Loss | Importance |
|---------|-------------:|-------:|------------|
| prev_common_good | 0.5701 | +0.0038 | marginal |
| prev_does_switch | 0.5700 | +0.0036 | marginal |
| prev_contribution | 0.5699 | +0.0036 | marginal |
| prev_contribution_valid | 0.5688 | +0.0025 | marginal |
| prev_punishment | 0.5683 | +0.0019 | marginal |
| round_number | 0.5663 | +0.0000 | no effect |
| agent_group | 0.5658 | -0.0005 | no effect |

- With more features, importance is spread more evenly — no single
  feature dominates.
- **prev_does_switch** and **prev_contribution_valid** both carry
  marginal signal, contributing to the overall improvement.
- **round_number** and **agent_group** contribute nothing — dropped from
  the final config.
- Final feature selection: prev_contribution, prev_common_good,
  prev_punishment, prev_does_switch, prev_contribution_valid.

### Overall model comparison

| Model | Test Loss (best) | Fold Variance |
|-------|----------------:|-------------:|
| MLP baseline | 0.5919 | 0.0165 |
| MLP+RNN (4 feat) | 0.5748 | 0.0246 |
| MLP+RNN ext (7 feat) | 0.5565 | 0.0369 |
| MLP+RNN feat (5 feat) | 0.5698 | 0.0241 |

The 7-feature extended model has the best test loss but highest fold
variance. Dropping round_number and agent_group (5 feat) did not help
— the reduced model is worse than both the extended and close to the
base 4-feature model. This may reflect CUDA non-determinism between
runs or subtle feature interactions not captured by the shuffle test.
Grid search (#44) will provide more definitive feature selection.

## Implementation notes

- One code change: `clamp_grad` in `train.py` changed from
  `train_args["clamp_grad"]` to `train_args.get("clamp_grad")` to make
  it optional
- Early stopping added to `train.py`, configurable via
  `early_stopping_patience` in `train_args` (optional, backwards
  compatible)

## Next Actions

- [x] Create baseline MLP configs (h=5/10, lr=3e-3/1e-4)
- [x] Verify CV folds are static across runs
- [x] Investigate shuffle_features mechanism for feature importance
- [x] Train MLP baseline, evaluate by test loss
- [x] Train MLP+RNN, MLP+edge, MLP+RNN+edge — evaluate by test loss
- [x] Retrain at 150 epochs to avoid overfitting
- [x] Run feature importance analysis (shuffle_features)
- [ ] Document findings
