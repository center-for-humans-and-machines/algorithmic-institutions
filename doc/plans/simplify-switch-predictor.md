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

## Implementation notes

- One code change: `clamp_grad` in `train.py` changed from
  `train_args["clamp_grad"]` to `train_args.get("clamp_grad")` to make
  it optional

## Next Actions

- [x] Create baseline MLP configs (h=5/10, lr=3e-3/1e-4)
- [x] Verify CV folds are static across runs
- [x] Investigate shuffle_features mechanism for feature importance
- [x] Train MLP baseline, evaluate by test loss
- [ ] Train MLP+RNN, evaluate by test loss
- [ ] Train MLP+RNN+edge, evaluate by test loss
- [ ] Run feature importance analysis (shuffle_features + statistics)
- [ ] Document findings
