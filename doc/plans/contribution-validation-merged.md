# [DRAFT] Merge contribution and validity models into a single joint model

Issue: #65

## Goal

Replace the two separate artificial human models (contribution predictor
with 21 output classes, validity predictor with 2 output classes) with a
single model that outputs 23 logits (21 contribution + 2 validity). This
simplifies the simulation pipeline (one model instead of two), allows
the contribution model to train on ALL samples (not just valid ones),
and lets the model learn joint contribution-validity structure.

Currently:
- Contribution model: `mask_name: contribution_valid`, 21 output
  classes, trained only on valid samples (~71% of data)
- Validity model: `mask_name: recorded`, 2 output classes, MLP-only,
  trained on all samples

After:
- Joint model: `mask_name: recorded`, 23 output classes, trained on all
  samples with a joint loss (contribution CE masked to valid samples +
  validity CE on all samples)

## Constraints

- The joint model must reproduce the same simulation behavior as the
  two separate models (contribution values, validity predictions)
- Backward compatibility: existing separate models must still work
  (do not remove the `artifical_humans_valid` code path)
- Loss is the primary metric (generative model, sampled from)
- 88-char line limit, Black formatter

## Plan

| # | Section | Change | Optional |
|---|---------|--------|----------|
| 1 | Baseline verification | Confirm modern validity config reproduces legacy | no |
| 2 | GraphNetwork output split | Teach model to split 23 logits into two heads | no |
| 3 | Training loop: joint loss | Add dual-head loss computation in `train.py` | no |
| 4 | Evaluation: dual-head metrics | Report separate metrics for each head | no |
| 5 | Prediction: dual-head sampling | Extract contribution + validity from joint output | no |
| 6 | Simulation: single model path | Use joint model in environment + RL training | no |
| 7 | Training config | New YAML config for joint model | no |
| 8 | Validation | Compare test loss and simulation behavior | no |

### 1. Baseline verification (done)

Modern config reproduces both legacy models exactly (all three
match to 4+ decimal places across all 6 CV folds):

| Model | Test Log Loss | Test Accuracy |
|-------|-------------:|-------------:|
| legacy_v4 | 0.0963 +/- 0.0216 | 97.98% |
| raven_22 | 0.0963 +/- 0.0216 | 97.98% |
| modern | 0.0963 +/- 0.0216 | 97.98% |

Best fold test loss: 0.0619 (fold 2). Model converges by ~epoch 400
but is trained to 10k. No overfitting (train/test loss track closely).
Naive base-rate CE: 0.602 — the model is far better than chance.

### 2. GraphNetwork output split

**File**: `src/aimanager/generic/graph.py`

The `GraphNetwork` currently produces `y_levels` output logits (set by
`y_levels` param, default 21). For the joint model, the output must
be split into two heads: contribution (first 21 logits) and validity
(last 2 logits).

- Add an optional `joint_output` config to `GraphNetwork.__init__`
  that describes the output decomposition. When absent, behavior is
  unchanged (single-head model). When present, it is a list of dicts
  describing each head, e.g.:
  ```
  joint_output:
    - name: contribution
      n_levels: 21
    - name: contribution_valid
      n_levels: 2
  ```
  When `joint_output` is set, `y_levels` becomes the sum of all head
  sizes (here 23). The `y_encoder` stays a single `IntEncoder` with
  `n_levels=21` (the primary target for encoding input y). The final
  layer outputs the full 23-dim vector.

- Add a `split_output(y_logit)` method that returns a dict mapping
  head names to their logit slices:
  `{"contribution": y_logit[..., :21], "contribution_valid": y_logit[..., 21:23]}`

- Update `predict_encoded` to handle joint output: return per-head
  predictions (contribution as int 0-20, validity as bool). The
  existing `y_encoder.decode` handles the contribution head; validity
  uses argmax/sampling on its own 2-class softmax.

- Add `joint_output` to the `save`/`load` attribute list so the
  config persists in the checkpoint.

### 3. Training loop: joint loss

**File**: `src/aimanager/artificial_humans/train.py`

The training loop currently computes a single CE loss masked by
`mask_name`. For joint output, it needs to compute two losses with
different masks.

- When `joint_output` is configured (detectable from `model.joint_output`
  being non-None):
  - Forward pass produces 23-dim logits
  - Split logits via `model.split_output(y_logit)`
  - **Contribution loss**: CE on the first 21 logits against
    `contribution` targets, masked by `contribution_valid` (only valid
    samples contribute to contribution loss)
  - **Validity loss**: CE on the last 2 logits against
    `contribution_valid` targets (cast to int64), masked by `recorded`
    (all samples contribute)
  - **Total loss**: `contribution_loss + validity_loss` (equal
    weighting to start; weighting can be tuned later)
  - The entropy regularization term (`l1_entropy`) applies to the
    contribution head only (matching current behavior)

- The `y_enc` in the encoded data currently encodes the primary target
  (contribution). For the joint model, we also need
  `contribution_valid` as an integer target. This can be extracted
  directly from `batch_data` since `contribution_valid` is already in
  the data dict.

- Config-level: add `joint_output` to `model_args` in the YAML config.
  The training loop detects joint mode from the model object, not from
  a separate config key.

### 4. Evaluation: dual-head metrics

**File**: `src/aimanager/artificial_humans/evaluation.py`

- `eval_model`: when `model.joint_output` is set, compute separate
  metrics for each head (contribution log_loss, validity log_loss,
  contribution accuracy, validity accuracy). Tag metrics with
  `head=contribution` or `head=contribution_valid`.
- `create_confusion_matrix`: generate separate confusion matrices for
  each head. The contribution confusion matrix is 21x21 (masked to
  valid), the validity one is 2x2 (all samples).

### 5. Prediction: dual-head sampling

**File**: `src/aimanager/generic/graph.py`

- `predict_encoded`: for joint models, return a dict or tuple with
  both predictions:
  - `contribution`: sampled/argmaxed from first 21 softmax outputs
  - `contribution_valid`: sampled/argmaxed from last 2 softmax outputs
    (as bool)
- `predict_independent` and `predict`: pass through the joint
  predictions. The return signature changes from
  `(y_pred, y_pred_proba)` to a richer structure when
  `joint_output` is set.

**Open question**: The return type change may ripple through many
callers. An alternative is to keep `predict` returning contribution
predictions and add a separate `predict_validity` method or a
`predict_joint` method. Decision depends on how many callers need
to change.

### 6. Simulation: single model path

**Files**:
- `src/aimanager/manager/environment.py`
- `src/aimanager/simulation/simulate.py`
- `src/aimanager/rl_manager.py`

**`environment.py` -- `update_contribution` (line 279-301)**:

Currently calls two separate models:
```python
contribution = self.artifical_humans.predict(self.state, ...)[0]
if self.artifical_humans_valid is not None:
    contribution_valid = self.artifical_humans_valid.predict(
        self.state, ...)[0]
    contribution_valid = contribution_valid.to(th.bool)
    contribution[~contribution_valid] = default_value
```

For the joint model, add a new code path: when
`artifical_humans_valid is None` AND the contribution model has
`joint_output`, call predict once and unpack both outputs. The
existing two-model path remains for backward compatibility.

- Add detection: `if hasattr(self.artifical_humans, 'joint_output') and self.artifical_humans.joint_output is not None`
- In this branch, a single `predict` call returns both contribution
  and validity. Apply the same masking logic (set invalid contributions
  to default value).
- When neither `joint_output` nor `artifical_humans_valid` is set,
  assume all contributions are valid (existing behavior, line 298-299).

**`simulate.py` -- `run_simulation` (line 154-194)**:

Currently loads two models per artificial human config:
```python
ah = GraphNetwork.load(hm_path, device=device)
ah_val = GraphNetwork.load(hmv_path, device=device)
```

- Add a new config key `joint_model` (path to the joint model) as an
  alternative to `contribution_model` + `valid_model`
- When `joint_model` is present, load only one model and pass
  `artifical_humans_valid=None` to the environment
- When `contribution_model` + `valid_model` are present, keep the
  existing two-model loading

**`rl_manager.py` -- `train_manager` (line 85-148)**:

Currently loads two models via `artificial_humans` and
`artificial_humans_valid` config keys.

- Add support for a `joint_model` config key. When present, load one
  model and pass `artifical_humans_valid=None` to the environment.
- Keep the existing two-key path for backward compatibility.

### 7. Training config

**File**: `configs/training/artificial_humans/contribution_joint.yml`

New config for the joint model. Key differences from `script_21`:
- `mask_name: recorded` (train on all samples, not just valid)
- `model_args.joint_output` as described in step 2
- `model_args.y_levels: 23` (21 contribution + 2 validity)
- Same architecture as script_21: MLP+edge+RNN, h=5
- Same features: prev_contribution, prev_punishment,
  prev_contribution_valid
- Same optimizer/training args as script_21

### 8. Validation

- Train the joint model on the cluster
- Compare contribution-head test loss against the standalone
  contribution model's test loss (script_21 baseline)
- Compare validity-head test loss against the standalone validity
  model's test loss (step 1 baseline)
- Run a simulation with the joint model and compare contribution
  distributions and common good trajectories against the two-model
  simulation
- If losses are comparable and simulation behavior matches, the joint
  model is validated

## Implementation notes

- The `y_encoder` in `GraphNetwork` serves dual purpose: it encodes
  the target variable for autoregressive input AND defines the output
  dimensionality. For joint output, keep `y_encoder` as the primary
  target encoder (contribution, 21 levels). The output layer size is
  set independently to `sum(head.n_levels for head in joint_output)`.
  This requires adjusting the `op2` NodeModel's `out_features` from
  `y_features` to the total output size.

- The `IntEncoder.decode` method handles sampling via `th.multinomial`
  on a probability vector. For the validity head, create a small
  `IntEncoder(encoding="onehot", name="contribution_valid", n_levels=2)`
  at model init time to handle decoding. Store it as
  `self.validity_encoder` (only present when `joint_output` is set).

- The `mask` in `batch_data` currently comes from a single mask field.
  For joint loss, the training loop needs two masks:
  `contribution_valid` for the contribution head and `recorded` for
  the validity head. Both are already available in the data dict
  before encoding. The simplest approach: pass `mask=recorded` to
  `model.encode` (so all samples are included in the batch), then
  extract `contribution_valid` from the data dict to mask the
  contribution loss inside the training loop.

- In the current `predict_encoded`, `y_pred_proba` has shape
  `[batch*agents, rounds, y_levels]`. For joint output, this becomes
  `[batch*agents, rounds, 23]`. The split into contribution (21) and
  validity (2) proba tensors happens inside `split_output`.

- Loss weighting: start with equal weight (1:1) for contribution and
  validity losses. If the validity task is much easier (as expected,
  since the standalone model achieves low loss with just a MLP), its
  gradient contribution will naturally be smaller. Consider adding a
  configurable `validity_loss_weight` parameter if tuning is needed.

## Next Actions

- [x] Verify baseline: all three models match to 4+ decimal places
- [ ] Implement `joint_output` in `GraphNetwork` (step 2)
- [ ] Implement `split_output` and dual-head prediction (step 5)
- [ ] Implement joint loss in training loop (step 3)
- [ ] Implement dual-head evaluation metrics (step 4)
- [ ] Add single-model simulation path (step 6)
- [ ] Create joint training config (step 7)
- [ ] Train on cluster and validate (step 8)
