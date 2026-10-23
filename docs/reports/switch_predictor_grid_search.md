# Switch Predictor Grid Search (issue #78)

This report documents the mild hyper-parameter sweep for the group-switching
predictor, run from branch `78-improve-ah-models-contribution-switch` on
2026-04-15. Artifacts under
`artifacts/artificial_humans/switch_pred_grid_search/` will be deleted after
this write-up; the derived `opt.yml` config carries the winning settings
forward.

## Motivation

Issue #78 flagged two simulation-vs-pilot gaps after the dataset correction
in #76:

1. Flat simulated contribution trajectory (stays ~8 vs pilot's 8 to 10).
2. Switch predictor under-predicts switches by ~6 to 9x at every decision
   round; the binary classifier collapses to the majority "no-switch" class.

The issue's investigation directions for the switch model included
re-visiting `hidden_size` (reduced from 10 to 5 in #76) and broadly checking
architecture sensitivity. This grid search targets that piece.

## Experimental setup

- **Config:** `configs/training/artificial_humans/switch_predictor/grid_search.yml`
- **Base:** the feature-rich variant (`mlp_rnn_feat.yml` at the time of the
  sweep), all 7 input features, shuffle + ablate perturbations enabled.
- **Data:** `experiments/group_switching_human_human_group_switching_8_agents.csv`,
  8 players, 2 groups, 24 rounds.
- **CV:** 8-fold cross-validation (`n_cross_val: 8`).
- **Grid (18 configs):**
  - `architecture`: `mlp`, `mlp+rnn`, `mlp+rnn+edge`
    (zipped with `add_rnn` / `add_edge_model`)
  - `hidden_size`: `5`, `10`, `20`
  - `lr`: `1e-3`, `3e-3`
- **Training:** 120 epochs, batch_size 10, eval every 5 epochs.
- **Submission:** 18 independent SLURM jobs, each requesting 1 A100, via
  `src/aimanager/artificial_humans/run.py`.

## Ranking (test log-loss, best to worst)

8-fold mean +/- std; epoch chosen per fold as the argmin of baseline test
log-loss.

| Rank | Architecture    | Hidden | LR    | Test log-loss     | Acc   | MAE   | Best epoch |
|-----:|:----------------|-------:|:------|:------------------|:------|:------|-----------:|
| 1    | mlp+rnn+edge    |     20 | 3e-3  | 0.5292 +/- 0.0585 | 0.721 | 0.279 |         79 |
| 2    | mlp+rnn         |     20 | 3e-3  | 0.5347 +/- 0.0432 | 0.725 | 0.275 |        101 |
| 3    | mlp+rnn         |     10 | 3e-3  | 0.5543 +/- 0.0375 | 0.721 | 0.279 |         93 |
| 4    | mlp+rnn+edge    |     10 | 3e-3  | 0.5558 +/- 0.0481 | 0.751 | 0.249 |         88 |
| 5    | mlp+rnn+edge    |     20 | 1e-3  | 0.5582 +/- 0.0443 | 0.734 | 0.266 |         93 |
| 6    | mlp+rnn+edge    |      5 | 3e-3  | 0.5668 +/- 0.0413 | 0.719 | 0.281 |         88 |
| 7    | mlp             |     20 | 3e-3  | 0.5695 +/- 0.0396 | 0.733 | 0.267 |         76 |
| 8    | mlp+rnn         |     20 | 1e-3  | 0.5706 +/- 0.0356 | 0.717 | 0.283 |        108 |
| 9    | mlp+rnn         |      5 | 3e-3  | 0.5735 +/- 0.0293 | 0.723 | 0.277 |        105 |
| 10   | mlp+rnn+edge    |     10 | 1e-3  | 0.5768 +/- 0.0325 | 0.714 | 0.286 |         99 |
| 11   | mlp             |      5 | 3e-3  | 0.5777 +/- 0.0390 | 0.715 | 0.285 |         90 |
| 12   | mlp             |     10 | 3e-3  | 0.5779 +/- 0.0447 | 0.721 | 0.279 |        103 |
| 13   | mlp             |     20 | 1e-3  | 0.5780 +/- 0.0358 | 0.715 | 0.285 |         91 |
| 14   | mlp+rnn         |     10 | 1e-3  | 0.5802 +/- 0.0388 | 0.719 | 0.281 |         96 |
| 15   | mlp+rnn+edge    |      5 | 1e-3  | 0.5895 +/- 0.0234 | 0.724 | 0.276 |         99 |
| 16   | mlp+rnn         |      5 | 1e-3  | 0.5919 +/- 0.0255 | 0.714 | 0.286 |        105 |
| 17   | mlp             |     10 | 1e-3  | 0.5921 +/- 0.0395 | 0.714 | 0.286 |        107 |
| 18   | mlp             |      5 | 1e-3  | 0.5943 +/- 0.0324 | 0.720 | 0.280 |        112 |

### Trends

- `lr=3e-3` beats `lr=1e-3` at every architecture x hidden_size pairing.
- Capacity helps: `hidden_size=20` dominates for both RNN-based
  architectures. The #76 decision to shrink from 10 to 5 was too aggressive.
- Adding the RNN and edge modules gives consistent gains over the
  `mlp`-only baseline; the `mlp+rnn+edge` combo takes the top spot.
- CV variance is noticeable (std ~ 0.03 to 0.06) given ~11 training
  episodes. The top four configs are within one std of each other; the
  ranking between them is not sharp.

## Feature importance across runs

Delta = test log-loss with the feature perturbed minus the baseline. Both
perturbations are evaluated at the baseline's best-test-loss epoch per
fold, then averaged across folds. Positive = perturbation hurt = feature
mattered.

### Ablate (replace feature with its mean)

| config (arch \| h \| lr)  | test loss | agent_group | prev_common_good | prev_does_switch | prev_contribution | prev_punishment | prev_contr_valid | round_number |
|:--------------------------|----------:|------------:|-----------------:|-----------------:|------------------:|----------------:|-----------------:|-------------:|
| mlp+rnn+edge \| 20 \| 3e-3 |    0.5292 |     +0.0181 |          -0.0034 |          +0.0044 |           -0.0002 |         -0.0017 |          +0.0009 |      -0.0000 |
| mlp+rnn      \| 20 \| 3e-3 |    0.5347 |     +0.0127 |          -0.0027 |          +0.0049 |           -0.0010 |         -0.0010 |          -0.0004 |      +0.0000 |
| mlp+rnn      \| 10 \| 3e-3 |    0.5543 |     +0.0076 |          -0.0007 |          +0.0021 |           -0.0018 |         -0.0003 |          +0.0011 |      +0.0000 |
| mlp+rnn+edge \| 10 \| 3e-3 |    0.5558 |     +0.0060 |          +0.0009 |          +0.0019 |           -0.0001 |         +0.0001 |          -0.0001 |      +0.0000 |
| mlp+rnn+edge \| 20 \| 1e-3 |    0.5582 |     +0.0019 |          +0.0021 |          +0.0003 |           -0.0002 |         +0.0006 |          -0.0001 |      +0.0000 |
| mlp+rnn+edge \| 5  \| 3e-3 |    0.5668 |     +0.0015 |          +0.0013 |          +0.0001 |           -0.0003 |         +0.0004 |          -0.0000 |      +0.0000 |
| mlp          \| 20 \| 3e-3 |    0.5695 |     +0.0020 |          +0.0017 |          +0.0000 |           -0.0002 |         +0.0008 |          -0.0002 |      +0.0000 |
| mlp+rnn      \| 20 \| 1e-3 |    0.5706 |     +0.0028 |          -0.0012 |          +0.0011 |           -0.0005 |         -0.0004 |          +0.0001 |      +0.0000 |
| mlp+rnn      \| 5  \| 3e-3 |    0.5735 |     +0.0022 |          -0.0022 |          +0.0017 |           -0.0017 |         -0.0008 |          +0.0010 |      +0.0000 |
| mlp+rnn+edge \| 10 \| 1e-3 |    0.5768 |     +0.0005 |          +0.0001 |          +0.0000 |           -0.0006 |         +0.0005 |          -0.0000 |      +0.0000 |
| mlp          \| 5  \| 3e-3 |    0.5777 |     -0.0003 |          +0.0023 |          +0.0000 |           -0.0017 |         +0.0007 |          -0.0007 |      +0.0000 |
| mlp          \| 10 \| 3e-3 |    0.5779 |     +0.0005 |          -0.0011 |          +0.0000 |           -0.0008 |         +0.0001 |          +0.0005 |      +0.0000 |
| mlp          \| 20 \| 1e-3 |    0.5780 |     +0.0007 |          +0.0016 |          +0.0000 |           -0.0001 |         +0.0004 |          -0.0003 |      +0.0000 |
| mlp+rnn      \| 10 \| 1e-3 |    0.5802 |     +0.0003 |          -0.0002 |          +0.0001 |           -0.0002 |         +0.0002 |          +0.0001 |      +0.0000 |
| mlp+rnn+edge \| 5  \| 1e-3 |    0.5895 |     -0.0001 |          +0.0003 |          +0.0000 |           -0.0002 |         +0.0001 |          +0.0000 |      +0.0000 |
| mlp+rnn      \| 5  \| 1e-3 |    0.5919 |     -0.0006 |          +0.0004 |          +0.0001 |           -0.0005 |         -0.0001 |          +0.0002 |      +0.0000 |
| mlp          \| 10 \| 1e-3 |    0.5921 |     -0.0030 |          -0.0012 |          +0.0000 |           -0.0007 |         +0.0002 |          +0.0006 |      +0.0000 |
| mlp          \| 5  \| 1e-3 |    0.5943 |     -0.0037 |          +0.0015 |          +0.0000 |           -0.0019 |         -0.0000 |          -0.0007 |      +0.0000 |

### Shuffle (permute feature across samples)

| config (arch \| h \| lr)  | test loss | prev_common_good | agent_group | prev_contribution | prev_punishment | prev_does_switch | prev_contr_valid | round_number |
|:--------------------------|----------:|-----------------:|------------:|------------------:|----------------:|-----------------:|-----------------:|-------------:|
| mlp+rnn+edge \| 20 \| 3e-3 |    0.5292 |          +0.0044 |     +0.0018 |           +0.0022 |         -0.0039 |          -0.0005 |          +0.0000 |      +0.0000 |
| mlp+rnn      \| 20 \| 3e-3 |    0.5347 |          +0.0112 |     +0.0010 |           +0.0059 |         -0.0017 |          +0.0011 |          +0.0010 |      +0.0000 |
| mlp+rnn      \| 10 \| 3e-3 |    0.5543 |          +0.0019 |     +0.0099 |           -0.0007 |         +0.0014 |          +0.0010 |          +0.0004 |      +0.0000 |
| mlp+rnn+edge \| 10 \| 3e-3 |    0.5558 |          +0.0036 |     +0.0020 |           +0.0007 |         -0.0001 |          +0.0053 |          -0.0004 |      +0.0000 |
| mlp+rnn+edge \| 20 \| 1e-3 |    0.5582 |          +0.0013 |     +0.0001 |           -0.0006 |         +0.0009 |          +0.0004 |          -0.0005 |      +0.0000 |
| mlp+rnn+edge \| 5  \| 3e-3 |    0.5668 |          +0.0029 |     +0.0012 |           +0.0001 |         +0.0010 |          +0.0005 |          -0.0001 |      +0.0000 |
| mlp          \| 20 \| 3e-3 |    0.5695 |          +0.0025 |     +0.0012 |           -0.0002 |         +0.0002 |          +0.0000 |          +0.0000 |      +0.0000 |
| mlp+rnn      \| 20 \| 1e-3 |    0.5706 |          -0.0008 |     +0.0000 |           +0.0005 |         -0.0004 |          +0.0009 |          -0.0005 |      +0.0000 |
| mlp+rnn      \| 5  \| 3e-3 |    0.5735 |          -0.0038 |     +0.0024 |           -0.0018 |         +0.0011 |          +0.0003 |          +0.0009 |      +0.0000 |
| mlp+rnn+edge \| 10 \| 1e-3 |    0.5768 |          +0.0011 |     +0.0015 |           -0.0007 |         +0.0010 |          +0.0003 |          +0.0001 |      +0.0000 |
| mlp          \| 5  \| 3e-3 |    0.5777 |          +0.0009 |     -0.0028 |           -0.0026 |         +0.0021 |          +0.0000 |          -0.0006 |      +0.0000 |
| mlp          \| 10 \| 3e-3 |    0.5779 |          -0.0032 |     +0.0009 |           -0.0004 |         +0.0004 |          +0.0000 |          -0.0001 |      +0.0000 |
| mlp          \| 20 \| 1e-3 |    0.5780 |          -0.0005 |     -0.0025 |           -0.0002 |         +0.0011 |          +0.0000 |          -0.0001 |      +0.0000 |
| mlp+rnn      \| 10 \| 1e-3 |    0.5802 |          -0.0003 |     +0.0004 |           +0.0001 |         -0.0000 |          -0.0000 |          -0.0004 |      +0.0000 |
| mlp+rnn+edge \| 5  \| 1e-3 |    0.5895 |          +0.0006 |     -0.0004 |           +0.0001 |         +0.0001 |          +0.0000 |          -0.0001 |      +0.0000 |
| mlp+rnn      \| 5  \| 1e-3 |    0.5919 |          +0.0004 |     +0.0005 |           +0.0000 |         -0.0002 |          +0.0001 |          +0.0001 |      +0.0000 |
| mlp          \| 10 \| 1e-3 |    0.5921 |          -0.0009 |     -0.0061 |           -0.0013 |         +0.0000 |          +0.0000 |          -0.0001 |      +0.0000 |
| mlp          \| 5  \| 1e-3 |    0.5943 |          +0.0034 |     -0.0057 |           -0.0004 |         -0.0002 |          +0.0000 |          +0.0000 |      +0.0000 |

### Key findings

- **`agent_group` is the most load-bearing input under ablate**, and its
  importance grows with model capacity (+0.018 at the winner, ~0 at the
  smallest models). That matches intuition: the switch decision depends on
  which of the two groups the player is in.
- **`prev_common_good` and `prev_contribution`** lead the shuffle ranking
  -- numeric signals that survive mean-substitution but lose meaning when
  permuted across samples.
- **`prev_does_switch`** matters only for higher-capacity RNN models; it
  carries temporal information that simpler architectures cannot exploit.
- **`round_number` delta is ~0 on every config, both methods** -- the
  model does not use temporal position directly. Dropping it simplifies
  the input without cost.
- All signals are small in absolute terms (|delta| < 0.02 log-loss),
  consistent with the small-dataset / high-variance regime noted in #78.

## Best run: `mlp+rnn+edge | h=20 | lr=3e-3`

### CV loss curves

- Train loss falls monotonically from ~0.7 to ~0.45 over 120 epochs.
- Test log-loss bottoms around epochs 60 to 80 at ~0.53 then drifts up --
  mild overfitting past that.
- Average best epoch across folds = 79. This is the primary motivation for
  capping `epochs: 75` in `opt.yml`.

### Confusion matrix

Aggregated across 8 folds (624 decision-round predictions total, `valid=True`):

|              | Pred: stay | Pred: switch |
|:-------------|:----------:|:------------:|
| True: stay   | 392 (88%)  |   54 (12%)   |
| True: switch | 123 (69%)  |   55 (31%)   |

Per-class (mean +/- std across folds):

| Class  | Precision       | Recall          | F1              | Support |
|:-------|:----------------|:----------------|:----------------|--------:|
| stay   | 76.8% +/- 5.2%  | 87.9% +/- 4.2%  | 81.9% +/- 4.0%  |      56 |
| switch | 50.7% +/- 13.6% | 33.0% +/- 16.6% | 39.2% +/- 16.0% |      22 |
| **macro F1** | | | **60.5%** | |

- Overall accuracy 72%.
- Positive-class (switch) recall = 33%. Issue #78 targets >= 50% -- the
  grid's best config improves on the #76 collapse but **does not yet meet
  the acceptance bar**.
- The gap is consistent with the imbalance in the data (2.5:1 stay/switch
  at decision rounds); class-weighted/focal loss and threshold calibration
  are the natural next steps called out in #78.

## Decisions carried forward

1. **Winning hyper-parameters locked in `opt.yml`:**
   - `architecture = mlp+rnn+edge` (`add_rnn=True`, `add_edge_model=True`)
   - `hidden_size = 20`
   - `lr = 3e-3`
   - `n_cross_val = 8`
2. **Dropped `round_number` feature** from `x_encoding`, `shuffle_features`,
   and `ablate_features` -- zero delta across every grid config under both
   perturbations.
3. **Trained for 75 epochs** (down from 120) -- stops before the test-loss
   drift that starts just past the per-fold best epoch (mean ~79).
4. **Old per-architecture configs removed** (`mlp.yml`, `mlp_rnn.yml`,
   `mlp_rnn_feat.yml`) -- the grid search subsumes them and `opt.yml`
   captures the winner.

## What this report does NOT close

Switch recall is still below the #78 acceptance criterion (0.33 vs >= 0.5).
Next experiments should tackle the class imbalance directly
(class-weighted cross-entropy, focal loss, and/or decision-threshold
calibration) rather than further hyper-parameter search -- the grid
plateau around test log-loss 0.53 suggests the bottleneck is the loss
function, not architecture or capacity.
