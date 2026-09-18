# Held-out teacher-forced RCB test: is the punishment response learned or memorised?

Branch `rcb-holdout-teacher-forced`, based on `origin/auto/contribution-punishment-response` (PR #181). A measurement, not an experiment: no model is proposed, nothing is evaluated in a stack, and no §2 gate applies.

## 1. The question

The RCB row ("reaction to punishment": mean next-round contribution change of punished non-full contributors, in four punishment-rate bins) scores 2.32 on the current best GNN contribution trunk (PR #179, `group_switching_contribution_50ep_group_vnode`). PR #181 measured the same trunk **teacher-forced** on the human trajectories and found an RCB statistic of 0.093, inside the 0.348 human-vs-human noise ceiling, against 0.797 in self-play, and read the gap as state drift: the conditional response is fine, the closed loop's states are not.

That 0.093 is an **in-sample** number. The shipped artifact is trained on all 50 games (flip-doubled), and a GRU over memorised trajectories could reproduce observed bin means without having learned a response that transfers. The test that separates "learned but lost in the closed loop" from "never learned, only memorised" is the held-out teacher-forced measurement: retrain the trunk with one CV fold held out, teacher-force it on the games it never saw, and compare with the same model on its own training games.

Quantities (all on the canonical single-copy human frame, one copy per game, model column = teacher-forced E[c_{t+1}] − c_t over the RCB population, the construction of `scripts/data_analysis/rcb_teacher_forced.py`):

- the RCB statistic (human-frequency-weighted mean |bin mean − human bin mean| over the four rate bins) and its score (statistic / 0.3479);
- the four RCB bin means (human: 0.892 / 1.341 / 1.679 / 2.014);
- the within-contribution-band OLS slopes of dc on punishment received, bands 0–4 / 5–9 / 10–14 / 15–19 (human: +0.140 / +0.104 / −0.077 / −0.161).

## 2. Setup

**Model under test.** The PR #179 bare trunk (group vnode, no stimulus skip): `configs/training/artificial_humans/contribution/group_switching_contribution_50ep_group_vnode.yml`, artifact `artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt`. Confirmed as PR #181's parent from its log (`contribution-punishment-response.md`, Declaration: the base model is the herding-copula stamp of exactly this trunk; the copula affects sampling only, so the bare trunk is the right object for a teacher-forced measurement). Secondary model, run after the primary result: the PR #181 stimulus-skip trunk (`..._50ep_vnode_stimulus_skip.yml`, shipped artifact `..._50ep_vnode_stimulus_skip/model/...575.pt`).

**Folds.** `configs/training/artificial_humans/contribution/group_switching_contribution_50ep_group_vnode_holdout_folds.yml` (and `..._vnode_stimulus_skip_holdout_folds.yml`): the parent config byte-for-byte, plus `holdout_fold: k` swept over k = 0..4 by the config grid, so `run.py` submits five parallel SLURM jobs labelled `fold_0..fold_4`. The mechanism, read from the code: `train.py` seeds `random` with the config seed (38381) before building the tensors; `get_cross_validations` (`src/aimanager/generic/data.py`) shuffles the episode index once, splits the flip pairs round-robin into five folds via `group_key=pair_id` (both copies of a game always land in the same fold), and only then looks at `holdout_fold`; with it set, it yields the single (train = other four folds, test = fold k) split with fold id `None`, which is the id `train.py` saves a model under. So each fold job trains on 40 games (80 doubled episodes) and saves that model, and fold k here is byte-identically fold k of the shipped artifact's own 5-fold CV (same seed). Every game is held out exactly once (asserted in the analysis script).

**Verification of the partition.** The analysis script (`scripts/data_analysis/rcb_holdout_teacher_forced.py`) rebuilds each fold's held-out episodes through `get_cross_validations` itself, then teacher-forces the fold model on them and compares the mean NLL with the held-out `log_loss` the training job recorded in its own metrics parquet. They agree within 1e-4 for every GPU fold and exactly for every CPU fold, so the games called held-out below are the games the model never saw.

**Hyperparameters.** Unchanged from the shipped artifact: node+edge+rnn, hidden 20, lr 3e-4, weight decay 1e-5, 575 epochs, batch 4, grad clamp 1, seed 38381, same `x_encoding` (`prev_contribution` numeric, `prev_punishment` numeric, `agent_group` one-hot), same data file and mask.

**Where it ran; the budget.** Raven, in the isolated dir `~/repros/ai-runs/rcb-holdout` (`AI_REMOTE_DIR`, `notes/autoresearch.md` §9; synced with `train_cluster.sh --sync-only`, submitted with the same command `train_cluster.sh` builds plus `WANDB_MODE=disabled` exported into the sbatch environment, because the fresh checkout's `.env` carries a placeholder `WANDB_API_KEY` that `train.py` would otherwise treat as a live key). Ten one-training jobs, SLURM 30302775–30302779 (vnode) and 30302781–30302786 (skip), each **2m25s** on an A100. The shipped six-training job takes 8m43s–11m12s (PR #181 note 7), so this is far inside the 3x iteration budget. Fold artifacts are committed under `artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode_holdout/` and `..._vnode_stimulus_skip_holdout/` (LFS, ~35 kB each, the repo's convention for GNN artifacts); the remote copies stay in the isolated dir.

**Local CPU cross-check.** Raven was unavailable when the test started, so the identical configs were first trained locally through `train.main` (device `cpu`, same job dicts `run.py` generates; 49 s per fold). PyG is not installable on macOS, so the local runs use the repo venv plus a scratchpad `torch_geometric` 2.5.0 and a pure-torch `scatter_mean` stand-in for `torch_scatter`; that stand-in reproduces PR #181's Raven-computed in-sample numbers for the shipped artifact to every printed decimal (0.09301064, slopes +0.1157 / +0.0791 / +0.0425 / −0.1659, self-check PASS). Because the seed fixes the batch order, the CPU and GPU fold models agree to three decimals on every pooled quantity (pooled held-out statistic 0.0953 CPU vs 0.0951 GPU; skip 0.1122 vs 0.1120); their summaries are committed as `plots/data_analysis/rcb_holdout_teacher_forced/*_local_cpu.csv`, the CPU artifacts themselves are not. A local full-data retrain (no folds) lands at in-sample 0.1029 with slopes +0.1146 / +0.0708 / +0.0437 / −0.2175 (shipped GPU artifact: 0.0930, +0.1157 / +0.0791 / +0.0425 / −0.1659): a retrain of the same config sits in the same regime, and the 15–19 slope is the least stable quantity between two trainings.

## 3. Results

RCB score = statistic / 0.3479. "human" rows are the observed human dc on exactly the same stimulus rows as the model row above them; on 10 games (~500 RCB rows) these are themselves noisy and are the honest yardstick for a fold's held-out row. Per-fold held-out test log loss of the GPU fold jobs: 2.0224 / 2.0751 / 2.0017 / 2.0337 / 1.9422, against the shipped artifact's own CV folds 2.0224 / 2.0574 / 2.0166 / 2.0474 / 1.8990.

### 3a. Primary model (group vnode), Raven GPU folds

`plots/data_analysis/rcb_holdout_teacher_forced/group_vnode_raven_gpu.csv`.

| set | fold | source | n | RCB stat | score | bin (0,.25] | (.25,.5] | (.5,1] | >1 | slope 0–4 | 5–9 | 10–14 | 15–19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped full-data, in-sample (d) | all | model | 2660 | 0.0930 | 0.267 | 0.773 | 1.341 | 1.524 | 1.917 | +0.116 | +0.079 | +0.043 | −0.166 |
| human reference | all | human | 2660 | 0 | 0 | 0.892 | 1.341 | 1.679 | 2.014 | +0.140 | +0.104 | −0.077 | −0.161 |
| held-out | 0 | model | 528 | 0.0867 | 0.249 | 0.841 | 1.477 | 1.665 | 1.763 | +0.030 | +0.119 | +0.047 | −0.316 |
| held-out | 0 | human | 528 | 0.1179 | 0.339 | 1.009 | 1.410 | 1.461 | 1.944 | +0.112 | +0.160 | −0.042 | −0.821 |
| in-sample | 0 | model | 2132 | 0.0892 | 0.256 | 0.791 | 1.316 | 1.536 | 1.913 | +0.126 | +0.083 | +0.038 | −0.193 |
| in-sample | 0 | human | 2132 | 0.0274 | 0.079 | 0.865 | 1.318 | 1.720 | 2.031 | +0.147 | +0.084 | −0.083 | −0.124 |
| held-out | 1 | model | 552 | 0.1804 | 0.519 | 0.790 | 1.541 | 1.569 | 2.617 | +0.123 | +0.097 | +0.029 | −0.342 |
| held-out | 1 | human | 552 | 0.2695 | 0.775 | 0.700 | 1.590 | 1.670 | 3.128 | +0.167 | +0.127 | −0.231 | −0.078 |
| in-sample | 1 | model | 2108 | 0.0688 | 0.198 | 0.881 | 1.342 | 1.535 | 1.650 | +0.113 | +0.065 | +0.027 | −0.215 |
| in-sample | 1 | human | 2108 | 0.0658 | 0.189 | 0.948 | 1.279 | 1.681 | 1.787 | +0.130 | +0.099 | −0.060 | −0.187 |
| held-out | 2 | model | 503 | 0.4070 | 1.170 | 0.460 | 1.056 | 1.509 | 3.015 | +0.149 | +0.140 | +0.016 | −0.000 |
| held-out | 2 | human | 503 | 0.2756 | 0.792 | 0.990 | 1.067 | 0.965 | 1.688 | +0.037 | +0.029 | −0.383 | +0.041 |
| in-sample | 2 | model | 2157 | 0.1176 | 0.338 | 0.772 | 1.404 | 1.548 | 1.800 | +0.141 | +0.083 | +0.027 | −0.206 |
| in-sample | 2 | human | 2157 | 0.0760 | 0.219 | 0.874 | 1.409 | 1.905 | 2.113 | +0.171 | +0.119 | −0.025 | −0.180 |
| held-out | 3 | model | 473 | 0.3726 | 1.071 | 1.249 | 1.161 | 2.114 | 2.817 | +0.152 | +0.013 | +0.149 | −0.023 |
| held-out | 3 | human | 473 | 0.7341 | 2.110 | 1.233 | 1.488 | 3.567 | 3.957 | +0.254 | +0.193 | +0.132 | −0.897 |
| in-sample | 3 | model | 2187 | 0.1882 | 0.541 | 0.782 | 1.388 | 1.379 | 1.324 | +0.078 | +0.049 | +0.010 | −0.199 |
| in-sample | 3 | human | 2187 | 0.1474 | 0.424 | 0.794 | 1.320 | 1.367 | 1.617 | +0.123 | +0.090 | −0.112 | −0.124 |
| held-out | 4 | model | 604 | 0.3452 | 0.992 | 0.692 | 1.784 | 1.332 | 1.260 | +0.126 | +0.091 | +0.096 | −0.080 |
| held-out | 4 | human | 604 | 0.4174 | 1.200 | 0.562 | 1.208 | 1.456 | 0.185 | +0.164 | +0.033 | −0.025 | −0.165 |
| in-sample | 4 | model | 2056 | 0.1250 | 0.359 | 0.834 | 1.373 | 1.859 | 2.573 | +0.117 | +0.094 | +0.099 | −0.119 |
| in-sample | 4 | human | 2056 | 0.1235 | 0.355 | 0.980 | 1.380 | 1.759 | 2.576 | +0.131 | +0.128 | −0.119 | −0.355 |
| **pooled held-out (c): 50 games, each by the model that never saw it** | all | model | 2660 | **0.0951** | **0.273** | 0.830 | 1.437 | 1.585 | 2.258 | +0.116 | +0.098 | +0.081 | −0.074 |
| pooled in-sample (c'): 50 games, each by the 4 models that saw it | all | model | 10640 | 0.0823 | 0.237 | 0.811 | 1.366 | 1.564 | 1.841 | +0.115 | +0.074 | +0.036 | −0.197 |

### 3b. Primary model, local CPU folds (cross-check)

`plots/data_analysis/rcb_holdout_teacher_forced/group_vnode_local_cpu.csv`. Pooled rows only; the per-fold rows differ from 3a in the third decimal at most.

| set | n | RCB stat | score | bin (0,.25] | (.25,.5] | (.5,1] | >1 | slope 0–4 | 5–9 | 10–14 | 15–19 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled held-out | 2660 | 0.0953 | 0.274 | 0.828 | 1.433 | 1.580 | 2.252 | +0.116 | +0.098 | +0.080 | −0.074 |
| pooled in-sample | 10640 | 0.0833 | 0.239 | 0.811 | 1.364 | 1.561 | 1.835 | +0.115 | +0.074 | +0.036 | −0.197 |

### 3c. Secondary model (stimulus skip, PR #181), Raven GPU folds

`plots/data_analysis/rcb_holdout_teacher_forced/stimulus_skip_raven_gpu.csv` (CPU cross-check in `stimulus_skip_local_cpu.csv`: pooled held-out 0.1122, pooled in-sample 0.0840). GPU fold held-out log loss 2.0345 / 2.0498 / 2.0131 / 2.0727 / 1.9215.

| set | fold | n | RCB stat | score | bin (0,.25] | (.25,.5] | (.5,1] | >1 | slope 0–4 | 5–9 | 10–14 | 15–19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped full-data, in-sample | all | 2660 | 0.1094 | 0.315 | 0.768 | 1.436 | 1.642 | 2.220 | +0.148 | +0.104 | +0.017 | −0.220 |
| held-out | 0 | 528 | 0.1247 | 0.358 | 1.004 | 1.426 | 1.859 | 2.194 | +0.063 | +0.124 | +0.038 | −0.290 |
| in-sample | 0 | 2132 | 0.1082 | 0.311 | 0.748 | 1.361 | 1.509 | 2.075 | +0.134 | +0.105 | +0.019 | −0.199 |
| held-out | 1 | 552 | 0.2219 | 0.638 | 0.820 | 1.764 | 1.743 | 2.686 | +0.127 | +0.108 | −0.001 | −0.268 |
| in-sample | 1 | 2108 | 0.0648 | 0.186 | 0.852 | 1.307 | 1.609 | 1.775 | +0.122 | +0.072 | +0.027 | −0.194 |
| held-out | 2 | 503 | 0.3764 | 1.082 | 0.538 | 1.246 | 1.890 | 3.455 | +0.172 | +0.152 | +0.023 | −0.014 |
| in-sample | 2 | 2157 | 0.1088 | 0.313 | 0.734 | 1.383 | 1.543 | 2.020 | +0.150 | +0.097 | +0.049 | −0.212 |
| held-out | 3 | 473 | 0.4188 | 1.204 | 1.249 | 1.267 | 2.109 | 3.525 | +0.147 | +0.092 | +0.200 | +0.089 |
| in-sample | 3 | 2187 | 0.2175 | 0.625 | 0.709 | 1.330 | 1.203 | 1.583 | +0.081 | +0.074 | +0.029 | −0.158 |
| held-out | 4 | 604 | 0.4083 | 1.174 | 0.660 | 1.790 | 1.176 | 1.076 | +0.111 | +0.079 | +0.086 | −0.086 |
| in-sample | 4 | 2056 | 0.1100 | 0.316 | 0.856 | 1.367 | 1.847 | 2.559 | +0.105 | +0.107 | +0.093 | −0.027 |
| **pooled held-out** | all | 2660 | **0.1120** | **0.322** | 0.873 | 1.521 | 1.699 | 2.532 | +0.129 | +0.111 | +0.083 | −0.087 |
| pooled in-sample | all | 10640 | 0.0829 | 0.238 | 0.779 | 1.349 | 1.532 | 1.993 | +0.118 | +0.090 | +0.040 | −0.179 |

### 3d. Slopes against the human reference (pooled rows, GPU folds)

| band | human | vnode shipped in-sample | vnode pooled in-sample | **vnode pooled held-out** | skip shipped in-sample | skip pooled in-sample | skip pooled held-out |
|---|---|---|---|---|---|---|---|
| 0–4 | +0.140 | +0.116 | +0.115 | **+0.116** | +0.148 | +0.118 | +0.129 |
| 5–9 | +0.104 | +0.079 | +0.074 | **+0.098** | +0.104 | +0.090 | +0.111 |
| 10–14 | −0.077 | +0.043 | +0.036 | **+0.081** | +0.017 | +0.040 | +0.083 |
| 15–19 | −0.161 | −0.166 | −0.197 | **−0.074** | −0.220 | −0.179 | −0.087 |

Closed-loop reference (PR #181, the same trunks in self-play): vnode +0.062 / +0.012 / −0.008 / −0.037, skip +0.073 / +0.054 / −0.047 / −0.025; RCB statistic 0.797 (vnode) and 0.718 (skip).

## 4. Verdict

**Learned and generalising, with one partial exception.** Held-out ≈ in-sample ≈ human on the quantity the RCB row scores: the pooled held-out statistic is 0.0951 (score 0.273) against 0.0823 in-sample and 0.0930 for the shipped artifact, all three inside the noise ceiling and within 0.013 of each other, and eight times smaller than the 0.797 the same trunk produces in self-play. The held-out bin means rise monotonically like the human ones. In the two low contribution bands, where 71% of the RCB population sits, the held-out slopes (+0.116, +0.098) equal or exceed the in-sample ones and reach 83–94% of the human values. The exception is the 15–19 band: the withdrawal response there is about 40% as strong out of sample (−0.074) as in-sample (−0.197), i.e. partially memorised; the 10–14 band has the wrong sign in every condition (in-sample, held-out, shipped, both trunks), i.e. never learned at all. The skip trunk tells the same story (held-out 0.112 vs in-sample 0.083, same band pattern).

**What this implies for the two candidate fixes.** The teacher-forced-vs-self-play gap (0.09 vs 0.80) is not an in-sample artefact: a model that never saw a game predicts its punishment response as well as it predicts its own training games. So PR #181's reading stands, and closed-loop drift fixes (gated stimulus skip, anything that keeps the recurrent state on the human manifold, or an evaluation of how the sim's punisher composition differs) remain the family that can move the RCB row, since the response is present out of sample and lost only in the loop. Generalisation fixes (features, regularisation, more data) cannot close a gap that is already absent held-out; they are relevant only to the two secondary deficits, the partially memorised 15–19 withdrawal and the missing 10–14 withdrawal, which together cover 29% of the RCB population and which no closed-loop fix will supply because the teacher-forced trunk does not have them either.

## 5. Notes

1. **Measured.** Every fold's reconstructed partition reproduces the fold job's own recorded held-out log loss (GPU: within 1e-4; CPU: exactly), every game is held out exactly once (asserted), and the observed-human column on the pooled held-out rows reproduces the declaration's human slopes and bin means to seven decimals: the model and human columns of every table are computed on identical rows.
2. **Measured.** Pooled held-out vs pooled in-sample vs shipped in-sample, vnode trunk, GPU folds: RCB statistic 0.0951 / 0.0823 / 0.0930; bin means (0.830, 1.437, 1.585, 2.258) / (0.811, 1.366, 1.564, 1.841) / (0.773, 1.341, 1.524, 1.917) against human (0.892, 1.341, 1.679, 2.014); slopes in 3d. The held-out `>1` bin overshoots the human mean (2.26 vs 2.01) where in-sample undershoots (1.84); the statistic is nonetheless within the ceiling.
3. **Measured.** Per-fold held-out statistics (0.087, 0.180, 0.407, 0.373, 0.345) sit within a factor of ~1.5 of the observed-human statistic on the same 10 games (0.118, 0.270, 0.276, 0.734, 0.417): on 10 games both sides of the comparison are noisy, and the pooled row is the one to read. The per-fold held-out 15–19 slopes are −0.32, −0.34, 0.00, −0.02, −0.08 (human on the same rows: −0.82, −0.08, +0.04, −0.90, −0.17).
4. **Inference.** The response the RCB row scores is learned, not memorised. The "0.093 is in-sample" objection is answered in the negative for the bin means and for the low-band slopes; the shipped artifact's teacher-forced number is what a held-out model produces too.
5. **Inference, with a caveat.** The 15–19 band's withdrawal is the one quantity that shrinks out of sample (−0.197 → −0.074) and it is also the least stable between two full-data trainings (shipped −0.166 vs CPU retrain −0.218); on 206 rows over five folds it cannot be pinned tighter here. It is where PR #181's skip candidate moved the closed loop least (−0.037 → −0.025 vs human −0.161) and where the RCB `>1` bin's composition is heaviest, so a regulariser or a feature aimed at that band could still pay, but it is second-order next to the closed-loop gap.
6. **Measured.** The 10–14 band slope is positive in every condition (shipped +0.043, in-sample +0.036, held-out +0.081; skip +0.017 / +0.040 / +0.083) against human −0.077: neither trunk carries the mid-contributor withdrawal, in-sample or out, consistent with PR #181's note 10. This is a capacity or feature issue independent of the closed loop.
7. **Measured.** The skip trunk is not better teacher-forced: shipped in-sample 0.1094 (vnode 0.0930), pooled held-out 0.1120 (vnode 0.0951); its low-band slopes are slightly stronger, its 15–19 held-out slope equally weak. Its closed-loop gain in PR #181 (0.797 → 0.718) therefore came from the loop, not from a better conditional, which is what the skip was designed for.
8. **Process.** Raven initially resolved to an account without a checkout; the local CPU pipeline (scratchpad PyG plus `scatter_mean` stand-in, validated against PR #181's numbers) produced the full result first and the GPU run then reproduced it to three decimals. `.env` on Raven was neither read nor modified; wandb was silenced through `WANDB_MODE=disabled` in the job environment.
