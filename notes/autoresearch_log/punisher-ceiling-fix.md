# The punisher at the contribution ceiling (RCC)

## 1. Declaration

**Slot:** punisher -- both families (the `lin_multinomial` the frontier stacks use, copula-stamped, and the GNN punisher), one feature added to each.

**Parent:** PR #184 (`auto/punisher-current-contribution`, `[REBASELINE]`, at `01f966a`): both punishers condition on round t's contribution, RCE exists and is protected, the suite scores 22 rows. Branch `auto/punisher-ceiling-fix` is created from it and the PR opens with `--base auto/punisher-current-contribution`. Isolated remote dir `~/repros/ai-runs/punisher-ceiling` (delete when this PR closes).

**Base models.** Punisher: `artifacts/baselines/punishment_multinomial_current_contr.joblib` (features `contribution, prev_contribution, prev_punishment, round_number, is_first`, C = 1.0, CV log loss 1.3465) and its severity-copula copy `punishment_multinomial_current_contr_severity_copula.joblib` (rho 0.4273); GNN punisher `artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_current_contr/model/architecture_node+edge+rnn__dataset_50ep_doubled.pt` (`x_encoding = contribution, prev_contribution, prev_punishment (numeric), is_first (bool)`, CV log loss 1.1756). Contributor and switch slots untouched.

**Evaluation stacks (§3 under the parent rule of §9).** The current frontier stack `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch` (PR #181's contributor x joint-exodus GNN switch x the copula-stamped lin_multinomial; the gated one, judged in full) and the GNN-punisher reference `23_2g8a_self_gnn_contr_gnn_switch` (main's gnn x gnn, runs `lin_multinomial_self` and `gnn_self`; reported, the GNN punisher's own before/after). Both re-run with the 23-family protocol (seed 42, 100 episodes, 24 rounds) from the parent's `_curpun` configs with only the punisher paths, `output_dir` and `figure_name` changed (`_ceiling` suffix).

**Baseline (the parent's stage D scores at full precision, `plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun/evaluation/scores.csv`; both gates are judged against these).**

| row | score | band |
|---|---|---|
| **RCC** (declared target) | **1.529762347729513** | 1-2 |
| RCE (protected; bands 0-4 / 5-9 / 10-14 / 15-19 slopes +0.095 / +0.020 / -0.058 / -0.160, signs ++--) | 0.8942480256952102 | <= 1 |
| RPA | 0.6930181766394568 | <= 1 |
| RCB | 1.5453991289093794 | 1-2 |
| mean over 22 rows | **1.0357** | |
| gate-2 ceiling (mean x 1.10) | **1.1393** | |
| rows <= 1 | 13/22 | |

Reference stack (`23_2g8a_self_gnn_contr_gnn_switch_curpun`): `lin_multinomial_self` mean 1.7405, 13/22, RCC 1.6181, RPA 0.6838, RCE 0.9976 (+0.064 / +0.092 / +0.044 / -0.098); `gnn_self` mean 1.7094, 8/22, RCC 1.3005, RPA 0.8884, RCE 1.0048 (+0.073 / +0.050 / +0.013 / +0.029).

**Target row:** RCC (gate 1: a band improvement, 1-2 -> <= 1, i.e. RCC < 1.0). Watch rows: RPA (the punished-at-20 tail is part of the p_t | c_t distribution), RCB, PA/PB (punishment marginals, since the punish rate at 20 changes), and RCE (protected).

### Hypothesis

**The defect (established by the parent, not re-derived).** Real managers almost never punish a player who contributed 20 (P(p>0 | c_t = 20) = 0.038) but when they do they punish hard (E[p | p>0] = 7.0). After the lag fix both simulated punishers still punish full contributors three to four times too often and too lightly (teacher-forced 0.136 / 0.105, self-play 0.141 / 0.142; E[p | p>0] at 20 3.9-5.1). RCC's population is by definition the punished full contributors, so the simulation fabricates that population; RCC was the one row the re-baseline did not move.

**The diagnosis (verified on the human data in step 1 below).** The multinomial is linear in `contribution` on every class logit, and the GNN encodes `contribution` numerically; neither can express a discontinuity at exactly 20 and both interpolate the ceiling from the 15-19 band.

**Behavioural rationale (one sentence, §5):** the human manager treats "gave everything" as a category of its own -- almost never punished, and heavily when punished -- not as the top of a linear scale, so the punisher gets an explicit indicator that the player contributed the maximum; the row that should move is RCC, with RPA's tail as a watch item.

**The change.** A derived feature `contribution_max = I(c_t = 20)` on both families: in the linear pool (`build_feature_pool`, added to `CURRENT_VALUED` and `PUNISHMENT_LEGAL_CURRENT`, so it is legal for the punishment target and illegal for the contribution target like `contribution` itself) and as a bool tensor on the GNN data path (`create_torch_data_new` for training, `api_manager.create_data` for the simulation, derived from the filled `contribution` tensor in both so absent cells read False). A `contribution_zero` indicator was tested on the human data and dropped (below). Everything else -- data, split, seed, C, architecture, epochs, copula rho -- is identical to the parent's artifacts.

### Artifact naming contract

| what | path |
|---|---|
| baseline config | `configs/training/baselines/punishment/multinomial_ceiling.yml` (`multinomial_current_contr.yml` with `contribution_max` in B1; a second B1 set with `contribution_zero` so the grid records the comparison; the artifact is the rank-1 row) |
| baseline artifact | `artifacts/baselines/punishment_multinomial_ceiling.joblib` |
| severity-copula copy | `artifacts/baselines/punishment_multinomial_ceiling_severity_copula.joblib` (`punishment_copula_rho.py --roundtrip --stamp-rho 0.4273`; the `--stamp-rho` flag is new on this branch) |
| GNN config | `configs/training/artificial_humans/punishment/rnn_edge_50ep_doubled_ceiling.yml` (`rnn_edge_50ep_doubled_current_contr.yml` with `- etype: bool, name: contribution_max` after `contribution`) |
| GNN artifact dir | `artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_ceiling/` (model file `model/architecture_node+edge+rnn__dataset_50ep_doubled.pt`) |
| sim configs | `configs/simulation/manager_testing/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_ceiling.yml`, `23_2g8a_self_gnn_contr_gnn_switch_ceiling.yml`; sim dirs `plots/simulation/<same>` |
| tables | `plots/data_analysis/evaluation/punisher_ceiling_fix/` (`human_ceiling_logit.csv`, `mechanism_teacher_forced.csv`, `mechanism_selfplay.csv`, `before_after.csv`) |

## 2. Plan

| # | step | implementer | status |
|---|---|---|---|
| 1 | Confirm or refute the diagnosis on the human data: fit the punish-or-not decision with and without a max indicator (and a zero indicator), report coefficient, significance, LR test, fitted P(p>0 \| c = 20); `scripts/data_analysis/punisher_ceiling_check.py`. | Fable | done |
| 2 | Add the indicator to the feature pool and the punishment legal set (`handcrafted_grid.py`), to the GNN data path (`data.py`, `api_manager.py`), document it (`baseline_feature_defs.md`), tests on both paths (`test_punisher_current_contribution.py`). | Fable | done |
| 3 | Retrain the linear punisher locally (`multinomial_ceiling.yml`, 4-fold CV, seed 38381), save the rank-1 bundle; report CV against 1.3465. | Fable | done |
| 4 | Stamp the severity copula carrying rho = 0.4273 over unchanged (`--stamp-rho`); report the refit for the record. | Fable | done |
| 5 | Retrain the GNN punisher on Raven (`rnn_edge_50ep_doubled_ceiling.yml`); report CV against 1.1756. | Fable | done |
| 6 | Teacher-forced mechanism check of both new punishers against the parent's artifacts and the human row. | Fable | done |
| 7 | Re-run the two stacks with the new punishers, fetch, evaluate all 22 rows (`PYTHONPATH=<worktree>/src`), self-play mechanism table. | Fable | done |
| 8 | Judge under the gates with RCE protected; log; PR against the parent. | Fable | done |

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-18 | (baseline) parent's current-contribution copula punisher, PR #181 stack (`_curpun`) | RCC 1.5298, RPA 0.6930, RCB 1.5454, RCE 0.8942 | 13/22 | 1.0357 | baseline |
| | 2026-09-19 | **the ceiling indicator on the copula lin_multinomial**, PR 181 stack (`_ceiling`) | RCC **1.2969**, RPA 0.6620, RCB 1.6591, RCE 0.8823 | 14/22 | 1.0331 | **FAIL** (gate 1: RCC stays in band 1-2) | |
| 2026-09-18 | (baseline) parent's plain multinomial, main gnn x gnn (`_curpun`, run `lin_multinomial_self`) | RCC 1.6181, RPA 0.6838, RCB 1.0206, RCE 0.9976 | 13/22 | 1.7405 | baseline (reference) |
| | 2026-09-19 | the ceiling indicator on the plain multinomial, main gnn x gnn (`_ceiling`, run `lin_multinomial_self`) | RCC 1.4615, RPA 0.6626, RCB 1.1492, RCE 0.9900 | 13/22 | 1.6616 | reported (not gated) | |
| 2026-09-18 | (baseline) parent's GNN punisher, main gnn x gnn (`_curpun`, run `gnn_self`) | RCC 1.3005, RPA 0.8884, RCB 1.2133, RCE 1.0048 | 8/22 | 1.7094 | baseline (reference) |
| | 2026-09-19 | the ceiling indicator on the GNN punisher, main gnn x gnn (`_ceiling`, run `gnn_self`) | RCC 1.1109, RPA 0.8066, RCB 1.0886, RCE 0.9157 | 11/22 | 1.6603 | reported (not gated) | |

### Step 1: the diagnosis on the human data (measured)

`scripts/data_analysis/punisher_ceiling_check.py` on the mechanism check's rows (single copy, 50 games, `punishment_valid & contribution_valid`, 8,914 rows; 1,232 at c_t = 20, 809 at c_t = 0). Observed P(p>0): c = 20 0.038, 15-19 0.196, 1-4 0.466, c = 0 0.468; E[p | p>0]: c = 20 7.00 (n 47), 15-19 3.87, c = 0 10.15.

Logit of P(p>0) on the linear punisher's own features (`contribution, prev_contribution, prev_punishment, round_number, is_first`), cluster-robust SE by episode:

| model | `contribution` | `contribution_max` | `contribution_zero` | LR vs linear (df) | fitted P(p>0 \| c=20) | fitted P(p>0 \| 15-19) | fitted P(p>0 \| c=0) | CV binary log loss (5 episode folds) |
|---|---|---|---|---|---|---|---|---|
| linear | -0.175 (0.024) | -- | -- | -- | 0.105 | 0.167 | 0.556 | 0.5215 |
| + max | -0.144 (0.028) | **-1.739 (SE 0.609, z -2.86, p = 0.004)** | -- | 127.4 (1), p = 1.6e-29 | 0.038 | 0.212 | 0.524 | 0.5151 |
| + max + zero | -0.159 (0.025) | -1.603 (0.629, p = 0.011) | -0.460 (0.421, p = 0.27) | 148.1 (2) | 0.038 | 0.201 | 0.468 | 0.5165 |

The diagnosis holds: the linear model puts P(p>0 | c = 20) at 0.105 -- interpolated from the 15-19 band, which it in turn under-fits (0.167 vs 0.196) -- while the indicator model lands on the observed 0.038 and fits the 15-19 band at 0.212. The indicator carries -1.74 logit beyond the linear term (an odds ratio of 0.18), significant under episode clustering, and improves out-of-sample binary log loss (0.5215 -> 0.5151). The zero indicator does not (p = 0.27; the linear term already puts c = 0 at 0.52-0.56) and worsens the CV. Severity given punished (OLS of p on the same regressors over the p > 0 rows): `contribution_max` +4.63 (SE 1.47, p = 0.002) -- the punished full contributor is punished about 4.6 points harder than the linear trend predicts, the "rarely but hard" pattern in one coefficient.

The 31-class multinomial the artifact actually is (4-fold CV on the locked train split, the grid's folds, C = 1.0): linear 1.3465 (se 0.058), + max 1.3446 (se 0.062), + max + zero 1.3492 (se 0.066). Decision: the max indicator only (ties and near-ties go to the simpler model; the zero indicator is not supported by the decision fit or the CV).

### Step 3: the linear punisher retrained (measured)

`configs/training/baselines/punishment/multinomial_ceiling.yml` run as its header says: 4-fold CV on the locked train split, seed 38381, the grid's folds, C = 1.0. The rank-1 row is the set carrying `contribution_max`, with CV log loss **1.3446** against 1.3465 for the parent's `punishment_multinomial_current_contr`, and locked test **1.2234** against 1.2468 (the grid's floor is 1.3561). The `contribution_zero` set ranks below both at 1.3492, which is the second independent reason it was dropped -- the first was its p = 0.27 in step 1's decision fit. The saved bundle's features are `contribution, contribution_max, prev_contribution, prev_punishment, round_number, is_first`; artifact `artifacts/baselines/punishment_multinomial_ceiling.joblib`, CV table `data/baselines/punishment_cv_multinomial_ceiling.csv`.

The margin is small in absolute terms (0.0019 of log loss over 31 classes) and it should be: the indicator changes the model's behaviour on 1,232 of 8,914 rows, and on most of those the change is from "punish a little, sometimes" to "almost never punish", which a 31-class log loss barely registers. It is the mechanism table, not the CV, that shows the size of the behavioural change.

### Step 4: the severity copula, stamped rather than refit (measured)

`scripts/baselines/punishment_copula_rho.py --roundtrip --stamp-rho 0.4273` writes `artifacts/baselines/punishment_multinomial_ceiling_severity_copula.joblib`: the plain bundle's weights bit-identical after reload (the script's assert plus an explicit coef / intercept / scaler comparison) plus the `copula_*` keys, carrying **rho = 0.4273 unchanged** from the parent. The protocol freezes rho when only the marginal is retrained, so the carried value is the stamped one.

The refit on the new marginal is recorded and deliberately **not** stamped: rho **0.4300**, 95% CI [0.357, 0.521], round-trip gate PASS (max |bias| 0.008, tolerance 0.03), out-of-sample MLE on the test split 0.368. The refit moving 0.4273 -> 0.4300 is well inside its own interval, so nothing about the ceiling indicator disturbs the within-round co-movement of punishments.

One thing a reader will trip over: the stamped bundle carries **NaN for `copula_rho_se` and the confidence interval**. That is not an unfitted copula. Those two fields describe the refit, and the refit is not what was stamped; the stamped rho is a carried constant and has no standard error of its own. A successor reading the bundle should take the interval above from this log, not from the file.

`artifacts/baselines/punishment_multinomial_ceiling_severity_copula.joblib` is the artifact the frontier stack runs.

### Step 5: the GNN punisher retrained (measured)

`configs/training/artificial_humans/punishment/rnn_edge_50ep_doubled_ceiling.yml` on Raven, job **30317122**, 6 min 54 s on one A100 for the 5-fold CV plus the full fit (within the ~12 min budget; wandb `bc4yqh66`). Final-epoch CV log loss **1.1743** against 1.1756 for `rnn_edge_50ep_doubled_current_contr`; best-epoch 1.1742 +- 0.1126 over folds against 1.1755 +- 0.1027; globally best epoch 1249 in both. `punishment_baseline.py`'s `GNN_REF` stays at 1.1756 -- this branch does not claim a new reference on a 0.0013 move.

The gain is smaller than the linear one on a model whose loss is 0.17 lower, and that is the prediction of the functional argument rather than a disappointment. The multinomial is linear in `contribution` on every class logit and literally cannot bend at 20; the GNN's MLP head can already approximate a bend, so an explicit bool sharpens what it was approximating instead of adding an expression it lacked. The mechanism table below shows the same thing behaviourally: the GNN's ceiling rate was already the better of the two before the fix (0.105 against 0.136) and improves by less.

Artifact `artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_ceiling/` (model, metrics, confusion matrix; LFS).

### Step 6: the mechanism, teacher-forced (measured)

`scripts/data_analysis/punisher_mechanism_check.py` replaying the 50 single-copy human games, 8,914 valid rows, no simulation; each model sees the human history and never its own draws, under its own stored `default_values`. The linears run locally, the GNNs on Raven's login node. OLS is of predicted expected punishment on c_t and c_{t-1} over the 8,431 rows with a valid previous contribution; E[p | p>0] per band is sum E[p] / sum P(p>0). Table: `plots/data_analysis/evaluation/punisher_ceiling_fix/mechanism_teacher_forced.csv`.

| punisher | P(p>0 \| c_t=20) | P(p>0 \| c_t<=4) | (c_t=20, c_{t-1}<=4) / (c_t<=4, c_{t-1}=20) | E[p \| p>0] 0-4 / 5-9 / 10-14 / 15-19 / 20 | OLS c_t / c_{t-1} | NLL |
|---|---|---|---|---|---|---|
| human | **0.038** | 0.467 | 0.179 / 0.571 | 7.99 / 4.98 / 4.27 / 3.87 / **7.00** | **-0.242** / +0.067 | -- |
| lin, c_t only (parent) | 0.136 | 0.448 | 0.309 / 0.651 | 7.20 / 5.60 / 4.77 / 4.22 / 3.94 | -0.125 / -0.030 | 1.283 |
| lin + ceiling | **0.053** | 0.431 | 0.224 / 0.635 | 7.43 / 5.46 / 4.43 / 3.76 / **7.47** | **-0.125** / -0.031 | 1.271 |
| gnn, c_t only (parent) | 0.105 | 0.432 | 0.301 / 0.532 | 7.26 / 5.48 / 4.80 / 4.63 / 5.11 | -0.128 / -0.022 | 1.144 |
| gnn + ceiling | **0.063** | 0.424 | 0.245 / 0.468 | 7.37 / 5.44 / 4.70 / 4.49 / **6.00** | **-0.144** / -0.010 | 1.132 |

The indicator does what it was added for and close to nothing else. The punish rate at the ceiling goes 0.136 -> 0.053 on the linear punisher and 0.105 -> 0.063 on the GNN, against the human 0.038, and the severity there goes 3.94 -> 7.47 and 5.11 -> 6.00 against the human 7.00. Rarely but hard, in both families, from one bool. Every band away from the ceiling moves by at most a few hundredths, and the NLL improves in both (1.283 -> 1.271, 1.144 -> 1.132), so the gain is not bought by damage elsewhere.

**The slope on the current contribution does not move**: -0.125 -> -0.125 on the linear punisher, -0.128 -> -0.144 on the GNN, against the human -0.242. The indicator fixes the break at 20. It does not fix the slope and was never designed to; see the note in section 4.

### Step 7: self-play and the 22 rows (measured)

Both stacks re-run with the 23-family protocol (seed 42, 100 episodes, 24 rounds) from the isolated remote dir `~/repros/ai-runs/punisher-ceiling`: Raven jobs **30317199** (frontier, 2 min 23 s) and **30317457** (reference, 2 min 25 s), one A100 each, then evaluated locally with the merged 22-row suite (500 repeats, seed 42). Sim dirs `plots/simulation/<stack>_ceiling/`; tables in `plots/data_analysis/evaluation/punisher_ceiling_fix/` (`mechanism_selfplay.csv`, `before_after.csv`, `before_after.md`, built by `scripts/data_analysis/punisher_ceiling_table.py`).

**Self-play mechanism table** (realised punishments over each sim's 19,200 agent-rounds, the same statistics as the teacher-forced table; the human row is the observed data on 8,914 rows):

| sim | P(p>0 \| c_t=20) | P(p>0 \| c_t<=4) | (c_t=20, c_{t-1}<=4) / (c_t<=4, c_{t-1}=20) | E[p \| p>0] 0-4 / 5-9 / 10-14 / 15-19 / 20 | OLS c_t / c_{t-1} |
|---|---|---|---|---|---|
| human | **0.038** | 0.467 | 0.179 / 0.571 | 7.99 / 4.98 / 4.27 / 3.87 / **7.00** | **-0.242** / +0.067 |
| frontier before | 0.122 | 0.473 | 0.167 / 0.612 | 7.05 / 5.83 / 4.88 / 3.92 / 3.84 | -0.142 / -0.019 |
| frontier after | **0.040** | 0.463 | 0.177 / 0.585 | 7.03 / 5.69 / 4.54 / 3.61 / **8.02** | **-0.143** / -0.015 |
| ref_lin before | 0.141 | 0.476 | 0.217 / 0.592 | 7.20 / 5.55 / 4.65 / 3.89 / 4.24 | -0.118 / -0.032 |
| ref_lin after | **0.059** | 0.451 | 0.141 / 0.515 | 7.42 / 5.47 / 4.30 / 3.67 / **7.98** | **-0.120** / -0.027 |
| ref_gnn before | 0.142 | 0.523 | 0.261 / 0.585 | 7.67 / 6.17 / 5.44 / 5.79 / 5.11 | -0.170 / -0.017 |
| ref_gnn after | **0.069** | 0.521 | 0.171 / 0.442 | 7.59 / 6.08 / 5.20 / 5.51 / **6.24** | **-0.187** / -0.007 |

Closed loop the fix lands harder than teacher-forced, because the contributors respond to it. P(p>0 | c_t = 20) on the frontier stack goes 0.122 -> **0.040** against the human 0.038 -- right to a thousandth, where the parent left it three and a half times too high -- and 0.141 -> 0.059, 0.142 -> 0.069 on the two references. E[p | p>0] at the ceiling goes 3.84 -> 8.02, 4.24 -> 7.98 and 5.11 -> 6.24 against the human 7.00, so the two linear punishers now slightly overshoot the human severity instead of undershooting it by half. The cross-tab moves onto the human ordering (0.177 / 0.585 against 0.179 / 0.571 on the frontier stack) and every non-ceiling band stays within a tenth or two of where the parent put it. **The OLS slope on c_t is unchanged in all three: -0.142 -> -0.143, -0.118 -> -0.120, -0.170 -> -0.187, against the human -0.242.**

**The 22 rows, frontier stack (the gated one).**

| row | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.8602 | 0.8563 | -0.0039 | <= 1 |
| CB | 0.7881 | 0.7941 | +0.0060 | <= 1 |
| CC | 0.8890 | 0.8958 | +0.0068 | <= 1 |
| CD | 0.8100 | 0.7949 | -0.0151 | <= 1 |
| CE | 1.0574 | 1.0579 | +0.0005 | 1-2 |
| CF | 0.8281 | 0.8169 | -0.0111 | <= 1 |
| CG | 1.5535 | 1.7588 | +0.2052 | 1-2 |
| SA | 0.7852 | 0.7687 | -0.0164 | <= 1 |
| SB | 1.0063 | 1.0105 | +0.0042 | 1-2 |
| SC | 1.4271 | 1.4632 | +0.0361 | 1-2 |
| PA | 0.6596 | 0.6526 | -0.0070 | <= 1 |
| PB | 0.9687 | 0.9558 | -0.0128 | <= 1 |
| PC | 0.9073 | 0.9349 | +0.0276 | <= 1 |
| PD | 0.7224 | 0.7598 | +0.0374 | <= 1 |
| RCA | 1.6327 | 1.6526 | +0.0198 | 1-2 |
| RCB | 1.5454 | 1.6591 | +0.1137 | 1-2 |
| **RCC** (target) | **1.5298** | **1.2969** | **-0.2329** | 1-2 (no upgrade) |
| RCD | 1.3091 | 1.2515 | -0.0576 | 1-2 |
| RCE (protected) | 0.8942 | 0.8823 | -0.0119 | <= 1 |
| RSA | 1.0701 | 0.9653 | -0.1048 | 1-2 -> <= 1 |
| RPA | 0.6930 | 0.6620 | -0.0310 | <= 1 |
| RPB | 0.8473 | 0.8380 | -0.0093 | <= 1 |
| **mean** | **1.0357** | **1.0331** | -0.0026 | |
| rows <= 1 | 13 | 14 | +1 | |

**The 22 rows, reference stack, plain multinomial (`lin_multinomial_self`).**

| row | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.9693 | 0.9572 | -0.0121 | <= 1 |
| CB | 0.8501 | 0.8310 | -0.0191 | <= 1 |
| CC | 1.8205 | 1.7150 | -0.1055 | 1-2 |
| CD | 0.8426 | 0.8240 | -0.0185 | <= 1 |
| CE | 1.3107 | 1.2212 | -0.0896 | 1-2 |
| CF | 0.8914 | 0.8988 | +0.0074 | <= 1 |
| CG | 10.2823 | 9.5199 | -0.7624 | > 5 |
| SA | 0.8208 | 0.8279 | +0.0071 | <= 1 |
| SB | 0.8375 | 0.8341 | -0.0033 | <= 1 |
| SC | 3.4400 | 3.0351 | -0.4049 | 2-5 |
| PA | 0.6479 | 0.6389 | -0.0090 | <= 1 |
| PB | 0.8080 | 0.8018 | -0.0062 | <= 1 |
| PC | 0.7643 | 0.7709 | +0.0066 | <= 1 |
| PD | 3.2046 | 2.7342 | -0.4704 | 2-5 |
| RCA | 1.9382 | 1.9767 | +0.0384 | 1-2 |
| RCB | 1.0206 | 1.1492 | +0.1286 | 1-2 |
| RCC | 1.6181 | 1.4615 | -0.1566 | 1-2 |
| RCD | 2.9329 | 3.0472 | +0.1142 | 2-5 |
| RCE | 0.9976 | 0.9900 | -0.0075 | <= 1 |
| RSA | 0.9439 | 0.9855 | +0.0416 | <= 1 |
| RPA | 0.6838 | 0.6626 | -0.0212 | <= 1 |
| RPB | 0.6649 | 0.6724 | +0.0075 | <= 1 |
| **mean** | **1.7405** | **1.6616** | -0.0789 | |
| rows <= 1 | 13 | 13 | 0 | |

**The 22 rows, reference stack, GNN punisher (`gnn_self`).**

| row | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.7884 | 0.7880 | -0.0004 | <= 1 |
| CB | 0.6814 | 0.7104 | +0.0290 | <= 1 |
| CC | 1.5694 | 1.5733 | +0.0039 | 1-2 |
| CD | 0.6637 | 0.6892 | +0.0255 | <= 1 |
| CE | 1.3009 | 1.4189 | +0.1181 | 1-2 |
| CF | 0.8087 | 0.8339 | +0.0252 | <= 1 |
| CG | 9.2206 | 9.3129 | +0.0922 | > 5 |
| SA | 0.8825 | 0.6718 | -0.2107 | <= 1 |
| SB | 0.8998 | 0.8398 | -0.0600 | <= 1 |
| SC | 2.9731 | 2.8059 | -0.1672 | 2-5 |
| PA | 1.3090 | 1.0599 | -0.2491 | 1-2 |
| PB | 1.1502 | 0.9919 | -0.1583 | 1-2 -> <= 1 |
| PC | 0.9947 | 0.9017 | -0.0930 | <= 1 |
| PD | 2.6898 | 3.1110 | +0.4212 | 2-5 |
| RCA | 2.0863 | 2.0314 | -0.0549 | 2-5 |
| RCB | 1.2133 | 1.0886 | -0.1246 | 1-2 |
| RCC | 1.3005 | 1.1109 | -0.1896 | 1-2 |
| RCD | 2.8464 | 2.8239 | -0.0225 | 2-5 |
| RCE | 1.0048 | 0.9157 | -0.0891 | 1-2 -> <= 1 |
| RSA | 1.0037 | 0.8890 | -0.1147 | 1-2 -> <= 1 |
| RPA | 0.8884 | 0.8066 | -0.0818 | <= 1 |
| RPB | 1.3317 | 1.1513 | -0.1804 | 1-2 |
| **mean** | **1.7094** | **1.6603** | -0.0492 | |
| rows <= 1 | 8 | 11 | +3 | |

**RCE band slopes, each with its own standard error and row count.** The protected-row rule is stated on slope magnitudes, so a band whose baseline slope is small can trip it on a change that is inside one seed's sampling error; the standard errors are here so a reader can tell erosion from noise. `change_in_se` is the before-to-after change divided by the pooled standard error of the two slopes.

| stack | band | human | before | after | change_in_se |
|---|---|---|---|---|---|
| frontier (gated) | 0-4 | +0.140 +- 0.018 (n 965) | +0.095 +- 0.014 (n 1873) | +0.087 +- 0.014 (n 1918) | 0.40 |
| frontier (gated) | 5-9 | +0.104 +- 0.024 (n 929) | +0.020 +- 0.013 (n 1997) | +0.038 +- 0.013 (n 2098) | 0.95 |
| frontier (gated) | 10-14 | -0.077 +- 0.035 (n 560) | -0.058 +- 0.022 (n 1221) | -0.043 +- 0.023 (n 1307) | 0.49 |
| frontier (gated) | 15-19 | -0.161 +- 0.079 (n 206) | -0.160 +- 0.064 (n 361) | -0.130 +- 0.067 (n 448) | 0.32 |
| ref_lin | 0-4 | +0.140 +- 0.018 (n 965) | +0.064 +- 0.016 (n 1626) | +0.067 +- 0.017 (n 1580) | 0.12 |
| ref_lin | 5-9 | +0.104 +- 0.024 (n 929) | +0.092 +- 0.015 (n 1801) | +0.094 +- 0.015 (n 1837) | 0.10 |
| ref_lin | 10-14 | -0.077 +- 0.035 (n 560) | +0.044 +- 0.021 (n 1332) | +0.036 +- 0.022 (n 1471) | 0.27 |
| ref_lin | 15-19 | -0.161 +- 0.079 (n 206) | -0.098 +- 0.070 (n 407) | -0.014 +- 0.061 (n 542) | 0.90 |
| ref_gnn | 0-4 | +0.140 +- 0.018 (n 965) | +0.073 +- 0.014 (n 2130) | +0.084 +- 0.014 (n 2079) | 0.56 |
| ref_gnn | 5-9 | +0.104 +- 0.024 (n 929) | +0.050 +- 0.013 (n 2123) | +0.060 +- 0.013 (n 2088) | 0.54 |
| ref_gnn | 10-14 | -0.077 +- 0.035 (n 560) | +0.013 +- 0.019 (n 1375) | -0.001 +- 0.020 (n 1414) | 0.49 |
| ref_gnn | 15-19 | -0.161 +- 0.079 (n 206) | +0.029 +- 0.046 (n 443) | +0.017 +- 0.041 (n 474) | 0.19 |

Protected-row checks, verbatim from the script:

- frontier (gated): `band_downgrade=False, sign_lost=[], magnitude_halved=[], signs ++-- -> ++--` -- **passes**.
- ref_lin (reported, not gated): `magnitude_halved=['15-19']`, signs `+++-` unchanged.
- ref_gnn (reported, not gated): `magnitude_halved=['10-14']`, signs `++++` -> `++-+`.

Both reference firings are discussed in section 4; neither is on the gated stack and neither is outside one standard error.

### Step 8: verdict -- `[FAIL]`

| gate | criterion | baseline | result | outcome |
|---|---|---|---|---|
| 1 (declared target) | RCC improves a band, 1-2 -> <= 1, i.e. RCC < 1.0 | 1.5298 | **1.2969**, still band 1-2 | **FAIL** |
| 2 | 22-row mean at or under 1.1393 | 1.0357 | 1.0331 | pass |
| protected | RCE: no band drop, no lost human sign, no slope magnitude at or below half | 0.8942, slopes +0.095 / +0.020 / -0.058 / -0.160 | 0.8823, +0.087 / +0.038 / -0.043 / -0.130 | pass |

**The branch fails.** RCC had to reach 1.0 and reached 1.2969. Gate 2 and the protected row both pass, and no combination of them rescues gate 1: the declared target is a band upgrade on a single row, and the row did not upgrade.

What is worth recording anyway, because it is measured and it is new. RCC moved **-0.2329**, the largest move that row has ever had on this stack -- the parent's whole re-baseline moved it between -0.11 and +0.08 across six stacks, which is why the parent listed it as the one row the punisher fix did not touch. The mechanism the hypothesis named is now essentially correct: the ceiling punish rate is 0.040 against the human 0.038, where it was 0.122. The 22-row mean is flat to three decimals, rows <= 1 goes 13 -> 14, and the GNN reference stack picks up three rows and drops its mean by 0.049. The hypothesis about the *mechanism* was right; the prediction that fixing it would carry RCE's sibling row across a band was wrong, and section 4 says why.

## 4. Notes

**Why RCC did not clear the band, measured rather than guessed.** RCC is the contrast in next-round contribution change between punished and unpunished full contributors. Decomposing it into its two means and their populations says exactly where the remaining distance sits:

| | contrast | dc, punished | n punished | dc, unpunished | n unpunished | punished share |
|---|---|---|---|---|---|---|
| human | **-7.035** | **-8.659** | 44 | -1.624 | 1096 | 3.9% |
| frontier before | -0.973 | -2.704 | 277 | -1.731 | 1964 | 12.4% |
| frontier after | **-1.978** | **-3.747** | 91 | -1.770 | 2178 | 4.0% |
| ref_lin after | -1.303 | -3.270 | 148 | -1.967 | 2292 | 6.1% |
| ref_gnn after | -2.841 | -4.718 | 170 | -1.877 | 2254 | 7.0% |

The fix did its half completely. The punished share at the ceiling was 12.4% against the human 3.9% and is now 4.0%; the fabricated population this experiment set out to remove is gone, and the dose those players receive is right too (E[p | p>0] at 20 is 8.02 against the human 7.00). What is left is the other half of the contrast, and it is not the punisher's: a punished full contributor in the simulation drops 3.75 the next round where a human drops 8.66. The contributor model under-reacts to a heavy punishment at the ceiling by a factor of about 2.3, and no change to the punisher can move that number -- the punisher chooses who gets punished and how hard, the contributor chooses the response. RCC is the only row in the suite that measures the ceiling dose-response, because RCE's population is the punished *non-full* contributors by construction, so this defect has nowhere else to show up. **The remaining RCC gap is a contributor-slot defect and should be declared against the contributor, not the punisher.**

**The slope on the current contribution is a separate defect that this experiment does not address.** The parent flagged it and it is unchanged here: the punisher's OLS response to c_t is -0.125 teacher-forced and -0.143 in self-play, against the human -0.242, so the simulated manager's punishment falls with contribution at a little over half the human rate. The ceiling indicator is a single bool at c = 20 and changes this by at most 0.016 in any condition, which is what section 3's step 6 shows. A successor must not read "the punisher was fixed at the ceiling" as "the punisher's contribution response was fixed". It was not. It is a live, separately declarable defect, and it plausibly drives part of RCB (which got *worse* on this branch, 1.5454 -> 1.6591, precisely because the ceiling rows left its numerator while the slope that shapes the rest stayed wrong).

**The protected-row rule fired on the two reference stacks; report and read carefully.** RCE is protected on the gated frontier stack and passes there cleanly. The rule also runs on the two reference stacks, which are reported rather than gated, and fires on both:

- `ref_lin`, band 15-19: -0.098 -> -0.014, magnitude below half. The change is **0.90 pooled standard errors** on slopes whose own standard errors are 0.070 and 0.061 over 407 and 542 rows. On one seed this is not distinguishable from noise.
- `ref_gnn`, band 10-14: +0.013 -> -0.001, magnitude below half. The change is **0.49 pooled standard errors**. Here the rule fires on what is arguably an *improvement*: the human slope in that band is -0.077, the before slope had the wrong sign at +0.013, and the after slope crosses zero to -0.001, moving toward the human. A relative-magnitude rule reads a slope passing through zero in the right direction as an erosion. That is a rule artefact, not a finding.

Both firings land on bands the suite barely constrains. PR #183 established that the 10-14 band is never learned -- wrong sign in every condition, teacher-forced and held out -- and the 15-19 band has the fewest rows of the four in every sim (361-542) and the widest standard error. A fixed relative threshold on a band whose baseline magnitude is 0.013 or 0.058 will fire on noise more often than on erosion. **This does not change the verdict**, which fails on gate 1 regardless and passes the protected row on the stack where it is gated; it is recorded so the maintainer can decide whether the protected-row rule needs an absolute floor or a significance test before it is applied to a band this thin.

**What went the wrong way.** On the frontier stack, RCB +0.114, CG +0.205, PD +0.037, SC +0.036, RCA +0.020. RCB and CG are the two to watch: RCB is a rate with `20 - c_t` in the denominator, so removing most of the punishments at c_t = 20 removes the rows where that denominator is smallest, and the row is now shaped entirely by the slope defect above. CG rises on the frontier stack while falling by 0.76 on the linear reference, which is the contributor and switch models meeting a punisher that behaves differently from the one they were accepted against -- the same systematic effect the parent documented for the re-baseline, at a tenth of the size.

**What this leaves for a successor.**

1. **The contributor's reaction to a heavy punishment at the ceiling is the live defect, and it is now cleanly isolated.** The punisher side of RCC is solved and measured; the residual is `dc | punished, c_t = 20` at -3.75 against the human -8.66 on a population that is now the right size. A successor declaring against the contributor slot can use RCC as its target row with the punisher held at this branch's artifacts, and has the decomposition table above as its baseline. Whether this branch's artifacts are worth merging for that purpose despite the failed gate is a maintainer call; the mechanism evidence says yes, the gate says no.
2. **The punisher's contribution slope is the second live defect**, at roughly half the human magnitude, untouched by anything on this branch or the parent. It wants a different change -- a non-linear response in c_t, not another indicator -- and RCB is the row it would most plausibly move.
3. **The protected-row rule wants an absolute floor.** Two of three stacks tripped it on bands with baseline magnitudes of 0.013 and 0.058, both inside one standard error, and one of the two firings was on a slope moving toward the human. Reporting slopes with their standard errors, as this log now does, should probably be the default for any experiment that touches RCE.
4. **Do not re-derive step 1.** The human-data diagnosis (indicator -1.739, SE 0.609, p 0.004, LR 127.4 on 1 df, fitted rate 0.038 against the linear model's 0.105) is settled, as is the rejection of the gave-nothing indicator.
5. **Housekeeping.** The isolated remote dir `~/repros/ai-runs/punisher-ceiling` can be deleted when this PR closes. `punishment_baseline.py`'s `GNN_REF` is deliberately left at 1.1756 rather than 1.1743.
