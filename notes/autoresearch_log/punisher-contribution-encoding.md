# The punisher's response to the current contribution (RPA, RCC)

## 1. Declaration

**Slot:** punisher -- both families (the copula-stamped `lin_multinomial` the frontier stack runs, and the GNN punisher), one change to each: how the current contribution is encoded.

**Parent:** PR #192 (`auto/punisher-ceiling-fix`, `[FAIL]` on its own gate but accepted by the maintainer as the new baseline, at `e230629`). Branch `auto/punisher-contribution-encoding` is created from it and the PR opens with `--base auto/punisher-ceiling-fix`. Isolated remote dir `~/repros/ai-runs/punisher-contr-bins` (delete when this PR closes).

**Base models.** Punisher: `artifacts/baselines/punishment_multinomial_ceiling.joblib` (features `contribution, contribution_max, prev_contribution, prev_punishment, round_number, is_first`, C = 1.0, CV log loss 1.3446) and its severity-copula copy `punishment_multinomial_ceiling_severity_copula.joblib` (rho 0.4273, stamped); GNN punisher `artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_ceiling/model/architecture_node+edge+rnn__dataset_50ep_doubled.pt` (`x_encoding = contribution (numeric), contribution_max (bool), prev_contribution, prev_punishment, is_first`, CV log loss 1.1743). Contributor and switch slots untouched.

**Evaluation stacks (§3 under the parent rule of §9).** The parent's two: the frontier stack `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch` (gated, judged in full) and the GNN-punisher reference `23_2g8a_self_gnn_contr_gnn_switch` (runs `lin_multinomial_self` and `gnn_self`; reported). Both re-run with the 23-family protocol (seed 42, 100 episodes, 24 rounds) from the parent's `_ceiling` configs with only the punisher paths, `output_dir` and `figure_name` changed (`_cbins` suffix).

**Baseline (the parent's confirmed scores, `plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_ceiling/evaluation/scores.csv`; both gates are judged against these).**

| row | score | band |
|---|---|---|
| **RPA** (declared target) | **0.6620** | <= 1 |
| **RCC** (declared target) | **1.2969** | 1-2 |
| RCE (protected) | 0.8823 | <= 1 |
| RCB | 1.6591 | 1-2 |
| mean over 22 rows | **1.0331** | |
| gate-2 ceiling (mean x 1.10) | **1.1364** | |
| rows <= 1 | 14/22 | |

Reference stack (`23_2g8a_self_gnn_contr_gnn_switch_ceiling`): `lin_multinomial_self` mean 1.6616, 13/22, RCC 1.4615, RPA 0.6626, RCE 0.9900; `gnn_self` mean 1.6603, 11/22, RCC 1.1109, RPA 0.8066, RCE 0.9157.

**Target rows:** RPA and RCC (gate 1: a band improvement on one of them). Watch rows: RCB (the row the parent's notes name as the slope's most plausible carrier), PA/PB/PC/PD (punishment marginals), and RCE (protected).

**A note on the declaration that the reader should have up front.** RPA already sits at 0.6620 on the gated stack, which is the best band there is (<= 1). A band upgrade on RPA is therefore not available, and gate 1 can only be carried by RCC on this stack -- the same single-row bar the parent failed. RPA is kept as a declared target because it is the row the hypothesis speaks to most directly and its movement is the honest read on whether the mechanism changed at all, but it cannot decide the gate. This is stated before the results rather than after them.

### Hypothesis

**The defect (established by the parent, not re-derived).** The parent fixed the punisher at the contribution ceiling: the punish rate at c_t = 20 went 0.122 -> 0.040 against the human 0.038 and the severity there 3.84 -> 8.02 against the human 7.00. It did not touch the other half of the same defect. The OLS weight of punishment on the current contribution is about -0.14 in every stack (frontier -0.143, ref_lin -0.120, ref_gnn -0.187) against the human -0.242, so the simulated manager's sensitivity to how much someone gave is roughly half a real manager's. The parent's own notes declare this a live, separately declarable defect.

**The diagnosis (to be confirmed on the human data in step 1).** Contribution enters both punishers as a single numeric feature, so its effect on the 31-level punishment distribution is one shape stretched across the whole range: on the multinomial it is exactly one weight per class, on the GNN one input unit. The maximum indicator worked precisely because it gave the model freedom at one contribution value. Giving it freedom across the range should let the middle of the range be as steep as the data wants.

**Behavioural rationale (one sentence, §5):** the human manager reads the contribution as a set of distinguishable amounts rather than as a number to be scaled -- "gave nearly nothing", "gave about half", "gave nearly all" carry their own punishment distributions -- so both punishers get the contribution as a full one-hot over its 21 possible values instead of a numeric input; the rows that should move are RPA (the manager-policy row that reads the punish rate as a function of contribution) and RCC, with RCB as a watch item.

**Why one-hot and not bins (§5 legality).** The manager-policy row RPA is defined on contribution bins of exactly {0}, 1-5, 6-10, 11-15, 16-19, {20}. Choosing those bins as features would be engineering at the metric's own definition and is illegal however well it fits. A full one-hot over the 21 possible contribution values is chosen instead on three grounds that have nothing to do with the metric: it is the maximally flexible encoding of a 21-level categorical variable and therefore the natural test of "one shape stretched across the range" as the hypothesis, it introduces no boundary the analyst picked, and it is the encoding the GNN already supports for every other categorical input (`encoding: onehot` in `generic/encoder.py`). A natural-cubic-spline basis is fitted alongside it in step 1 as the smooth alternative, and the choice between them is made on cross-validated log loss, not on any evaluation row.

**The maximum indicator is replaced, not kept.** A one-hot that carries `contribution == 20` as its own column is `contribution_max` exactly, so keeping both would be a duplicated column. Ties go to the simpler model (§5): the one-hot replaces the indicator in both families, and the parent's ceiling behaviour is verified to survive in the mechanism check (step 6) rather than assumed.

**The change.** Twenty-one derived features `contribution_is_00 ... contribution_is_20` in the linear pool (`build_feature_pool`, added to `CURRENT_VALUED` and `PUNISHMENT_LEGAL_CURRENT`, so they are legal for the punishment target and illegal for the contribution target like `contribution` itself), replacing `contribution` and `contribution_max` in the punisher's feature set; and `encoding: onehot` in place of `encoding: numeric` on the GNN's `contribution` node feature, with the `contribution_max` bool dropped. Everything else -- data, split, seed, C, architecture, epochs, copula rho -- identical to the parent's artifacts.

### Artifact naming contract

| what | path |
|---|---|
| human-data check | `scripts/data_analysis/punisher_contribution_encoding_check.py` |
| baseline config | `configs/training/baselines/punishment/multinomial_contr_bins.yml` |
| baseline artifact | `artifacts/baselines/punishment_multinomial_contr_bins.joblib` |
| severity-copula copy | `artifacts/baselines/punishment_multinomial_contr_bins_severity_copula.joblib` (rho 0.4273 stamped) |
| GNN config | `configs/training/artificial_humans/punishment/rnn_edge_50ep_doubled_contr_bins.yml` |
| GNN artifact dir | `artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_contr_bins/` |
| sim configs | `configs/simulation/manager_testing/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_cbins.yml`, `23_2g8a_self_gnn_contr_gnn_switch_cbins.yml`; sim dirs `plots/simulation/<same>` |
| tables | `plots/data_analysis/evaluation/punisher_contribution_encoding/` |

## 2. Plan

| # | step | status |
|---|---|---|
| 1 | Human-data check before any cluster time: fit the 31-class punishment model with the parent's encoding, with the one-hot and with a spline basis; report cross-validated log loss and the implied OLS slope on c_t against the human -0.242. | done -- **refuted** |
| 2 | Add the one-hot to the linear feature pool and the punishment legal set (`handcrafted_grid.py`), switch the GNN's `contribution` to `onehot` in the config, document it (`baseline_feature_defs.md`), tests on both paths. | done |
| 3 | Retrain the linear punisher locally (`multinomial_contr_bins.yml`, 4-fold CV, seed 38381); report CV against the parent's 1.3446. | done |
| 4 | Stamp the severity copula carrying rho = 0.4273 over unchanged; report the refit for the record, do not stamp it. | done |
| 5 | Retrain the GNN punisher on Raven (`rnn_edge_50ep_doubled_contr_bins.yml`); report CV against the parent's 1.1743. | done |
| 6 | Teacher-forced mechanism check of both new punishers against the parent's artifacts and the human row. | done |
| 7 | Re-run the two stacks, fetch, evaluate all 22 rows (`PYTHONPATH=<worktree>/src`), self-play mechanism table. | done |
| 8 | Judge under the gates with RCE protected (amended magnitude clause); log; PR against the parent. | done |

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-19 | (baseline) the parent's ceiling punisher, frontier stack (`_ceiling`) | RPA 0.6620, RCC 1.2969, RCB 1.6591, RCE 0.8823 | 14/22 | 1.0331 | baseline |
| 2026-09-19 | **the 21-level one-hot contribution on the copula lin_multinomial**, PR 181 stack (`_cbins`) | RPA **0.6932**, RCC **1.4237**, RCB 1.3568, RCE 0.8835 | 15/22 | **0.9970** | **FAIL** (gate 1: both targets worse; protected row: RCE 10-14 slope eroded) |
| 2026-09-19 | (baseline) the parent's ceiling multinomial, main gnn x gnn (`_ceiling`, run `lin_multinomial_self`) | RPA 0.6626, RCC 1.4615, RCB 1.1492, RCE 0.9900 | 13/22 | 1.6616 | baseline (reference) |
| 2026-09-19 | the one-hot on the plain multinomial, main gnn x gnn (`_cbins`, run `lin_multinomial_self`) | RPA 0.6405, RCC 1.4430, RCB 1.0469, RCE 1.0045 | 12/22 | 1.6747 | reported (not gated) |
| 2026-09-19 | (baseline) the parent's ceiling GNN punisher, main gnn x gnn (`_ceiling`, run `gnn_self`) | RPA 0.8066, RCC 1.1109, RCB 1.0886, RCE 0.9157 | 11/22 | 1.6603 | baseline (reference) |
| 2026-09-19 | the one-hot on the GNN punisher, main gnn x gnn (`_cbins`, run `gnn_self`) | RPA 0.9289, RCC 1.1745, RCB 1.1374, RCE 0.9561 | 8/22 | 1.7386 | reported (not gated) |

### Step 1: the human-data check -- the hypothesis is refuted before any cluster time (measured)

`scripts/data_analysis/punisher_contribution_encoding_check.py`, which reproduces three existing conventions exactly so the comparison is like for like: `cv_logloss_train` is what `run_baseline_cv.py` scores (locked train split, 7,345 rows, the grid's 4 folds, seed 38381, `mask: punishment_valid`, C = 1.0); `slope_artifact` fits on the whole train split as `inspect_best_model.py --save-best` does and replays on the mechanism check's 8,914 rows, so it is `punisher_mechanism_check.py`'s `OLS c_t`; `slope_oof50` is the same OLS from out-of-fold predictions over all 50 games (5 folds by episode, mechanism mask). The fidelity check passes on both parent numbers: the parent's encoding reproduces CV **1.3446**, locked test **1.2234** and slope **-0.1255** to four decimals.

| contribution encoding | cols | CV log loss (train) | locked test | slope (artifact-equivalent) | slope (out of fold, 50 games) | P(p>0 \| c=20) | E[p \| p>0] at 20 |
|---|---|---|---|---|---|---|---|
| human (observed) | -- | -- | -- | **-0.2417** | -0.2417 | 0.038 | 7.00 |
| numeric (grandparent) | 1 | 1.3465 (se 0.058) | 1.2468 | -0.1249 | -0.2150 | 0.136 | 3.94 |
| **numeric + max (parent)** | 2 | **1.3446** (se 0.062) | **1.2234** | **-0.1255** | -0.2141 | 0.053 | 7.47 |
| **onehot (this branch's candidate)** | 21 | **1.4157** (se 0.068) | 1.2544 | **-0.1329** | -0.2204 | 0.053 | 7.52 |
| onehot + numeric | 22 | 1.4166 (se 0.068) | 1.2546 | -0.1339 | -0.2221 | 0.053 | 7.55 |
| natural spline, 4 df | 4 | 1.3555 (se 0.062) | 1.2279 | -0.1286 | -0.2195 | 0.070 | 5.97 |
| natural spline, 4 df + max | 5 | 1.3599 (se 0.062) | 1.2189 | -0.1281 | -0.2195 | 0.053 | 7.57 |
| natural spline, 6 df | 6 | 1.3658 (se 0.062) | 1.2264 | -0.1303 | -0.2201 | 0.061 | 6.46 |
| natural spline, 6 df + max | 7 | 1.3663 (se 0.063) | 1.2226 | -0.1293 | -0.2200 | 0.053 | 7.57 |

Both halves of the hypothesis fail. **The fit gets worse, not better**: the one-hot costs **+0.0711** of cross-validated log loss against the parent (1.3446 -> 1.4157), more than one standard error, and every richer encoding tested -- one-hot, one-hot plus the numeric term, and natural splines at 4 and 6 df, with and without the maximum indicator -- is worse than the parent's two-column encoding. **And the slope barely moves**: -0.1255 -> -0.1329, which closes 6% of the 0.116 gap to the human -0.2417. The spline rows say the same thing with a tenth of the parameters, so this is not the one-hot overfitting where a smoother basis would have succeeded: the flexibility is simply not what the slope is short of. The ceiling behaviour the parent won does survive the re-encoding (P(p>0 \| c=20) 0.053 and E[p \| p>0] at 20 7.52 against the parent's 0.053 and 7.47), so the one-hot's level-20 column does subsume `contribution_max` exactly as predicted -- that part of the reasoning was right and is the only part that was.

### Step 1b: where the slope deficit actually comes from (measured)

The refutation raised a sharper question, because the same model class fitted out of fold over all 50 games lands at **-0.214 to -0.222 whatever the encoding** -- within 12% of the human -0.2417 -- while the shipped artifact replays at -0.1255. If the model class can express the human slope, the deficit is not in the model class. `--decompose` holds the encoding fixed at the parent's and changes only the rows the model is fitted on (`human_encoding_check_decomposition.csv`):

| fitted on | rows | OLS c_t on the mechanism rows |
|---|---|---|
| the locked train split, `punishment_valid` (**the artifact**) | 7,377 | **-0.1234** |
| the same 40 episodes, `punishment_valid & contribution_valid` | 7,167 | -0.1693 |
| all 50 episodes, `punishment_valid` | 9,193 | -0.1568 |
| all 50 episodes, `punishment_valid & contribution_valid` | 8,914 | -0.2127 |
| 20 random 40-episode subsets, `punishment_valid & contribution_valid` | ~7,100 | -0.2141 +- 0.0226 (min -0.1790, max -0.2532) |
| human (observed) | 8,914 | **-0.2417** |

Two things that are not the encoding account for almost the whole gap. **The training mask is worth about -0.046**: the punisher trains on `punishment_valid` rather than `punishment_valid & contribution_valid`, so roughly 210 of its 7,377 rows are rounds where the manager punished a player whose contribution is missing and the model is shown the imputed default of 9 next to a real punishment. Those rows are pure attenuation on exactly the coefficient in question, and both punisher families carry the same mask (`mask_name: punishment_valid` in the GNN config). **The locked split's particular draw is worth about another -0.043**: the same features on the same mask give -0.1693 on the locked 40 episodes against -0.2141 +- 0.0226 over random 40-episode subsets, so the split the whole baseline family is trained on is roughly two standard deviations flat on this coefficient. What is left between -0.2127 and the human -0.2417 is 24% of the original gap, and that residual is the only part a richer model of c_t could ever have addressed.

(The table's -0.1234 is the artifact's -0.1255 rebuilt from the full 50-game file restricted to the train episodes rather than from the train CSV, which is why its row count is 7,377 and not 7,345; the 32-row difference comes from `create_torch_data`'s per-file defaults and does not move the number materially.)

**Decision.** The declared hypothesis is refuted on the human data. Per section 9 the experiment is still run to its verdict -- a refuted hypothesis with a measured end-to-end failure is what stops the next agent retrying it -- and the one-hot is kept as the candidate rather than swapping in the least-bad spline, because a fully saturated encoding of c_t is the decisive test: if 21 free levels do not steepen the response, no encoding of c_t will. The indicator is replaced, not kept, as declared.

### Step 2: the change on both paths

`CONTRIBUTION_ONEHOT` (21 names, `contribution_is_00 ... contribution_is_20`) joins the linear feature pool in `build_feature_pool`, `CURRENT_VALUED` and `PUNISHMENT_LEGAL_CURRENT`, so the columns are legal for the punishment target and rejected with the same hard error as `contribution` for the contribution target. The GNN needed no code change at all: `generic/encoder.py`'s `IntEncoder` has supported `encoding: onehot` since before this campaign, so the config switches `contribution` from `numeric` to `onehot` and drops the `contribution_max` bool. Four tests pin it, two of them new: the linear adapter's design matrix is `np.eye(21)[c_t]` row for row, the GNN's encoded node feature is the same matrix, and its level-20 column equals the parent's `contribution_max` tensor exactly. 132 tests pass on Raven (`remote_test.sh --test-only` against the isolated dir, with `plots/simulation/22_2g8a_linear_self_ridge_contr/per_round.parquet` shipped separately because the sync excludes `plots/`).

### Step 3: the linear punisher retrained (measured)

`configs/training/baselines/punishment/multinomial_contr_bins.yml` run as its header says: 4-fold CV on the locked train split, seed 38381, the grid's folds, C = 1.0. The rank-1 row is the declared one-hot set (the B1 block carries it alone, so the grid cannot re-select the parent's encoding and save the wrong artifact), with CV log loss **1.4157** against the parent's **1.3446**, and locked test **1.2544** against **1.2234**. Both numbers reproduce step 1 to four decimals.

The size of the loss is worth stating plainly. The constant floor -- the marginal punishment distribution with no features at all -- is **1.4355** on the same folds. The parent's six features beat that floor by 0.091; the one-hot's twenty-five beat it by 0.020. Twenty-one dummies times thirty-one classes is 651 coefficients fitted on about 5,500 rows per fold at C = 1.0, and the model spends nearly all of its advantage over the marginal on them. Train log loss goes the other way (1.2043 against the parent's, so the extra parameters do fit the training rows), which is what overfitting looks like from both sides.

Artifact `artifacts/baselines/punishment_multinomial_contr_bins.joblib`, features `contribution_is_00 ... contribution_is_20, prev_contribution, prev_punishment, round_number, is_first`; CV table `data/baselines/punishment_cv_multinomial_contr_bins.csv`.

### Step 4: the severity copula, stamped rather than refit (measured)

`scripts/baselines/punishment_copula_rho.py --roundtrip --stamp-rho 0.4273` writes `artifacts/baselines/punishment_multinomial_contr_bins_severity_copula.joblib`: the plain bundle's weights bit-identical after reload (the script's own `predict_proba` comparison over the first 100 rows) plus the `copula_*` keys, carrying **rho = 0.4273 unchanged** from the parent. The protocol freezes the copula parameters per model family when only the marginal is retrained, so the carried value is the stamped one.

The refit on the new marginal is recorded and deliberately **not** stamped: rho **0.4623**, 95% CI [0.3909, 0.5522], out-of-sample MLE on the test split 0.3643. That is higher than the parent's refit of 0.4300 [0.357, 0.521] and than the stamped 0.4273, though 0.4273 sits inside the new interval. A marginal that over-fits its contribution levels leaves more of the within-round co-movement unexplained for the copula to absorb, which is the direction the refit moved; it is one more reading of the same over-fitting and not an independent finding. As on the parent branch the stamped bundle carries NaN for `copula_rho_se` and the interval, because those describe the refit and the refit is not what was stamped.

### Step 5: the GNN punisher retrained (measured)

`configs/training/artificial_humans/punishment/rnn_edge_50ep_doubled_contr_bins.yml` on Raven, job **30318234**, 7 min 38 s on one A100 for the 5-fold CV plus the full fit (the parent's was 6 min 54 s; the extra time is the wider input layer, 24 encoded dimensions against 5). Final-epoch CV log loss **1.1940** against the parent's **1.1743**; best-epoch 1.1936 +- 0.1112 over folds against 1.1742 +- 0.1126; globally best epoch 1240 against 1249.

The GNN loses **+0.0197** of log loss where the linear lost +0.0711, and the ratio is the point. The multinomial is linear in `contribution` on every class logit and gains 651 free coefficients from the one-hot; the GNN's MLP head could already bend its numeric input into any shape it wanted, so replacing that input with 21 units adds parameters to a model that was not short of expressiveness and costs a quarter as much. Both move the same way, which is the reading that matters: on this data, more freedom in c_t is a cost, not a gain, in both families.

Artifact `artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_contr_bins/`. `punishment_baseline.py`'s `GNN_REF` stays at 1.1756, as on the parent branch.

### Step 6: the mechanism, teacher-forced (measured)

`scripts/data_analysis/punisher_mechanism_check.py` replaying the 50 single-copy human games, 8,914 valid rows, no simulation; each model sees the human history and never its own draws, under its own stored `default_values`. The linears run locally, the GNNs on Raven's login node. Tables: `mechanism_teacher_forced_linear.csv` and `mechanism_teacher_forced_gnn.csv` in `plots/data_analysis/evaluation/punisher_contribution_encoding/`.

| punisher | P(p>0 \| c_t=20) | P(p>0 \| c_t<=4) | (c_t=20, c_{t-1}<=4) / (c_t<=4, c_{t-1}=20) | E[p \| p>0] 0-4 / 5-9 / 10-14 / 15-19 / 20 | OLS c_t / c_{t-1} | NLL |
|---|---|---|---|---|---|---|
| human | 0.038 | 0.467 | 0.179 / 0.571 | 7.99 / 4.98 / 4.27 / 3.87 / 7.00 | **-0.242** / +0.067 | -- |
| lin ceiling (parent) | 0.053 | 0.431 | 0.224 / 0.635 | 7.43 / 5.46 / 4.44 / 3.76 / 7.47 | **-0.125** / -0.031 | 1.271 |
| lin onehot | 0.053 | 0.420 | 0.218 / 0.588 | 7.63 / 5.22 / 4.26 / 4.04 / 7.52 | **-0.133** / -0.026 | 1.238 |
| gnn ceiling (parent) | 0.063 | 0.424 | 0.245 / 0.468 | 7.37 / 5.44 / 4.70 / 4.49 / 6.00 | **-0.144** / -0.010 | 1.132 |
| gnn onehot | 0.058 | 0.431 | 0.238 / 0.503 | 7.55 / 5.30 / 4.65 / 4.47 / 6.66 | **-0.159** / +0.001 | 1.117 |

**The ceiling the parent won survives in both families**, which was the one thing the replacement of `contribution_max` by the one-hot's level-20 column had to preserve: the linear punisher is unchanged at 0.053 with severity 7.47 -> 7.52, and the GNN improves slightly, 0.063 -> 0.058 with severity 6.00 -> 6.66 against the human 7.00. The design reasoning about the indicator being subsumed was correct.

**The slope moves by 0.008 and 0.015.** -0.125 -> -0.133 on the linear punisher and -0.144 -> -0.159 on the GNN, against the human -0.242: 7% and 15% of each model's own gap. The prediction of the human-data check (-0.1255 -> -0.1329 for the linear, to four decimals) is reproduced by the shipped artifact.

One number needs reading carefully so it is not mistaken for a win. The teacher-forced **NLL improves in both** (1.271 -> 1.238, 1.132 -> 1.117) while the cross-validated log loss got clearly worse (1.3446 -> 1.4157, 1.1743 -> 1.1940). There is no contradiction: 40 of the 50 replayed episodes are in the training split, so this NLL is largely in-sample, and a model with 651 extra coefficients fits the rows it was trained on better by construction. The CV numbers are the honest ones and they point the other way. A successor reading only the mechanism table would draw the wrong conclusion.

### Step 7: self-play and the 22 rows (measured)

Both stacks re-run with the 23-family protocol (seed 42, 100 episodes, 24 rounds) from the isolated remote dir `~/repros/ai-runs/punisher-contr-bins`: Raven jobs **30318407** (frontier, 1 min 50 s) and **30318408** (reference, 2 min 57 s), one A100 each, then evaluated locally with the merged 22-row suite (500 repeats, seed 42) under `PYTHONPATH=<worktree>/src`. Sim dirs `plots/simulation/<stack>_cbins/`; tables in `plots/data_analysis/evaluation/punisher_contribution_encoding/`.

**Self-play mechanism table** (realised punishments over each sim's 19,200 agent-rounds; the human row is the observed data on 8,914 rows).

| sim | P(p>0 \| c_t=20) | P(p>0 \| c_t<=4) | (c_t=20, c_{t-1}<=4) / (c_t<=4, c_{t-1}=20) | E[p \| p>0] 0-4 / 5-9 / 10-14 / 15-19 / 20 | OLS c_t / c_{t-1} |
|---|---|---|---|---|---|
| human | 0.038 | 0.467 | 0.179 / 0.571 | 7.99 / 4.98 / 4.27 / 3.87 / 7.00 | **-0.242** / +0.067 |
| frontier before | 0.040 | 0.463 | 0.177 / 0.585 | 7.03 / 5.69 / 4.54 / 3.61 / 8.02 | **-0.143** / -0.015 |
| frontier after | 0.045 | 0.421 | 0.157 / 0.731 | 7.03 / 5.35 / 4.39 / 3.94 / 7.38 | **-0.136** / -0.003 |
| ref_lin before | 0.059 | 0.451 | 0.141 / 0.515 | 7.42 / 5.47 / 4.30 / 3.67 / 7.98 | **-0.120** / -0.027 |
| ref_lin after | 0.055 | 0.462 | 0.100 / 0.452 | 7.44 / 5.38 / 4.41 / 3.75 / 7.36 | **-0.140** / -0.023 |
| ref_gnn before | 0.069 | 0.521 | 0.171 / 0.442 | 7.59 / 6.08 / 5.20 / 5.51 / 6.24 | **-0.187** / -0.007 |
| ref_gnn after | 0.066 | 0.565 | 0.207 / 0.530 | 8.34 / 5.74 / 5.09 / 5.77 / 6.71 | **-0.224** / -0.014 |

Closed loop the slope does move, and it moves in three different directions. **On the GNN reference stack it nearly closes**: -0.187 -> **-0.224** against the human -0.242, 80% of the remaining gap, the largest movement this quantity has had in the campaign. On the linear reference it moves -0.120 -> -0.140. **On the gated frontier stack it moves the wrong way**, -0.143 -> -0.136. The ceiling the parent won is intact everywhere (0.040 -> 0.045, 0.059 -> 0.055, 0.069 -> 0.066 against the human 0.038) and the ceiling severity moves toward the human on both linear stacks (8.02 -> 7.38 and 7.98 -> 7.36 against 7.00), so nothing the parent achieved was lost.

That divergence matters for the conclusion and is set out in section 4: the one stack where the hypothesis' own mechanism did what it promised is also the stack whose evaluation got worst.

**The 22 rows, frontier stack (the gated one).**

| row | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.8563 | 0.9296 | +0.0734 | <= 1 |
| CB | 0.7941 | 0.9378 | +0.1438 | <= 1 |
| CC | 0.8958 | 0.8844 | -0.0113 | <= 1 |
| CD | 0.7949 | 0.8983 | +0.1034 | <= 1 |
| CE | 1.0579 | 1.0093 | -0.0486 | 1-2 |
| CF | 0.8169 | 0.9034 | +0.0864 | <= 1 |
| CG | 1.7588 | 0.9655 | **-0.7932** | 1-2 -> <= 1 |
| SA | 0.7687 | 0.7755 | +0.0067 | <= 1 |
| SB | 1.0105 | 0.8860 | -0.1246 | 1-2 -> <= 1 |
| SC | 1.4632 | 1.4615 | -0.0017 | 1-2 |
| PA | 0.6526 | 0.6929 | +0.0404 | <= 1 |
| PB | 0.9558 | 0.8768 | -0.0791 | <= 1 |
| PC | 0.9349 | 0.8786 | -0.0563 | <= 1 |
| PD | 0.7598 | 0.7245 | -0.0353 | <= 1 |
| RCA | 1.6526 | 1.5157 | -0.1369 | 1-2 |
| RCB | 1.6591 | 1.3568 | -0.3023 | 1-2 |
| **RCC** (target) | **1.2969** | **1.4237** | **+0.1268** | 1-2 (no upgrade) |
| RCD | 1.2515 | 1.1408 | -0.1108 | 1-2 |
| RCE (protected) | 0.8823 | 0.8835 | +0.0011 | <= 1 |
| RSA | 0.9653 | 1.2950 | +0.3297 | <= 1 -> 1-2 |
| **RPA** (target) | **0.6620** | **0.6932** | **+0.0311** | <= 1 (no upgrade available) |
| RPB | 0.8380 | 0.8009 | -0.0372 | <= 1 |
| **mean** | **1.0331** | **0.9970** | **-0.0361** | |
| rows <= 1 | 14 | 15 | +1 | |

**The 22 rows, reference stack, plain multinomial (`lin_multinomial_self`).**

| row | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.9572 | 0.9517 | -0.0055 | <= 1 |
| CB | 0.8310 | 0.8412 | +0.0102 | <= 1 |
| CC | 1.7150 | 1.7208 | +0.0058 | 1-2 |
| CD | 0.8240 | 0.8203 | -0.0038 | <= 1 |
| CE | 1.2212 | 1.2587 | +0.0375 | 1-2 |
| CF | 0.8988 | 0.8636 | -0.0352 | <= 1 |
| CG | 9.5199 | 9.4988 | -0.0211 | > 5 |
| SA | 0.8279 | 0.7780 | -0.0499 | <= 1 |
| SB | 0.8341 | 0.8381 | +0.0040 | <= 1 |
| SC | 3.0351 | 2.7710 | -0.2641 | 2-5 |
| PA | 0.6389 | 0.5891 | -0.0498 | <= 1 |
| PB | 0.8018 | 0.7700 | -0.0318 | <= 1 |
| PC | 0.7709 | 0.7581 | -0.0128 | <= 1 |
| PD | 2.7342 | 3.2025 | +0.4683 | 2-5 |
| RCA | 1.9767 | 2.0802 | +0.1035 | 1-2 -> 2-5 |
| RCB | 1.1492 | 1.0469 | -0.1023 | 1-2 |
| RCC | 1.4615 | 1.4430 | -0.0185 | 1-2 |
| RCD | 3.0472 | 3.2341 | +0.1869 | 2-5 |
| RCE | 0.9900 | 1.0045 | +0.0145 | <= 1 -> 1-2 |
| RSA | 0.9855 | 0.9940 | +0.0085 | <= 1 |
| RPA | 0.6626 | 0.6405 | -0.0221 | <= 1 |
| RPB | 0.6724 | 0.7373 | +0.0649 | <= 1 |
| **mean** | **1.6616** | **1.6747** | +0.0131 | |
| rows <= 1 | 13 | 12 | -1 | |

**The 22 rows, reference stack, GNN punisher (`gnn_self`).**

| row | before | after | delta | band |
|---|---|---|---|---|
| CA | 0.7880 | 0.8348 | +0.0467 | <= 1 |
| CB | 0.7104 | 0.6950 | -0.0154 | <= 1 |
| CC | 1.5733 | 1.6044 | +0.0311 | 1-2 |
| CD | 0.6892 | 0.7017 | +0.0125 | <= 1 |
| CE | 1.4189 | 1.3289 | -0.0900 | 1-2 |
| CF | 0.8339 | 0.8147 | -0.0192 | <= 1 |
| CG | 9.3129 | 9.3209 | +0.0080 | > 5 |
| SA | 0.6718 | 0.9681 | +0.2963 | <= 1 |
| SB | 0.8398 | 0.9005 | +0.0607 | <= 1 |
| SC | 2.8059 | 2.9270 | +0.1212 | 2-5 |
| PA | 1.0599 | 1.5547 | +0.4948 | 1-2 |
| PB | 0.9919 | 1.2156 | +0.2237 | <= 1 -> 1-2 |
| PC | 0.9017 | 1.1202 | +0.2186 | <= 1 -> 1-2 |
| PD | 3.1110 | 2.7104 | -0.4005 | 2-5 |
| RCA | 2.0314 | 2.0744 | +0.0431 | 2-5 |
| RCB | 1.0886 | 1.1374 | +0.0488 | 1-2 |
| RCC | 1.1109 | 1.1745 | +0.0636 | 1-2 |
| RCD | 2.8239 | 2.6575 | -0.1664 | 2-5 |
| RCE | 0.9157 | 0.9561 | +0.0404 | <= 1 |
| RSA | 0.8890 | 1.0843 | +0.1953 | <= 1 -> 1-2 |
| RPA | 0.8066 | 0.9289 | +0.1223 | <= 1 |
| RPB | 1.1513 | 1.5397 | +0.3884 | 1-2 |
| **mean** | **1.6603** | **1.7386** | +0.0784 | |
| rows <= 1 | 11 | 8 | -3 | |

**RCE band slopes, each with its own standard error and row count.**

| stack | band | human | before | after | change_in_se | closer to human? |
|---|---|---|---|---|---|---|
| frontier (gated) | 0-4 | +0.140 +- 0.018 (n 965) | +0.087 +- 0.014 (n 1918) | +0.112 +- 0.015 (n 1601) | 1.23 | yes |
| frontier (gated) | 5-9 | +0.104 +- 0.024 (n 929) | +0.038 +- 0.013 (n 2098) | +0.030 +- 0.015 (n 2101) | 0.41 | no |
| frontier (gated) | 10-14 | -0.077 +- 0.035 (n 560) | -0.043 +- 0.023 (n 1307) | **-0.007 +- 0.021 (n 1398)** | **1.14** | **no** |
| frontier (gated) | 15-19 | -0.161 +- 0.079 (n 206) | -0.130 +- 0.067 (n 448) | -0.157 +- 0.052 (n 409) | 0.32 | yes |
| ref_lin | 0-4 | +0.140 +- 0.018 (n 965) | +0.067 +- 0.017 (n 1580) | +0.060 +- 0.016 (n 1649) | 0.29 | no |
| ref_lin | 5-9 | +0.104 +- 0.024 (n 929) | +0.094 +- 0.015 (n 1837) | +0.069 +- 0.014 (n 2140) | 1.23 | no |
| ref_lin | 10-14 | -0.077 +- 0.035 (n 560) | +0.036 +- 0.022 (n 1471) | -0.027 +- 0.020 (n 1448) | 2.15 | yes |
| ref_lin | 15-19 | -0.161 +- 0.079 (n 206) | -0.014 +- 0.061 (n 542) | +0.037 +- 0.058 (n 419) | 0.60 | no |
| ref_gnn | 0-4 | +0.140 +- 0.018 (n 965) | +0.084 +- 0.014 (n 2079) | +0.082 +- 0.013 (n 2189) | 0.07 | no |
| ref_gnn | 5-9 | +0.104 +- 0.024 (n 929) | +0.060 +- 0.013 (n 2088) | +0.056 +- 0.013 (n 2449) | 0.23 | no |
| ref_gnn | 10-14 | -0.077 +- 0.035 (n 560) | -0.001 +- 0.020 (n 1414) | +0.015 +- 0.020 (n 1363) | 0.55 | no |
| ref_gnn | 15-19 | -0.161 +- 0.079 (n 206) | +0.017 +- 0.041 (n 474) | -0.017 +- 0.047 (n 437) | 0.54 | yes |

Protected-row checks, verbatim from the script:

- frontier (gated): `band_downgrade=False, sign_lost=[], magnitude_eroded=['10-14'], signs ++-- -> ++--` -- **fails the protected row.**
- ref_lin (reported, not gated): `band_downgrade=True, sign_lost=['15-19'], magnitude_eroded=[]`, signs `+++-` -> `++-+`.
- ref_gnn (reported, not gated): `sign_lost=['10-14']`, signs `++-+` -> `+++-`, `magnitude_eroded=[]`.

The amended magnitude clause is the one that fires on the gated stack, and it fires on all three of its conditions at once, which is what the amendment was for: the 10-14 slope is halved (-0.043 -> -0.007, and 0.007 <= 0.5 x 0.043), it is **not** closer to the human -0.077 after the change (0.070 away against 0.034 before), and the change is 1.14 pooled standard errors. The parent's relative-only version would have fired on the same band; the amendment does not rescue it, and the honest reading is that this is the marginal end of erosion rather than a rule artefact -- 1.14 SE is barely past the threshold, and a reader is entitled to weigh it as such. RCE's own score is flat (0.8823 -> 0.8835), which is why the row-level check and the slope check disagree here.

### Step 8: verdict -- `[FAIL]`

| gate | criterion | baseline | result | outcome |
|---|---|---|---|---|
| 1 (declared targets) | a band improvement on RPA or RCC | RPA 0.6620 (already <= 1), RCC 1.2969 | RPA **0.6932** (worse, no better band exists), RCC **1.4237** (worse, still 1-2) | **FAIL** |
| 2 | 22-row mean at or under 1.1364 | 1.0331 | **0.9970** | pass |
| protected | RCE: no band drop, no lost human sign, no eroded slope magnitude (amended clause) | 0.8823, slopes +0.087 / +0.038 / -0.043 / -0.130 | 0.8835, +0.112 / +0.030 / -0.007 / -0.157; `magnitude_eroded=['10-14']` | **FAIL** |

**The branch fails twice over.** Both declared targets moved the wrong way on the gated stack -- RCC by +0.127, the opposite of what the hypothesis predicted, and RPA by +0.031 on a row that had no better band to reach in the first place -- and the protected row's 10-14 slope eroded on the amended clause. Gate 2 passes comfortably, and in fact the 22-row mean **improved** to 0.9970 with rows <= 1 going 14 -> 15, which is the best mean this frontier stack has recorded. That is not a rescue: gate 1 is a band upgrade on a declared row and there was none, and a protected-row failure is disqualifying on its own.

## 4. Notes

1. The human-data check refuted the hypothesis before any cluster time, which is what it is for. It was run anyway to a full verdict because the protocol has no path for abandoning a declared experiment silently, and because the decomposition in step 1b only exists because the refutation forced the question of where the slope deficit really lives.
2. The parent's numbers reproduce exactly (CV 1.3446, locked test 1.2234, teacher-forced slope -0.1255), so the refutation is not an artefact of a different pipeline. That fidelity check was run first, and the first version of this check -- which used out-of-fold predictions over all 50 games as the slope -- disagreed with the parent by a factor of 1.7 until the locked split and the training mask were identified as the cause.
3. **The declaration was partly unwinnable and that should have been caught at plan time.** RPA was declared a target row at 0.6620 on the gated stack, which is already the best band. No band upgrade on RPA was available at any point, so gate 1 rested entirely on RCC -- the same single row the parent failed -- and the two-target declaration was one target in substance. It is recorded in section 1 before the results and is a check the next declaration should run: a target row already at <= 1 on the gated stack cannot carry gate 1.
4. **The largest finding is closed-loop and cuts against the whole family of slope-chasing hypotheses.** On the GNN reference stack the one-hot did what the hypothesis said it would: the self-play OLS response to c_t went -0.187 -> **-0.224** against the human -0.242, closing 80% of the remaining gap -- much more than the teacher-forced number (-0.144 -> -0.159) predicted, because the contributors respond to the changed punisher. And that stack's evaluation got **worse than any other**: mean 1.6603 -> 1.7386, rows <= 1 eleven -> eight, with PA +0.49, RPB +0.39, PC +0.22, PB +0.22 and RPA +0.12. The rows that degraded are the punishment marginals -- the distribution of punishment overall and by group size -- which is exactly what a punisher that fits its 21 contribution levels too closely would damage. **Closing the contribution slope did not buy a single evaluation row anywhere, and where it closed most it cost most.** A successor should treat "the punisher's slope on c_t is half the human's" as a described discrepancy whose causal link to the evaluation suite is now measured and negative, not as a defect worth another experiment on its own.
5. **The gated stack moved the slope the wrong way while improving its mean.** The frontier self-play slope went -0.143 -> -0.136, away from the human, at the same time as the 22-row mean improved to 0.9970 and CG fell by 0.79 into band <= 1. The three stacks therefore disagree about the sign of the slope change (-0.007, -0.020, -0.037) while agreeing about the encoding being worse out of sample. The slope a stack realises in self-play is a joint property of punisher, contributor and switch model, not a property of the punisher alone, and one stack's measurement of it does not transfer.
6. **What went the right way on the gated stack, and why it is not a claim.** CG 1.7588 -> 0.9655 (a band upgrade, but CG is not a declared target and the protocol counts upgrades only on declared rows), RCB 1.6591 -> 1.3568, SB 1.0105 -> 0.8860, RCA -0.14, RCD -0.11, and the best 22-row mean this stack has recorded. RCB's -0.30 is the one the parent predicted would follow a slope change, and it is the only one of these that the hypothesis can claim any credit for; CG is the contributor and switch models meeting a punisher that behaves differently from the one they were accepted against, the same systematic effect the parent and grandparent both documented, and CG is anti-correlated with the individual-fit rows by construction (§6). What went the wrong way: RSA +0.33 (a band downgrade), RCC +0.13, CB +0.14, CD +0.10, CF +0.09.
7. **The over-fitting is visible in four independent places and they all agree.** Cross-validated log loss +0.0711 (linear) and +0.0197 (GNN); the linear model's advantage over the constant floor collapsing from 0.091 to 0.020 while its training loss improves to 1.2043; the teacher-forced NLL improving on rows the model largely trained on while the CV got worse; and the closed-loop punishment-marginal rows (PA, PB, PC, RPA, RPB) degrading on the GNN reference stack. Any one of these alone would be arguable; together they are one story. A successor who wants a flexible c_t response should not repeat this with a smaller basis expecting a different sign -- the spline rows in step 1 already tested that at 4 and 6 df and lost too.
8. **The `punishment_valid`-only training mask is the best-supported live defect this branch found, and it belongs to a successor as a bug fix.** About 210 of the punisher's 7,377 training rows are rounds where the manager punished a player whose contribution is missing; the model is shown the imputed default of 9 next to a real punishment, and dropping those rows alone moves the implied slope from -0.1234 to -0.1693, 40% of the gap to the human. Both families carry the mask (`mask: punishment_valid` in the baseline config, `mask_name: punishment_valid` in the GNN config). Under §4 a bug fix is its own experiment with before/after scores; note that section 4's own note 4 above says the slope is not a rewarding target, so the case for this fix is correctness, not the expected score movement.
9. **Housekeeping.** The isolated remote dir `~/repros/ai-runs/punisher-contr-bins` can be deleted when this PR closes. `punishment_baseline.py`'s `GNN_REF` is left at 1.1756. The artifacts are committed (`.joblib` raw, `.pt` and `.parquet` under LFS) so a successor can reproduce the tables without retraining, but neither is a candidate the maintainer should adopt: both are worse than the parent's out of sample.
