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

### Hypothesis

**The defect (established by the parent, not re-derived).** The parent fixed the punisher at the contribution ceiling: the punish rate at c_t = 20 went 0.122 -> 0.040 against the human 0.038 and the severity there 3.84 -> 8.02 against the human 7.00. It did not touch the other half of the same defect. The OLS weight of punishment on the current contribution is about -0.14 in every stack (frontier -0.143, ref_lin -0.120, ref_gnn -0.187) against the human -0.242, so the simulated manager's sensitivity to how much someone gave is roughly half a real manager's. The parent's own notes declare this a live, separately declarable defect.

**The diagnosis (to be confirmed on the human data in step 1).** Contribution enters both punishers as a single numeric feature, so its effect on the 31-level punishment distribution is one shape stretched across the whole range: on the multinomial it is exactly one weight per class, on the GNN one input unit. The maximum indicator worked precisely because it gave the model freedom at one contribution value. Giving it freedom across the range should let the middle of the range be as steep as the data wants.

**Behavioural rationale (one sentence, §5):** the human manager reads the contribution as a set of distinguishable amounts rather than as a number to be scaled -- "gave nearly nothing", "gave about half", "gave nearly all" carry their own punishment distributions -- so both punishers get the contribution as a full one-hot over its 21 possible values instead of a numeric input; the rows that should move are RPA (the manager-policy row that reads the punish rate as a function of contribution) and RCC, with RCB as a watch item.

**Why one-hot and not bins (§5 legality).** The manager-policy row RPA is defined on contribution bins of exactly {0}, 1-5, 6-10, 11-15, 16-19, {20}. Choosing those bins as features would be engineering at the metric's own definition and is illegal however well it fits. A full one-hot over the 21 possible contribution values is chosen instead on three grounds that have nothing to do with the metric: it is the maximally flexible encoding of a 21-level categorical variable and therefore the natural test of "one shape stretched across the range" as the hypothesis, it introduces no boundary the analyst picked, and it is the encoding the GNN already supports for every other categorical input (`encoding: onehot` in `generic/encoder.py`). A natural-cubic-spline basis is fitted alongside it in step 1 as the smooth alternative, and the choice between them is made on cross-validated log loss, not on any evaluation row.

**The maximum indicator is replaced, not kept.** A one-hot that carries `contribution == 20` as its own column is `contribution_max` exactly, so keeping both would be a duplicated column. Ties go to the simpler model (§5): the one-hot replaces the indicator in both families, and the parent's ceiling behaviour is verified to survive in the mechanism check (step 4) rather than assumed.

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
| 2 | Add the one-hot to the linear feature pool and the punishment legal set (`handcrafted_grid.py`), switch the GNN's `contribution` to `onehot` in the config, document it (`baseline_feature_defs.md`), tests on both paths. | |
| 3 | Retrain the linear punisher locally (`multinomial_contr_bins.yml`, 4-fold CV, seed 38381); report CV against the parent's 1.3446. | |
| 4 | Stamp the severity copula carrying rho = 0.4273 over unchanged; report the refit for the record, do not stamp it. | |
| 5 | Retrain the GNN punisher on Raven (`rnn_edge_50ep_doubled_contr_bins.yml`); report CV against the parent's 1.1743. | |
| 6 | Teacher-forced mechanism check of both new punishers against the parent's artifacts and the human row. | |
| 7 | Re-run the two stacks, fetch, evaluate all 22 rows (`PYTHONPATH=<worktree>/src`), self-play mechanism table. | |
| 8 | Judge under the gates with RCE protected (amended magnitude clause); log; PR against the parent. | |

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-19 | (baseline) the parent's ceiling punisher, frontier stack (`_ceiling`) | RPA 0.6620, RCC 1.2969, RCB 1.6591, RCE 0.8823 | 14/22 | 1.0331 | baseline |

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

## 4. Notes

1. The human-data check refuted the hypothesis before any cluster time, which is what it is for. It was run anyway to a full verdict because the protocol has no path for abandoning a declared experiment silently, and because the decomposition in step 1b only exists because the refutation forced the question of where the slope deficit really lives.
2. The parent's numbers reproduce exactly (CV 1.3446, locked test 1.2234, teacher-forced slope -0.1255), so the refutation is not an artefact of a different pipeline. That fidelity check was run first, and the first version of this check -- which used out-of-fold predictions over all 50 games as the slope -- disagreed with the parent by a factor of 1.7 until the locked split and the training mask were identified as the cause.
