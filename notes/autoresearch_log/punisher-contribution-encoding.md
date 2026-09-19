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
| 1 | Human-data check before any cluster time: fit the 31-class punishment model with the parent's encoding, with the one-hot and with a spline basis; report cross-validated log loss and the implied OLS slope on c_t against the human -0.242. | |
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

## 4. Notes
