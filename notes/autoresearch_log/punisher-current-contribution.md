# The punisher must see the contribution it punishes (re-baseline)

## 1. Declaration

**Slot:** punisher -- both families (the `lin_multinomial` baseline every top stack uses and the GNN punisher), plus the shared legality rule in `scripts/baselines/handcrafted_grid.py` / `src/aimanager/simulation/linear_ah.py` that enforced the lag. A shared-code bug fix in the sense of §4: fix first, then before/after scores.

**Parent:** PR #181 (`auto/contribution-punishment-response`, `[FAIL]`, at `81c8002`): it carries the group-vnode trunk (PR #179), the stimulus-skip trunk, and the copula scripts, and its RCB investigation is where the lag was found. Branch `auto/punisher-current-contribution` is created from it and merges `origin/rcb-alternative-response-slope` (`699f921`, based on `main` `b174f90`), which adds the RCE row to the evaluation suite; the RCF cell row that branch also carried is dropped again (the user chose RCE only; `reports/rcb_alternative_comparison.md` still discusses both). Isolated remote dirs: `~/repros/ai-runs/punisher-current-contr-tests` (stage A tests), stage C names its own.

**Base models.** Punisher: `artifacts/baselines/punishment_multinomial_best_with_contr.joblib` (features `prev_contribution, prev_punishment, round_number, is_first`, C = 1.0, the `handcrafted_grid_cat.yml` grid's best entry with `prev_contribution`; commit 03507c4) and its severity-copula copy `punishment_multinomial_severity_copula.joblib`; GNN punisher `artifacts/artificial_humans/punishment_rnn_edge_50ep_doubled/model/architecture_node+edge+rnn__dataset_50ep_doubled.pt` (`x_encoding = prev_contribution, prev_punishment, is_first`). Contributor and switch slots untouched.

**Target rows:** RCE (new, protected from this branch on: §2) and RPA, the two rows that are by definition the distribution of p_t given c_t; RCC (the population of punished full contributors) and RCB (whose rate `p / (20 - c_t)` mixes a c_t denominator with a c_{t-1}-driven numerator). Every other row is a watch item, because the punisher artifact sits in every stack and this change moves them all -- which is why the ledger is re-baselined (stage D) rather than gated.

### Hypothesis

**The finding (verified, `punisher_lag_check.md`).** Both artificial punishers decide round t's punishment from round t-1's contribution, while the human manager punishes round t's. In the data, `common_good_t = 1.6 * sum(c_t) - sum(p_t)` holds in 100.0% of valid rows and `payoff_t = 20 - c_t - p_t + common_good_t / n` in 89% (57% with p_{t-1}), so the `punishment` column at (episode, t, player) is the punishment applied to round t's contribution. Behaviourally: corr(p_t, c_t) = -0.277 vs corr(p_t, c_{t-1}) = -0.189 (n = 8,431 after the eval suite's pair dedup); OLS `p_t ~ c_t + c_{t-1}` gives -0.242 on c_t and +0.067 on c_{t-1}; and P(p > 0 | c_t <= 4, c_{t-1} = 20) = 0.571 (n 49) against P(p > 0 | c_t = 20, c_{t-1} <= 4) = 0.179 (n 39). The human manager reacts to this round.

**Where the lag lived.** Training: `generic/data.py::shift` rolls every tensor by one round, so `prev_contribution[t] = contribution[t-1]`; the GNN config's `x_encoding` and the linear grid's legal set (`validate_feature_legality`, which hard-erred on any current-valued feature for the punishment target) only ever admitted the shifted tensor, and the bundle `punishment_multinomial_best_with_contr.joblib` accordingly carries `prev_contribution` ("with_contr" meant the lag). Simulation: `linear_ah.py::get_punishments` and `api_manager.py::create_data` both had round t's contribution at the last index and did not read it; the env's ordering (`update_contribution` -> `punish` -> the `prev_` copy in `step`) was already right, in the simulation loop and in RL training. Training and simulation were consistently lagged: a faithful model of the wrong mechanism, inherited from the contributor's leak rule (where `prev_` is correct) with no behavioural rationale recorded; `reports/recommended_ah_configs_50ep.md` had flagged it and recommended `contribution` (current), never implemented.

**Every sim shows it.** OLS of p on (c_t, c_{t-1}): human -0.242 / +0.067; PR 179 +0.011 / -0.083; PR 181 +0.002 / -0.081; PR 170 -0.006 / -0.079; `main` gnn x gnn x lin_multinomial +0.008 / -0.078; `main` with the GNN punisher -0.009 / -0.018. P(p > 0 | c_t = 20): human 0.038, sims 0.215-0.334. The cross-tab flips sign in the multinomial stacks (0.34 / 0.23 against the human 0.18 / 0.57).

**Behavioural rationale (one sentence, §5):** the manager punishes what a player just contributed, so the punisher must read round t's contribution -- the rows that should move are RPA, RCC, RCB and RCE.

**The change.** `contribution` (round t, the same-round input the manager has) is admitted for the punishment target in both families; `prev_contribution`, `prev_punishment`, `round_number` and `is_first` stay; `punishment`, `payoff`, `common_good` at round t and everything built from them remain illegal (they contain p_t). Both punishers are retrained with the new feature (stage C), the severity copula is re-stamped on the new multinomial bundle, the most promising stacks are re-run, and the ledger is re-baselined with RCE as the first protected row (stage D).

### Artifact naming contract (stages B-D depend on these exact names)

| what | path |
|---|---|
| baseline config | `configs/training/baselines/punishment/multinomial_current_contr.yml` (the `best_with_contr` grid entry, `contribution` added; one feature set: `contribution, prev_contribution, prev_punishment, round_number, is_first`) |
| baseline artifact | `artifacts/baselines/punishment_multinomial_current_contr.joblib` (`run_baseline_cv.py` on the config, then `inspect_best_model.py <cv csv> --config <config> --save-best --name punishment_multinomial_current_contr.joblib`; commands in the config header) |
| severity-copula copy | `artifacts/baselines/punishment_multinomial_current_contr_severity_copula.joblib` (`punishment_copula_rho.py --roundtrip --bundle <baseline artifact> --out <this path>`; the `--bundle` / `--out` flags are new on this branch, the defaults are the old paths) |
| GNN config | `configs/training/artificial_humans/punishment/rnn_edge_50ep_doubled_current_contr.yml` (`rnn_edge_50ep_doubled.yml` with `contribution` numeric first in `x_encoding`) |
| GNN artifact dir | `artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_current_contr/` -- set explicitly as `output_dir` in the config (train.py takes the artifact dir from `output_dir`, not from the config's file name); the model file inside is `model/architecture_node+edge+rnn__dataset_50ep_doubled.pt`, from the unchanged `labels` |

## 2. Plan

| # | step | stage |
|---|---|---|
| 1 | Branch from PR #181, merge the RCE branch, drop RCF, verify the eval-suite tests locally. | A (done) |
| 2 | Admit round t's contribution for the punishment target in both families: `PUNISHMENT_LEGAL_CURRENT` / `illegal_current_features` in `handcrafted_grid.py`, the load-time assert in `linear_ah.py`, `punishment_baseline.py` FEATS; document the timing in `create_data`, `_pool_from_rounds`, `notes/baseline_feature_defs.md`. No change to the env ordering (already correct) or to the contributor / switch slots. | A (done) |
| 3 | The two configs of the naming contract; `--bundle` / `--out` on `punishment_copula_rho.py` so the stamped copy can carry its name. | A (done) |
| 4 | Tests that round t's contribution reaches the punisher at round t on the linear and the GNN path, plus the legality rule (`src/aimanager/tests/test_punisher_current_contribution.py`); all tests green on Raven. | A (done) |
| 5 | Protocol: RCE as the first protected row in `notes/autoresearch.md` §2, the row count 21 -> 22, the re-baseline paragraph, §5's feature rule corrected. | A (done) |
| 6 | Train the multinomial punisher on the new config, stamp the copula, train the GNN punisher on Raven; record CV / test log-loss against the old bundle's 1.3031 (floor 1.3561) and the GNN's 1.2030. | C (done) |
| 7 | Re-run the most promising stacks with the new punishers (23-family protocol), fetch, evaluate. | D (done: cases a-e, `punisher-current-contribution-cases.md`) |
| 8 | Re-baseline the ledger: new score matrix and ranking with RCE, reset the confirmed scores of the frontier PRs, record the before/after of every row for the top stack. | D (done: the before/after of every row for all six runs; the ledger's top-stack figures and the frontier PRs' baselines reset in `notes/autoresearch.md` §3; the full 32-stack matrix was not re-run, see caveat 5) |

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-18 | (before) lagged punisher, PR #181 stack (case b, rescored with 22 rows) | RCE 0.9064, RPA 1.2271, RCC 1.6133, RCB 2.0867 | 13/22 | 1.0955 | baseline |
| 2026-09-18 | current-contribution multinomial + copula, PR #181 stack (case b) | RCE 0.8942, RPA 0.6930, RCC 1.5298, RCB 1.5454 | 13/22 | 1.0357 | **REBASELINE** |
| 2026-09-18 | (before) lagged punisher, PR #179 stack (case a) | RCE 1.0997, RPA 1.3112, RCC 1.6596, RCB 2.3152 | 12/22 | 1.0989 | baseline |
| 2026-09-18 | current-contribution multinomial + copula, PR #179 stack (case a) | RCE 1.2682, RPA 0.7232, RCC 1.7110, RCB 1.4744 | 9/22 | 1.1079 | **REBASELINE** |
| 2026-09-18 | (before) lagged punisher, PR #177 stack (case c) | RCE 0.9886, RPA 1.2750, RCC 1.3266, RCB 1.4621 | 12/22 | 1.1187 | baseline |
| 2026-09-18 | current-contribution multinomial + copula, PR #177 stack (case c) | RCE 0.8508, RPA 0.7350, RCC 1.3814, RCB 0.9662 | 12/22 | 1.1012 | **REBASELINE** |
| 2026-09-18 | (before) lagged punisher, PR #174 stack (case d) | RCE 0.8320, RPA 1.3141, RCC 1.0880, RCB 1.9103 | 12/22 | 1.2049 | baseline |
| 2026-09-18 | current-contribution multinomial + copula, PR #174 stack (case d) | RCE 0.7046, RPA 0.7361, RCC 1.1695, RCB 1.3315 | 8/22 | 1.1880 | **REBASELINE** |
| 2026-09-18 | (before) lagged plain multinomial, main gnn x gnn (case e) | RCE 1.0918, RPA 1.2683, RCC 1.5388, RCB 1.9282 | 11/22 | 1.7292 | baseline |
| 2026-09-18 | current-contribution plain multinomial, main gnn x gnn (case e) | RCE 0.9976, RPA 0.6838, RCC 1.6181, RCB 1.0206 | 13/22 | 1.7405 | **REBASELINE** |
| 2026-09-18 | (before) lagged GNN punisher, main gnn x gnn (case e) | RCE 0.9860, RPA 1.5552, RCC 1.4074, RCB 1.8858 | 7/22 | 1.8659 | baseline |
| 2026-09-18 | current-contribution GNN punisher, main gnn x gnn (case e) | RCE 1.0048, RPA 0.8884, RCC 1.3005, RCB 1.2133 | 8/22 | 1.7094 | **REBASELINE** |

### Stage C: training and simulation

Stage C trained the two punishers and re-stamped the copula (commits `224c86e`, `44997a6`); the simulations of plan step 7 are stage D's. Remote isolated dir: `~/repros/ai-runs/punisher-current-contr-train` (left in place for stage D).

**Multinomial punisher** (`multinomial_current_contr.yml`, locally, 4-fold CV on the locked train split, seed 38381, 7,345 rows). The full 5-feature set ranks first; no block-OFF subset outranks it, so the artifact is the rank-1 row (`data/baselines/punishment_cv_multinomial_current_contr.csv`, force-added past the `data/` ignore):

| rank | features | CV log loss |
|---|---|---|
| 1 | contribution, prev_contribution, prev_punishment, round_number, is_first | 1.3465 (se 0.058) |
| 2 | contribution, prev_contribution, prev_punishment | 1.3534 |
| 3 | floor | 1.4355 |
| 4 | round_number, is_first | 1.4411 |

Against the lagged bundle `punishment_multinomial_best_with_contr.joblib` on the same split and seed: CV 1.3465 vs 1.3661, train 1.2716 vs 1.3007, locked test 1.2468 vs 1.3031 (floor 1.3561): the old bundle beat the floor by 0.053 on test, the new one by 0.109. Permutation importance (in-sample delta log loss): prev_punishment 0.195, contribution 0.122, prev_contribution 0.026, round_number 0.025, is_first 0.012. The fitted coefficients say where the lag went: on the p = 0 logit, standardized, `contribution` +1.165 (per raw unit +0.180) against `prev_contribution` -0.339 (per raw unit -0.054); in the old bundle `prev_contribution` carried +0.411 (per raw unit +0.065). The level-weighted mean coefficient over the p > 0 classes is -0.183 on `contribution` and +0.015 on `prev_contribution`: the current contribution now carries the whole "contributed more -> punished less and less severely" response, and the lag flips to a small opposite-sign residual, the same pattern as the human OLS (-0.242 / +0.067).

**Severity copula** (`punishment_copula_rho.py --roundtrip` on the new bundle, 15,291 within-cell pairs): rho = 0.4273, SE 0.0459, 95% CI [0.3514, 0.5283], against the old 0.3508 [0.2780, 0.4232]. Round-trip gate PASS (max |bias| 0.013, tolerance 0.03); out-of-sample MLE on the test split 0.332 (old script run: not recorded). The stamped copy is weight-identical to the plain bundle plus the eight `copula_*` keys (the script's reload assert plus an explicit coef/intercept/scaler comparison). The prediction that rho would *drop* was wrong: the within-round co-movement of punishments is not explained by the shared current contribution, it gets stronger once each marginal conditions on c_t. A plausible reading: the manager's per-round severity level is a group-level mood that the marginal cannot absorb, and conditioning on c_t sharpens the marginals so the residual latent correlation is less attenuated.

**GNN punisher** (`rnn_edge_50ep_doubled_current_contr.yml`, Raven job 30304102, 7 min 47 s on one A100 for the 5-fold CV plus the full fit, within the ~12 min budget; wandb `uzh2g9ju`). Final-epoch CV log loss 1.1756 (best-epoch 1.1755 +- 0.1027 over folds) against 1.2030 (1.2028 +- 0.0868) for the lagged `punishment_rnn_edge_50ep_doubled`; `punishment_baseline.py`'s `GNN_REF` is now 1.1756. Artifact fetched to `artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_current_contr/` (model, metrics, confusion matrix; LFS).

**Mechanism check** (`scripts/data_analysis/punisher_mechanism_check.py`, teacher-forced on the 50 single-copy human games, 8,914 valid rows, no simulation; the human row is the observed data on the same rows, the model rows are predicted P(p > 0) and E[p] under each model's own round-0 defaults; the linears run locally, the GNNs on Raven's login node). OLS is of (predicted) expected punishment on c_t and c_{t-1} over the 8,431 rows with a valid previous contribution; E[p | p>0] per band is sum E[p] / sum P(p>0):

| punisher | P(p>0 \| c_t=20) | P(p>0 \| c_t<=4) | P(p>0 \| c_t=20, c_{t-1}<=4) / (c_t<=4, c_{t-1}=20) | E[p \| p>0] 0-4 / 5-9 / 10-14 / 15-19 / 20 | OLS c_t / c_{t-1} | NLL |
|---|---|---|---|---|---|---|
| human | 0.038 | 0.467 | 0.179 / 0.571 | 7.99 / 4.98 / 4.27 / 3.87 / 7.00 | -0.242 / +0.067 | -- |
| lin old (lagged) | 0.207 | 0.380 | 0.561 / 0.282 | 6.57 / 5.82 / 5.48 / 5.36 / 5.29 | +0.054 / -0.172 | 1.328 |
| lin new (c_t) | 0.136 | 0.448 | 0.309 / 0.651 | 7.20 / 5.60 / 4.77 / 4.22 / 3.94 | -0.125 / -0.030 | 1.283 |
| gnn old (lagged) | 0.158 | 0.379 | 0.386 / 0.302 | 6.49 / 5.77 / 5.44 / 5.39 / 5.97 | +0.019 / -0.131 | 1.188 |
| gnn new (c_t) | 0.105 | 0.432 | 0.301 / 0.532 | 7.26 / 5.48 / 4.80 / 4.63 / 5.11 | -0.128 / -0.022 | 1.144 |

Both new punishers now respond to the current round: the OLS on c_t goes from ~0 to about -0.13 (human -0.24) and the lag coefficient from -0.13/-0.17 to ~-0.03 (human +0.07); the cross-tab flips to the human ordering (punish the low contributor who was high last round more than the reverse); the severity gradient across bands reappears (7.2 -> 4.2/4.6 against the flat ~5.5 of the lagged models; human 8.0 -> 3.9). What remains short: P(p>0 | c_t=20) is 0.10-0.14 against the human 0.04 -- a linear-in-c_t logit cannot produce the human step at 20 (E[p | p>0] at c_t = 20 is 7.0 in the humans, a few heavy punishments of full contributors, which the models spread into many light ones), and the response slope on c_t is half the human one. NLL (in-sample, 40 of the 50 episodes trained on) drops 1.328 -> 1.283 and 1.188 -> 1.144.

### Stage D: re-baseline

**Verdict: `[REBASELINE]`.** No gate applies (§2 re-baseline paragraph): the punisher artifact sits in every stack, so this branch resets the ledger rather than competing against it. The five `_curpun` configs of the cases note (`punisher-current-contribution-cases.md`) were run with the 23-family protocol (seed 42, 100 episodes, 24 rounds; Raven jobs 30305004 a, 30305005 b, 30305017 c, 30305018 d, 30305019 e, about two minutes each on one A100) and evaluated locally with the merged 22-row suite (500 repeats, seed 42). Cases a, b and e ran from this checkout (`~/repros/ai-runs/punisher-current-contr`); c and d, whose gaussian_mlp contributors unpickle classes that exist only on the gmlp lineage, ran from the checkout of branch `auto/punisher-current-contribution-gmlp` (`d639970` = `origin/auto/contribution-inflated-gmlp` `c91f9bb` + stage A's `7ad1ddd`, `c74bc0e`, `ce70a09` cherry-picked, plus the retrained copula punisher bundle and the two configs; `~/repros/ai-runs/punisher-current-contr-gmlp`). Tables: `plots/data_analysis/evaluation/punisher_current_contr/rebaseline_table.{csv,md}`, `rce_bands.csv`, `mechanism_selfplay.csv`; sim dirs `plots/simulation/<source>_curpun/`.

**New baselines (before -> after; before = the source sim rescored with the 22-row suite):**

| case | stack | punisher | mean | rows <= 1 | RCE | RPA | RCB | RCC |
|---|---|---|---|---|---|---|---|---|
| a | PR 179 group vnode x joint-exodus GNN switch | lin_multinomial copula | 1.0989 -> **1.1079** | 12 -> **9** | 1.0997 -> **1.2682** | 1.3112 -> 0.7232 | 2.3152 -> 1.4744 | 1.6596 -> 1.7110 |
| b | PR 181 stimulus skip x joint-exodus GNN switch | lin_multinomial copula | 1.0955 -> **1.0357** | 13 -> **13** | 0.9064 -> **0.8942** | 1.2271 -> 0.6930 | 2.0867 -> 1.5454 | 1.6133 -> 1.5298 |
| c | PR 177 inflated gmlp copula x k-one-hot switch | lin_multinomial copula | 1.1187 -> **1.1012** | 12 -> **12** | 0.9886 -> **0.8508** | 1.2750 -> 0.7350 | 1.4621 -> 0.9662 | 1.3266 -> 1.3814 |
| d | PR 174 gmlp v2 copula x k-one-hot switch | lin_multinomial copula | 1.2049 -> **1.1880** | 12 -> **8** | 0.8320 -> **0.7046** | 1.3141 -> 0.7361 | 1.9103 -> 1.3315 | 1.0880 -> 1.1695 |
| e | main gnn x gnn (sweep top stack) | lin_multinomial (plain) | 1.7292 -> **1.7405** | 11 -> **13** | 1.0918 -> **0.9976** | 1.2683 -> 0.6838 | 1.9282 -> 1.0206 | 1.5388 -> 1.6181 |
| e | main gnn x gnn (sweep top stack) | gnn | 1.8659 -> **1.7094** | 7 -> **8** | 0.9860 -> **1.0048** | 1.5552 -> 0.8884 | 1.8858 -> 1.2133 | 1.4074 -> 1.3005 |

**All 22 rows, every case** (bold: moved by more than 0.1; band changes in parentheses):

| row | a_vnode | b_skip | c_infl | d_kexo | e_lin | e_gnn |
|---|---|---|---|---|---|---|
| CA | **0.961 -> 1.198 (<= 1 -> 1-2)** | 0.848 -> 0.860 | 1.442 -> 1.508 | 1.609 -> 1.703 | **0.772 -> 0.969** | 0.842 -> 0.788 |
| CB | **0.953 -> 1.213 (<= 1 -> 1-2)** | 0.830 -> 0.788 | **1.017 -> 1.174** | **0.936 -> 1.144 (<= 1 -> 1-2)** | **0.691 -> 0.850** | 0.685 -> 0.681 |
| CC | 0.920 -> 0.991 | 0.821 -> 0.889 | **0.860 -> 0.968** | 1.036 -> 1.134 | **1.606 -> 1.821** | **1.712 -> 1.569** |
| CD | **0.922 -> 1.153 (<= 1 -> 1-2)** | 0.796 -> 0.810 | **0.961 -> 1.100 (<= 1 -> 1-2)** | **1.116 -> 1.322** | **0.650 -> 0.843** | 0.670 -> 0.664 |
| CE | **1.111 -> 1.010** | **0.910 -> 1.057 (<= 1 -> 1-2)** | 0.803 -> 0.850 | 0.947 -> 1.022 (<= 1 -> 1-2) | 1.332 -> 1.311 | 1.356 -> 1.301 |
| CF | 1.076 -> 1.026 | 0.887 -> 0.828 | 0.855 -> 0.884 | 1.345 -> 1.369 | 0.814 -> 0.891 | 0.852 -> 0.809 |
| CG | 0.899 -> 0.884 | **1.310 -> 1.554** | 1.842 -> 1.854 | **2.079 -> 1.665 (2-5 -> 1-2)** | **9.850 -> 10.282** | **10.138 -> 9.221** |
| SA | 0.864 -> 0.866 | 0.784 -> 0.785 | 0.916 -> 0.887 | **0.810 -> 1.082 (<= 1 -> 1-2)** | 0.721 -> 0.821 | **0.659 -> 0.883** |
| SB | **1.111 -> 1.007** | 1.040 -> 1.006 | 0.965 -> 0.896 | 0.891 -> 0.931 | 0.754 -> 0.837 | **0.744 -> 0.900** |
| SC | **0.977 -> 1.356 (<= 1 -> 1-2)** | **1.023 -> 1.427** | **1.233 -> 1.579** | 0.980 -> 1.076 (<= 1 -> 1-2) | **3.270 -> 3.440** | **3.455 -> 2.973** |
| PA | **0.582 -> 0.706** | 0.630 -> 0.660 | **0.621 -> 0.782** | **0.619 -> 0.856** | 0.634 -> 0.648 | 1.267 -> 1.309 |
| PB | 0.919 -> 0.912 | 0.901 -> 0.969 | 0.952 -> 1.014 (<= 1 -> 1-2) | 0.960 -> 1.011 (<= 1 -> 1-2) | 0.878 -> 0.808 | 1.114 -> 1.150 |
| PC | 0.865 -> 0.933 | 0.877 -> 0.907 | 0.891 -> 0.950 | 0.888 -> 0.975 | 0.778 -> 0.764 | **1.160 -> 0.995 (1-2 -> <= 1)** |
| PD | **0.775 -> 0.999** | **0.854 -> 0.722** | 0.919 -> 0.920 | **0.865 -> 0.732** | **2.935 -> 3.205** | **2.823 -> 2.690** |
| RCA | 1.400 -> 1.423 | **1.469 -> 1.633** | **1.862 -> 1.690** | 3.507 -> 3.428 | 2.035 -> 1.938 (2-5 -> 1-2) | **2.367 -> 2.086** |
| RCB | **2.315 -> 1.474 (2-5 -> 1-2)** | **2.087 -> 1.545 (2-5 -> 1-2)** | **1.462 -> 0.966 (1-2 -> <= 1)** | **1.910 -> 1.332** | **1.928 -> 1.021** | **1.886 -> 1.213** |
| RCC | 1.660 -> 1.711 | 1.613 -> 1.530 | 1.327 -> 1.381 | 1.088 -> 1.169 | 1.539 -> 1.618 | **1.407 -> 1.301** |
| RCD | 1.340 -> 1.428 | **2.205 -> 1.309 (2-5 -> 1-2)** | 1.353 -> 1.262 | 0.732 -> 0.760 | **2.772 -> 2.933** | 2.893 -> 2.846 |
| RCE | **1.100 -> 1.268** | 0.906 -> 0.894 | **0.989 -> 0.851** | **0.832 -> 0.705** | 1.092 -> 0.998 (1-2 -> <= 1) | 0.986 -> 1.005 (<= 1 -> 1-2) |
| RSA | 1.355 -> 1.284 | **1.236 -> 1.070** | **1.335 -> 1.166** | **1.317 -> 1.196** | 0.909 -> 0.944 | **1.137 -> 1.004** |
| RPA | **1.311 -> 0.723 (1-2 -> <= 1)** | **1.227 -> 0.693 (1-2 -> <= 1)** | **1.275 -> 0.735 (1-2 -> <= 1)** | **1.314 -> 0.736 (1-2 -> <= 1)** | **1.268 -> 0.684 (1-2 -> <= 1)** | **1.555 -> 0.888 (1-2 -> <= 1)** |
| RPB | 0.758 -> 0.809 | 0.847 -> 0.847 | 0.731 -> 0.810 | 0.727 -> 0.791 | **0.814 -> 0.665** | 1.343 -> 1.332 |
| mean | **1.099 -> 1.108** | **1.096 -> 1.036** | **1.119 -> 1.101** | **1.205 -> 1.188** | **1.729 -> 1.740** | **1.866 -> 1.709** |
| rows <= 1 | 12 -> 9 | 13 -> 13 | 12 -> 12 | 12 -> 8 | 11 -> 13 | 7 -> 8 |

**RCE band slopes** (OLS of the next-round contribution change on the punishment received, per own-contribution band, RCB population; human +0.140 / +0.104 / -0.077 / -0.161, ++--):

| case | before 0-4 / 5-9 / 10-14 / 15-19 | signs | after 0-4 / 5-9 / 10-14 / 15-19 | signs |
|---|---|---|---|---|
| a_vnode | +0.062 / +0.012 / -0.008 / -0.037 | ++-- (4/4) | +0.049 / -0.007 / -0.017 / +0.040 | +--+ (2/4) |
| b_skip | +0.073 / +0.054 / -0.047 / -0.025 | ++-- (4/4) | +0.095 / +0.020 / -0.058 / -0.160 | ++-- (4/4) |
| c_infl | +0.033 / +0.061 / -0.051 / -0.231 | ++-- (4/4) | +0.068 / +0.109 / -0.020 / -0.193 | ++-- (4/4) |
| d_kexo | +0.103 / +0.037 / -0.116 / -0.174 | ++-- (4/4) | +0.120 / +0.071 / -0.112 / -0.205 | ++-- (4/4) |
| e_lin | +0.062 / +0.043 / +0.047 / -0.042 | +++- (3/4) | +0.064 / +0.092 / +0.044 / -0.098 | +++- (3/4) |
| e_gnn | +0.090 / +0.021 / -0.005 / -0.000 | ++-- (4/4) | +0.073 / +0.050 / +0.013 / +0.029 | ++++ (2/4) |

Sign checks against the human ++--: b, c and d keep 4/4 and every one of their bands moves toward the human magnitude (b's 15-19 band -0.025 -> -0.160 against the human -0.161; d's 0-4 +0.103 -> +0.120 and 15-19 -0.174 -> -0.205); e_lin keeps its 3/4 (the 10-14 band stays positive at +0.044) with the 5-9 and 15-19 bands roughly doubling toward the human. Two cases lose sign matches: a_vnode's 5-9 band goes to -0.007 and its 15-19 band flips to +0.040 (both were within 0.04 of zero before, so these are the small-magnitude bands going through zero, not a reversal of a strong response), and e_gnn's 10-14 / 15-19 bands go from -0.005 / -0.000 to +0.013 / +0.029, so it now reads ++++. These would be RCE violations for a gated experiment (sign flips away from the human), which is exactly why this branch is a re-baseline rather than a gated one: from here on, the protected-row check for a's and e_gnn's successors starts from the after column.

**The rows the lag fabricated directly (RPA, RCC, RCB):**

- **RPA** (the distribution of p_t given c_t) is the clean signal: 1.23-1.56 -> 0.68-0.89 in every case, band 1-2 -> <= 1 in all six. The row was a pure artefact of the lag.
- **RCB** (rate `p / (20 - c_t)`, whose denominator was a c_t while its numerator was driven by c_{t-1}) drops 0.50-0.91 everywhere: a 2.315 -> 1.474 and b 2.087 -> 1.545 (both 2-5 -> 1-2; PR 181's declared target, missed by 4.34% there, is now cleared by the punisher fix alone), c 1.462 -> 0.966 (<= 1), e_lin 1.928 -> 1.021. PR 181's contributor-side hypothesis was aimed at a row that the punisher was fabricating.
- **RCC** (the population of punished full contributors) does *not* improve: +0.05 to +0.08 in a, c, d, e_lin, -0.08 / -0.11 in b and e_gnn. The self-play table below says why: P(p > 0 | c_t = 20) falls from 0.22-0.33 to 0.12-0.16 but the human rate is 0.038, so full contributors are still punished three to four times too often, and the row measures who they are, not how many. The teacher-forced check already put this at the functional form (a linear-in-c_t logit cannot produce the step at 20); it survives self-play unchanged.

**Self-play mechanism table** (`punisher_mechanism_check.py --sim`, realised punishments of each sim's 19,200 agent-rounds, same statistics as stage C's teacher-forced table; human = observed data, 8,914 rows):

| sim | P(p>0 \| c_t=20) | P(p>0 \| c_t<=4) | cross-tab (c_t=20, c_{t-1}<=4) / (c_t<=4, c_{t-1}=20) | E[p \| p>0] 0-4 / 5-9 / 10-14 / 15-19 / 20 | OLS c_t / c_{t-1} |
|---|---|---|---|---|---|
| human | 0.038 | 0.467 | 0.179 / 0.571 | 7.99 / 4.98 / 4.27 / 3.87 / 7.00 | -0.242 / +0.067 |
| a_vnode before | 0.217 | 0.356 | 0.344 / 0.232 | 5.84 / 6.35 / 6.03 / 5.78 / 5.10 | +0.011 / -0.083 |
| a_vnode after | 0.134 | 0.450 | 0.161 / 0.609 | 7.13 / 5.73 / 5.12 / 4.01 / 3.95 | -0.145 / +0.001 |
| b_skip before | 0.215 | 0.364 | 0.397 / 0.300 | 6.10 / 5.98 / 5.69 / 5.74 / 5.13 | +0.002 / -0.081 |
| b_skip after | 0.122 | 0.473 | 0.167 / 0.612 | 7.05 / 5.83 / 4.88 / 3.92 / 3.84 | -0.142 / -0.019 |
| c_infl before | 0.223 | 0.363 | 0.405 / 0.325 | 6.16 / 6.16 / 5.55 / 5.53 / 5.16 | +0.005 / -0.087 |
| c_infl after | 0.144 | 0.456 | 0.151 / 0.603 | 6.94 / 5.75 / 4.82 / 3.91 / 3.57 | -0.127 / -0.020 |
| d_kexo before | 0.256 | 0.356 | 0.611 / 0.486 | 6.05 / 6.29 / 5.51 / 5.23 / 6.04 | +0.004 / -0.085 |
| d_kexo after | 0.157 | 0.449 | 0.167 / 0.889 | 6.71 / 5.88 / 4.58 / 3.93 / 3.84 | -0.136 / -0.011 |
| e_lin before | 0.246 | 0.368 | 0.370 / 0.230 | 6.45 / 5.89 / 5.89 / 5.28 / 6.43 | +0.008 / -0.078 |
| e_lin after | 0.141 | 0.476 | 0.217 / 0.592 | 7.20 / 5.55 / 4.65 / 3.89 / 4.24 | -0.118 / -0.032 |
| e_gnn before | 0.334 | 0.401 | 0.378 / 0.494 | 6.29 / 5.86 / 5.97 / 6.56 / 6.36 | -0.009 / -0.018 |
| e_gnn after | 0.142 | 0.523 | 0.261 / 0.585 | 7.67 / 6.17 / 5.44 / 5.79 / 5.11 | -0.170 / -0.017 |

Closed loop, the fix does what the teacher-forced check predicted: the OLS on c_t goes from ~0 to -0.12 ... -0.17 (human -0.24; the teacher-forced new punishers gave -0.13), the lag coefficient from -0.08 to ~-0.02 (human +0.07), the cross-tab flips to the human ordering in every case (0.15-0.26 / 0.59-0.89 against the human 0.18 / 0.57, where before it was 0.34-0.61 / 0.23-0.49), and the severity gradient over the bands reappears in the linear cases (7.1 -> 3.9 against the human 8.0 -> 3.9; before, flat at ~5.5-6.3). The GNN punisher keeps a flatter severity profile (7.7 / 6.2 / 5.4 / 5.8 / 5.1) but has the steepest c_t slope. What stays short in every case: P(p > 0 | c_t = 20) at 0.12-0.16 against 0.04, E[p | p > 0] at c_t = 20 at 3.6-5.1 against 7.0 (the humans punish full contributors rarely but hard; the models often and lightly), and the c_t slope at half to two thirds of the human one.

**What improved and what got worse -- an honest reading.** The punisher rows and the punishment-response rows the lag fabricated move as they should: RPA into the noise band everywhere, RCB down by 0.5-0.9 everywhere, RSA down in five of six, RCD 2.205 -> 1.309 in b, RCA down in c, e_lin, e_gnn. The GNN punisher stack (e_gnn) is the largest net winner (mean 1.866 -> 1.709, CG 10.14 -> 9.22, SC 3.46 -> 2.97, RCA 2.37 -> 2.09). What got worse is systematic, not noise: the contribution marginals (CA/CB/CD) rise by ~0.2 in a, d and e_lin (a loses three <= 1 rows there), PA rises 0.12-0.24 in a, c, d, and SC rises by 0.35-0.40 in a, b, c. These are the contributor and switch models meeting a punisher that now behaves differently from the one they were simulated against when their PRs were accepted: a punisher that punishes low contributors harder and full contributors less shifts the contribution distribution the contributors produce in closed loop (more compliance from low contributors, a different punished population), and the group composition the switch model sees with it. None of the contributor / switch models was retrained against the new punisher (they are trained on human data, not on the punisher, so retraining is not required for correctness -- but their closed-loop states have moved, and their reported baselines with them). The means barely move for the linear-punisher stacks (-0.060 to +0.011); rows <= 1 fall in a (12 -> 9) and d (12 -> 8) and rise in e (11 -> 13, 7 -> 8).

**Caveats.**

1. Cases c and d ran on the gmlp code tree (`auto/punisher-current-contribution-gmlp`) with stage A's commits cherry-picked, not on this branch's tree; the cherry-picks were clean and the four linear-path tests of `test_punisher_current_contribution.py` pass there. Their before sims were produced by the same lineage, so before/after is like-for-like in code except for the punisher fix.
2. The severity-copula rho rose from 0.351 to 0.427 when the marginal gained c_t (stage C); the copula-stamped punisher in a-d therefore carries a stronger within-round latent than before. PD (the row the copula targets) moves -0.13 in b and d, +0.22 in a, 0 in c: no consistent direction, so the stronger rho is not visibly buying or costing anything at the row level.
3. The PR-lineage stacks' contributor and switch models were not retrained against the new punisher (see above); the new baselines are the closed-loop states of the old contributor / switch models with the new punisher.
4. The two sign-flip cases (a's 15-19 band, e_gnn's 10-14 / 15-19 bands) are bands whose |slope| was below 0.04 before and after; the protected-row rule would count them as flips, and successors of those stacks inherit the after column as their RCE baseline.
5. The sweep's score matrix (`23_stack_sweep_updated`, 32 stacks x 21 rows) was not re-run; only the top stack (e) has a post-fix row. The ranking rule of §3 stays defined on the pre-fix matrix until the maintainer refreshes it.

**What this leaves for a successor.**

1. **Retrain / re-evaluate the frontier contributors against the fixed punisher.** The contribution marginals and SC in a, b, c moved because the closed-loop states moved; PR 179's group vnode and PR 181's stimulus skip were chosen in a world with a lagged punisher. The first thing to check is whether their copula recalibrations (rho_p 0.044 / 0.039) still hold with the new punisher, and whether the stimulus skip's CG cost (1.31 -> 1.55 here) is now worth its RCB gain given that the punisher fix alone clears RCB's band.
2. **The P(p > 0 | c_t = 20) residual** (0.12-0.16 vs 0.038) is a functional-form limit shared by both families and the reason RCC does not move. A punisher that separates "whether" (a full-contributor indicator, or a hinge at 20) from "how much" is the obvious next punisher-slot experiment; its target rows are RCC and RPA-at-20.
3. **The copula rho question.** Rho rose with c_t in the marginal, against the prediction; the plausible reading is a per-round severity mood the marginal cannot absorb. Whether that mood is a manager-level latent (one per episode) or a round-level one is testable on the human data with the copula scripts and matters for PD.
4. **The GNN punisher is now competitive** (e_gnn mean 1.709 vs e_lin 1.740, but 8 vs 13 rows <= 1; PA/PB/RPB worse, CG/SC/RCA/RSA better). A sweep row for it inside the frontier stacks (copula-stamped or not) would tell whether its steeper c_t response is worth its worse marginals.
5. **RCE in the gnn-punisher stack reads ++++.** Whether the contributor's withdrawal response at high contribution is being masked by the GNN punisher's flat severity profile is a one-run question.

## 4. Notes

1. Stage A: the lag was a modelling error, not an indexing bug -- training and simulation were consistently lagged, so no artifact trained before this branch is affected in any way other than by conditioning on the wrong round. The fix is a legal-set change plus two configs; every code path already carried c_t at the right index.
2. Stage A: `punishment_baseline.py` (the GNN-parity diagnostic) now lists `contribution` too, so its printed baseline is comparable to the new GNN config, not the old one; its `GNN_REF` constant still quotes the lagged GNN's 1.2030 until stage C replaces it.
3. Stage A: RCF was dropped after the merge (its `KINDS` entry, methods, tests, and definition paragraph); the comparison report keeps its RCF discussion as a record of the choice.
4. Stage C: the copula prediction failed in the predicted direction -- rho rose from 0.351 to 0.427 with c_t in the marginal instead of dropping -- so the copula copy is stamped as calibrated, not as a smaller correction; stage D should read RPA/RCC/RCB/RCE from the sims before anyone reasons further about why. The residual gap the mechanism check leaves (P(p>0 | c_t = 20) 0.10-0.14 vs human 0.04, half the human OLS slope on c_t) is a functional-form limit of both families, a separate experiment, not a stage of this one.
5. Stage D: the sims confirm the teacher-forced prediction in closed loop (OLS on c_t -0.12 ... -0.17 in every stack; cross-tab in the human ordering everywhere) and the rows the lag fabricated move accordingly (RPA to <= 1 in all six runs, RCB down 0.5-0.9 everywhere), but RCC does not move because the residual P(p > 0 | c_t = 20) = 0.12-0.16 (human 0.04) survives self-play. The re-baseline is taken as is -- no gate, no second run -- and the contributor-side collateral (CA/CB/CD +0.2 in a, d, e_lin; SC +0.35-0.40 in a, b, c) is recorded as the new closed-loop state of the old contributor / switch models, not as a regression to fix here.
6. Stage D: cases c and d were run from a second checkout of the gmlp lineage (`auto/punisher-current-contribution-gmlp`, d639970) because the `git worktree`-based runner needs the gaussian_mlp classes; the branch is pushed so the code that produced those two sims is reproducible, and it is not opened as a PR. The `squeue -u certuer` in the cases note is wrong for this account (`squeue --me`).
