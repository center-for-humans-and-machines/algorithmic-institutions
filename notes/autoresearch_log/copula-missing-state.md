# Missing state behind the copula: how much of rho is observable group state?

## 1. Declaration

**Slot:** contribution (measurement only -- no model is trained, no simulation is run).

**Branch:** `auto/copula-missing-state`, from `origin/auto/punisher-current-contribution` (the fixed punisher, the RCE row, the PR #181 stimulus-skip contributor trunk and the copula scripts). One of three parallel experiments on the "copula question"; the siblings are `auto/copula-seed-ensemble` and `auto/copula-closed-loop-variance`. Isolated Raven dir `~/repros/ai-runs/copula-missing-state` (login node only, used for the one PyG step) -- delete it when this PR closes.

**Base model:** the PR #181 stimulus-skip trunk, `artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt` (bare trunk, weight-identical to the copula-stamped copy for a teacher-forced measurement). `x_encoding = prev_contribution (numeric), prev_punishment (numeric), agent_group (onehot)`; no `edge_encoding` (edge messages over all 8 agents carry no same-group bit); `group_vnode: True`; `stimulus_skip: True`. Its stamped copula: `rho = 0.03949863621805423`, `phi = 1.0`, `switch_every = 1`, rho CI [0.0181, 0.0566].

### The question

Players decide independently given the true situation. Any correlation left in the model's errors *within* a group therefore means the model is missing part of the situation. The herding copula (rho about 0.04, one static latent per (episode, group)) stands in for that missing part with a random number. How much of the residual within-group correlation is explained by observable group state the trunk does not currently see, and how much is genuinely unobservable (and so legitimately a random latent)?

### Method

1. **Residual table** on the canonical human frame (50 games, the min-`episode_id` copy per `pair_id`, identical to the union of the copula script's 40 train + 10 test episodes): for every valid agent-round the realised contribution, the skip trunk's teacher-forced expected contribution and full 21-level marginal. Three residual scales, all "given the model": the level residual `r = c - E[c]`; the copula's latent, taken as the mid-point normal score of the model's own marginal `z = Phi^-1(F(c-1) + p(c)/2)`; and the exchangeable-Gaussian-copula `rho` by the calibration script's pairwise rectangle-likelihood MLE (`punishment_copula_rho.rho_mle`, imported unchanged). Cells are (game, round, group_id); co-movement is the correlation across cross-member pairs inside a cell, by three estimators: plain pooled Pearson over the stacked pair list (PR #140's style), the calibration script's pair-weighted moment estimator, and the MLE.
2. **Candidates**, all group-level and computable from history up to t-1 plus the membership at t (which the player sees before contributing): (a) `grp_pun_last`, `grp_pun_last3`, `grp_share_pun_last`; (b) `rounds_since_change`, `left_at_last`, `joined_at_last`, `size_now`; (c) `other_mean_last`, `other_size_now`, `gap_last`; (d) `own_mean_last`, `own_mean_last3`, `own_trend` (t-1 minus t-3), `own_sd_last`; (e) `early_type` (rounds 0-2 mean, running mean before round 3), `cum_mean`; (t) `round`; (f) fixed effects for decomposition only, never features: `fe_game`, `fe_game_group` (one manager per (game, group_id), i.e. the exact shape of the static copula latent), `fe_session`. Round-0 priors are the training defaults (contribution 9, punishment 0). Each candidate is regressed on the residual (pooled OLS with game-clustered SEs, plus a cell-mean WLS: the shared component on the candidate), then partialled out (`r - X b`; on the MLE scale the fitted `X b` shifts both rectangle bounds of the latent) and the co-movement recomputed. Share explained = `1 - rho_after / rho_before`, 95% CI from 200 game-resamples (50 for the MLE), cells re-keyed per draw so a game drawn twice never pairs with itself.
3. **Forward selection** over the legal candidates on the latent pair-moment scale; stop when the incremental drop's bootstrap CI includes 0; the joint set is then confirmed on all three scales.
4. **Persistence** of the leftover: lag-k autocorrelation of the cell-mean residual within (game, group), the static-vs-round-local variance split of that cell mean, and the calibration script's `phi = rho_lag1_cross / rho` (cross-player lag-1 pairs), before and after partialling.

Scripts: `scripts/data_analysis/copula_missing_state.py` (`predict` on Raven's login node, `analyse` locally) and `scripts/data_analysis/copula_missing_state_analysis.py`. Outputs: `plots/data_analysis/evaluation/copula_missing_state/`.

## 2. Plan

| # | step | implementer |
|---|---|---|
| 1 | Teacher-force the skip trunk on the 50 canonical games; assert the train-split MLE reproduces the stamped rho | done |
| 2 | Baseline reproduction: raw co-movement, residual co-movement, MLE; by round third, by membership change | done |
| 3 | Candidate table with bootstrap CIs; forward selection; joint model | done |
| 4 | Persistence check | done |
| 5 | Feature recommendation, log, PR | done |

## 3. Results

**Verdict (no simulation, so no gates; a `[RESULT]` PR).** On the human data the skip trunk's residual within-group dependence is `rho = 0.0479` (pairwise MLE, 50 games, CI [0.028, 0.067]; 0.0395 on the 40 train games, reproducing the stamped value to full precision). Observable group state the trunk does not see explains **14% of it** (the best 3-feature set, MLE scale, CI [9%, 29%]) and **23% at most** (all 16 legal candidates jointly). The remaining three quarters is **not a static, unobservable group effect**: the cross-player correlation of the residual latent is 0.036 within a round, 0.024 one round later, and **-0.005 [-0.010, +0.000] pooled over every pair of rounds two or more apart** (373,310 pairs; the MLE sits on its 0 boundary). The static share of the shared residual is 0 (CI [-0.33, +0.01]). What the copula stands in for is a round-local shared shock with a two-thirds echo into the next round -- the opposite shape from the static `phi = 1` latent that is stamped.

### 3.1 Baseline reproduction

| quantity (cross-member pairs within (game, round, group), 50 canonical games) | value | 95% CI (200 game resamples) |
|---|---|---|
| raw contribution, plain Pearson | 0.4832 | [0.397, 0.567] |
| raw contribution, ICC(1) / pair moment / corr with leave-one-out group mean | 0.484 / 0.456 / 0.579 | |
| group-spread ratio SD(group mean) / SD(individual) | 0.848 | (PR #140: 0.85) |
| level residual `c - E[c]` given the skip trunk, plain Pearson | **0.0316** | [0.013, 0.050] |
| latent (mid-point PIT) residual, pair moment (the calibration script's diagnostic) | 0.0363 | [0.018, 0.053] |
| latent, randomised PIT, pair moment | 0.0299 | |
| latent, pairwise MLE, 50 games | **0.0479** | [0.028, 0.067] (50 resamples) |
| latent, pairwise MLE, 40 train games | **0.03949863621805423** | = the stamped rho, exactly |
| level residual, 40 train games | 0.0262 | |

PR #140 quoted 0.507 raw and 0.073 residual. The raw figure is not reproduced by any of the four estimators (0.456-0.579; the spread ratio is reproduced exactly at 0.85), so it was a different estimator or population; the residual 0.073 came from a *linear* state fit and the maintainer called it a lower bound on the explained share -- the GNN trunk leaves 0.032, i.e. the trunk already removes **93%** of the raw co-movement (the linear fit removed 86%).

By split (point estimates; `plots/data_analysis/evaluation/copula_missing_state/baseline_splits.csv`):

| split | rows | pairs | raw Pearson | residual Pearson (level) | latent moment | MLE |
|---|---|---|---|---|---|---|
| all | 9320 | 19133 | 0.483 | 0.032 | 0.036 | 0.048 |
| round 0 excluded | 8934 | 18572 | 0.496 | 0.038 | 0.040 | 0.053 |
| rounds 0-7 | 3108 | 5992 | 0.269 | 0.021 | 0.012 | **0.014** |
| rounds 8-15 | 3105 | 6686 | 0.519 | 0.034 | 0.048 | **0.068** |
| rounds 16-23 | 3107 | 6455 | 0.599 | 0.049 | 0.055 | **0.084** |
| membership changed at the block's switch round (t >= 4) | 6987 | 14854 | 0.488 | 0.033 | 0.036 | 0.048 |
| membership unchanged at the block's switch round (t >= 4) | 786 | 2031 | 0.750 | 0.108 | 0.093 | **0.125** |
| membership changed this very round | 1723 | 3615 | 0.421 | 0.043 | 0.032 | 0.054 |

The residual dependence grows through the game (0.014 -> 0.068 -> 0.084), consistent with the round-thirds rise PR #149 / #170 reported. The "unchanged" split is 91% vs 9% of cells because a single switcher changes both groups; the unchanged cells are large late-game groups (size 6-8, blocks 16-20) and the excess is block-wise inconsistent (MLE 0.00 / 0.22 / 0.02 / 0.20 over blocks 8 / 12 / 16 / 20), so it is recorded as suggestive only (note 7).

### 3.2 Candidate table

Share of the residual within-group co-movement explained by partialling the candidate out alone, `1 - rho_after / rho_before`, three scales; 95% CI over 200 game resamples (50 for the MLE); `t` is the game-clustered t of the candidate in the cell-mean regression (the shared component on the candidate). Full table with coefficients: `candidates.csv`.

| candidate | family | MLE share [CI] | latent-moment share [CI] | level share [CI] | cell-mean t |
|---|---|---|---|---|---|
| `grp_share_pun_last` | a | **+0.063 [+0.008, +0.146]** | +0.076 [+0.012, +0.198] | +0.006 [-0.026, +0.149] | +2.30 |
| `grp_pun_last` | a | +0.012 [-0.005, +0.053] | +0.013 [-0.008, +0.074] | -0.003 [-0.010, +0.088] | +0.91 |
| `grp_pun_last3` | a | +0.004 [-0.003, +0.038] | +0.004 [-0.005, +0.052] | +0.001 [-0.007, +0.089] | +0.60 |
| `rounds_since_change` | b | +0.026 [+0.004, +0.120] | +0.019 [-0.001, +0.114] | +0.032 [+0.006, +0.135] | -2.03 |
| `left_at_last` | b | -0.001 [-0.007, +0.040] | -0.002 [-0.010, +0.026] | -0.004 [-0.016, +0.021] | +0.46 |
| `joined_at_last` | b | +0.001 [-0.003, +0.036] | +0.000 [-0.005, +0.034] | +0.007 [-0.005, +0.051] | +0.40 |
| `size_now` | b | -0.006 [-0.023, +0.034] | -0.010 [-0.034, +0.015] | -0.011 [-0.042, +0.042] | -0.14 |
| `stable_block` | b | +0.003 [-0.003, +0.044] | -0.001 [-0.007, +0.025] | +0.001 [-0.008, +0.028] | -0.94 |
| `other_mean_last` | c | +0.001 [-0.006, +0.044] | +0.003 [-0.003, +0.069] | +0.010 [-0.002, +0.100] | -1.04 |
| `other_size_now` | c | -0.006 [-0.023, +0.034] | -0.010 [-0.034, +0.015] | -0.011 [-0.042, +0.042] | +0.14 |
| `gap_last` | c | +0.016 [-0.001, +0.085] | +0.010 [-0.001, +0.083] | +0.046 [+0.002, +0.203] | +1.26 |
| `own_mean_last` | d | +0.015 [-0.011, +0.080] | +0.007 [-0.003, +0.038] | +0.038 [+0.003, +0.112] | +1.18 |
| `own_mean_last3` | d | -0.003 [-0.021, +0.042] | -0.002 [-0.007, +0.013] | +0.013 [-0.010, +0.071] | -0.14 |
| `own_trend` | d | **+0.040 [+0.016, +0.127]** | +0.041 [+0.004, +0.126] | +0.041 [-0.001, +0.149] | +2.85 |
| `own_sd_last` | d | +0.014 [-0.008, +0.074] | +0.024 [-0.006, +0.116] | +0.034 [-0.030, +0.143] | **-3.58** |
| `early_type` | e | -0.002 [-0.012, +0.039] | -0.001 [-0.005, +0.030] | +0.010 [-0.005, +0.061] | -0.04 |
| `cum_mean` | e | -0.006 [-0.025, +0.036] | -0.003 [-0.010, +0.014] | +0.011 [-0.008, +0.074] | -0.31 |
| `round` | t | +0.006 [-0.003, +0.061] | +0.006 [-0.007, +0.041] | +0.007 [-0.016, +0.059] | -0.69 |
| `fe_session` (decomposition only) | f | +0.010 [-0.012, +0.037] | +0.012 [-0.015, +0.050] | +0.008 [-0.029, +0.041] | |
| `fe_game` (decomposition only) | f | +0.153 [+0.053, +0.387] | +0.142 [+0.036, +0.341] | +0.199 [+0.097, +0.512] | |
| `fe_game_group` (decomposition only) | f | +0.319 [+0.171, +0.715] | +0.305 [+0.157, +0.673] | +0.483 [+0.255, +1.153] | |

Only two candidates have a share whose CI excludes 0 on the copula's own (MLE) scale: the share of the group punished last round (6%) and the group's contribution trend (4%); `rounds_since_change` (3%) does on the MLE and level scales. Everything the trunk can already see through the vnode and the edges -- the group's own mean, its history, its early type, the other group -- explains nothing further, which is the check that the partialling is honest: the vnode absorbed the level of the group, as PR #179 measured. The fixed effects are in-sample and confound *individual* residual persistence with a group latent (note 5); the unbiased read of the static component is section 3.4.

### 3.3 Joint model

Forward selection on the latent pair-moment scale, stopping when the incremental drop's CI includes 0 (`forward_selection.csv`):

| step | added | rho before | rho after | drop [CI] | |
|---|---|---|---|---|---|
| 1 | `grp_share_pun_last` | 0.0363 | 0.0335 | +0.0028 [+0.0003, +0.0083] | keep |
| 2 | `own_sd_last` | 0.0335 | 0.0316 | +0.0020 [+0.0003, +0.0044] | keep |
| 3 | `own_trend` | 0.0316 | 0.0300 | +0.0015 [+0.0002, +0.0040] | keep |
| 4 | `rounds_since_change` | 0.0300 | 0.0293 | +0.0007 [-0.0000, +0.0035] | stop |

The joint set on every scale (`joint_model.csv`), and the reference sets (`reference_sets.csv`):

| set | MLE rho before -> after | MLE share [CI] | latent-moment share [CI] | level share [CI] |
|---|---|---|---|---|
| **joint set** (`grp_share_pun_last`, `own_sd_last`, `own_trend`) | 0.0479 -> 0.0410 | **0.143 [0.092, 0.288]** | 0.173 [0.097, 0.368] | 0.101 [0.024, 0.373] |
| all 16 legal candidates | 0.0479 -> 0.0368 | 0.231 | 0.254 | 0.260 |
| game FE (illegal) | 0.0479 -> 0.0406 | 0.153 | 0.142 | 0.199 |
| (game, group) FE (illegal) | 0.0479 -> 0.0326 | 0.319 | 0.305 | 0.483 |
| joint set + (game, group) FE | 0.0479 -> 0.0257 | 0.464 | 0.478 | 0.607 |

Joint coefficients on the latent scale (game-clustered t): `grp_share_pun_last` +0.141 (t 2.48), `own_sd_last` -0.021 (t -4.68), `own_trend` +0.012 (t 2.95). Read: a group whose members were mostly punished last round contributes *more* than the trunk expects, a group whose members are far apart contributes *less*, a group on the way up keeps going.

### 3.4 Persistence of the leftover

Cross-player pair correlation of the residual latent between rounds r and r + k of the same (game, group) -- a static latent gives the same value at every lag, a round-local shock nothing beyond its echo (`persistence_boot.csv`, 200 game resamples; MLE on the same pair lists):

| lag | pairs | before partialling: moment [CI] / MLE | after the joint set: moment [CI] / MLE |
|---|---|---|---|
| 0 (within round) | 19,133 | 0.0363 [0.018, 0.053] / 0.0479 | 0.0300 [0.013, 0.049] / 0.0410 |
| 1 | 36,371 | 0.0243 [0.013, 0.038] / 0.0333 | 0.0195 [0.008, 0.031] / 0.0285 |
| 2 | 34,430 | -0.0017 [-0.014, +0.012] / 0.0 | -0.0022 [-0.014, +0.010] / 0.0 |
| 3 | 32,529 | -0.0063 [-0.019, +0.005] / 0.0 | -0.0001 [-0.011, +0.012] / 0.0 |
| **>= 2, pooled** | **373,310** | **-0.0051 [-0.0100, +0.0002] / 0.0** | **-0.0033 [-0.0083, +0.0014] / 0.0** |
| >= 4, pooled | 306,351 | -0.0054 [-0.0106, -0.0001] / 0.0 | -0.0038 [-0.0100, +0.0021] / 0.0 |
| static share = rho(>= 2) / rho(0) | | -0.14 [-0.33, +0.006] | -0.11 [-0.33, +0.07] |
| phi at lag 1 = rho(1) / rho(0) | | 0.67 (MLE 0.70) | 0.65 (MLE 0.69) |

An AR(1) latent with the lag-1 ratio 0.67 would put lag 2 at 0.45 x 0.036 = 0.016; observed -0.002 [-0.014, +0.012]. Partialling the joint set out does not change the shape. The cell-mean autocorrelation says the same thing more noisily (lag 1: 0.12, lag 2: -0.01, lag 4: 0.05, lag 8: 0.03; `persistence.csv`).

## 4. Notes

1. **Measured.** The prediction stage asserts the canonical 50-game copy equals the evaluation suite's (min `episode_id` per `pair_id`) and equals the union of the copula script's train and test episodes, and the MLE on the 40 train games reproduces the stamped rho to every digit (0.03949863621805423), so the residual table is the calibration's own population and marginals.
2. **Measured.** The trunk already explains 93% of the raw within-group co-movement (0.483 -> 0.032 on the level scale). The residual dependence the copula was fitted to is small and almost entirely round-local, so "the model is missing part of the situation" is true but the missing part is not a slowly varying group state.
3. **Measured, the headline.** Observable state explains 14% (best set, CI 9-29%) and at most 23% (everything). If all three joint-set features were added to the trunk and worked exactly as the linear partialling does, the stamped rho would move from 0.0395 to about 0.034 -- inside the current CI [0.018, 0.057]. The features are worth having for what they say about behaviour, not for the dose.
4. **Measured.** The static component of the shared residual is zero: the cross-player correlation pooled over all round pairs at lag >= 2 is -0.005 [-0.010, +0.000] and the MLE hits its 0 boundary. The shared latent that the calibration measures within a round has a one-round echo (two thirds) and no memory beyond it. The stamped `phi = 1` (a latent held fixed for 24 rounds) therefore has no counterpart in the human residuals. This does not contradict PR #179's sim finding that `phi = 1` beats `phi = 0.618` on CG -- PR #179's own ablation read the static latent as "free-running variance the deterministic trunk cannot generate", a sim-side variance source, not the human dependence shape; the two statements are about different objects.
5. **Inferred, with the mechanism stated.** The (game, group) fixed effect "explains" 31% and the game FE 15%, far above the 0% the lag profile gives. A fixed effect demeans by the mean of every row of the slot over 24 rounds, so the subtraction carries each *player's own* residual persistence across rounds (the marginal conditions on `prev_contribution`, but in-sample per-player deviations still autocorrelate) into the cross-member covariance -- the same self-pair confound the calibration script excludes when it estimates phi from cross-player pairs only. The FE rows are therefore an upper bound contaminated by individual persistence, and the difference between 31% and 0% is a per-*player* effect (a CA-family question), not a per-group one.
6. **Inferred.** A round-local shared deviation with a lag-1 echo is what a mis-modelled *joint response to the previous round's shared events* looks like: the event is in the history the trunk sees (all eight agents' `prev_contribution` and `prev_punishment` reach every node through the edges), the members react to it together at t and partly at t+1, and the trunk's per-agent response to it is too weak or the wrong shape -- the same diagnosis PR #181 made for the punishment response of an individual, now at the group level. The two candidates that carry the explained share are both *group-level summaries of the previous round's events* (how many of us were punished; which way the group is moving), which is what a per-agent readout of edge messages without a same-group bit would be expected to under-use. A common input at t that no history contains (timing, screen state) cannot be excluded on this data and would look the same.
7. **Suggestive only.** Cells whose membership did not change at the block's switch round show a residual MLE of 0.125 against 0.048 elsewhere; they are 9% of the cells, large late-game groups, and the excess is inconsistent across blocks (0.00 / 0.22 / 0.02 / 0.20). A linear candidate for it explains nothing (the effect is in the *dependence*, not the mean), so if it is real it is a case for a state-dependent rho rather than a feature.
8. **Process.** The PyG step ran on Raven's login node in `~/repros/ai-runs/copula-missing-state` (46 s; the artifact is excluded from the rsync and was copied by hand); everything else ran locally in ~11 min. No training, no simulation, no seed was touched.

### Feature recommendation

For a follow-up retrain of the skip trunk (each is a group-level summary of the previous round over the agent's *current* group, so it is legal under §5 -- history up to t-1 plus the membership at t, which the player sees before contributing; the same information already reaches the node through the edges, so no new observability assumption is made). Expected drops are the linear partialling's shares applied to the stamped rho 0.0395, an upper bound for a linear add-on and no bound at all for what a retrained nonlinear trunk does with the same input:

| feature (name in `x_encoding`) | how computed | share (MLE) | expected rho drop |
|---|---|---|---|
| `own_grp_prev_share_punished` | share of the agent's current-group members (leave-one-out, valid rows) with `prev_punishment > 0`; 0 at round 0 and with no valid peers | 6% | 0.0395 -> 0.037 |
| `own_grp_prev_trend` | current group's mean valid contribution at t-1 minus at t-3 (group id's members at those rounds); 0 before round 3 | 4% | 0.0395 -> 0.038 |
| `own_grp_prev_sd_contr` | within-group SD of valid contributions at t-1 over the current group's members; 0 with fewer than two | 1-2% | 0.0395 -> 0.039 |
| all three | | 14% [9%, 29%] | **0.0395 -> 0.034** |

The `own_grp_prev_mean_contr` feature that `parse_agent_rounds` already builds (#114 / M3) explains 1.5% here -- the vnode has it. The recommendation that actually follows from the numbers is not a feature but the latent's shape: on the human data the shared residual is a round shock with a two-thirds echo and no static part, so a `phi = 1` latent is the wrong object to calibrate against these residuals; a lag-1-echo latent (`phi` ~ 0.67 at lag 1, 0 beyond, i.e. an MA(1) rather than an AR(1)) is what the human dependence supports, and whether the closed loop still needs a static variance source on top is the sibling experiments' question, not this one's.
