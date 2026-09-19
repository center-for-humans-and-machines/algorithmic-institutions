# The punisher and the timed-out player (a data-handling fix, RCC)

## 1. Declaration

**Slot:** punisher -- both families (the copula-stamped `lin_multinomial` the frontier stack runs, and the GNN punisher), one data-handling fix and one feature, judged as one change.

**Parent:** PR #192 (`auto/punisher-ceiling-fix`, `[FAIL]`, at `e230629`), the model the maintainer has accepted as current: both punishers carry the ceiling indicator `contribution_max`, the severity copula is stamped at rho 0.4273, and the suite scores 22 rows. Branch `auto/punisher-timeout-feature` is created from it and the PR opens with `--base auto/punisher-ceiling-fix`. Isolated remote dir `~/repros/ai-runs/punisher-timeout` (delete when this PR closes).

**Base models.** Punisher: `artifacts/baselines/punishment_multinomial_ceiling.joblib` (features `contribution, contribution_max, prev_contribution, prev_punishment, round_number, is_first`, C = 1.0, CV log loss 1.3446, locked test 1.2234) and its severity-copula copy `punishment_multinomial_ceiling_severity_copula.joblib` (rho 0.4273); GNN punisher `artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_ceiling/model/architecture_node+edge+rnn__dataset_50ep_doubled.pt` (`x_encoding = contribution (numeric), contribution_max (bool), prev_contribution, prev_punishment (numeric), is_first (bool)`, CV log loss 1.1743). Contributor and switch slots untouched.

**Evaluation stacks (§3 under the parent rule of §9).** The frontier stack `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch` (the gated one, judged in full) and the GNN-punisher reference `23_2g8a_self_gnn_contr_gnn_switch` (runs `lin_multinomial_self` and `gnn_self`; reported, not gated). Both re-run with the 23-family protocol (seed 42, 100 episodes, 24 rounds) from the parent's `_ceiling` configs with only the punisher paths, `output_dir` and `figure_name` changed (`_timeout` suffix).

**Baseline (the parent's confirmed `_ceiling` scores; both gates are judged against these).**

| row | score | band |
|---|---|---|
| **RCC** (declared target) | **1.2969** | 1-2 |
| RCE (protected; bands 0-4 / 5-9 / 10-14 / 15-19 slopes +0.087 / +0.038 / -0.043 / -0.130, signs ++--) | 0.8823 | <= 1 |
| RPA (**not** declared) | 0.6620 | <= 1 |
| RCB | 1.6591 | 1-2 |
| mean over 22 rows | **1.0331** | |
| gate-2 ceiling (mean x 1.10) | **1.1364** | |
| rows <= 1 | 14/22 | |

Reference stack (`23_2g8a_self_gnn_contr_gnn_switch_ceiling`): `lin_multinomial_self` mean 1.6616, 13/22, RCC 1.4615, RPA 0.6626, RCE 0.9900; `gnn_self` mean 1.6603, 11/22, RCC 1.1109, RPA 0.8066, RCE 0.9157.

**Target row:** RCC (gate 1: a band improvement, 1-2 -> <= 1, i.e. RCC < 1.0). RPA is deliberately **not** declared: at 0.6620 it already sits in the best band, so no band upgrade is available on it and declaring it would make gate 1 unwinnable in part (PR #193's follow-up 2). Watch rows: RCB and RCD (the two rows the punisher's contribution response shapes), PA/PB/PC (punishment marginals), RCE (protected).

### Hypothesis

**Behavioural rationale (one sentence, §5):** the human manager could see that a player had given no input at all and never punished one of them, while punishing a player who chose to give nothing 46.9% of the time, so the artificial punisher is given the same distinction -- the recorded contribution of a timed-out player (0) and a flag saying the input was missing; the row that should move is RCC, through the punisher's response to low contributions, with RCB and RCD as watch items.

**The change.** Two parts, one coherent fix:

1. **The punisher's contribution feature takes the recorded value, 0, for a timed-out player**, rather than the imputed default of 9. Targets are untouched.
2. **Both punisher families gain a boolean `contribution_valid`**, the name and `etype: bool` convention the RL manager already uses (`configs/training/rl_manager/03_2g8a_sum.yml`), so the model can tell a timeout apart from a genuine zero.

Step 0 verified the premise of part 1 and **relocated it**: the imputed 9 never enters the training data at all. It is injected by the *environment* at simulation time, so part 1 is a fix to what the manager is *served*, not to what it is *trained on*. Part 2 is the training-side change. See note 1 in section 4 -- this is the single most important thing a successor must understand about this branch.

### Artifact naming contract

| what | path |
|---|---|
| baseline config | `configs/training/baselines/punishment/multinomial_timeout.yml` (`multinomial_ceiling.yml` with a second B1 set carrying `contribution_valid`; the artifact is the rank-1 row) |
| baseline artifact | `artifacts/baselines/punishment_multinomial_timeout.joblib` |
| severity-copula copy | `artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib` (`punishment_copula_rho.py --roundtrip --stamp-rho 0.4273`) |
| GNN config | `configs/training/artificial_humans/punishment/rnn_edge_50ep_doubled_timeout.yml` |
| GNN artifact dir | `artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_timeout/` |
| sim configs | `configs/simulation/manager_testing/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_timeout.yml`, `23_2g8a_self_gnn_contr_gnn_switch_timeout.yml`; sim dirs `plots/simulation/<same>` |
| tables | `plots/data_analysis/evaluation/punisher_timeout_feature/` |

## 2. Plan

| # | step | status |
|---|---|---|
| 0 | Verify on the human data that 0 is the game's real value for a timed-out player (accounting identity), and locate where the imputed 9 actually reaches a model. | **done** |
| 1 | Add `contribution_valid` to the linear feature pool and the punishment legal set; make the manager's view of a timed-out player read the recorded 0 on both simulation paths; unit tests on both paths. | **done** |
| 2 | Retrain the linear punisher locally (`multinomial_timeout.yml`, 4-fold CV, seed 38381); report CV against 1.3446. | **done** |
| 3 | Stamp the severity copula carrying rho = 0.4273 over unchanged; report the refit for the record. | **done** |
| 4 | Retrain the GNN punisher on Raven (`rnn_edge_50ep_doubled_timeout.yml`); report CV against 1.1743. | **done** (session 2, job 30319268) |
| 5 | Teacher-forced mechanism check of both new punishers against the parent's artifacts and the human row. | **linear done; GNN needs Raven** |
| 6 | Re-run the two stacks, fetch, evaluate all 22 rows (`PYTHONPATH=<worktree>/src`), self-play mechanism table, RCE band slopes with standard errors. | **BLOCKED -- needs Raven** (table script written and self-tested; baseline half verified) |
| 7 | Judge under the gates with RCE protected; measure the blast radius on the other slots; log; PR against the parent. | **blast radius done; verdict BLOCKED** |

The cluster was unreachable for session 1 and went down again 24 minutes into session 2 (the persistent SSH ControlMaster to Raven dies and only the maintainer can restore it -- see section 5). Everything that can be established without a live cluster is established and committed. **No verdict is claimed and no PR is opened**, because the result is not established until step 6 runs.

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-19 | punisher reads the recorded 0 for a timeout and gains `contribution_valid` | pending -- both stacks unrun (training done: linear CV 1.3446 -> 1.3271, GNN 1.1743 -> 1.1719) | pending | pending | **pending -- see section 5** |

### Step 0: is 0 the game's real value for a timed-out player? (measured -- **premise confirmed**)

The premise is that the game charged a timed-out player 0, so 0 (not the imputed median 9) is what the human manager saw. The accounting identity settles it without appeal to intent: the stored `common_good` of a group in a round must equal 1.6 x the group's contributions minus its punishments.

The group in that identity is **`group_id` (0/1) within `(episode_id, round_number)`**, not `global_group_id` -- players switch groups every 4 rounds, so the roster changes within an episode and group sizes run 1-8. `common_good` is the group **total** and is unique per `(episode_id, round_number, group_id)`. On `experiments/2group_8agent_50ep.csv` (19,200 agent-rounds, 4,512 group-rounds):

| contribution of a timed-out player | group-rounds the identity holds on | max abs residual | mean abs residual |
|---|---|---|---|
| **the recorded 0** | **4,512 / 4,512 (100%)** | 2.8e-14 | 0.0 |
| the imputed 9 | 3,996 / 4,512 (88.6%) | 43.2 | -- |

Restricted to the **516 group-rounds that contain at least one timed-out player**: with the recorded 0 the identity holds on **all 516** (max abs residual 1.4e-14, i.e. floating-point exact); with the imputed 9 it fails on **every one of them** (0/516), mean abs residual **15.63**. The 516 failures under the 9 are exactly the 4,512 - 3,996 above, so the two views agree. The residual is flat across group-rounds with 1, 2 or 3 timed-out players (474 / 40 / 2, all exact under 0).

**Conclusion: 0 is the game's real value, not a placeholder. The premise of the fix holds.** Supporting facts on the same file: all 560 timed-out agent-rounds record `contribution == 0` in the raw file (no other value ever appears); the human manager punished **0.00%** of them; players who chose to contribute 0 were punished **43.5%** of the time over all rows and **46.9%** over the punisher's own mask. The mirror case is harmless as declared: 814 rows have `manager_no_input == 1`, the raw punishment is 0 on every one, and the imputed default is also 0.

*(Two small corrections to the declaration's arithmetic, neither of which affects anything: `manager_no_input == 1` holds on **814** rows, not 812; and the two punish rates for a genuine zero, 43.5% and 46.9%, are the same quantity over all rows and over the punisher's mask respectively -- both appear in this log and they are not in conflict.)*

### Step 1: the source change (done; scope confirmed punisher-only)

`get_default_values`' median of 9 is **not** what overwrites a timed-out player's contribution in training -- see note 1. The substitution that matters happens at simulation time, and the fix is applied at the two points where the manager is served, leaving the environment's own dynamics and its common-good accounting untouched:

- `src/aimanager/generic/data.py` -- adds the named constant `MISSING_CONTRIBUTION = 0` with the step-0 evidence recorded next to it. No behaviour change in this file.
- `src/aimanager/manager/api_manager.py` -- `create_data`'s `create_tensor` gains a `missing` argument; an **own-group** cell whose input flag is False now reads `MISSING_CONTRIBUTION` instead of the model's default fill. Other-group cells are masked out and keep the default fill as before. Applied to `contribution` only; `punishment` keeps the default (which is 0 there anyway, so the mirror case is a no-op either way). `create_data` is called from exactly one place, `MultiManager.get_punishments` -- **this is the GNN punisher's serving path and nothing else's.**
- `src/aimanager/simulation/linear_ah.py` -- `_pool_from_rounds` (the **rounds-driven path, which is the linear punisher's and only the linear punisher's**) puts the recorded 0 back before the feature pool is built, and passes the validity mask through to `_pool_from_arrays`. The env-driven path `_pool_from_env`, which serves the contribution and switch models, passes `cv=None` and is unchanged.
- `src/aimanager/simulation/simulate.py` -- `make_round` takes the env's realised `contribution_valid` instead of inferring it from `c is not None` (which is always True in simulation), so the flag reaches the round dicts the punisher reads.
- `scripts/baselines/handcrafted_grid.py` -- `build_feature_pool` gains the `contribution_valid` feature; it is added to `CURRENT_VALUED` (so it is **illegal for the contribution target**, which is prev-anchored) and to `PUNISHMENT_LEGAL_CURRENT` (legal for the punishment target, which the manager sets after seeing round t's inputs). Adding a feature to the pool changes no existing model; only a config that selects it.
- `scripts/baselines/punishment_baseline.py` -- `FEATS` gains `contribution_valid`, keeping the standalone script in step with the grid.

**Scope is punisher-only and the result stays attributable.** The two substitution sites are each reachable by exactly one punisher family, the contributor and switch paths are untouched, and the shared change (the feature pool) is inert for any model that does not select the new feature. No shared-code behaviour change was made.

### Step 2: the linear punisher retrained (measured)

`configs/training/baselines/punishment/multinomial_timeout.yml`, 4-fold CV grouped by pair, seed 38381, locally. The grid deliberately carries the parent's feature set alongside the candidate so the comparison is internal to one run:

| rank | CV log loss | features |
|---|---|---|
| 1 | **1.3271** | contribution, contribution_max, **contribution_valid**, prev_contribution, prev_punishment, round_number, is_first |
| 2 | 1.3340 | contribution, contribution_max, **contribution_valid**, prev_contribution, prev_punishment |
| 3 | **1.3446** | contribution, contribution_max, prev_contribution, prev_punishment, round_number, is_first *(= the parent)* |
| 4 | 1.3500 | contribution, contribution_max, prev_contribution, prev_punishment |
| 5 | 1.4355 | *(marginal floor)* |

Rank 3 reproduces the parent's stored `cv_metric` to all 16 digits (1.3445988078115791), which is the control that says the two runs differ only in the added feature. **CV log loss 1.3446 -> 1.3271** (-0.0175), and the flag helps with and without the structural block. On the locked test fold, untouched during selection: **1.2234 -> 1.1934** (-0.0300) against an unchanged floor of 1.3561. Artifact `artifacts/baselines/punishment_multinomial_timeout.joblib`, C = 1.0, verified to carry the 7-feature rank-1 set.

### Step 3: the severity copula, stamped rather than refit (measured)

`artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib` verified to carry **`copula_rho = 0.4273`, identical to the parent's**, as the freeze requires, together with the same estimator metadata (`pairwise_mle`, `episode_round_group` cells, `copula_n_pairs = 15291`, same train file). The marginals are the new ones; only the dependence parameter is carried over.

**For the record, what a refit would give:** re-running `punishment_copula_rho.py` on the new bundle without stamping returns a pairwise-likelihood MLE of **rho = 0.4821** on the same 15,291 pairs (7,345 masked rows, 1,550 cells of size >= 2), against the stamped 0.4273 -- a move of +0.055 that the protocol deliberately does not take. Out-of-sample on the held-out test file: rho 0.4216. Diagnostic splits run 0.398-0.641 across cell sizes and 0.452-0.559 across round thirds, so 0.4821 sits inside the ordinary spread of the estimator rather than signalling a changed dependence structure. The attenuated randomized-PIT diagnostic, which is not a selection criterion, reads 0.163.

### Step 4: the GNN punisher retrained (measured -- and the move is inside the fold noise)

`configs/training/artificial_humans/punishment/rnn_edge_50ep_doubled_timeout.yml` on Raven, job **30319268**, 7 min 30 s on one A100 (`ravg1111`, exit 0), 5-fold CV plus the full fit, seed 38381, the parent's config plus the one bool node feature.

| punisher | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | final-epoch CV | best-epoch CV |
|---|---|---|---|---|---|---|---|
| GNN + ceiling (parent) | 1.0874 | 1.2420 | 1.1000 | 1.1012 | 1.3409 | **1.1743** +- 0.1127 | 1.1742 +- 0.1126 |
| GNN + ceiling + timeout | 1.0859 | 1.2399 | 1.0879 | 1.0883 | 1.3574 | **1.1719** +- 0.1229 | 1.1716 +- 0.1231 |
| delta | -0.0015 | -0.0021 | -0.0120 | -0.0129 | +0.0165 | **-0.0024** | -0.0026 |

The parent's stored 1.1743 reproduces to four decimals from its own committed metrics parquet, which is the control that the two numbers are computed the same way. Globally best epoch is 1249 in both, so final-epoch and best-epoch agree.

**Read honestly, the GNN move is nothing.** Four of five folds improve and the fifth worsens by more than any of them improves; the paired per-fold difference is -0.0024 with a standard deviation of 0.0118 over the five folds, i.e. a standard error of 0.0053 and **t = -0.46**. The fold-to-fold spread of the metric itself (sd 0.113-0.123) is fifty times the mean move. `punishment_baseline.py`'s `GNN_REF` stays at 1.1756, as the parent left it -- on a move this size there is no new reference to claim.

This is a real asymmetry with the linear family, where the same feature moved CV log loss 1.3446 -> 1.3271 (-0.0175) and the locked test 1.2234 -> 1.1934 (-0.0300). The obvious reading is that the graph punisher, with its RNN state and edge model, can already infer "this player gave no input" from context the linear model has no way to represent -- the flag tells it something it had largely reconstructed. That is a hypothesis, not a measurement; the teacher-forced mechanism table of step 5 is what would test it, and it has not run.

**Artifact provenance, stated plainly because it is not the usual one.** The committed artifact under `artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_timeout/` is job 30319268's, which ran at 03:13-03:20 on 2026-09-19 -- *before* this branch's five commits were made at 08:06-08:08, from a sync of session 1's uncommitted working tree. It is attributable anyway, and here is why: the remote `job.yml` reproduces the committed config's `params` block field for field, including the `contribution_valid` bool in `x_encoding`; a full re-sync of the committed tree in session 2 transferred only five files (`simulate.py`, `test_punisher_current_contribution.py`, the two `_timeout` sim configs and this log), none of which is on the GNN training path; and the artifact's own weights show the extra input channel -- the edge MLP's first layer is 20x12 against the parent's 20x10 (two endpoints x one new node feature) and the node MLP's 20x26 against 20x25. Session 2 nevertheless resubmitted the identical training from the verified committed tree as **job 30323407**, which was still `PENDING` ("nodes required are DOWN, DRAINED or reserved") when the connection died. That job will overwrite the remote artifact dir; see section 5.

### Step 2 (cluster unit tests): submitted, no result

The four cluster-only tests were dispatched with `scripts/remote_test.sh --test-only -- -k test_punisher_current_contribution -v` after the missing fixture `plots/simulation/22_2g8a_linear_self_ridge_contr/per_round.parquet` was shipped by hand to the isolated dir (confirmed present on the cluster, 274,539 bytes). **The run produced an empty log**: the ControlMaster passed its liveness check when the script started and was gone when the ssh call returned. There is no pass and no failure to report -- the tests did not run, and nothing about them should be inferred from this session.

### Step 4/5 (linear half): the mechanism, teacher-forced (measured)

`scripts/data_analysis/punisher_mechanism_check.py` replaying the 50 single-copy human games, **8,914 valid rows**, no simulation; each model sees the human history and never its own draws, under its own stored `default_values`. OLS is of predicted expected punishment on c_t and c_{t-1} over the 8,431 rows with a valid previous contribution. The GNN rows need Raven and are pending.

| punisher | P(p>0 \| c_t=20) | P(p>0 \| c_t<=4) | (c_t=20, c_{t-1}<=4) / (c_t<=4, c_{t-1}=20) | E[p \| p>0] 0-4 / 5-9 / 10-14 / 15-19 / 20 | OLS c_t / c_{t-1} | NLL |
|---|---|---|---|---|---|---|
| human | **0.038** | 0.467 | 0.179 / 0.571 | 7.99 / 4.98 / 4.27 / 3.87 / **7.00** | **-0.242** / +0.067 | -- |
| lin + ceiling (parent) | **0.053** | 0.431 | 0.224 / 0.635 | 7.43 / 5.46 / 4.43 / 3.76 / **7.47** | **-0.125** / -0.031 | 1.2705 |
| lin + ceiling + timeout | **0.053** | 0.475 | 0.202 / 0.784 | 7.43 / 5.45 / 4.38 / 3.68 / **7.44** | **-0.173** / +0.005 | 1.2664 |

**The ceiling behaviour PR #192 won survives intact**, which was the explicit condition on this change: the punish rate at c_t = 20 is 0.0530 against the parent's 0.0528 (human 0.038), and the severity there is 7.44 against the parent's 7.47 (human 7.00). Both move by less than a third of a percent. The fix does not spend the parent's gain.

**The slope on the current contribution moves, and by the predicted amount.** OLS c_t goes **-0.125 -> -0.173** against the human -0.242: the change closes **0.048** of a 0.116 gap, or 41% of it. PR #193 independently attributed **-0.046** of the punisher's under-strength contribution response to exactly this mask, and the measured move is -0.0476. That is the quantitative prediction this branch was built on, confirmed to within 0.002 on a number nobody tuned.

This is also a **correction to the parent's note for successors**, which recorded that the contribution slope was "untouched by anything on this branch or the parent" and predicted it "wants a different change -- a non-linear response in c_t, not another indicator". A second indicator did move it, because this one does not reshape the response to c_t at all; it removes 558 rows that were mislabelled within it. The parent's reasoning was about curvature and remains right about curvature; the slope was also carrying a labelling error, and that part is now fixed.

Elsewhere: P(p>0 | c_t <= 4) goes 0.431 -> 0.475 against the human 0.467, an improvement that slightly overshoots; the lagged coefficient recovers the human's sign (-0.031 -> +0.005, human +0.067); every E[p | p>0] band moves by at most 0.08; and the NLL improves 1.2705 -> 1.2664. The one row that moves against the human is the cross-tab P(p>0 | c_t <= 4, c_{t-1} = 20), 0.635 -> 0.784 against the human 0.571 -- worth watching in the self-play table, since the timed-out rows concentrate in the low-c_t class this cell conditions on.

Table: `plots/data_analysis/evaluation/punisher_timeout_feature/mechanism_teacher_forced_linear_only.csv` (the human row and the two linear rows, as measured this session). The command in section 5 regenerates it as `mechanism_teacher_forced.csv` at the contracted path with the GNN rows included; the two linear rows must reproduce exactly.

### Step 5: unit tests (done for everything that runs without the graph libraries)

`src/aimanager/tests/test_punisher_current_contribution.py`. The two the change is judged on:

- `test_linear_punisher_sees_zero_and_the_flag_for_a_timeout` -- with agent 0 timed out (the env having already overwritten its contribution with 9) and agent 1 having genuinely contributed 0, the linear punisher's design matrix reads value 0 / flag 0 for the first and value 0 / flag 1 for the second, and no other agent's row is disturbed. **Passes locally.**
- `test_gnn_punisher_sees_zero_and_the_flag_for_a_timeout` -- the same two cases through `create_data` and the bool `Encoder`, including that other-group cells still read the model's own default fill. **Needs Raven** (`torch_scatter`).

Also added: `test_linear_punisher_timeout_lag_is_zero_too` (the round after a timeout, the lag reads 0 -- passes locally), `test_contribution_valid_legality` (legal for the punishment target, rejected for the contribution target -- passes locally), and `test_simulation_round_carries_the_env_validity_flag` (`make_round` carries the env flag rather than inferring it; **needs Raven**, because importing `simulate` pulls in `graph.py`).

Local run, `PYTHONPATH=src pytest src/aimanager/tests/test_punisher_current_contribution.py`: **8 passed, 4 failed, every failure a bare `ModuleNotFoundError: No module named 'torch_scatter'`** and nothing else. The four are the three `test_gnn_*` and `test_simulation_round_carries_the_env_validity_flag`. This matches the repo's existing convention (`test_linear_manager.py` documents the same thing in its docstring rather than skipping), so no skip markers were added. They must be run on Raven via `scripts/remote_test.sh` before the PR is judged.

`black --check` and `flake8 --max-line-length=88 --extend-ignore=E203,W503` are clean over `src/`. (`src/aimanager/artificial_humans/train.py` and `src/aimanager/rl_manager.py` are unformatted on the parent and were deliberately left alone -- they are not part of this change.)

## 4. Notes

**Note 1 -- the imputed 9 never reaches the training data; it is injected at simulation time. This relocates the fix and a successor must not re-derive it.**

The declaration said `get_default_values` overwrites a timed-out player's contribution with the median of the valid rows, 9, and that the punisher trains on that. **That is not what happens.** Measured directly on the tensors:

- The raw file records `contribution == 0` for all 560 timed-out agent-rounds, and `data.py:46` does `df["contribution"].fillna(0)`. After `create_torch_data`, the contribution on every invalid cell is **0**, not 9.
- `default_values["contribution"] = 9` is used as the pivot's `fill_value` for agent-rounds **absent** from the frame, and as `shift`'s round-0 default for `prev_contribution`. Neither is a timed-out player.
- Therefore **the punisher already trained on the recorded 0.** Part 1 of the change is a no-op in training, and the whole training-side gain in step 2 comes from part 2, the flag -- which is exactly what the grid shows, since the flag is the only difference between rank 1 and rank 3.
- The 9 enters at **`src/aimanager/manager/environment.py:332`**: when a `valid_model` is configured, the env samples who times out and then does `contribution[~contribution_valid] = default_values["contribution"]` before the value is passed on. The env's own common-good accounting is correct (it zeroes invalid contributions separately at line 197); it is the value handed onward to the manager that is wrong. Both gated stacks configure a `valid_model` (`artifacts/artificial_humans/raven_script_22/...`), so timeouts really occur in these simulations and the fix is live rather than inert.

The consequence for reading this branch: **step 2's CV gain and step 4's mechanism table are measured on the training/replay side, where only the flag is active. Part 1 can only show up in the self-play numbers of step 6.** A successor who sees a small self-play move should not conclude the flag did nothing; the two parts are doing different work in different places and the teacher-forced table cannot see part 1 at all.

**Note 2 -- blast radius on the contributor and switch slots (measured, reported, not changed).** The question was how many rows those two models would be affected on if the same substitution were corrected for them. It has two different answers, and the difference is note 1 again:

*Training side: zero rows, for all three models.* Rebuilding the entire 63-feature hand-crafted pool from `experiments/baseline/2group_8agent_50ep_bline_train.csv` under both treatments -- contribution left as stored, versus contribution forced to 0 on every invalid cell -- moves **not one feature by more than 1e-9**, so 0 of the contributor's 7,457 masked rows, 0 of the switch's 1,515, and 0 of the punisher's 7,345. The stored value already *is* 0. There is nothing to correct in training for anybody.

*Serving side: nonzero for both, and still unfixed.* At simulation time the env's imputed 9 is real, and this branch intercepts it only on the two punisher paths. `linear_ah._pool_from_env`, which serves the contribution and switch models, still passes `cv=None` and still reads the env's 9; the GNN contributor reads `self.state` directly and likewise. So a timed-out player is shown to the **switch** model as having contributed 9 that round (it reads current-round contribution directly, block B1) and to the **contributor** model as having contributed 9 the round before (via `prev_contribution` and the group means, it being prev-anchored). The affected population is whatever fraction of agent-rounds the `valid_model` fires on: in the human data that is **560 / 19,200 = 2.9%** of agent-rounds, and the simulated rate can only be measured by running a stack, which needs the cluster. **This was left unfixed on purpose**, so that any movement in the 22 rows is attributable to the punisher alone; it is a genuine, separately declarable defect in the other two slots and a successor should have it.

**Note 3 -- what this does not touch.** `punishment_baseline.py`'s `GNN_REF` stays at 1.1756, as on the parent. The contributor's under-reaction to a heavy punishment at the ceiling, which the parent isolated as the live owner of the remaining RCC gap (`dc | punished, c_t = 20` at -3.75 against the human -8.66), is untouched here and remains the strongest candidate for the next declaration. If RCC fails to clear its band on this branch despite the slope moving, that decomposition is where to look first, and the parent's table is the baseline for it.

## 5. Status after session 2 -- what ran, what the cluster took away, what is left

Session 2 opened with Raven reachable (`ssh raven`, user `levinb`, empty queue) and lost it again **24 minutes in**. The failure is worth recording exactly, because the next session will meet it: the local SSH **ControlMaster for raven died** between 08:22:58, when `remote_test.sh` checked it and reported it active, and 08:23, when the next call returned `Permission denied (gssapi-with-mic,password)`. It cannot be rebuilt by an agent. Raven offers only GSSAPI and password auth; the local Kerberos cache is empty (`klist` -> "Cache not found"), so there is no credential to authenticate with. The `gate2` master is still alive with a delegated ticket valid to 2026-10-17, but raven's host key is in neither `~/.ssh/known_hosts` nor `/etc/ssh/ssh_known_hosts` on gate2, so hopping from there would mean accepting a host key on the maintainer's behalf -- not an agent's decision. **The maintainer restores this by running `ssh raven` locally; nothing else works.**

### What session 2 got

| step | result |
|---|---|
| 4 -- GNN punisher trained | **done**, final-epoch CV **1.1719** against the parent's 1.1743, paired t = -0.46 over folds. Artifact committed. See section 3. |
| baseline half of step 6's table | **done and verified.** `punisher_timeout_table.py` written and self-tested; its "before" column reproduces every declared baseline number exactly -- frontier RCC 1.2969, RCB 1.6591, RCD 1.2515, RCE 0.8823, mean 1.0331, rows <= 1 14/22; ref_lin mean 1.6616, 13/22, RCC 1.4615, RCE 0.9900; ref_gnn mean 1.6603, 11/22, RCC 1.1109, RCE 0.9157. Only the "after" column is missing. |
| 2 -- cluster unit tests | **no result.** Fixture shipped, run dispatched, connection died mid-call, empty log. |
| 5 -- mechanism table with the GNN rows | **not run** (needs `torch_scatter`, i.e. the cluster). |
| 6 -- the two simulations, fetch, evaluate | **not run.** |
| 7 -- verdict | **not reached.** No PR. |

### The seed noise floor, and how the result must be read when it arrives

PR #195 (`auto/seed-spread-noise-floor`) measured, after session 1 stopped, how far this evaluation moves when nothing changes but a training seed: **a typical row travels sd 0.138**, the 22-row mean sd 0.047 against a gate-2 margin of 0.103, and the rows <= 1 count sd 3.16 with a range of 6 to 14. It changed the contributor, not the punisher, so it is not this experiment's own variance -- but it is the right scale for reading any row that moves, and no movement below it should be called a result.

The numbers this branch will be judged on, each with its floor:

| quantity | baseline | seed sd | note |
|---|---|---|---|
| **RCC** (declared target, gate 1: band 1-2 -> <= 1, i.e. < 1.0) | 1.2969 | **0.1631** | gateable (boundary 2.6 sd away), but the baseline 1.2969 is the **six-arm minimum**, 1.7 sd below the arm mean -- the target is a favourable draw to beat |
| **22-row mean** (gate 2: <= 1.1364) | 1.0331 | **0.0473** | the whole 10% allowance is 2.2 seed sd wide |
| **RCE** (protected) | 0.8823 | **0.1063** | **ungateable on one run**: the band boundary sits 0.64 sd away and only one arm of six reaches `<= 1` -- and that one arm is this baseline |
| RCE band slopes +0.087 / +0.038 / -0.043 / -0.130 | | 0.0182 / 0.0223 / 0.0263 / 0.0564 | the 10-14 and 15-19 slopes change sign across six retrains of an unchanged model |
| RCB / RCD (watch) | 1.6591 / 1.2515 | 0.1425 / 0.2698 | both gateable |
| rows <= 1 | 14/22 | **3.16** | report, never conclude |

`punisher_timeout_table.py` carries these as `SEED_SD` / `UNGATEABLE` and emits `seed_sd`, `in_seed_sd`, `legible` and `ungateable` columns, so the table itself marks any movement smaller than a retrain. It also applies the **amended magnitude clause** the maintainer set for this branch -- a halved RCE band slope fires only when the candidate's slope is *not* closer to the human value **and** the change exceeds one pooled standard error -- while still printing the raw `magnitude_halved` list beside it, so the parent's reading stays visible.

### The commands that are left

Run from this worktree, `/Users/brinkmann/repros/algorithmic-institutions/.claude/worktrees/agent-abeff61f385511d28`. Local python is `/Users/brinkmann/repros/algorithmic-institutions/.venv/bin/python` (this worktree has no venv of its own).

```bash
ssh raven                                        # maintainer, separate terminal
export AI_REMOTE_DIR=~/repros/ai-runs/punisher-timeout

# 0. FIRST, before any sync. Job 30323407 (the identical training resubmitted
#    from the committed tree) was PENDING when the connection died. If it has
#    since run it has overwritten the remote artifact dir, and a full sync
#    deletes it. Fetch it to a scratch path and compare against the committed
#    metrics parquet: same seed, same config, same data, so any difference is
#    GPU non-determinism and the two should agree to ~1e-3. If they do, the
#    committed artifact is confirmed; if they diverge, prefer 30323407's and
#    say so. Then `scancel 30323407` if it is still queued.
rsync -az raven:'~/repros/ai-runs/punisher-timeout/artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_timeout' /tmp/gnn_replication/

# 1. The four cluster-only unit tests. The fixture is already on the cluster;
#    re-ship it after any full sync, which excludes plots/ and does not carry it.
rsync -az plots/simulation/22_2g8a_linear_self_ridge_contr/per_round.parquet \
  raven:'~/repros/ai-runs/punisher-timeout/plots/simulation/22_2g8a_linear_self_ridge_contr/'
scripts/remote_test.sh --test-only -- -k test_punisher_current_contribution -v

# 2. Step 5 -- the mechanism table with the GNN rows, on Raven's login node.
#    The two linear rows are already measured (section 3) and must reproduce
#    exactly; that reproduction is the control on the whole table.
PYTHONPATH=src python scripts/data_analysis/punisher_mechanism_check.py \
  --linear ceiling_parent=artifacts/baselines/punishment_multinomial_ceiling.joblib \
           timeout_new=artifacts/baselines/punishment_multinomial_timeout.joblib \
  --gnn ceiling_parent=artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_ceiling/model/architecture_node+edge+rnn__dataset_50ep_doubled.pt \
        timeout_new=artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_timeout/model/architecture_node+edge+rnn__dataset_50ep_doubled.pt \
  --out plots/data_analysis/evaluation/punisher_timeout_feature/mechanism_teacher_forced.csv

# 3. Step 6 -- both stacks (the parent took ~2 min 25 s each on one A100).
#    Use --no-sync once artifacts are on the cluster and unfetched.
scripts/simulate_cluster.sh \
  configs/simulation/manager_testing/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_timeout.yml
scripts/simulate_cluster.sh --no-sync \
  configs/simulation/manager_testing/23_2g8a_self_gnn_contr_gnn_switch_timeout.yml
scripts/fetch_cluster.sh \
  plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_timeout/
scripts/fetch_cluster.sh plots/simulation/23_2g8a_self_gnn_contr_gnn_switch_timeout/

# 4. Evaluate all 22 rows locally. PYTHONPATH=src is not optional: without it
#    the shared venv's editable install resolves `aimanager` to the main
#    checkout, which lacks the RCE row and silently scores 21.
PYTHONPATH=src /Users/brinkmann/repros/algorithmic-institutions/.venv/bin/python \
  -m aimanager evaluate \
  configs/simulation/manager_testing/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_timeout.yml
PYTHONPATH=src /Users/brinkmann/repros/algorithmic-institutions/.venv/bin/python \
  -m aimanager evaluate \
  configs/simulation/manager_testing/23_2g8a_self_gnn_contr_gnn_switch_timeout.yml

# 5. Step 7 -- the before/after table. Already written, already self-tested,
#    no edits needed: it reads the `_ceiling` dirs as before and the `_timeout`
#    dirs as after, and prints the gate verdict with the seed floor beside it.
PYTHONPATH=src /Users/brinkmann/repros/algorithmic-institutions/.venv/bin/python \
  scripts/data_analysis/punisher_timeout_table.py
```

**Then judge.** Gate 1: RCC < 1.0. Gate 2: the 22-row mean <= 1.1364. RCE protected at 0.8823 with the amended magnitude clause, every band slope reported with its standard error and row count. RPA (0.6620) is reported, **not** declared, and is not a gate -- it already sits in the best band. The reference stack is reported, not gated. And say plainly, in the verdict itself, whether each movement clears the seed floor in the table above; a band upgrade that moves RCC by less than 0.163 is a draw, not a finding.

## 6. For a successor

1. **The serving-side defect in the other two slots is real, measured, and still there.** This branch intercepts the environment's imputed 9 only on the two punisher paths. `linear_ah._pool_from_env`, which serves the contribution and switch models, still passes `cv=None`, and the GNN contributor reads `self.state` directly. So during simulation a timed-out player is shown to the **switch** model as having contributed 9 that round (it reads the current-round contribution directly, block B1) and to the **contributor** model as having contributed 9 the round before (it is prev-anchored, through `prev_contribution` and the group means). In the human data the `valid_model` fires on **560 / 19,200 = 2.9% of agent-rounds**, and the simulated rate can only be measured by running a stack. This was left unfixed **on purpose**, so that any movement in the 22 rows on this branch is attributable to the punisher alone. It is a separately declarable defect in the contributor and switch slots and a successor should take it: the fix is the same one-line shape already applied on the punisher side, and the natural target rows are the S family and CG.

2. **The graph punisher barely moved and the linear one did.** CV log loss -0.0024 (t = -0.46 paired over folds) against -0.0175 for the linear family, on the same feature and the same data. Before concluding the feature is worthless for the GNN, run the teacher-forced mechanism table (section 5, command 2) and look at the OLS slope on the current contribution: the linear punisher closed 41% of its gap to the human -0.242 on that slope, and the question is whether the GNN's slope was already closer because the RNN and edge model reconstruct "gave no input" from context. If it was, that is a finding about what the graph architecture already encodes, and it is worth a paragraph whichever way the gates land.

3. **Two experiments are now pending on this branch that nobody should re-derive.** Job **30323407** on Raven is the identical GNN training resubmitted from the committed tree; fetch it before any sync (section 5, command 0) and use it as a determinism check on the committed artifact. And the accounting identity of step 0 is settled -- 0 is the game's real value on all 4,512 group-rounds, the imputed 9 fails on all 516 timeout group-rounds -- so the premise never needs re-testing.

4. **Read every row against the seed floor, and say so out loud.** The table script now emits `seed_sd`, `in_seed_sd`, `legible` and `ungateable` per row. The protected row RCE is in the ungateable ten: its baseline 0.8823 is the only one of six same-config retrains that reaches band `<= 1`, so a "band drop" on RCE here is as likely to be the baseline's luck as the candidate's damage. PR #195's section 6 asks the maintainer for three protocol changes on exactly this point; until they are adopted, the honest move is to report the clause firing *and* its distance from the floor, which this branch's script does.

5. **Housekeeping.** The isolated remote dir `~/repros/ai-runs/punisher-timeout` can be deleted when this PR closes. `punishment_baseline.py`'s `GNN_REF` is deliberately left at 1.1756. No copula was recalibrated: the severity copula carries rho 0.4273 unchanged, as the freeze requires, and the refit value 0.4821 in section 3 is recorded for the maintainer, not applied.
