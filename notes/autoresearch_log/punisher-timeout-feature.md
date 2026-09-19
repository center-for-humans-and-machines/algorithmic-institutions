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
| 4 | Retrain the GNN punisher on Raven (`rnn_edge_50ep_doubled_timeout.yml`); report CV against 1.1743. | **BLOCKED -- needs Raven** |
| 5 | Teacher-forced mechanism check of both new punishers against the parent's artifacts and the human row. | **linear done; GNN needs Raven** |
| 6 | Re-run the two stacks, fetch, evaluate all 22 rows (`PYTHONPATH=<worktree>/src`), self-play mechanism table, RCE band slopes with standard errors. | **BLOCKED -- needs Raven** |
| 7 | Judge under the gates with RCE protected; measure the blast radius on the other slots; log; PR against the parent. | **blast radius done; verdict BLOCKED** |

The cluster was unreachable for this session (the persistent SSH connection to Raven is down and only the maintainer can restore it). Everything that can be established without a GPU is established and committed; section 5 lists what is left, with the exact commands. **No verdict is claimed and no PR is opened**, because the result is not established until steps 4 and 6 run.

## 3. Results

| date | change (one line) | target scores | rows <= 1 | mean | verdict |
|---|---|---|---|---|---|
| 2026-09-19 | punisher reads the recorded 0 for a timeout and gains `contribution_valid` | pending (needs Raven) | pending | pending | **pending** |

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

## 5. Remaining work -- all of it needs Raven

Nothing below was attempted: the persistent SSH connection to Raven was down for this session and only the maintainer can restore it. Commands are given in full so whoever resumes runs them without rediscovery. Run from the worktree root, `/Users/brinkmann/repros/algorithmic-institutions/.claude/worktrees/agent-abeff61f385511d28`. Local python is `/Users/brinkmann/repros/algorithmic-institutions/.venv/bin/python` (this worktree has no venv of its own).

```bash
# 0. Prerequisite: bring the ControlMaster up in a separate terminal (persists 12h).
ssh raven

# Isolated remote dir for every remote step below (delete when the PR closes).
export AI_REMOTE_DIR=~/repros/ai-runs/punisher-timeout

# 1. Step 4 -- train the GNN punisher (parent took 6 min 54 s on one A100).
scripts/train_cluster.sh ah \
  configs/training/artificial_humans/punishment/rnn_edge_50ep_doubled_timeout.yml
scripts/fetch_cluster.sh \
  artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_timeout/
# Report final-epoch CV log loss against the parent's 1.1743. Leave GNN_REF at 1.1756.

# 2. The four cluster-only unit tests (all fail locally on torch_scatter only).
scripts/remote_test.sh -- -k test_punisher_current_contribution -v

# 3. Step 5 -- the mechanism table including the GNN rows, on Raven's login node,
#    written to the contracted path. The two linear rows are already measured
#    (section 3) and must reproduce exactly.
PYTHONPATH=src python scripts/data_analysis/punisher_mechanism_check.py \
  --linear ceiling_parent=artifacts/baselines/punishment_multinomial_ceiling.joblib \
           timeout_new=artifacts/baselines/punishment_multinomial_timeout.joblib \
  --gnn ceiling_parent=artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_ceiling/model/architecture_node+edge+rnn__dataset_50ep_doubled.pt \
        timeout_new=artifacts/artificial_humans/punishment/rnn_edge_50ep_doubled_timeout/model/architecture_node+edge+rnn__dataset_50ep_doubled.pt \
  --out plots/data_analysis/evaluation/punisher_timeout_feature/mechanism_teacher_forced.csv

# 4. Step 6 -- re-run both stacks (parent took ~2 min 25 s each on one A100).
#    Both configs are already written and their paths validated; only the GNN
#    punisher artifact is missing until step 1 lands.
scripts/simulate_cluster.sh \
  configs/simulation/manager_testing/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_timeout.yml
scripts/simulate_cluster.sh \
  configs/simulation/manager_testing/23_2g8a_self_gnn_contr_gnn_switch_timeout.yml
scripts/fetch_cluster.sh \
  plots/simulation/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_timeout/
scripts/fetch_cluster.sh plots/simulation/23_2g8a_self_gnn_contr_gnn_switch_timeout/

# 5. Step 6 -- evaluate all 22 rows locally (evaluate takes the simulation config
#    and reads that sim's per_round.parquet; 500 repeats, seed 42 as on the parent).
PYTHONPATH=src /Users/brinkmann/repros/algorithmic-institutions/.venv/bin/python \
  -m aimanager evaluate \
  configs/simulation/manager_testing/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_timeout.yml
PYTHONPATH=src /Users/brinkmann/repros/algorithmic-institutions/.venv/bin/python \
  -m aimanager evaluate \
  configs/simulation/manager_testing/23_2g8a_self_gnn_contr_gnn_switch_timeout.yml

# 6. Step 7 -- the before/after table, self-play mechanism rows and RCE band
#    slopes with standard errors. Copy the parent's table script and retarget it:
#      cp scripts/data_analysis/punisher_ceiling_table.py \
#         scripts/data_analysis/punisher_timeout_table.py
#    then in the copy change exactly four things --
#      line 38  OUT_DIR -> "plots/data_analysis/evaluation/punisher_timeout_feature"
#      line 99  before: src + "_curpun"  -> src + "_ceiling"
#      line 100 after:  src + "_ceiling" -> src + "_timeout"
#      line 113 the same two suffixes in the rce_check(...) call
#    so that "before" is this branch's parent and "after" is this branch.
PYTHONPATH=src /Users/brinkmann/repros/algorithmic-institutions/.venv/bin/python \
  scripts/data_analysis/punisher_timeout_table.py
```

**The judgement, once those have run.** Declared target **RCC**, baseline **1.2969** in band 1-2; gate 1 is a band upgrade, RCC **< 1.0**. Gate 2: the mean over the 22 rows must stay within 10% of the parent's **1.0331**, i.e. **<= 1.1364** (the parent's own margin rule, `b174f90`). **RCE is protected** at **0.8823** with band slopes +0.087 / +0.038 / -0.043 / -0.130 (signs + + - -): report each band's slope with its standard error and check for a band downgrade, a slope losing the human sign, or a slope magnitude halving. Watch RCB (1.6591) and RCD, the PA/PB/PC marginals, and the cross-tab flagged at the end of step 4. RPA (0.6620) is reported but **not** declared and must not be treated as a gate. The reference stack is reported, not gated, and the parent's note that the protected-row rule fires on thin bands there applies unchanged.

Only after all of that: open the PR with `--base auto/punisher-ceiling-fix`, add `Closes` for the issue, comment the PR link on the issue, drop the `*-agent-ready` label and add `human-review`.
