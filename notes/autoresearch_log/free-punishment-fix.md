# The free punishment on a player who gave no input (the fourth and last instance)

## 1. Declaration

**Slot:** shared simulation code -- a bug fix under §4's "a bug fix in shared code is legal but is its own experiment". **Nothing is trained on this branch at all**: no artificial human, no manager, no copula. No weights, features or training configs change, and no copula parameter is recalibrated.

**Parent:** `auto/rl-manager-timeout-view` at `56bf6097e91ed431263ac690ae13b12105267c5b` (PR #205, `[FIX]`, the third serving-path instance), which sits on `auto/sim-timeout-imputation` (PR #196) and on `auto/punisher-timeout-feature` beneath that. Branch `auto/free-punishment-fix` is created from it and the PR opens with `--base auto/rl-manager-timeout-view`. Isolated remote dir `~/repros/ai-runs/free-punishment` (delete when this PR closes).

**The defect.** `ArtificialHumanEnv.punish` stored the manager's action verbatim for all eight agents, including the ones the validity model had just marked as timed out. The accounting then discarded it -- `compute_common_good_per_group` zeroes punishment at `~contribution_valid`, `compute_payoff_per_group` zeroed the whole invalid contributor's payoff -- so the action cost the manager nothing. But `step()` copied the raw value into `prev_punishment`, the contribution model's only channel from the manager, and `served_state()` corrected only `contribution` and `prev_contribution`. Free deterrence: a behavioural lever the real game did not offer, on cells the manager can identify through `contribution_valid`, which is in its own `x_encoding`. This is `notes/reviews/rl-manager-review.md` D1, scoped to its own branch by the parent's note 2 and section 6.2, and it is the fourth and last known instance of one defect -- a model being shown a value for a timed-out player that the game never used.

**What makes it its own experiment, and the one thing that separates it from the three before it.** The three serving-path fixes all left `self.state` alone: they corrected what a model was *shown* and left what the run *recorded* exactly as it was, which is why the parent could prove a byte-identical `per_round.parquet`. This one cannot and must not. The value being corrected is the manager's own action, and the run's record of the manager's action is `per_round.parquet`'s `punishment` column. Zeroing it where the action is realised moves that column, contributions diverge from the round after, and every one of the 22 rows moves with them. Section 3 step 2 argues why moving the record is right here and was wrong there; the two cases genuinely differ, and the difference is in the human data, not in convenience.

### Hypothesis

**Behavioural rationale (§5, one sentence):** a punishment aimed at a player who gave no input was never charged and never shown -- all 560 timed-out rows of the human data carry `punishment == 0.0` exactly -- so the artificial humans must be shown 0 there and the run must record 0 there; the rows that should move are the group-spread row CG and the response family, because the correction removes a wrong value from the contribution model's only channel from the manager.

**Declared targets (before the first simulation):** CG and the R family -- RCA, RCB, RCC, RCD, RCE, RSA, RPA, RPB. RCE is the protected row and is judged under the amended clauses of §2 on `docs/post-rebaseline-program`.

**The change.** Two hunks, in the idiom of the corrections beneath:

1. `ArtificialHumanEnv.punish` zeroes the punishment at `~contribution_valid` where the action is realised -- the same rule the common-good and payoff accounting already applied, applied one step earlier. Because `punish()` is the single point every path goes through, this corrects the contribution model's `prev_punishment`, the switch model's `punishment`, the RL manager's replay view, the RL run's metrics and the recorded `per_round.parquet` at once.
2. `simulate.py` builds the manager's own round record from the env's realised punishment instead of from the raw action. That record is where `api_manager.create_data` shifts the punisher's `prev_punishment` feature from, and the punisher is an artificial human too. One reordering, no new rule.

### Artifact naming contract

| what | path |
|---|---|
| probe | `scripts/data_analysis/free_punishment_probe.py`; output `plots/data_analysis/evaluation/free_punishment/{env,sim,run}_*.json` |
| before/after table | `scripts/data_analysis/free_punishment_table.py`; output `plots/data_analysis/evaluation/free_punishment/before_after.{csv,md}` |
| tests | `src/aimanager/tests/test_free_punishment.py` |
| sim configs | `configs/simulation/manager_testing/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_freepun.yml` (gated), `..._self_gnn_contr_gnn_switch_freepun.yml` (reported, not gated) |

## 2. Plan

| # | step | status |
|---|---|---|
| 1 | Confirm the lever independently on the unchanged tree: that the action is free, that it is shown, and how often the path fires. | **done** |
| 2 | Fix it at the point the action is realised; state what that does to the recorded output and why it is right here. | **done** |
| 3 | Unit tests in the shape of `test_rl_manager_timeout_view.py`, run against the unfixed code first. | **done** -- 4 fail on the parent, 8 pass here, 181 pass on Raven |
| 4 | Re-run the frontier stack and two reference stacks, evaluate all 22 rows, quote every movement against the seed floor. | **done** |
| 5 | The protected row RCE under the amended clauses, every band with its SE and row count. | **done** |
| 6 | The interaction with `auto/manager-common-pool-reward` (PR #206), which also edits `punish()`. | **done** -- section 4 |

## 3. Results

### Step 1: the lever, measured on the unchanged tree

**The human evidence, reconfirmed independently.** `experiments/2group_8agent_50ep.csv`, 19,200 agent-rounds: 560 rows carry `player_no_input == 1`, a rate of **2.917%**, and the punishment on every one of them is exactly **0.0** (`value_counts` returns a single class, max |p| = 0.0). On the rows where the player did give input, 30.2% carry a punishment above zero. The real game charged nothing and showed nothing on those cells; the two populations are not close.

**The counterfactual, in the env, with plain torch and mock models** (`free_punishment_probe.py --mode env`, agent 0 timed out, agent 1 genuinely contributing 0):

| manager's action on the timed-out agent | group common good | group payoff sum | recorded `punishment` | served `punishment` (switch model) | served `prev_punishment` next round (contribution model) |
|---|---|---|---|---|---|
| 0 | 11.266666 | 67.800003 | 0 | 0 | 0 |
| 30 | 11.266666 | 67.800003 | **30** | **30** | **30** |

Identical reward to the digit, a different input to every downstream model and a different value in the record. This reproduces the review's table and the parent's reproduction of it, on a third independent run.

**How free it is, over a whole simulation rather than one round.** The probe recomputes both accounting functions on every round of a 100-episode frontier rollout, once with the punisher's action as played and once with it zeroed at `~contribution_valid`, and reports the largest difference either way: **max change in common good 0.0, max change in group payoff sum 0.0** over all 19,200 agent-rounds. Not "small" -- zero, because the accounting already applies the same mask.

**Which model reads which channel** (read off the loaded artifacts, not assumed): the contribution model's encoding is `agent_group, prev_contribution, prev_punishment`; the switch model's is `agent_group, common_good, punishment, round_number`; the multinomial punisher's feature set includes `prev_punishment`. So all three of the artificial humans read one of the two punishment channels, and `prev_punishment` -- the one the fix reaches a round later -- is the contribution model's *only* channel from the manager.

**How often the path fires.** Three measurements, all on the frontier stack, 19,200 agent-rounds each:

| measurement | timeouts | rate | cells the punisher aims a punishment at |
|---|---|---|---|
| probe rollout, unfixed code (`--mode sim`) | 357 | 1.86% | 2 (values 15 and 24) |
| probe rollout, fixed code (`--mode sim`) | 357 | 1.86% | 2 (identical -- the timeout process is untouched) |
| the scored run itself (`--mode run`, `run_simulation` wrapped) | 374 | 1.95% | **1** (value 30) |

The first two roll the stack out from the probe and so do not consume the RNG in exactly the order `simulate.py` does; the third wraps `ArtificialHumanEnv.punish` around the unmodified `run_simulation`, so it reports the rate on the very trajectory that is scored. **On the gated stack the fix corrects one cell in 19,200 at source.** That number is the headline of the measurement section and it is the reason the gate outcome reads as it does.

It is a property of the punisher, not of the defect. The frontier's punisher is `punishment_multinomial_timeout_severity_copula`, which carries the `contribution_valid` feature `auto/punisher-timeout-feature` added for exactly this population and has learned from the human data that those cells are never punished. The GNN punisher, which carries the same feature, has not: on the `gnn_self` reference pairing the fix zeroes **326** punishment cells of 19,200 (1.70%). Same defect, two orders of magnitude apart in how much of it the artificial manager was actually using.

### Step 2: the change, and what it does to the recorded output

`environment.punish` now realises the action the way the game realised it:

```python
self.punishment = th.where(
    self.contribution_valid, punishment, th.zeros_like(punishment)
)
```

and `simulate.py` builds the manager's round record from `state["punishment"]` after `punish()` rather than from the raw action, so the punisher's own `prev_punishment` feature carries the charged value too. Nothing else changes: `punishment_valid` stays True (the manager *did* give input; the human data agrees -- `convert.load_human` only NaNs punishment where `manager_no_input == 1`), the accounting is untouched, and the contribution substitution the parent chain deliberately left in the record is untouched.

**What this does to `per_round.parquet`, stated plainly: it changes it.** `recorder.add(**state)` runs immediately after `env.punish`, so the `punishment` column now carries 0 on every timed-out cell, and because `prev_punishment` feeds the contribution model the contributions diverge from the following round on. Measured on the gated stack: 14 of 19,200 punishment cells differ, 35 contribution cells differ, mean punishment 1.8464 -> 1.8417, mean contribution 9.6595 -> 9.6585. On the `gnn_self` pairing, where the lever was actually being used, 444 punishment cells and 505 contribution cells differ and 484 rows change group.

**Is moving the record right here, when the three fixes beneath it deliberately did not?** Yes, and the reason is in the human data rather than in preference. The parent chain's constraint was specific: `convert.load_human` NaNs a timed-out player's **contribution**, so those human rows are *absent* from every metric, while `load_sim` has no validity column and the simulated ones are *present*. Recording 0 for a simulated contribution would therefore have pushed ~2% of the scored rows to a hard zero against human rows that are not in the comparison at all -- a change to the scored population, not a correction. **Punishment is the opposite case.** `load_human` keeps a timed-out player's punishment; it only drops it where the *manager* gave no input. So on the human side those 560 rows are in the frame, carrying exactly 0. Before this change the simulated rows beside them carried whatever the artificial manager felt like. After it they carry 0, which is what the human rows carry. The record moves toward the comparison, not away from it. The one asymmetry that remains -- the sim rows are scored in a contribution bin of 9 where the human rows are dropped for having no contribution -- is the parent chain's open issue and is untouched here.

**Every consumer of `punishment` in the loop, checked rather than assumed.**

1. Contribution AH, via `served_state()["prev_punishment"]` -- **corrected**, the round after the action.
2. Switch AH, via `served_state()["punishment"]` -- **corrected**, the same round.
3. Punisher AH, via the `simulate.py` round record -> `api_manager.create_data` -> `prev_punishment` -- **corrected** by the second hunk. Measured after the fix: all 346 previously-timed-out cells the punisher is served read 0.
4. `Memory.add` -> `mem_to_df` -> `per_round.parquet` -> the evaluation suite -- **corrected, deliberately**, argued above and measured in step 4.
5. `compute_common_good_per_group` / `compute_payoff_per_group` -- **unchanged in effect.** They apply the same mask themselves, so feeding them the already-charged value is idempotent; step 1's zero-difference measurement is the proof, and a test pins the common good against the env's own accounting.
6. `rl_manager.run_batch`'s replay buffer (`prev_punishment`, `punishment` in `replay_keys`) -- **corrected**, and correctly: the manager should learn from what the game charged.
7. `rl_manager`'s Q-learning target -- **untouched.** `replay_mem.add(action=action, ...)` stores the raw action separately and `manager.update` gathers the Q-values on that, so the action indexing is unaffected by the zeroing. This was checked because zeroing the state copy alone would have made the update attribute the outcome to an action that was not taken.
8. `rl_manager`'s recorded metrics -- **changed**, by the same argument as (4): they are the run's record of what the game charged.
9. `api_manager.create_data`'s `punishment` for the round being punished -- **untouched**; it is a placeholder, the round's punishments are not known yet.

### Step 3: unit tests

`src/aimanager/tests/test_free_punishment.py`, eight tests, in the shape of `test_rl_manager_timeout_view.py` and with the same fixture discipline -- agent 0 times out, agent 1 genuinely contributes 0, so a cell the manager may punish sits next to one it may not in every assertion:

- `test_the_recorded_punishment_on_a_timeout_is_zero` -- the one recorded value the fix changes;
- `test_the_switch_model_is_served_zero_this_round`;
- `test_the_contribution_model_is_served_zero_next_round` -- the channel the lever actually acted on;
- `test_the_punisher_record_carries_the_charged_value` -- through the real `make_round` / `add_punishments` / `create_data`, so it pins the `simulate.py` seam rather than the env alone;
- `test_a_punishment_on_a_genuine_zero_is_untouched`, `test_the_reward_is_unchanged`, `test_round_zero_prev_punishment_default_is_untouched`, `test_the_recorded_contribution_still_carries_the_imputed_value` -- the four guards, including the boundary of the parent's fix.

**The four defect tests fail on the parent and pass here; the four guards pass on both** -- measured by running the new file against the parent's `environment.py` and `simulate.py` on Raven (4 failed, 4 passed) and against this branch's (8 passed), not merely asserted. `scripts/remote_test.sh`-equivalent over the whole of `src/` and `scripts/tests` on Raven: **181 passed**, 0 failed. (The isolated dir needs `plots/simulation/22_2g8a_linear_self_ridge_contr/per_round.parquet` shipped by hand, as the parent's note 4 warns, or five evaluation-suite tests fail on a missing fixture and it is not a regression.)

`flake8 --max-line-length=88 --extend-ignore=E203,W503` is clean over `src/`. `black --check` leaves `rl_manager.py` and `artificial_humans/train.py` as the two files already unformatted on the parent; neither is touched here, and no other file in `src/` would be reformatted. The two new analysis scripts carry a handful of long lines in the same places their predecessors `sim_timeout_table.py` and `punisher_timeout_table.py` do (21 between them); the pre-commit hooks scope black and flake8 to `^src/`, so this is the existing convention rather than an exception made here.

### Step 4: the 22 rows, before and after, with the seed floor beside every movement

Before = the parent chain's `_simtimeout` sims, after = the `_freepun` sims. The two configs differ in `output_dir` and `figure_name` and in nothing else -- same seed 42, same 100 episodes, same 24 rounds, the same four artifacts on disk. **Nothing is retrained, so this branch's own run-to-run variation is exactly zero**: the simulation is bit-reproducible, the parent demonstrated it byte for byte, and every number below is the fix and nothing else. The PR #195 seed floor is quoted beside each row for a different reason -- it is the scale on which the *model* is uncertain, so a movement under it is not distinguishable from the same model retrained with another draw, and ten rows cannot be gated on a single run at all.

Raven jobs **30400961** (frontier) and **30400962** (reference), both COMPLETED, exit 0, 00:02:16 and 00:02:29.

**Frontier stack** `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch`, run `lin_multinomial_copula_self` -- the gated one:

| row | before | after | delta | in seed sd | distinguishable | band | target |
|---|---|---|---|---|---|---|---|
| CA | 0.8422 | 0.8417 | -0.0006 | 0.00 | no | <= 1 | |
| CB | 0.8164 | 0.8161 | -0.0003 | 0.00 | no | <= 1 | |
| CC | 0.8154 | 0.8151 | -0.0003 | 0.00 | no | <= 1 | |
| CD | 0.7665 | 0.7659 | -0.0005 | 0.00 | no | <= 1 | |
| CE | 0.9675 | 0.9675 | 0.0000 | 0.00 | no | <= 1 | |
| CF | 0.8152 | 0.8144 | -0.0008 | 0.01 | no | <= 1 | |
| CG | 1.8449 | 1.8461 | +0.0012 | 0.00 | no | 1-2 | **target** |
| SA | 0.7741 | 0.7741 | 0.0000 | 0.00 | no | <= 1 | |
| SB | 1.0695 | 1.0695 | 0.0000 | 0.00 | no | 1-2 | |
| SC | 1.8295 | 1.8295 | 0.0000 | 0.00 | no | 1-2 | |
| PA | 0.6408 | 0.6413 | +0.0005 | 0.01 | no | <= 1 | |
| PB | 0.8995 | 0.8999 | +0.0004 | 0.02 | no | <= 1 | |
| PC | 0.9147 | 0.9146 | -0.0001 | 0.00 | no | <= 1 | |
| PD | 0.6860 | 0.6804 | -0.0057 | 0.10 | no | <= 1 | |
| RCA | 1.7761 | 1.7730 | -0.0031 | 0.02 | no | 1-2 | **target** |
| RCB | 1.0668 | 1.0596 | -0.0072 | 0.05 | no | 1-2 | **target** |
| RCC | 1.4983 | 1.4974 | -0.0009 | 0.01 | no | 1-2 | **target** |
| RCD | 1.3368 | 1.3372 | +0.0004 | 0.00 | no | 1-2 | **target** |
| RCE | 0.9474 | 0.9260 | -0.0215 | 0.20 | no | <= 1 | **target, protected** |
| RSA | 1.1529 | 1.1529 | 0.0000 | 0.00 | no | 1-2 | **target** |
| RPA | 0.6426 | 0.6417 | -0.0009 | 0.05 | no | <= 1 | **target** |
| RPB | 0.7624 | 0.7613 | -0.0011 | 0.04 | no | <= 1 | **target** |
| **22-row mean** | **1.0393** | **1.0375** | **-0.0018** | **0.04** | **no** | | |
| rows <= 1 | 14 | 14 | 0 | 0.00 | no | | |

**Not one of the 22 rows moves as much as a quarter of its own seed sd.** The largest movement in the table is the protected row RCE at -0.0215, which is 0.20 of its floor of 0.106; fourteen rows move by less than 0.001; no row changes band. Sixteen of the 22 move in the improving direction and six against, which is the pattern of one cell's worth of divergence rather than of a mechanism. Under the symmetry rule none of it counts either way.

**Reference stack** `23_2g8a_self_gnn_contr_gnn_switch`, reported and not gated. Two pairings, and they are the interesting part of this experiment:

| | `lin_multinomial_self` | | `gnn_self` | |
|---|---|---|---|---|
| row | delta | in seed sd | delta | in seed sd |
| CA | 0.0000 | 0.00 | -0.0112 | 0.06 |
| CB | 0.0000 | 0.00 | -0.0069 | 0.03 |
| CC | 0.0000 | 0.00 | +0.0276 | 0.21 |
| CD | 0.0000 | 0.00 | -0.0073 | 0.04 |
| CE | 0.0000 | 0.00 | +0.0630 | **1.08** |
| CF | 0.0000 | 0.00 | -0.0406 | 0.29 |
| CG | 0.0000 | 0.00 | +0.3435 | **1.14** |
| SA | 0.0000 | 0.00 | -0.1722 | **1.06** (1-2 -> <= 1) |
| SB | 0.0000 | 0.00 | -0.0608 | **1.33** |
| SC | 0.0000 | 0.00 | +0.2774 | **2.05** |
| PA | -0.0004 | 0.01 | **-0.3928** | **9.72** (2-5 -> 1-2) |
| PB | +0.0001 | 0.01 | **-0.2147** | **9.29** |
| PC | +0.0001 | 0.00 | **-0.1692** | **4.71** |
| PD | -0.0007 | 0.01 | +0.0382 | 0.64 |
| RCA | 0.0000 | 0.00 | -0.0342 | 0.24 |
| RCB | -0.0019 | 0.01 | -0.0655 | 0.46 |
| RCC | 0.0000 | 0.00 | +0.0171 | 0.11 |
| RCD | 0.0000 | 0.00 | +0.0620 | 0.23 |
| RCE | -0.0001 | 0.00 | -0.0012 | 0.01 |
| RSA | 0.0000 | 0.00 | -0.0051 | 0.03 |
| RPA | +0.0001 | 0.01 | **-0.1270** | **7.26** (1-2 -> <= 1) |
| RPB | 0.0000 | 0.00 | **-0.2912** | **10.29** (2-5 -> 1-2) |
| **mean** | **-0.0001** | 0.00 | **-0.0350** | 0.74 |
| rows <= 1 | 14 -> 14 | | **6 -> 8** | |

The `lin_multinomial_self` pairing zeroes **one** cell in 19,200 and moves nothing. The `gnn_self` pairing zeroes **326** and moves four rows across a band, all four toward the humans, three of them by seven to ten seed sd: PA 2.09 -> 1.70, PB 1.50 -> 1.28, RPA 1.09 -> 0.97, RPB 2.02 -> 1.73. Its mean falls by 0.035 and its rows at the ceiling go from 6 to 8. It also pays for it -- CG +0.34, SC +0.28, CE +0.06, all above their floors. **This is the same correction reading as nothing on one stack and as a large, clearly legible improvement on another, and the difference between them is exactly how much of the free lever the artificial manager in the stack was spending.** It is not evidence about the frontier, it is not eligible for a gate, and it is the strongest evidence on this branch that the fix does what it claims.

### Step 5: the protected row, under the amended clauses

The amended rule of §2 on `docs/post-rebaseline-program`: the band-drop clause fires only on a drop larger than RCE's own seed sd (0.106); the sign clause is retired on the 10-14 and 15-19 bands, where retraining an unchanged model flips the sign on its own, and kept on 0-4 and 5-9; the magnitude clause fires only when the candidate's slope is not closer to the human value *and* the change exceeds one pooled standard error. Every band with its slope, its standard error, its row count, and the change in pooled SE and in the band's own seed sd:

**Frontier stack (gated).** RCE score 0.9474 -> 0.9260, a *fall* of 0.0215 (an improvement), 0.20 seed sd, band `<= 1` throughout.

| band | human | before | after | change in pooled SE | change in band seed sd |
|---|---|---|---|---|---|
| 0-4 | +0.140 +- 0.018 (n 965) | +0.102 +- 0.014 (n 2045) | +0.103 +- 0.014 (n 2043) | 0.04 | 0.04 |
| 5-9 | +0.104 +- 0.024 (n 929) | +0.019 +- 0.015 (n 1906) | +0.024 +- 0.015 (n 1904) | 0.26 | 0.25 |
| 10-14 | -0.077 +- 0.035 (n 560) | +0.000 +- 0.022 (n 1364) | +0.000 +- 0.022 (n 1364) | 0.00 | 0.00 |
| 15-19 | -0.161 +- 0.079 (n 206) | -0.085 +- 0.054 (n 474) | -0.085 +- 0.054 (n 474) | 0.00 | 0.00 |

Signs `+++-` before and after, matching the human `++--` on three of four bands in both arms. **No clause fires, raw or amended**: no band downgrade (the score improves), no sign flip on any band let alone the two that still carry the clause, no halved magnitude. The largest band movement is 0.26 pooled SE. **RCE is protected.**

For the record, the two reference pairings: `lin_multinomial_self` moves no band slope by more than 0.01 pooled SE; `gnn_self` moves 10-14 by 0.44 pooled SE (+0.012 -> +0.024, away from the human -0.077 but well inside the noise and on the band the amended rule declares unlearnable) and the other three by 0.14 to 0.17. No clause fires on either.

## 4. The interaction with `auto/manager-common-pool-reward` (PR #206)

That branch also edits `punish()`: it moves the reward computation into it, as a pure function of contribution, punishment and validity, and it corrects the payoff of a timed-out player. It is not a base for this work and this branch was not rebased onto it -- doing so would put two changes into one before/after, which is the thing §4 exists to prevent.

**The base this experiment is measured against is stated once and plainly: `auto/rl-manager-timeout-view` at `56bf6097`, without PR #206's reward rewrite and without its payoff correction.** The before and after arms are both on that base; the reward mode in both sim configs is the env default and no simulation on this branch reads the reward at all (`run_simulation` never touches `env.reward`), so the payoff correction cannot have entered either arm of the table in section 3.

**The two changes were checked together rather than argued about**, on a scratch tree carrying PR #206's `environment.py` with this branch's one-line change applied to it (`interaction_check.py`, plain torch, mock models, agent 0 timed out, the manager aiming 30 at it and 5 at everyone else):

| reward mode | reward with 0 aimed | reward with 30 aimed | recorded punishment | served `prev_punishment` |
|---|---|---|---|---|
| `common_pool` | 21.800000 | 21.800000 | 0.0 | 0.0 |
| `sum` | 71.066666 | 71.066666 | 0.0 | 0.0 |
| `avg` | 17.766666 | 17.766666 | 0.0 | 0.0 |

All three channels -- what the artificial humans are served, what the run records, and what enters the reward -- read 0 on those cells under all three modes, and the arms are identical in every field. **Nothing double-counts**: the reward computed from the already-charged state equals, bit for bit, the reward recomputed from a manually zeroed copy of the action (`th.equal` True), because `th.where(valid, p, 0)` applied twice is `th.where(valid, p, 0)`. The two changes are complementary -- PR #206 makes the reward a pure function of the round's values, this branch makes those values the ones the game used -- and after both land the accounting side and the shown/recorded side finally agree.

**What the merge will look like.** Both branches insert a docstring at the top of `punish()` and both touch its body, so a textual conflict there is likely; it is the trivial kind, and the resolution is to keep both docstrings' content and both code changes, with `self.punishment = th.where(...)` above `self.update_common_good()` and `self.reward = self.compute_reward_per_group(...)` below `self.update_payoff()`, exactly as the scratch tree above has it. **A caution for whoever merges: a silent auto-merge here is not self-evidently correct.** A `git merge-tree` of the *uncommitted* state of this work against PR #206 reported no conflict and produced a `punish()` containing PR #206's changes and none of this branch's -- correct behaviour for the inputs it was given, and a good illustration of how quietly this particular hunk can disappear. Whoever merges should read the merged `punish()` and re-run `test_free_punishment.py`, which fails loudly if the zeroing is lost.

One trap from PR #206's log, repeated here because it cost this branch a double-take: the env's `common_good` field is the **per-capita share**, while the column of the same name in the human CSV is the **undivided pool**. In the interaction table above the `common_pool` reward is 21.8 and the state's `common_good` is 7.267 = 21.8 / 3 valid players in the group.

## 5. Notes

1. **The fix went at the point the action is realised, not at the serving boundary, and that is the whole reason it is two lines.** `punish()` is the single door every path walks through -- the simulation, the RL rollout, the tests -- so correcting the value there corrects the contribution model, the switch model, the punisher's record, the replay buffer and `per_round.parquet` at once. Putting it in `served_state()` instead would have corrected four of those and left the record wrong, which is the failure mode that produced three of the four instances of this defect in the first place.

2. **The rate at which this defect bites is a property of the punisher, not of the code.** 1 cell in 19,200 on the frontier's multinomial punisher, 326 on the reference stack's GNN punisher, and both of those punishers carry the `contribution_valid` feature that `auto/punisher-timeout-feature` added for precisely this population. The linear model learned the rule from it; the GNN did not. That is worth someone's attention on its own -- it is a measured expressiveness gap on a feature that was added at cost -- and it is recorded in step 1 rather than pursued, because it is not this experiment.

3. **The RL manager is the consumer this matters most for and it is the one that cannot be measured here.** The simulation's punishers are fixed artifacts that will never learn to exploit a free lever; a Q-learner with epsilon 0.1 and a cost-bearing reward is exactly the thing that will. The review's D1 argument is about a manager that does not exist yet, and nothing in section 3 confirms or refutes it. What section 3 does establish is that the lever was real, that it was free to the penny over 19,200 agent-rounds, and that it is now gone.

4. **The timeout rate keeps coming out lower in simulation than in the humans.** Human 2.917%, this branch's frontier scored run 1.95%, the probe rollout 1.86%, the parent's RL stack 1.84%, the parent's frontier 2.24%. Five measurements now, all below the human rate. Nobody has declared on it; the trail is in this note and in the parent's note 3.

5. **Nothing was trained and no copula was recalibrated.** Every model in every run on this branch loads the same file on disk the parent's runs loaded. The evaluation suite was not touched.

6. **Housekeeping.** Remote dir `~/repros/ai-runs/free-punishment` can be deleted when this PR closes. The two `_freepun` sim configs and their outputs are additions; no existing sim directory was overwritten.

## 6. The verdict: correct, and not a gate

The correctness outcome and the gate outcome are separate and both are reported, because they disagree.

**Correctness: the defect was real, it is measured, and it is closed.** A punishment aimed at a player who gave no input cost the manager exactly nothing -- max change in common good and in group payoff 0.0 over 19,200 agent-rounds, measured against the env's own accounting -- while the value reached the contribution model's only channel from the manager, the switch model, the punisher's own lag feature and the recorded output. The human game never charged it and never showed it: all 560 timed-out rows carry punishment 0.0 exactly. After the fix every one of those channels reads 0, eight unit tests pin it, four of them fail on the parent, and 181 tests pass on Raven. **This is the right change whether or not a band moves.**

**Gate: `[FAIL]`.** On the gated frontier stack no declared target row changes band, and no row in the table moves as much as a quarter of its own seed standard deviation -- the largest movement in the 22 is RCE at -0.0215 against a floor of 0.106. Gate 1 is not met. Gate 2 holds comfortably (mean 1.0393 -> 1.0375, a fall of 0.0018 = 0.04 seed sd, against a ceiling of 1.1433). The protected row is protected: RCE improves, no band slope flips sign, none is halved, and the largest band movement is 0.26 pooled SE. Under the symmetry rule the honest summary of the frontier table is **no distinguishable movement in either direction**, and that is a statement about how little of the lever the frontier's punisher was using, not about whether the lever was there.

**The reason the gate reads that way is measured, not guessed: one cell in 19,200.** On the reference stack's GNN-punisher pairing, where the same fix zeroes 326 cells, four rows cross a band toward the humans and three of them do so by seven to ten seed sd. The mechanism works; the frontier's punisher had already almost stopped using the lever, so there was almost nothing left there to take away.

### Verdict tag

**[FAIL]** on the gates, with a confirmed defect closed and a large, legible effect demonstrated on a non-gated stack. No band upgrade is claimed on the frontier and none exists. This is the fourth and last known instance of the timed-out-player defect.

## 7. For a successor

1. **This closes the fourth and last known instance of one defect, and the family is complete.** The punisher's two serving paths (`auto/punisher-timeout-feature`), the contribution and switch models' (`auto/sim-timeout-imputation`), the RL manager's three consumers (`auto/rl-manager-timeout-view`), and the punishment channel in both directions (this branch). Every model in the system is now shown, and the run now records, the value the game actually used for a player who gave no input.

2. **The structural fix that would have prevented all four is to make the corrected view opt-out rather than opt-in.** `served_state()` is a helper a call site has to remember to call, and three of the four instances are a call site that did not: the RL path read `self.state` directly, the punisher read a record built beside the env, and this one read a field the env itself had written raw. The parent's successor note already names it; this branch is the fourth data point and should settle it. The shape: `self.state` stops being reachable from outside the env, `reset()` / `punish()` / `step()` return the served view, and the raw dict is exposed under a name that says what it is for -- `recorded_state()`, read by `Memory.add` and by the RL metrics and by nothing else. Then adding a consumer gets the corrected view by default and recording the raw one is the thing you have to ask for, which is the right way round. It is a mechanical change to a handful of call sites and it is worth doing before the manager training run, not after.

3. **D2 and D3 of `notes/reviews/rl-manager-review.md` are still open and both are upstream of any number a manager training run produces.** D2 (the reward discards a timed-out player's payoff) is addressed by `auto/manager-common-pool-reward`, PR #206 -- read its log before assuming anything about the reward. D3 (the RL opponent is the pre-fix lagged punisher) is untouched by anyone.

4. **The GNN punisher does not use its `contribution_valid` feature the way the linear one does.** Measured here: on the same stack, with the same feature available, the multinomial punisher aims a punishment at 1 timed-out cell in 19,200 and the GNN at 326. Whether that is an expressiveness gap, a training-data weighting, or the GNN spending its punishment budget more freely everywhere is not established. It is a punisher-slot experiment with a clear measurement already in hand.

5. **The one-cell frontier result is a warning about the evaluation protocol, not just about this fix.** A correction whose effect on the scored stack is one cell in 19,200 cannot be judged by the 22 rows at all -- the floor of the instrument is 0.02 to 0.30 per row and the signal is three orders of magnitude below it. The right evidence for a change like this is the mechanism measurement (what fires, how often, what it costs) plus a stack where the mechanism is actually exercised, which is what section 3 provides. A successor with a similarly rare correction should plan for that rather than discovering it after the simulation has run.
