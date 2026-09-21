# The reinforcement-learning manager and the timed-out player (the third and last serving path)

## 1. Declaration

**Slot:** shared simulation code -- a bug fix under §4's "a bug fix in shared code is legal but is its own experiment". **Nothing is trained on this branch at all**: no artificial human, no manager, no copula. No weights, features or training configs change.

**Parent:** `auto/sim-timeout-imputation` at `3fe1f449f2a97ad77899affcaf8eeeec0722398d` (PR #196, `[FAIL]` on the gates with a confirmed defect and a real correctness gain), which carries the contribution and switch half of this same fix plus the punisher half beneath it. Branch `auto/rl-manager-timeout-view` is created from it and the PR opens with `--base auto/sim-timeout-imputation`. Isolated remote dir `~/repros/ai-runs/rl-manager-timeout` (delete when this PR closes).

**The defect.** When a player times out the real game charged 0, paid out on 0 and showed everyone 0; the training data stores 0. At simulation time `environment.update_contribution` overwrites those cells with `default_values["contribution"]` -- 9 -- before the state is handed on. `auto/punisher-timeout-feature` corrected the punisher's two serving paths; `auto/sim-timeout-imputation` corrected the contribution and switch models' through the new `ArtificialHumanEnv.served_state()`. The RL path was the one left: `reset()`, `punish()` and `step()` return `self.state`, the raw one, and that is what `rl_manager.run_batch` reads. The parent's own log names it in step 2, consumer 6: "`rl_manager`'s observations. Unchanged; `served_state()` is opt-in and the RL call sites were not touched." So after the parent's fix the contribution and switch models saw the recorded 0 while a manager trained in the same environment saw the made-up 9 -- two models looking at different games, and the manager trained on a value the real game never showed.

**The constraint that still holds.** The parent did not remove the substitution at its source, and that decision was load-bearing and is unchanged here: `self.state["contribution"]` is also what `Memory.add` records, what `mem_to_df` turns into `per_round.parquet` and what the frozen evaluation suite scores, and the suite drops a human timeout from every metric (`convert.load_human` NaNs it) while scoring a simulated one (`load_sim` has no validity column). Recording 0 there would push ~2.2% of the scored simulated rows to a hard zero against human rows that are not present. The fix therefore had to reach the manager without touching what the env records, and that is exactly the seam it uses.

**No gate table, and the reason is structural rather than convenient.** `src/aimanager/rl_manager.py` is imported by nothing in the simulation or evaluation path (`simulate.py`, `evaluation_suite/`, `api_manager.py` do not reference it; the only importer in the tree is `cli.py`'s dispatch). The 22 rows cannot move, and section 3 step 4 proves it rather than asserting it: the frontier stack re-run on this branch produces a `per_round.parquet` byte-identical to the parent's. What this branch changes is what a *future* manager training run learns from; there is no trained manager on the frontier to re-score, and training one would be a different experiment with a different declaration.

### Hypothesis

**Behavioural rationale (§5, one sentence):** a player who gave no input contributed nothing, and that is what the manager saw on its screen, so the reinforcement-learning manager -- and the replay buffer its TD update trains from, and the fixed opponent punisher beside it -- must be served the recorded 0 rather than an imputed 9; no evaluation row should move, and the row that would move is in a manager training run that does not exist yet.

**The change.** One file, `src/aimanager/rl_manager.py`, in the idiom the parent established:

1. The manager's observation is taken from `env.served_state()` after `env.reset()` and after `env.step()`, instead of from their return value. Because `run_batch` carries that one `state` variable to all three consumers, this corrects the manager's own input, the `statecopy` cloned into the replay buffer, and the opponent punisher's input together.
2. The state returned by `env.punish()` is bound to a local named `recorded` and keeps feeding the metrics, unchanged. Those metrics are the manager run's record of what the game charged -- the same role `per_round.parquet` plays for a simulation -- so they stay on the imputed value for exactly the parent's reason.

`environment.py` is not touched. `served_state()` is used as it was designed to be used.

### Artifact naming contract

| what | path |
|---|---|
| probe | `scripts/data_analysis/rl_manager_timeout_probe.py`; output `plots/data_analysis/evaluation/rl_manager_timeout/probe_{before,after}.json` |
| tests | `src/aimanager/tests/test_rl_manager_timeout_view.py` |
| sim config (re-run unchanged, for the byte-identity check) | `configs/simulation/manager_testing/23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_simtimeout.yml` (the parent's, used as-is) |

## 2. Plan

| # | step | status |
|---|---|---|
| 1 | Confirm the defect on the unchanged tree: what the manager, the replay buffer and the opponent are actually served at a timed-out cell, and how often the path fires. | **done** (probe, unchanged tree) |
| 2 | Fix the serving path; check every other consumer of the state the manager path touches. | **done** |
| 3 | Unit tests mirroring `test_sim_timeout_serving.py`, including the replay contents; run the suite on Raven. | **done** -- 151 pass |
| 4 | Prove nothing else moved: re-run the frontier stack and compare `per_round.parquet` byte for byte against the parent's. | **done** -- identical |
| 5 | Answer the `review/rl-manager` findings D5 and D1 that bear on this branch. | **done** -- section 3 step 5, note 2 |

## 3. Results

### Step 1: what the manager is actually served (measured, on the unchanged tree)

`scripts/data_analysis/rl_manager_timeout_probe.py` drives the real training rollout -- `rl_manager.run_batch`, the real `ArtificialHumanEnv`, the real `ArtificalManager`, the real replay `Memory` -- with `manager.get_action`, `replay_mem.add` and the opponent's `predict` wrapped, and reads `env.state` at the same instant for the side-by-side. Config `configs/training/rl_manager/03_2g8a_sum.yml`, seed 42, `batch_size` 32 in place of 1000 so it runs on a login node: 32 episodes x 24 rounds x 8 agents = 6,144 agent-rounds on CPU. Output: `plots/data_analysis/evaluation/rl_manager_timeout/probe_before.json`.

**How often the path fires.** 6,144 agent-rounds, **113 timeouts = 1.84%**. The human rate is 2.9% (560 / 19,200) and the frontier simulation's is 2.24% (parent, step 1); this stack's is measured here rather than assumed, and it is lower again. The realised count is **identical in both arms** (113 / 6,144), which is what the parent's finding predicts: the validity model in this config (`raven_script_22`) reads only `prev_contribution_valid`, so the timeout process itself is untouched by anything the manager is shown.

**What each seat at the table was handed.** Every one of the 113 cells, in both arms, with no exceptions and no mixed values:

| consumer | before | after | env's own record |
|---|---|---|---|
| `manager.get_action` -- the live observation | **9.0 x 113** | **0.0 x 113** | 9.0 x 113 |
| replay buffer -- what `manager.update` computes the TD error on | **9.0 x 113** | **0.0 x 113** | 9.0 x 113 |
| opponent punisher -- the fixed AH playing the other group | **9.0 x 113** | **0.0 x 113** | 9.0 x 113 |
| `env.state` -- what a recorded output would carry | 9.0 x 113 | **9.0 x 113 (unchanged)** | -- |

**The manager's encoder reads the value channel, and it already had the flag.** Measured from the config's `x_encoding` rather than assumed: the manager consumes `contribution` (numeric, 21 levels), `contribution_valid` (bool), `in_group` (bool), `prev_punishment` (numeric) and `round_number` (one-hot). So the defect is carried by `contribution`, encoded as 9/20 instead of 0, on a cell the network could already identify through `contribution_valid`. The flag was there before this branch and is there after it; what changes is only that the value beside it is now the one the game used.

**The manager's own behaviour diverges.** The digest of every punishment chosen over the rollout moves from `fb6d4cce...` (sum 101,774) to `edd6e795...` (sum 101,610). With a randomly initialised Q-network that number means little in itself; it is recorded because it settles the one question the served-value table cannot -- that the corrected input reaches the argmax and is not absorbed somewhere before it.

### Step 2: the change, and every consumer of the state the manager path touches

The parent checked its six consumers rather than assuming them; the same discipline, for the state this branch moves. **Corrected (three), deliberately unchanged (the rest):**

1. `manager.get_action` (`rl_manager.py:76`) -- **corrected.** Measured above.
2. `statecopy` -> `replay_mem.add` -> `manager.update` (`rl_manager.py:74`, `:131`) -- **corrected.** `replay_keys` is derived from the manager's own encoding config and contains `contribution`, so the TD error was being computed on the imputed 9. The online view and the replayed view now agree; they would have disagreed had only `get_action` been switched, which is worse than either alone (`review/rl-manager`, D5).
3. `opponent_manager.predict` (`rl_manager.py:88`) -- **corrected.** It is the one model in the loop whose corrected artifact (`rnn_edge_50ep_doubled_timeout`) carries a `contribution_valid` feature built for this population.
4. `metrics` from the state `env.punish()` returns (`rl_manager.py:98-140`) -- **deliberately unchanged**, and bound to a local named `recorded` so the separation is visible in the code. These are the run's record of what the game charged and paid out; they play the role `per_round.parquet` plays for a simulation and they keep the imputed value for the parent's reason. The env's own state is never mutated, so this is unchanged by construction; a test pins it.
5. The reward (`update_reward` -> `group_payoff_sum`) -- **unchanged.** It is computed inside `punish()` from `self.state` before `served_state()` is ever called, and `served_state()` returns a shallow copy with two keys replaced; it cannot reach the reward.
6. `compute_common_good_per_group` and `compute_payoff_per_group` -- **unchanged**, and they were already correct: both zero the invalid cell themselves (`th.where(contribution_valid, ...)`). A test pins the common good against a value computed from the env's own accounting.
7. `env.agent_groups` -- **unchanged.** The two-manager masks, the group sizes and `rl_mask` are read off the env object, not out of the state dict.
8. The replay `Memory` itself -- **never reaches disk.** Neither RL config passes `output_file` in `replay_memory_args`, and `Memory.write` is a no-op without it. The only disk outputs of a manager training run are `metrics/<job_id>.parquet` (consumer 4, unchanged) and the model checkpoint.
9. `simulate.py` -> `Memory` -> `mem_to_df` -> `per_round.parquet` -> the evaluation suite -- **unchanged.** `simulate.py` does not import `rl_manager`; step 4 proves the output byte-identical rather than resting on that.
10. `manager/artificial_human_group.py` -- **unchanged.** It calls `group.step()` and discards the return value (and is dead code besides; `review/rl-manager` D6).
11. `manager.expand_obs_for_groups` -- **unchanged.** It derives `in_group` from `agent_group`, a key `served_state()` does not touch.

**An unexpected result of the audit, and the strongest argument for the fix.** `api_manager.create_data` -- the path a *trained* manager checkpoint is deployed through, in simulation -- already serves `MISSING_CONTRIBUTION` (0) at own-group timeouts. So before this branch the RL manager was trained on 9 and deployed on 0 at the same cell: not merely a wrong value but a train/serve skew, on a cell the network can identify through `contribution_valid` and could therefore learn a rule about that does not survive deployment. The two sides now agree. (They still disagree about the *other* group -- `create_data` overwrites every out-of-group cell with the default and forces `contribution_valid` False, while training shows the true values flagged as out-of-group. That is `review/rl-manager` S3, it is not this defect, and it is untouched here.)

### Step 3: unit tests

`src/aimanager/tests/test_rl_manager_timeout_view.py`, seven tests, mirroring `test_sim_timeout_serving.py` in shape and fixtures (agent 0 times out, agent 1 genuinely contributes 0, so a real zero sits next to a fabricated one in every assertion):

- `test_manager_sees_zero_for_a_timeout` -- the timed-out agent reaches `get_action` as 0 with `contribution_valid` False beside it, the genuine zero is unchanged with its flag set, nobody else moves;
- `test_replay_buffer_stores_the_recorded_zero` -- read back out of the real `Memory`, so it pins the buffer and not just the call;
- `test_opponent_manager_sees_zero_for_a_timeout`;
- `test_round_zero_defaults_are_untouched` -- round 0 has no previous round to have timed out and its `prev_*` cells keep the dataset default, exactly as `create_torch_data`'s `shift()` puts them there in training;
- `test_recorded_metrics_and_env_state_keep_the_imputed_value` and `test_common_good_is_unchanged_by_the_fix` -- the guards on the dependency argument;
- `test_env_return_values_are_unchanged` -- the boundary guard: `reset`, `punish` and `step` still hand back `self.state` itself, so `simulate.py`, which records their return value, cannot have moved.

**The three defect tests fail on the parent and pass here; the four guard tests pass on both** -- measured by running the new file against the parent's `rl_manager.py` on Raven (3 failed, 4 passed) and against this branch's (7 passed). That is the signature a regression test should have, and it is worth more than the green run alone.

`scripts/remote_test.sh --test-only` over the whole of `src/` on Raven: **151 passed**, 0 failed. Locally only `test_env_return_values_are_unchanged` runs; the other six import `rl_manager`, which pulls in `torch_geometric`, the repo's convention. (The isolated remote dir needs `plots/simulation/22_2g8a_linear_self_ridge_contr/per_round.parquet` shipped by hand -- `simulate_cluster.sh` excludes `plots/` -- or an unrelated fixture-missing failure appears and is not a regression.)

`flake8 --max-line-length=88 --extend-ignore=E203,W503` is clean over `src/` and over the new probe. `black --check` leaves the same three hunks in `rl_manager.py` as on the parent and none of them is in the changed region (verified by diffing black's output against the parent's file); `rl_manager.py` and `artificial_humans/train.py` remain the two files unformatted on the parent and were not reformatted.

### Step 4: the byte-identical re-run (the proof that nothing else moved)

The frontier stack `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_simtimeout`, re-run from the parent's config **unmodified**, on this branch's code, in the isolated dir. Raven job **30400661**, COMPLETED, exit 0, 00:01:57.

| output | parent (`~/repros/ai-runs/sim-timeout-imputation`, 2026-09-19) | this branch (job 30400661) | |
|---|---|---|---|
| `per_round.parquet` -- what the evaluation suite scores | `adc8108c22782098e9c0e68e0112c480` | `adc8108c22782098e9c0e68e0112c480` | **same** |
| `aggregates.csv` | `9db8db81c1b420171da5646f9ea44c75` | `9db8db81c1b420171da5646f9ea44c75` | **same** |
| `switch_count_per_round.jpg` | `10086c01acff5a66ec2e40cb3bb777a4` | `10086c01acff5a66ec2e40cb3bb777a4` | same |
| `comparison_pairing_side.jpg` | `ab2f8923ea0a5e006f0f20427949e8e6` | `3aec4afe14dc8b36d5702a154cdc1d28` | differ |
| `group_size_evolution_global.jpg` | `dbc9afbe0f09d5c821dec4d8d2796288` | `21f528a82fbf301762992c56346d2986` | differ |

**`per_round.parquet` is byte-identical.** The simulation is bit-reproducible, which the campaign established, so this is the strongest available statement that no evaluation row can have moved: not "the scores are the same" but "the scored file is the same file". No evaluation was run, because there is nothing for it to distinguish.

**Two of the five outputs are two JPEGs, and they differ in their encoded bytes while the data does not.** Reported rather than glossed, because a silent difference would matter. It is rendering nondeterminism, not a change in the numbers, and the file that settles it is `aggregates.csv`: it is computed from the same per-round frame the figures plot and it *is* identical, as is the third figure. Nothing on this branch can reach a matplotlib call -- the diff is one function in `rl_manager.py`, which `simulate.py` does not import.

### Step 5: the `review/rl-manager` findings that bear on this branch

**D5 -- "fixing `get_action` alone will not close it", naming `rl_manager.py:66` (`statecopy`) and `:80` (the opponent).** Agreed, and both are closed here, because the fix was made at the assignment of `state` rather than at `get_action`: `run_batch` carries one `state` variable to all three consumers, so correcting it once corrects all three. This was measured, not reasoned -- the probe reports the replay buffer and the opponent separately from the manager, and all three go 9.0 -> 0.0 on all 113 cells (step 1), and two of the seven tests assert on them directly.

**D1 -- punishment aimed at a timed-out player is free to the manager and still shown to every artificial human.** Real, reproduced independently, and **deliberately not fixed here**; note 2 has the reasoning and section 6 scopes it. Reproduced with a local probe (plain torch, mock models), one round with agent 0 timed out:

| manager's action on the timed-out agent | group common good | `group_payoff_sum` | `prev_punishment` raw | `prev_punishment` served | `prev_contribution` served |
|---|---|---|---|---|---|
| 0 | 16.000 | 78.000 | 0 | 0 | 0 |
| 30 | 16.000 | 78.000 | 30 | 30 | 0 |

Identical reward, different input to every downstream model -- the review's table, reproduced to the digit. The mechanism is visible in the code: `compute_common_good_per_group` zeroes punishment at `~contribution_valid` and `compute_payoff_per_group` zeroes the whole invalid contributor's payoff, while `punish()` stores the action verbatim and `step()` copies it into `prev_punishment`, which `served_state()` does not correct. The last column is this branch's fix working beside it: the contribution channel reads the recorded 0 and the punishment channel does not.

## 4. Notes

1. **The fix was made at the assignment, not at the call sites, and that is why it is three lines rather than three patches.** `run_batch` threads one `state` through the manager, the replay clone and the opponent, so the place to correct it is where it is bound. This is also what makes the change auditable: `grep -n "state" rl_manager.py` shows exactly two producers of that variable and they are both `served_state()`. A call-site fix would have left the reader checking every consumer by hand, and D5 is a record of what happens when one is missed.

2. **D1 is the same defect seen from the other side and it does not belong in this change.** Three reasons, in order of weight. (a) **It changes the simulation.** The review's own shape of the fix -- zero the punishment at `~contribution_valid` inside `punish()`, "so the recorded value matches the game" -- writes into `self.state["punishment"]`, which `Memory.add` records into `per_round.parquet`, and into `prev_punishment`, which is the contribution AH's *only* channel from the manager. Contributions would diverge from round 1 onward and every one of the 22 rows would move. This branch's whole claim is that the scored file does not change, proven byte for byte in step 4; bundling D1 would destroy that proof and leave the two fixes inseparable in the scores -- the same reason the parent refused to combine its serving fix with a copula recalibration. Under §4 a shared-code bug fix is its own experiment with its own before/after on the top-ranked stack, and D1 needs one. (b) **This branch neither creates nor widens the lever.** The coordinator's reading was that shipping the state correction without D1 "leaves an exploitable asymmetry that did not exist before". Measured: `contribution_valid` is in the manager's `x_encoding` in both RL configs on the parent and is unchanged here (step 1), so the network could already identify the free cells before this branch. What changes is the *value* on the channel beside the flag, not the flag. The asymmetry is exactly as large after this change as before it. (c) **It is a different mechanism on a different surface** -- the punishment channel, in `punish()`, in the env, touching the recorded output -- and it is shared with the simulation path, where it is the artificial punisher rather than the RL manager that spends the free punishment. Scoping it as its own branch also lets it be judged on the rows it should move, which is what §4 exists for.

3. **The timeout rate is a property of the stack, not a constant, and it keeps coming out lower in simulation than in the humans.** Human 2.9%, frontier simulation 2.24%, this RL stack 1.84%. All three are measured. Nobody has declared on it; it is recorded here because three independent measurements now point the same way and a successor working on the validity model would want the trail.

4. **Two of `03_2g8a_sum.yml`'s four artifacts are not in the tree** (`review/rl-manager` D4): the contributor's `epochs_1000` checkpoint and the autoregressive opponent. The probe resolves the first to the one checkpoint in the same directory (`epochs_575`) and reports the substitution in its output; the second is passed on the command line, pointing at `punishment_rnn_edge_50ep_doubled` exactly as the config's own comment instructs. Neither choice can affect what this branch measures -- the probe reads what a model is *served*, not what it predicts -- but the config cannot be run as written and a successor should not discover that the hard way.

5. **Nothing was trained, and nothing was evaluated.** Every model in the probe and in the re-run loads the same file on disk the parent's runs loaded. There is no seed floor to quote against because there is no movement to interpret: the scored file is identical.

6. **Housekeeping.** Remote dir `~/repros/ai-runs/rl-manager-timeout` can be deleted when this PR closes. No copula was recalibrated; the evaluation suite was not touched.

## 5. The verdict: a confirmed defect, closed, with nothing else moved

**The defect was real, it was measured on the unchanged tree, and it is fixed.** In an RL rollout the validity model fires on **1.84%** of agent-rounds, and on every one of them the manager's live observation, the replay buffer its TD update trains on, and the fixed opponent punisher beside it all read **9.0** -- a value the game never used and one that never occurs in the training data, where the same cell is 0. After the fix all three read 0.0 on all 113, the env's recorded value is untouched, and round 0's defaults are untouched. Seven unit tests pin it, three of which fail on the parent; 151 tests pass on Raven.

**Nothing else moved, and that is proven rather than argued.** The frontier stack re-run on this branch (job 30400661) produces a `per_round.parquet` with the same md5 as the parent's, `adc8108c22782098e9c0e68e0112c480`, and an identical `aggregates.csv`. No evaluation row can have moved, so no gate applies and none is claimed. Two of the three figures differ in their JPEG bytes and neither the parquet nor the aggregates do; that is reported in step 4 rather than glossed.

**The fix also closes a train/serve skew nobody had noticed:** `api_manager.create_data`, the deployment path, already served the recorded 0 at own-group timeouts, so a manager trained before this branch would have been deployed on inputs it was never trained on, on precisely the cells `contribution_valid` lets it identify.

### Verdict tag

**[FIX]** -- third and last of three instances of one defect, closed. No band upgrade is claimed and no gate is engaged, because the change cannot move an evaluation row and the byte-identical re-run demonstrates it. The one adjacent defect found by `review/rl-manager` (D1) is reproduced, reported and deliberately scoped to a successor branch.

## 6. For a successor

1. **This closes the third of three instances of one defect, and the family is now complete.** The punisher's two serving paths (`auto/punisher-timeout-feature`), the contribution and switch models' (`auto/sim-timeout-imputation`), and the RL manager's three consumers (this branch). Every model in the system is now served the recorded 0 for a player who gave no input, and `ArtificialHumanEnv.served_state()` is the single place that decides it. **If a fourth consumer of `self.state` is ever added, it goes through `served_state()`** -- the helper is opt-in, which is how the RL path was missed, and the one structural improvement left on this axis would be to make it opt-out instead.

2. **D1 is the next experiment on this axis and it is ready to declare.** A punishment aimed at a timed-out player costs the manager nothing (the accounting zeroes it) and is still shown to every artificial human through `prev_punishment` -- free deterrence the real game did not offer, on the 1.8-2.9% of agent-rounds that time out, on cells the manager can identify. The human evidence is already gathered in `notes/reviews/rl-manager-review.md` (on branch `review/rl-manager`): all 560 `player_no_input == 1` rows carry `punishment == 0.0` exactly and the group identity holds to 2.8e-14 with that zero in place, so no artificial human was ever trained on a row where a timed-out player had been punished. **Shape:** zero the punishment at `~contribution_valid` in `punish()`, where the action is realised, so the recorded value matches the game too. **Scope:** it is a simulation-behaviour change -- `per_round.parquet` moves and every one of the 22 rows moves with it -- so it is its own experiment under §4, with the frontier stack re-run and a full before/after. Declare CG and the R family as targets: the contribution AH's only channel from the manager is `prev_punishment`, and this removes a wrong value from 2% of it. It should **not** be folded into a serving-path branch, for the reasons in note 2.

3. **The first manager trained against the corrected players should be trained on the corrected observations too, and that is now true.** Anyone starting a run should read `notes/reviews/rl-manager-review.md` (on branch `review/rl-manager`) first: D2 (the reward discards a timed-out player's payoff, a ~31% shortfall on group-rounds containing a timeout, and it is exogenous to the policy) and D3 (the opponent is the pre-fix lagged punisher) both change what a manager learns and neither is fixed. D1, D2 and D3 are three separate declarable experiments and all three are upstream of any number a training run produces.

4. **`03_2g8a_sum.yml` cannot run as written** (note 4, review D4) and its switch model carries the pre-#123 anchoring. `03_2g8a_sum_d_lr1e3_freq500.yml` is the config to start from.
