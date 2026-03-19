# [ACTIVE] Remove group_payoff formula

Issue: #42

## Goal

Remove the `group_payoff` reward formula and make `group_payoff_round` the sole
reward computation in `update_reward()`. After issue #41 separated the two
formulas into clean code paths, `group_payoff_round` is the correct and preferred
formula: it uses same-round data following standard RL conventions, while
`group_payoff` embeds incorrect temporal semantics and was only kept for
historical reasons. Removing it simplifies the environment, eliminates a source
of confusion, and makes the `reward_formula` config parameter unnecessary.

## Plan

| # | Section | Change | Optional |
|---|---------|--------|----------|
| 1 | Simplify `update_reward()` | Remove branching; inline `group_payoff_round` logic | No |
| 2 | Remove `reward_formula` parameter | Drop from `__init__` signature and instance | No |
| 3 | Remove dead helper usage | Stop calling `compute_average_per_group` inside `update_reward`; keep the method | No |
| 4 | Update active configs | Remove `reward_formula` from all configs in `configs/` and default config | No |
| 5 | Delete obsolete config | Delete `01_rnn_node_group_payoff.yml` | No |
| 6 | Leave `run/` configs untouched | Historical experiment definitions stay as-is | No |
| 7 | Guard against stale configs | Add `pop("reward_formula")` guard in `rl_manager.py` | No |
| 8 | Update tests | Remove `group_payoff`-specific tests; simplify `_make_env` | No |

### 1. Simplify `update_reward()`

- **Where**: `src/aimanager/manager/environment.py`, `update_reward()` (lines 257-307)
- Replace the entire if/elif/else block with a single assignment:
  `self.reward = self.group_payoff`
- This works for both terminal and non-terminal rounds because `self.group_payoff`
  is computed during `punish()` (via `update_payoff()`) from same-round data,
  and is not overwritten by `step()` (no `prev_group_payoff` key exists).
- Remove the inline comments about stable-group assumptions and manual payoff
  recomputation, as they only applied to the `group_payoff` formula.

### 2. Remove `reward_formula` parameter

- **Where**: `src/aimanager/manager/environment.py`, `__init__` (line 42, 72)
- Remove `reward_formula="group_payoff"` from the `__init__` parameter list.
- Remove `self.reward_formula = reward_formula` from the constructor body.

### 3. Remove dead helper usage (keep `compute_average_per_group`)

- **Where**: `src/aimanager/manager/environment.py`, `update_reward()` (lines 266,
  277, 283)
- The three calls to `self.compute_average_per_group()` inside the `group_payoff`
  branch will be deleted as part of section 1.
- **Keep** the `compute_average_per_group` method itself. It is a general-purpose
  utility used by `test_multi_group_env` in the test suite (line 146) and could
  be used by future code. It has no dependency on `reward_formula`.

### 4. Update active configs

Remove the `reward_formula` line from these files:

- `src/aimanager/rl_manager.yml` (line 45): delete `reward_formula: group_payoff`
- `configs/training/rl_manager/01_rnn_node.yml` (line 49): delete
  `reward_formula: group_payoff`
- `configs/training/rl_manager/01_rnn_node_group_payoff_round.yml` (line 47):
  delete `reward_formula: group_payoff_round`
- `configs/training/rl_manager/test_01_rnn_node.yml` (line 49): delete
  `reward_formula: group_payoff`

### 5. Delete obsolete config

- **Where**: `configs/training/rl_manager/01_rnn_node_group_payoff.yml`
- This config is identical to `01_rnn_node_group_payoff_round.yml` except for the
  formula and output directory. Since `group_payoff` no longer exists, the config
  has no purpose. Delete the file entirely.
- Note: both files are currently untracked (shown as `??` in git status), so
  deleting `01_rnn_node_group_payoff.yml` simply means not committing it.

### 6. Leave `run/` configs untouched

- **Where**: `run/manager/*.yml` (7 files referencing `reward_formula`)
- These are historical experiment definitions for past DJX runs. They reference
  formulas that no longer exist in the code (`payoff`, `impact_on_group_payoff`,
  `true_common_good`, `group_payoff`). They serve as documentation of past
  experiments and should not be modified.
- They will not cause runtime errors because they are not used as input to the
  current training pipeline unless explicitly selected.

### 7. Guard against stale configs

- **Where**: `src/aimanager/rl_manager.py`, around line 132-137
- If an old config still contains `reward_formula` inside `env_args`, passing
  `**config["env_args"]` to `ArtificialHumanEnv()` will raise a `TypeError`
  (unexpected keyword argument).
- Add a `pop("reward_formula", None)` call on the `env_args` dict before passing
  it to `ArtificialHumanEnv`, with a deprecation warning via `warnings.warn()`.
  This prevents old configs from breaking while making it clear the parameter is
  no longer used.

### 8. Update tests

- **Where**: `src/aimanager/tests/test_environment.py`
- **Delete** `test_group_payoff_uses_same_round_data` (line 375) -- tests
  the removed `group_payoff` formula.
- **Delete** `test_group_payoff_terminal` (line 406) -- tests the removed
  `group_payoff` terminal branch.
- **Update** `_make_env` (line 305): remove the `reward_formula` parameter and
  stop passing it to `ArtificialHumanEnv`.
- **Update** `test_group_payoff_round_non_terminal` (line 341) and
  `test_group_payoff_round_terminal` (line 355): call `_make_env(n_rounds=...)`
  without the formula argument.
- **Update** `test_multi_group_env` (line 22): remove
  `reward_formula="group_payoff"` from the `ArtificialHumanEnv` constructor call.
  Update the terminal reward assertion (line 145-149) to assert
  `th.allclose(reward, env.group_payoff)` instead of the old
  `-avg_prev_punishment/32` formula. This requires capturing `group_payoff`
  before the final step.
- **Update** `test_artificial_human_env` (line 154): update the reward assertions
  (lines 270-275, 291-292) to match the new formula: reward equals
  `self.group_payoff` (already computed during `punish()`), not the manually
  recomputed payoff/32.

## Implementation notes

- The `test_multi_group_env` terminal reward assertion needs care. The test loop
  calls `env.punish(test_punish)` after `env.step()` returns `done=True`
  (line 133). At that point `group_payoff` is re-set by that final `punish()`.
  But `reward` was already captured from the `step()` that returned `done=True`.
  So the assertion should compare `reward` against the `group_payoff` from the
  *last non-terminal* `punish()`, not the post-done `punish()`. The simplest
  approach: capture `env.group_payoff.clone()` right before the final `step()`
  inside the while loop and assert against that.

- The `test_artificial_human_env` reward assertion at line 270-275 currently
  manually recomputes the `group_payoff` formula and divides by 32. The new
  assertion should simply check `reward == env.group_payoff` (captured before the
  step). The `/32` normalization was part of the old `group_payoff` formula; the
  `group_payoff_round` path assigns `self.reward = self.group_payoff` without any
  scaling.

## Next Actions

- [ ] Review and approve this plan
- [ ] Implement sections 1-2: simplify `update_reward()` and remove
      `reward_formula` parameter in `environment.py`
- [ ] Implement section 4: remove `reward_formula` from active config files
- [ ] Implement section 5: delete `01_rnn_node_group_payoff.yml`
- [ ] Implement section 7: add `pop("reward_formula")` guard in `rl_manager.py`
- [ ] Implement section 8: update test file
- [ ] Run tests via `scripts/remote_test.sh`
- [ ] Close issues #15 and #39 as resolved
