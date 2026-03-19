# [DONE] Align reward timing with manager's action round

Issue: #41 (supersedes #40)

## Summary

| # | Section | Change | Priority |
|---|---------|--------|----------|
| 1 | Terminal reward branch | Split terminal logic so `group_payoff_round` uses its own formula | High |
| 2 | `group_payoff` lookahead | Compute `group_payoff` reward from same-round state, not next-round contributions | High |
| 3 | `update_reward` structure | Refactor into a clean top-level fork between reward formulas | Medium |
| 4 | Replay memory timing | Verify replay stores reward for the correct (state, action) pair | Low |
| 5 | Manual regression test | Run training with both reward formulas and compare learning curves | High |

## Context: Current Flow

The round loop in `run_batch()` is: `reset()` then repeat `[get_action -> punish() -> step()]`.

- `punish()` sets punishment, then calls `update_common_good()` and `update_payoff()`, computing `group_payoff` for the CURRENT round.
- `step()` increments round, copies current state to `prev_*`, generates NEXT round's contributions via `update_contribution()`, then calls `update_reward()`.
- `update_reward()` has a shared terminal branch (when `done=True`) and a formula-specific else branch.

## 1. Terminal reward branch

- **What**: When `done=True`, both `group_payoff` and `group_payoff_round` currently share the same terminal reward: `-average_masked_punishment_per_group / 32`. This is a punishment-cost-only value.
- **Where**: `src/aimanager/manager/environment.py`, `update_reward()`, lines 256-260.
- **Why**: For `group_payoff_round`, the terminal reward should be `self.group_payoff` (already computed during the final `punish()` call), matching the formula used in all non-terminal rounds. The current shared branch silently changes the reward semantics on the last step.

## 2. `group_payoff` lookahead removal

- **What**: The `group_payoff` formula currently computes reward using `self.contribution` (which, after `step()` runs `update_contribution()`, is the NEXT round's contribution) combined with `self.prev_punishment` (the manager's action from the current round). This means the reward for action at round t includes contributions from round t+1.
- **Where**: `src/aimanager/manager/environment.py`, `update_reward()`, lines 262-280; and the ordering in `step()`, lines 328-342.
- **Why**: As described in issue #40, this breaks the standard RL convention that reward for action a_t reflects the outcome of that action within the same timestep. The fix requires computing the `group_payoff` reward from the same-round contribution and punishment (both available before `step()` advances).
- **Note**: `group_payoff_round` already avoids this problem because it uses `self.group_payoff`, which was computed during `punish()` from same-round data. The fix for `group_payoff` should bring it in line with this behavior.

## 3. `update_reward` structure

- **What**: Restructure `update_reward()` so that each reward formula is a self-contained block handling both normal and terminal cases, rather than sharing a terminal branch.
- **Where**: `src/aimanager/manager/environment.py`, `update_reward()`.
- **Why**: The current shared terminal branch makes it easy to introduce bugs when adding new formulas and obscures the reward semantics. A top-level fork by formula, each handling its own terminal case, is clearer and less error-prone.

## 4. Replay memory timing

- **What**: Verify that the state captured in `statecopy` (taken before `punish()`) and the reward (taken from `step()`) are correctly paired in replay memory for both formulas after the timing changes.
- **Where**: `src/aimanager/rl_manager.py`, `run_batch()`, lines 50-67.
- **Why**: Changing when reward is computed could create a mismatch between the state the manager saw when choosing an action and the reward attributed to that action. After the changes in sections 1-3, replay storage should be audited to confirm correctness. No change is expected here, but it should be verified.

## 5. Manual regression test

- **What**: After implementing sections 1-3, run RL training with both the original `group_payoff` config (`run/manager/17_exp2_group_payoff_heavy_optimize.yml`) and the new `group_payoff_round` config (`run/manager/22_exp2_group_payoff_round_heavy_optimize.yml`). Compare the resulting learning curves (episode reward over training steps) to validate behavioral equivalence.
- **Where**: Training runs on cluster using both configs; comparison via evaluation notebook or manual plot inspection.
- **Why**: The plan argues that `group_payoff_round` already has correct same-round timing and that the fix to `group_payoff` is a shift in attribution, not magnitude. Running both formulas side-by-side after the fix confirms that (a) `group_payoff_round` trains successfully with the terminal branch fix, and (b) the corrected `group_payoff` produces comparable learning dynamics to the pre-fix version. Divergent curves would indicate an implementation error or an incorrect assumption about equivalence.
- **Config**: The `group_payoff_round` config is identical to `17_exp2_group_payoff_heavy_optimize.yml` except for `reward_formula: group_payoff_round`. It is checked in at `run/manager/22_exp2_group_payoff_round_heavy_optimize.yml`.

## Open Questions

None -- the investigation resolved all five concerns. The `group_payoff_round` formula already has correct same-round timing but needs its terminal branch fixed. The `group_payoff` formula needs its reward computation moved to use same-round data. Both benefit from the structural refactor.

## Impact Assessment

- `group_payoff_round` is not used in any existing config or run definition. The terminal branch fix is correctness-oriented but has no impact on past experiments.
- `group_payoff` is the primary formula used in all existing experiments. Changing its timing will alter Q-value targets during training. As noted in issue #40, the total undiscounted return over a full episode is unchanged (it is a shift in attribution, not magnitude), so the practical impact on training outcomes is expected to be minimal.
- Other formulas referenced in run configs (`payoff`, `impact_on_group_payoff`, `true_common_good`) are not present in the current `update_reward()` code and appear to be from older versions. They are out of scope.

## Next Actions

- [x] Review and approve this plan
- [x] Implement terminal branch split for `group_payoff_round`
- [x] Implement same-round reward computation for `group_payoff`
- [x] Refactor `update_reward()` into top-level formula fork
- [x] Audit replay memory pairing after changes (verified, no changes needed)
- [x] Add or update tests in `src/aimanager/tests/test_environment.py` covering both formulas and terminal/non-terminal rounds
- [x] Run manual regression test: trained both formulas (100k steps each), simulated against pr7 baseline. All three managers produce comparable behavioral dynamics (punishment, contribution, common good, payoff). Results at `plots/simulation/02_reward_timing/`
