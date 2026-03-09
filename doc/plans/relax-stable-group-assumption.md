# [DRAFT] Relax Stable-Group Assumption in Reward Computation

GitHub Issue: #15
Depends on: #14

## Summary

| # | Section | Change | Priority |
|---|---------|--------|----------|
| 1 | `update_reward` -- group_payoff formula | Use per-round validity masks instead of current-round mask for previous-round data | High |
| 2 | `update_reward` -- terminal reward | Audit mask usage for the done-branch | Medium |
| 3 | `compute_average_per_group` | No change expected; already group-aware via `agent_group_mask` | -- |
| 4 | Tests | Add test proving reward invariance when group membership is fixed | High |
| 5 | Validation | Run existing training config and confirm identical results | Medium |

## Goal

Remove the implicit assumption that group composition is stable across rounds from the reward computation in `ArtificialHumanEnv`. Behavior must remain identical when groups are in fact fixed (which is the only case today). This is a pure refactor for conceptual correctness and future extensibility.

## Current State

- **`update_reward`** (`environment.py`, lines 252-285) has two code comments explicitly flagging the stable-group assumption:
  - Line 264: "this assumes all groups in the batch to be identically compositioned"
  - Line 267: "this additional assumes that groups do not change throughout the game"
- **Root cause**: The `"group_payoff"` branch (lines 262-280) computes `prev_punishment_per_group` by calling `compute_average_per_group(self.prev_punishment, self.contribution_valid)`. It uses **current-round** `contribution_valid` as the validity mask for **previous-round** punishment. This is only correct if group membership (and therefore which agents are valid in which group) has not changed between rounds.
- **Similarly**, `contribution_per_group` on line 264 uses `self.contribution_valid` which is correct for current-round contribution, but the comment suggests a broader concern about identical composition across the batch.
- **Terminal reward branch** (lines 256-260): Uses `compute_average_per_group(masked_prev_punishment)` with no explicit validity mask (defaults to all-ones). This may also embed assumptions worth auditing.
- **`compute_average_per_group`**, `compute_common_good_per_group`, and `compute_average_payoff_per_group` are already group-aware through `self.agent_group_mask`, but `agent_group_mask` is set once in `update_groups` and never updated per-round. If groups changed between rounds, the mask would also need updating -- but that is out of scope for this issue.

## Proposed Changes

### 1. Fix validity mask in `update_reward` group_payoff branch
- **Where**: `src/aimanager/manager/environment.py`, `update_reward` method, lines 262-280
- **What**: Use `self.prev_contribution_valid` (instead of `self.contribution_valid`) when computing `prev_punishment_per_group`. This correctly pairs previous-round data with its own validity mask.
- **Why**: Removes the assumption that valid agents are the same across consecutive rounds.

### 2. Audit terminal reward branch
- **Where**: `src/aimanager/manager/environment.py`, `update_reward` method, lines 256-260
- **What**: Determine whether the terminal reward also implicitly assumes stable groups. If the default validity mask (all-ones) is intentional regardless of group changes, document that with a comment. If not, pass the appropriate mask.
- **Why**: Completeness -- the issue asks to remove the stable-group assumption from reward computation generally.

### 3. Audit the "identical composition" comment (line 264)
- **Where**: Same method, line 264
- **What**: Clarify or remove the comment "this assumes all groups in the batch to be identically compositioned". If the code is correct after fix 1, update or remove the comment. If there is a real remaining assumption, document what would need to change.
- **Why**: Stale comments are worse than no comments.

### 4. Add regression test
- **Where**: `src/aimanager/manager/test/test_environment.py`
- **What**: Add a test that runs the environment with fixed groups and asserts that rewards from the refactored code match the old formula exactly (within floating-point tolerance). The existing `test_multi_group_env` partially covers this but does not specifically assert on the `group_payoff` reward formula mid-game.
- **Why**: The issue explicitly requires "evidence of invariance under the refactor".

### 5. Validate with training run
- **Where**: Notebook or script (same location as issue #14 validation, TBD)
- **What**: Run training with the existing config and compare learning curves before and after the refactor.
- **Why**: End-to-end confirmation that the refactor is behavior-preserving.

## Open Questions

1. **Dependency on #14**: Issue #15 explicitly depends on #14 being concluded. Should this plan be implemented before or after #14 is merged? (The plan for #14 is currently DRAFT.)
2. **`agent_group_mask` updates**: If groups were to change between rounds (future work), `agent_group_mask` itself would need per-round updates. Should this plan add any infrastructure for that (e.g., storing per-round masks), or is fixing the validity mask sufficient for the stated goal?
3. **Terminal reward semantics**: In the terminal branch, `masked_prev_punishment` is computed using `self.prev_contribution_valid`, but `compute_average_per_group` is called without a validity mask. Is the intent that all agents count equally for the terminal penalty, or should `prev_contribution_valid` be passed as the validity mask?

## Next Actions

- [ ] Resolve dependency: confirm #14 is merged or determine that this can proceed independently
- [ ] Resolve open questions 2 and 3
- [ ] Implement changes 1-3 (small diff in `update_reward`)
- [ ] Add regression test (change 4)
- [ ] Run validation (change 5) and document results
- [ ] Create PR referencing issue #15
