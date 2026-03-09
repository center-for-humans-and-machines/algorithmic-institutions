# [DRAFT] Relax Stable-Group Assumption in Reward Computation

GitHub Issue: #15
Depends on: #14
Related: #39 (standalone bug -- `prev_contribution_valid` mask mismatch, extracted from this plan)

## Summary

| # | Section | Change | Priority |
|---|---------|--------|----------|
| 1 | `update_reward` -- group_payoff formula | Ensure per-round validity masks and group masks are correct when group composition varies between rounds | High |
| 2 | `update_reward` -- terminal reward | Audit mask usage for the done-branch under changing groups | Medium |
| 3 | `compute_average_per_group` | No change expected; already group-aware via `agent_group_mask` | -- |
| 4 | Tests | Add test proving reward invariance when group membership is fixed | High |
| 5 | Validation | Run existing training config and confirm identical results | Medium |

## Goal

Remove the implicit assumption that group composition is stable across rounds from the reward computation in `ArtificialHumanEnv`. Behavior must remain identical when groups are in fact fixed (which is the only case today). This is a pure refactor for conceptual correctness and future extensibility.

## Relationship to Issue #39

The `prev_contribution_valid` mask mismatch on line 269 of `environment.py` (using `self.contribution_valid` instead of `self.prev_contribution_valid` for previous-round punishment) was originally identified during analysis for this plan. It has been extracted into issue #39 as a standalone bug fix because the mismatch is incorrect regardless of whether groups change. This plan assumes #39 is resolved first and focuses only on the remaining stable-group assumptions.

## Current State

- **`update_reward`** (`environment.py`, lines 252-285) has two code comments explicitly flagging the stable-group assumption:
  - Line 264: "this assumes all groups in the batch to be identically compositioned"
  - Line 267: "this additional assumes that groups do not change throughout the game"
- **Core concern**: Even after #39 is fixed, the group_payoff branch still assumes that the set of agents belonging to each group is the same across rounds. The `agent_group_mask` used by `compute_average_per_group` is set once in `update_groups` and never updated per-round. If groups changed between rounds, averages would be computed over the wrong set of agents.
- **`contribution_per_group`** (line 264) uses `self.contribution_valid` for current-round contribution, which is correct for the current round. The comment about "identical composition" flags that the formula assumes each group has the same members across the batch dimension.
- **Terminal reward branch** (lines 256-260): Uses `compute_average_per_group(masked_prev_punishment)` with no explicit validity mask (defaults to all-ones). This may also embed assumptions worth auditing.
- **`compute_average_per_group`**, `compute_common_good_per_group`, and `compute_average_payoff_per_group` are already group-aware through `self.agent_group_mask`, but `agent_group_mask` is set once in `update_groups` and never updated per-round. If groups changed between rounds, the mask would also need updating -- but that is out of scope for this issue.

## Proposed Changes

### 1. Remove stable-group assumption in `update_reward` group_payoff branch
- **Where**: `src/aimanager/manager/environment.py`, `update_reward` method, lines 262-280
- **What**: After #39 fixes the validity mask bug, determine what additional changes are needed so that the group_payoff formula produces correct results when group membership varies between rounds. This likely involves ensuring that `agent_group_mask` reflects the correct grouping for the round whose data is being aggregated.
- **Why**: Removes the assumption documented in the line 267 comment.

### 2. Audit terminal reward branch
- **Where**: `src/aimanager/manager/environment.py`, `update_reward` method, lines 256-260
- **What**: Determine whether the terminal reward also implicitly assumes stable groups. If the default validity mask (all-ones) is intentional regardless of group changes, document that with a comment. If not, pass the appropriate mask.
- **Why**: Completeness -- the issue asks to remove the stable-group assumption from reward computation generally.

### 3. Update or remove stale comments
- **Where**: Same method, comments at lines 264 and 267
- **What**: After changes 1 and 2, update or remove the comments that flag the stable-group assumption. If any residual assumptions remain, document what would need to change.
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

1. **Dependency on #14**: Issue #15 explicitly depends on #14 being concluded. Should this plan be implemented before or after #14 is merged?
2. **Dependency on #39**: The validity mask bug fix (#39) should land first. Confirm sequencing.
3. **`agent_group_mask` updates**: If groups were to change between rounds (future work), `agent_group_mask` itself would need per-round updates. Should this plan add any infrastructure for that (e.g., storing per-round masks), or is fixing the conceptual assumptions in `update_reward` sufficient for the stated goal?
4. **Terminal reward semantics**: In the terminal branch, `masked_prev_punishment` is computed using `self.prev_contribution_valid`, but `compute_average_per_group` is called without a validity mask. Is the intent that all agents count equally for the terminal penalty, or should `prev_contribution_valid` be passed as the validity mask?

## Next Actions

- [ ] Resolve dependency: confirm #14 is merged or determine that this can proceed independently
- [ ] Resolve dependency: confirm #39 is merged or determine that this can proceed independently
- [ ] Resolve open questions 3 and 4
- [ ] Implement changes 1-3 (small diff in `update_reward`)
- [ ] Add regression test (change 4)
- [ ] Run validation (change 5) and document results
- [ ] Create PR referencing issue #15
