# [DRAFT] Validate Parallel Training of Independent Groups via Configuration

GitHub Issue: #14

## Summary

| # | Section | Change | Priority |
|---|---------|--------|----------|
| 1 | Environment init | Support group assignment from config | High |
| 2 | Training loop | Pass group assignment to `env.update_groups()` | High |
| 3 | New YAML config | Config with two groups of 2 agents each | High |
| 4 | Validation | Compare learning curves against single-group baseline | High |

## Goal

Configure an experiment where two managers each manage a distinct, fixed group of agents, trained in parallel. Because groups are independent and rewards are already separated per group (PR #7), learning dynamics should be identical to the single-group baseline.

## Current State

- **Infrastructure exists**: PR #7 added multi-group Q-values, `in_group` feature, `expand_obs_for_groups`, per-group reward computation, and `env.update_groups()`.
- **Group assignment is hardcoded to all-zeros**: `environment.py` line 80 initializes `agent_groups = th.zeros(...)`, placing all agents in group 0 regardless of `n_groups`.
- **No config key for group assignment**: Neither the YAML configs nor `rl_manager.py` read or apply a group assignment. The `update_groups` method is never called with a nontrivial split.
- **Existing config `01_rnn_node.yml`** already sets `n_groups: 2` but this has no real effect since all agents remain in group 0.

## Proposed Changes

### 1. Environment: Accept group assignment from constructor
- **Where**: `src/aimanager/manager/environment.py`, `__init__` method
- **What**: Add an optional `agent_groups` parameter to `env_args`. When provided, use it instead of the all-zeros default. When absent, keep current behavior.
- **Why**: Allows YAML-driven group configuration without changing the environment's internal logic.

### 2. Training loop: Wire group assignment through
- **Where**: `src/aimanager/rl_manager.py`, `train_manager` function (around line 132)
- **What**: After constructing the environment, read the group assignment from `env_args` and call `env.update_groups()` with the proper tensor. This may also be handled inside `__init__` per change 1.
- **Why**: The training loop currently never sets a nontrivial group assignment.

### 3. New YAML configuration
- **Where**: `configs/training/rl_manager/` (new file, e.g. `02_parallel_groups.yml`)
- **What**: Duplicate `01_rnn_node.yml` with the addition of a group assignment field (e.g. `agent_groups: [0, 0, 1, 1]`) that splits 4 agents into two groups of 2. All other hyperparameters remain identical.
- **Why**: This is the primary deliverable of issue #14 -- a config-only change that enables parallel group training.

### 4. Validation
- **Where**: Notebook or script (location TBD)
- **What**: Run both the baseline config (`01_rnn_node.yml` with all agents in group 0) and the new config. Compare learning curves (reward, contribution, punishment, loss) to confirm they are statistically equivalent.
- **Why**: The issue explicitly requires verifying that parallel groups do not alter learning dynamics.

## Open Questions

1. **Group assignment format in YAML**: Should this be a flat list (e.g. `[0, 0, 1, 1]`) that gets broadcast across the batch, or should the environment constructor handle it? The flat list approach is simplest and consistent with fixed groups.
2. **Where does validation live?** Should it be a notebook in `notebooks/test_manager/`, a script in `scripts/manager/`, or both?
3. **Batch-level variation**: Currently all episodes in a batch share the same group assignment. Is this the intended behavior, or should different batches have different group compositions?
4. **Number of agents**: The existing config uses `n_agents: 4` with `n_groups: 2`. Should each group have exactly 2 agents, or should validation also cover uneven splits?

## Next Actions

- [ ] Resolve open questions (especially 1 and 2) before implementation
- [ ] Implement changes 1-3 (minimal code change + new config)
- [ ] Run validation (change 4) and document results
- [ ] Create PR referencing issue #14
