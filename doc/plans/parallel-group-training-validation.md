# [DRAFT] Parallel Group Training Validation

GitHub Issue: #14
Dependency: #42 (simplified reward computation) must land first

## Summary

| # | Section | Change | Priority |
|---|---------|--------|----------|
| 1 | Environment init | Accept `agent_groups` list from config, broadcast to batch tensor | High |
| 2 | Training loop | Wire config-provided group assignment into env construction | High |
| 3 | DJX run config | New run config: 2 groups x 4 agents, based on reference config | High |
| 4 | Regression test | Compare 2-group run against single-group baseline | High |

## Goal

Train two independent groups of 4 agents each (8 agents total, `n_groups=2`) in parallel within a single environment instance. Because groups share no information, learning dynamics must be statistically equivalent to the single-group baseline. This validates that the multi-group infrastructure from PR #7 works end-to-end when activated via configuration.

## Reference Config

`configs/training/rl_manager/01_rnn_node.yml` — the current canonical RL training config. It replicates the old DJX config `run/manager/17_exp2_group_payoff_heavy_optimize.yml` with only expected differences: `reward_formula` removed (#42) and paths updated from `data/` to `artifacts/`. Key parameters: `n_agents: 4`, `n_groups: 1`, `batch_size: 1000`, `n_rounds: 24`, seed 42.

## Current State

- **Multi-group infrastructure exists** (PR #7): per-group Q-values, `agent_group_mask`, `update_groups()`, group-aware reward/payoff computation.
- **Group assignment is always all-zeros**: `environment.py` line 80 hardcodes `agent_groups = th.zeros(...)`. The `n_groups` parameter is accepted but has no practical effect.
- **No config path for group assignment**: `rl_manager.py` passes `env_args` straight through to `ArtificialHumanEnv.__init__` but there is no `agent_groups` key in any config.

## Proposed Changes

### 1. Environment: accept group assignment from config
- **Where**: `src/aimanager/manager/environment.py`, `__init__`
- **What**: Add optional `agent_groups` parameter (a flat list like `[0,0,0,0,1,1,1,1]`). When provided, convert to a `(batch_size, n_agents)` tensor and pass to `update_groups()`. When absent, keep the current all-zeros default.
- **Why**: Enables YAML-driven group configuration with zero changes to the training loop.

### 2. Training loop: no changes expected
- **Where**: `src/aimanager/rl_manager.py`
- **What**: Because `env_args` is already splatted into the env constructor, the new `agent_groups` key flows through automatically once change 1 is in place. Verify this works; add explicit handling only if needed.
- **Why**: Minimizes code changes as the issue requests.

### 3. New DJX run config
- **Where**: `run/manager/` (new file, naming TBD)
- **What**: Based on the reference config with these modifications: `n_agents: 8`, `n_groups: 2`, `agent_groups: [0,0,0,0,1,1,1,1]`. All hyperparameters (lr, gamma, eps, hidden_size, encodings, batch_size, n_rounds) stay identical.
- **Why**: Two groups of 4 agents each is the regression test configuration. Keeping everything else identical isolates the multi-group change.

### 4. Regression test
- **Where**: Notebook in `notebooks/test_manager/` (new file)
- **What**: Run the reference config (1 group, 4 agents) and the new config (2 groups, 4 agents each) with the same seed. Compare learning curves (reward, contribution, punishment, loss) across training steps. Curves should be statistically equivalent.
- **Why**: The issue explicitly requires confirming that parallelization does not alter learning dynamics or introduce unintended coupling between groups.

## Constraints

- After #42 lands, `reward_formula` will no longer exist. The new config should not include it. If #42 is not yet merged when implementation starts, use `reward_formula: group_payoff_round` and drop it in a follow-up.
- The graph edge index in `update_groups()` is constructed per-group (agents only connect to same-group members). With 8 agents this produces within-group fully-connected edges, which is correct.
- `batch_edge_index` construction in `update_groups()` uses a triple loop. With 8 agents and `batch_size=1000` this may be slow. If it becomes a bottleneck, flag for optimization but do not block on it.

## Next Actions

- [ ] Wait for #42 to land
- [ ] Implement change 1 (environment accepts `agent_groups` from config)
- [ ] Verify change 2 (training loop needs no modification)
- [ ] Create new run config (change 3)
- [ ] Run regression test (change 4), document results in notebook
- [ ] Create PR referencing issue #14
