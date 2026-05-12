# [ACTIVE] Reward mode: sum vs avg for group-switching RL manager

## Goal

The group-switching RL manager currently optimises the **average** payoff per
group member, which makes the manager indifferent to group size. With
group-switching enabled, group size is endogenous — managers that retain or
attract members should be rewarded for it. Switch the reward to the **sum** of
in-group payoffs so the manager is incentivised on both per-member payoff and
group size. Keep `avg` available as a config-selectable mode so the two reward
schemes can be compared empirically. Default is `sum` — this is the new
intended behaviour; legacy configs must explicitly set `reward_mode: avg` to
reproduce their original reward signal.

## Changes

| # | Section | Change | Optional |
|---|---------|--------|----------|
| 1 | Env constructor | New `reward_mode` kwarg | no |
| 2 | Payoff computation | Compute sum alongside avg | no |
| 3 | State dict | Add `group_payoff_sum` field | no |
| 4 | Reward selection | Pick sum vs avg per `reward_mode` | no |
| 5 | Manager wiring | Pipe `reward_mode` through config | no |
| 6 | Unit test | Cover both modes on a tiny batch | no |

### 1. Env constructor — `src/aimanager/manager/environment.py:43,75-105`
- Add `reward_mode: str = "sum"` kwarg to the env `__init__`; store as
  `self.reward_mode`.
- Validate it's one of `{"avg", "sum"}`; raise `ValueError` otherwise.

### 2. Payoff computation — `src/aimanager/manager/environment.py:216-243`
- In `compute_average_payoff_per_group`, the existing numerator
  (`payoff * agent_group_mask * contribution_valid` summed over agents) is the
  per-group **sum** we want. Return both:
  ```
  group_payoff_sum = numerator                              # (B, G, 1)
  group_payoff_avg = numerator / contribution_valid.sum(...) # existing value
  ```
- Rename internally or add a sibling method — whichever keeps the diff small.
  The avg path must remain numerically unchanged.

### 3. State dict — `src/aimanager/manager/environment.py:131-156`
- In `reset_state`, add `group_payoff_sum` with the same shape as the existing
  `group_payoff` field: `(batch_size, n_groups, 1)`, zero-initialised.
- Keep the existing `group_payoff` key for `avg` (don't rename, to avoid
  breaking downstream loggers/notebooks).

### 4. Reward selection — `src/aimanager/manager/environment.py:270-282`
- In `update_payoff`: write both `group_payoff` (avg) and `group_payoff_sum`
  into state every round, regardless of `reward_mode`. This keeps diagnostics
  comparable across runs.
- In `update_reward`: select the field to use as the optimisation signal based
  on `self.reward_mode` (`group_payoff` for `"avg"`, `group_payoff_sum` for
  `"sum"`).

### 5. Manager wiring — `src/aimanager/manager/manager.py` and `src/aimanager/simulation/simulate.py:184`
- Wherever the env is built from config (constructor / factory in this file,
  and the `ArtificialHumanEnv(...)` call in `simulate.py`), read `reward_mode`
  from the config dict and forward it to the env. Default to `"sum"` if the
  key is absent — matches the env-constructor default.

## Test

Add one test in `src/aimanager/tests/test_environment.py` that constructs the
env twice on a tiny hand-built batch (two groups, asymmetric sizes, known
contributions and payoffs): once with `reward_mode="avg"` and once with
`reward_mode="sum"`. Assert (a) `state["group_payoff"]` matches the existing
average formula in both runs, (b) `state["group_payoff_sum"]` equals the
numerator (sum over valid in-group payoffs) in both runs, and (c) the reward
returned by `update_reward` equals avg in the first run and sum in the second.
Run on Raven via `scripts/remote_test.sh` — the suite cannot run locally
(PyG/torch-scatter is Linux-only).

## Config migration note

Default is `sum`, so legacy RL manager configs (e.g.
`configs/training/rl_manager/02_rnn_node_2groups.yml`,
`configs/training/rl_manager/02_rnn_node_1group.yml`) will switch reward
signal unless they explicitly set `reward_mode: avg`. If we want to keep the
existing comparison runs reproducible, add `reward_mode: avg` to those YAMLs
as part of this PR. New configs can omit the key.

## Out of scope

A new 2g8a RL manager config (using the 50ep AHs) with `reward_mode: sum` is a
separate follow-up, not part of this plan.

## Next Actions

- [x] Step 1: env constructor `reward_mode` kwarg + validation
- [x] Step 2: payoff fn returns sum and avg in one pass
- [x] Step 3: add `group_payoff_sum` to state dict
- [x] Step 4: `update_payoff` writes both; `update_reward` selects via `reward_mode`
- [x] Step 5: wire `reward_mode` through `simulate.py` (rl_manager already spreads `**env_args`); add `reward_mode: avg` to all 5 legacy RL manager YAMLs
- [ ] Step 6: unit test in `src/aimanager/tests/test_environment.py`
- [ ] Run `scripts/remote_test.sh` on Raven
- [ ] Lint pass (`pre-commit run --all-files`) and commit via `/commit`
