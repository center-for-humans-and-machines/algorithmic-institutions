# [ACTIVE] Train RL Manager with Group-Switching Artificial Humans (Issue #69)

## Goal

Run the first RL manager training on the new 2-group, 8-agent, group-switching
stack. The trained manager controls 4 agents; the other 4 are managed by a
fixed opponent — the autoregressive punishment AH from PR #90
(`artifacts/artificial_humans/punishment_autoregressive_50ep/...`). Both
managers act simultaneously on global state each round; the RL manager learns
only from its own group's reward. Success = a reward curve that improves over
training, captured in the parquet/log artefacts so improvement is verifiable
post hoc. We ship a `reward_mode: sum` config (primary) and a `reward_mode: avg`
twin for sanity comparison.

## Architecture diagram in words

Per round `t`:

1. **State** carries `prev_*` features (last round's contributions + last
   round's punishments by both managers) for all 8 agents, plus current
   `agent_groups` (possibly mutated this round by the switch predictor).
2. **RL manager** consumes global state, outputs Q-values over all 8 agents,
   selects `rl_action` (shape `(B, 8, 1)`).
3. **Opponent AH** (`predict_autoreg`) runs over all 8 agents and returns
   `opp_action` (shape `(B, 8, 1)`). Its internal autoregression only crosses
   its own predictions; we discard the 4 entries that belong to the RL group
   (Approach A: ~2x compute waste, accepted).
4. **Mask + merge**: `rl_mask = (agent_groups == rl_group_id)`; final
   punishments are `th.where(rl_mask, rl_action, opp_action)`.
5. **Env**: `env.punish(final_p)` applies punishments and updates
   `common_good`; `env.step()` advances the round, applies the switch
   predictor (which may rewrite `agent_groups` and recompute
   `agent_group_mask`), and returns `(state, reward, done)` with `reward`
   shape `(B, n_groups, 1)`.
6. **Replay**: store `(prev_state, rl_action, reward[:, rl_group_id], state,
   rl_mask)`. Opponent transitions are NOT stored; the AH is frozen.

Reference pseudocode for the change in `src/aimanager/rl_manager.py` — the
implementation should mirror this:

```python
# One-time setup, alongside existing AH loads
opponent_manager = GraphNetwork.load(config["opponent_manager"], device=device)
# For this issue the opponent is the autoregressive 50ep punishment AH.
# Same loader path will work later for an RL-manager checkpoint (self-play),
# so the config key stays generic.
rl_group_id = config.get("rl_group_id", 0)

# Training loop, replacing the current single-manager env.punish(action) call
state = env.reset()
while not done:
    rl_action  = trained_manager.act(state)                        # (B, 8, 1)
    opp_action = opponent_manager.predict(state, edge_index=env.batch_edge_index)
                                                                   # (B, 8, 1)

    # Recompute mask every round — env.agent_groups mutates at switch boundaries
    rl_mask    = (env.agent_groups.squeeze(-1) == rl_group_id).unsqueeze(-1)
    final_p    = th.where(rl_mask, rl_action, opp_action)

    state = env.punish(final_p)
    state, reward, done = env.step()

    # Only RL transitions stored; reward sliced to the RL manager's group
    replay.add(prev_state, rl_action, reward[:, rl_group_id], state, rl_mask)
```

The autoregressive vs non-autoregressive distinction lives inside
`opponent_manager.predict()` (dispatched by `GraphNetwork`'s `autoregressive`
flag at load time); the calling code above is identical either way.

## Changes

| # | File | Change | Type |
|---|------|--------|------|
| 1 | `src/aimanager/rl_manager.py` ~L140-160 (env construction area) | Load opponent manager via `GraphNetwork.load(config["opponent_manager"], device=device)` next to the existing AH loads. Read `rl_group_id = config.get("rl_group_id", 0)`. | new code |
| 1b | `src/aimanager/rl_manager.py` (same area + env construction) | Load the switch AH via `config["switch_model"]` (same `AH_MODELS[...].load(...)` pattern), and pass it as `artifical_humans_switch=ah_switch` to the `ArtificialHumanEnv(...)` call. `switch_every` already flows through `**env_args`. Without this, the env never invokes the switch predictor and groups stay static — defeats the point of #69. | new code |
| 2 | `src/aimanager/rl_manager.py` (training loop, after `manager.act`) | Insert merge logic: call `opponent_ah.predict(state, edge_index=env.batch_edge_index)`, build `rl_mask` from `env.agent_groups`, compute `final_p = th.where(rl_mask, rl_action, opp_action)`, pass `final_p` (not `rl_action`) to `env.punish(...)`. | new code |
| 3 | `src/aimanager/rl_manager.py` (replay-add call) | Index reward as `reward[:, rl_group_id]` before passing to the replay buffer; also forward `rl_mask` so the loss only counts RL-group agents. | new code |
| 4 | `src/aimanager/manager/environment.py` | No change. `punish()`/`step()` split, per-round `agent_groups` mutation, and `(B, n_groups, 1)` reward already implemented in PRs #88/#90/#68. | no change |
| 5 | `src/aimanager/generic/graph.py` `predict_autoreg` | No change. Approach A means we just call `predict` on the full 8-agent state. | no change |
| 6 | `configs/training/rl_manager/03_2g8a_sum.yml` | New YAML. Keys: `artificial_humans`, `artificial_humans_valid`, `switch_model` (the three 50ep AH paths), `opponent_manager` (the autoreg punishment .pt), `rl_group_id: 0`, `env_args: {n_agents: 8, n_groups: 2, agent_groups: [0,0,0,0,1,1,1,1], n_rounds: 24, switch_every: 4, reward_mode: sum}` (values pulled from `configs/simulation/ah_testing/group_switching_ah_punishment_50ep.yml` and `configs/training/artificial_humans/punishment/autoregressive_50ep.yml`), `manager_args` mirroring `02_rnn_node_2groups.yml` (RNN + edge model, same hidden_size, output over 8 agents — optimisation is future work), `n_update_steps`, `eval_period`, `batch_size`, `replay_buffer_size`, `output_dir`. | new config |
| 7 | `configs/training/rl_manager/03_2g8a_avg.yml` | New YAML, identical to (6) except `reward_mode: avg` and a distinct `output_dir`. | new config |

## Sanity-check plan

Primary signal: the per-update training reward (already logged to parquet by
`rl_manager.py`) should trend upward over `n_update_steps`. Read it off the
parquet artefact in `output_dir`. As a cross-check, train the `sum` and `avg`
variants in parallel: both should improve, and their curves should be
proportional up to the `n_agents_in_group` scale factor — large qualitative
divergence (e.g. one improves, one doesn't) flags a reward-indexing bug.

## Test

No new unit tests. The env path (group switching, two-group reward, mask
recompute) is already covered by tests landed with issues #68 and #88, and the
AH side is untouched. Before launching a long run, do a smoke training on
Raven via `scripts/train_cluster.sh manager configs/training/rl_manager/03_2g8a_sum.yml`
with a small override (`n_update_steps` ~50, tiny batch) to confirm: opponent
manager loads, shapes line up at the merge step, replay accepts the indexed
reward, training completes a few updates without NaN. Then submit the full run.

## PR evidence (final verification)

The PR opened for this work must include a simulation run using the trained
checkpoint — not just the training-time reward curve. Concretely:

1. Take the trained checkpoint from `output_dir` after the full run.
2. Run a simulation via `python -m aimanager simulate <sim_config>` where the
   trained manager plays against the same opponent AH, on the 50ep AH stack.
   Reuse / adapt `configs/simulation/ah_testing/group_switching_ah_punishment_50ep.yml`
   as a template; swap the manager slot to point at the new checkpoint.
3. Produce visualisations in `plots/` covering at minimum:
   - per-round contribution and punishment trajectories (RL group vs opponent
     group),
   - reward / group-payoff curves over rounds,
   - group-size dynamics (consequence of the switch predictor — does the RL
     manager retain/attract members under `reward_mode: sum`?).
4. Embed the key plots in the PR body so reviewers can eyeball the
   improvement vs the no-training baseline.

## Out of scope

- Approach B (autoregressing only over the opponent's 4 agents).
- Retraining a non-autoregressive opponent AH.
- Self-play / two learning managers.
- More than 2 groups, or asymmetric group sizes.
- Using a trained RL-manager checkpoint as the opponent instead of the AH.

## Next Actions

- [x] Step 1: load opponent manager + `rl_group_id` in `src/aimanager/rl_manager.py`.
- [x] Step 1b: load switch AH and pass it to `ArtificialHumanEnv(...)`.
- [x] Step 2: merge logic in training loop (`run_batch` or its caller) — opponent.predict, mask, th.where, env.punish(final_p).
- [ ] Step 3: index reward by `rl_group_id` and forward `rl_mask` to the replay buffer.
- [ ] Add `configs/training/rl_manager/03_2g8a_sum.yml` and `03_2g8a_avg.yml`.
- [ ] Smoke run on Raven via `scripts/train_cluster.sh manager 03_2g8a_sum.yml` with reduced steps.
- [ ] Launch full `sum` + `avg` training runs in parallel.
- [ ] Inspect parquet reward curves to confirm improvement.
- [ ] Run simulation with the trained checkpoint vs the opponent AH; generate plots (contribution/punishment trajectories, reward curves, group-size dynamics).
- [ ] Embed plots in the PR body and report back on issue #69.
