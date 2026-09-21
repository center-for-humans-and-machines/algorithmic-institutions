# rl-manager-two-worlds

## Declaration

**Not a slot experiment.** This branch does not change an artificial-human
model and is not judged by the §2 gates of `notes/autoresearch.md`. It is the
first reinforcement-learning manager training since the long run of
correctness fixes, run as a two-arm comparison, and its question is
behavioural: **does the trained manager punish at all, and does that differ
between the old and the corrected simulated players?**

**Base branch.** `origin/auto/sim-timeout-imputation` (head `3fe1f44`).
`origin/auto/rl-manager-timeout-view` was requested as the base but did not
exist on the remote when this branch was created (only a local sibling
worktree at the same `3fe1f44`), so the documented fallback was taken. The
serving fix that branch carries is therefore reproduced here directly (note
3) rather than inherited.

**Arms.** Identical in every key except the four artificial-human artifact
paths and the seed; the configs are generated from one template by
`scripts/rl_two_worlds/make_configs.py` so that claim is mechanical rather
than asserted. Hyperparameters are copied verbatim from
`configs/training/rl_manager/03_2g8a_sum.yml`; nothing is tuned.

| slot | arm OLD | arm NEW |
|---|---|---|
| contribution | `group_switching_contribution_50ep/.../epochs_1000.pt` | `group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/.../epochs_575.pt` |
| valid | `raven_script_22/.../rnn_False__dataset_full.pt` | same |
| switch | `switch_pred_opt_50ep/.../epochs_375.pt` | `switch_joint_exodus/.../dataset_50ep_doubled.pt` |
| opponent punisher | `punishment_autoregressive_50ep/.../epochs_7500.pt` | `baselines/punishment_multinomial_ceiling_severity_copula.joblib` |

Three seeds per arm (42 / 43 / 44). The seed spread is a first-class result,
not a footnote: this project has measured that retraining a *supervised*
model moves a typical evaluation row by more than most experiments move it,
and RL is noisier.

**What is measured**, in priority order: (1) does it punish — share of
agent-rounds with punishment > 0, mean punishment, mean given punishment > 0,
against the human managers in `experiments/2group_8agent_50ep.csv` through
the evaluation suite's canonical frame; (2) the policy shape, punishment by
the contribution it responds to, on the suite's RPA bins, learned vs human vs
artificial punisher; (3) a rule-based control that decides whether a manager
which declines to punish is badly trained or correct about this world;
(4) the common good achieved, always named with the world it was measured in.

## Plan

1. **Recover the deleted arm-OLD artifacts.** Done — note 1.
2. **Serve both managers the recorded 0 for a timeout** in
   `rl_manager.run_batch`. Done — note 3.
3. **Batched linear punisher opponent** so arm NEW's frontier punisher can
   occupy the opponent slot. Done — note 2.
4. **Price one run** with a short arm-NEW pilot before committing to six.
5. Six training runs, three seeds per arm.
6. Cross-evaluate all six managers in both worlds, plus the rule-based
   control, the zero-punishment floor and each world's own artificial
   punisher scored on the same axes.
7. Tables under `plots/data_analysis/evaluation/rl_manager_two_worlds/`.

## Results

| date | run | world | punish rate | mean p | mean p given p>0 | common good | verdict |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Notes

1. **The two deleted arm-OLD artifacts came from different places.** The
   autoregressive opponent punisher
   (`punishment_autoregressive_50ep/.../epochs_7500.pt`) survives on
   `origin/includes-stale-work` and was restored from there. The contribution
   checkpoint the config names, `epochs_1000.pt`, survives *nowhere* on any
   remote branch: `git log --all --diff-filter=D` puts its deletion in
   `7d104cc` ("Retrain AH stack on doubled data", #103), and it was restored
   from that commit's parent. Both are real torch archives, not LFS pointers.

2. **The frontier punisher is a `.joblib` linear baseline, and the RL loop
   could not load one.** `rl_manager` loaded the opponent through
   `AH_MODELS[...]` (GNN only), and the simulation's `LinearAHAdapter` serves
   punishments through `get_punishments(rounds)` — a Python round-dict
   history for **one** episode, rebuilding the whole `build_feature_pool`
   every round. RL training runs `batch_size: 1000` episodes in parallel, so
   that path is both the wrong shape and far too slow to sit inside the loop.
   The way out is that every feature the current punishment bundles select is
   *local*: `punishment_multinomial_ceiling_severity_copula` uses exactly
   `contribution`, `contribution_max`, `prev_contribution`,
   `prev_punishment`, `round_number`, `is_first` — no group aggregates, no
   tenure windows. `manager/linear_opponent.py` recomputes those columns
   straight from the env state, batched over episodes, asserts at load that
   the bundle selects nothing outside that set, and is pinned to the
   simulation path by `tests/test_linear_opponent.py` (exact equality on all
   four punishment bundles, every round of a 12-round episode with timeouts
   and a mid-episode reshuffle).

3. **Both managers now see the recorded 0 for a timed-out player.**
   `environment.served_state()` already did this for the contribution, valid
   and switch models, but `run_batch` handed the *raw* state to both
   `manager.get_action` and `opponent_manager.predict` — so a manager read a
   contribution of 9 (the dataset default) where the game had charged 0 and
   where the punishers' own training data stores 0. Fixed by serving
   `env.served_state()` to both, and by taking the replay copy from the same
   view so the policy is trained on what it acted on. This is a correctness
   fix to the simulator rather than a model choice, so it is in **both**
   arms and cannot favour either.
