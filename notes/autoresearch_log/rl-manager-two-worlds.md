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
| switch | `switch_pred_opt_50ep/.../epochs_375.pt` **(open — note 7)** | `switch_joint_exodus/.../dataset_50ep_doubled.pt` |
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
   Config written (`two_worlds_pilot_new.yml`), **not submitted** — note 11.
5. Six training runs, three seeds per arm. **Blocked** — notes 4-7.
6. Cross-evaluate all six managers in both worlds, plus the rule-based
   control, the zero-punishment floor and each world's own artificial
   punisher scored on the same axes as the learned managers, so the clone of
   a human manager is a baseline and not only an opponent.
7. Tables under `plots/data_analysis/evaluation/rl_manager_two_worlds/`.

## Successor

Whoever picks this up inherits a branch where the setup is done and nothing
has been run. In order:

1. **Decide S1** (note 9) — is the manager trained on `group_payoff_sum` or on
   the common good? This changes what a punish rate means and should be
   settled before, not after.
2. **Take D1 and D2 from wherever they land** (a sibling is extending the
   serving fix to D1's call sites and the replay buffer). This branch's own
   `run_batch` change covers D5's three call sites only; reconcile rather
   than duplicate.
3. **Decide arm OLD's switch slot** (note 7). Recommendation:
   `OLD_SWITCH_REANCHORED`, already named in the generator.
4. **Price it** — submit `two_worlds_pilot_new.yml` (40 update steps) and
   read the wall clock before committing six A100-days. Six jobs fit one
   wave, nine do not (note 11).
5. Then, and only then, the six runs.

Self-play (note 10) is a separate change and should not land in the same run
as the first RL training.

## Status: BLOCKED, nothing has been run

No training job has been submitted and no compute has been spent. The setup
below is committed so the work is not lost and so the blockers can be judged
against something concrete, not because it is ready to run. Every generated
config carries a `NOT YET RUN -- BLOCKED` banner.

## Results

| date | run | world | punish rate | mean p | mean p given p>0 | common good | verdict |
|---|---|---|---|---|---|---|---|
| — | nothing run yet | — | — | — | — | — | blocked (notes 4-8) |

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

4. **The pre-run review stopped the experiment before any compute was spent.**
   `notes/reviews/rl-manager-review.md` (branch `review/rl-manager`) reports 6
   confirmed defects. Four bear on this design, and two of them —  D1 and D2 —
   would invalidate both arms equally, which does *not* make them harmless:
   they change what the learner optimises, so "does the manager punish" would
   be answered about a game the humans never played. Working around them was
   explicitly ruled out, so nothing was launched.

5. **D1, the free-punishment lever, is the one that most directly attacks this
   experiment's headline question.** A punishment aimed at a timed-out player
   is discarded by the common-good and payoff accounting but is still copied
   into `prev_punishment` and shown to every artificial human. The manager's
   own `x_encoding` contains `contribution_valid`, so it can identify exactly
   the cells where the action is free, and the action is not inert there —
   `prev_punishment` is the contribution model's only channel from the manager.
   A measured punish rate could then be partly an artefact of a lever the real
   game did not offer. The review's diagnostic (mean punishment conditioned on
   `contribution_valid == False`) should be part of the measurement plan even
   after the fix lands, as a regression check.

6. **D2 changes the reward, not just the bookkeeping.** With `reward_mode:
   sum` the timed-out player's payoff — which the real game paid, mean 33.94
   over all 560 timed-out human rows — is dropped from the group total. On the
   516 human group-rounds containing a timeout that is a 30.7% shortfall, and
   the validity model is exogenous to the policy, so it is pure unlearnable
   noise on roughly a fifth of group-rounds. Fixing this changes the
   optimisation problem, which is why no run should precede it.

7. **D4 is the one that changes how arm OLD has to be built, and it is a
   decision rather than a fix.** `03_2g8a_sum.yml`'s switch model,
   `switch_pred_opt_50ep`, was trained under the pre-#123 arrival-round
   anchoring; `environment.step()` now calls the predictor at the end of round
   s on round-s post-punish features. Feeding the old artifact there hands it
   round s-1 values where it expects round s. The damage is specific: that lag
   would sit in **arm OLD only**, so the two arms would differ by a code
   convention as well as by the clones, and the comparison the experiment
   exists to make would be confounded. Recommendation, stated as a
   recommendation: keep arm OLD's contribution model (`epochs_1000`) and its
   autoregressive punisher, and move the switch slot to
   `switch_pred_opt_50ep_doubled_reanchored` — the same switch trunk at the
   earliest artifact that is anchoring-correct, already on the branch. Arm OLD
   then means "the old contributor and the old punisher", which is what the
   question is about, rather than "the old contributor, the old punisher and a
   one-round switch lag". Both candidate paths are named in
   `scripts/rl_two_worlds/make_configs.py` with the decision left open.

8. **D3 reads differently for this experiment than for the reviewer.** The
   review flags the lagged GNN punisher in `03_2g8a_sum_d_lr1e3_freq500.yml`.
   That is not the config this branch builds on: arm OLD's opponent is the
   autoregressive punisher of `03_2g8a_sum.yml`, and arm OLD is *supposed* to
   be the old world, an old punisher included. The honest framing is that arm
   OLD is a straw-man world by construction and its purpose is to be compared
   against a corrected one — not that its opponent should be upgraded, which
   would collapse the two arms towards each other. What should be recorded is
   that the arm-OLD opponent has never been checked for the
   current-contribution defect, so its own policy shape must be measured and
   reported (item 2 of the measurement plan) rather than assumed.

9. **S1 is a specification call the maintainer should make before any run, and
   it bears directly on the headline claim.** The reward is
   `group_payoff_sum = 20*n_valid + 0.6*sum(c) - 2*sum(p)`, not the common
   pool `1.6*sum(c) - sum(p)`. Its marginal rate of substitution makes the
   manager about five times more reluctant to punish than the stated objective
   implies, and 63% of its variance across human group-rounds is the headcount
   term. So "the manager barely punishes" would have (at least) three live
   explanations, not the two the experiment was framed around: the clones
   under-react to punishment; punishment genuinely does not pay in this world;
   or the training reward prices punishment far above what the common good
   does. The planned rule-based control separates the first two but not the
   third. The cheap way to separate the third is to report the common good and
   the training reward side by side for every run and for the control, which
   the measurement plan now does — but if the maintainer's success criterion
   is the policy against the common good, training on `group_payoff_sum` is
   the wrong objective and should be changed before the run, not explained
   afterwards.

10. **Self-play (the requested third arm) needs code, not configuration.**
    `run_batch` calls the opponent through `.predict()` and never collects its
    experience: there is one `Memory`, one `get_action` call, one `update`, and
    the metric schema hard-codes a fixed `opp_*` side. A second learner needs
    its own action path in the rollout (own epsilon, own RNN reset), its own
    replay stream, its own target-network schedule and its own `update` call —
    `ArtificalManager.update` already slices the TD error by `rl_group_id`, so
    the TD machinery is the part that mostly exists. It also needs two saved
    checkpoints and a metrics/parquet schema that does not assume one side is
    frozen. Estimate: a few hundred lines concentrated in `rl_manager.py` plus
    config keys, and a new failure surface (non-stationarity, poaching, one
    manager ceding its group) that has no baseline to be read against. Landing
    it in the same change as the first RL run would make either one
    undiagnosable, so it is left unwritten pending a decision.

11. **Cluster capacity, measured.** `sacctmgr` gives the `mpib_gpu`
    association `MaxJobs=8`, `MaxSubmit=300`, and the `gpu` partition's
    `MaxTime` is 24 h. So at most **8 concurrent GPU jobs**, one of which is
    currently taken by an unrelated job. Six runs fit in a single wave; nine
    would need two waves. `scripts/manager/run_training.sh` requests
    `--time=20:00:00` per job, which is the project's own standing estimate
    for one manager training and the only cost figure that exists — there are
    **no** `train-manager` jobs in this account's recent SLURM history, and
    `.log/training/manager/` on the shared checkout is empty, so nothing has
    been measured. The pilot that would have measured it
    (`two_worlds_pilot_new.yml`, 40 update steps) is written and committed but
    was not submitted.
