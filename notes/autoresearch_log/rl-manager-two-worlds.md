# rl-manager-two-worlds

## Declaration

**Not a slot experiment.** This branch does not change an artificial-human
model and is not judged by the §2 gates of `notes/autoresearch.md`.

**Scope, as revised (note 12).** It began as a two-arm old-vs-new comparison.
The maintainer dropped arm OLD as too hard to replicate faithfully, so what
remains is one world and one question: **train a manager against the current
corrected clones and answer whether it punishes at all, what its policy looks
like, and whether it beats the artificial punisher — this project's clone of
a human manager.** Three seeds, no self-play.

**Base branch.** `origin/auto/rl-manager-timeout-view` (PR #205), which serves
the manager, its replay buffer and the opponent the recorded 0 for a
timed-out player, verified on 113 timed-out cells. This branch was first
built on `origin/auto/sim-timeout-imputation` because #205 was not yet on the
remote, and was rebased once it was; #205's version of that fix supersedes
and replaced this branch's own (note 13).

**The one world.**

| slot | artifact |
|---|---|
| contribution | `group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/.../epochs_575.pt` |
| valid | `raven_script_22/.../rnn_False__dataset_full.pt` |
| switch | `switch_joint_exodus/.../dataset_50ep_doubled.pt` |
| punisher | `baselines/punishment_multinomial_ceiling_severity_copula.joblib` |

The punisher is both the opponent the manager trains against **and** a
baseline scored on the same axes as the learned policies — the comparison the
maintainer's success criterion turns on.

Three seeds (42 / 43 / 44), differing in nothing else. The seed spread is a
first-class result, not a footnote: this project has measured that retraining
a *supervised* model moves a typical evaluation row by more than most
experiments move it, and RL is noisier.

**Reward: the common pool.** Training on `group_payoff_sum` priced punishment
about five times too high and was ~63% headcount by variance (review S1). The
common-pool mode is being added on `auto/manager-common-pool-reward`, which
also fixes the discarded payoff for timed-out players (review D2). The
generator carries `REWARD_MODE` as a single named constant so the switch is
one line.

## Measurement plan

Six outputs, in priority order. Every one is computed per seed and reported
with its spread across seeds, never pooled into a single number.

1. **Does it punish.** Share of agent-rounds with punishment > 0, mean
   punishment, and mean given punishment > 0 — for each of the three learned
   managers, for the artificial punisher, for the rule-based comparison points
   and for the zero-punishment floor. Against the human managers in
   `experiments/2group_8agent_50ep.csv` read through
   `evaluation_suite.convert.load_human` (which drops the flip duplicates and
   marks a manager timeout as a NaN punishment, so "punished 0" and "gave no
   input" stay distinct). Human reference, measured: mean punishment 1.79,
   P(p = 0) 0.694 over 18,386 rows with a valid manager input.
2. **Mean punishment conditioned on `contribution_valid`, per seed and across
   training.** Required output, not a diagnostic afterthought: while the
   free-punishment defect is unfixed, punishment on timed-out cells is free
   while punishment everywhere else is costly under a common-pool reward, so a
   free lever strictly dominates a paid one. If the learned policy concentrates
   there, the headline number is an artefact and must be reported as one.
3. **The policy shape.** Punishment by the contribution it responds to, on the
   evaluation suite's own RPA bins (`ResponseMetrics.rpa`, edges
   `[-1, 0, 5, 10, 15, 19, 20]`, so 0 and 20 are isolated), mean per bin.
   Learned vs human vs artificial punisher on the same axes. Human reference,
   measured: 4.76 / 2.97 / 1.67 / 0.98 / 0.69 / 0.27 over bins {0}, 1-5, 6-10,
   11-15, 16-19, {20}.
4. **The comparison points.** The artificial punisher scored as a baseline,
   the zero-punishment floor, and the best rules from the sibling sweep
   (`notes/autoresearch_log/rule-based-manager-sweep.md` when it lands). If a
   rule earns more common good than the learned manager, that is a training
   failure; if it earns less, the world is saying punishment does not pay.
5. **The common good achieved**, reported alongside the training reward so the
   two are never confused.
6. **Convergence**, not only final numbers: reward, loss and the punish rate
   over update steps, per seed.

Cross-evaluation is one world, so it collapses to three learned managers plus
the baselines in a single simulation config, one pairing per manager against
the artificial punisher.

## Plan

1. **Batched linear punisher opponent** so the frontier multinomial can
   occupy the opponent slot. Done — note 2.
2. **Rebase onto PR #205** for the served-state fix. Done — note 13.
3. **Drop arm OLD; regenerate for one arm.** Done — note 12.
4. **Price one run** with a 40-step pilot before committing three A100-days.
   Done — notes 14-15.
5. Three training runs, seeds 42/43/44. **Not started** — see Status.
6. One cross-evaluation simulation: the three learned managers, the
   artificial punisher as a baseline on the same axes, the zero floor and the
   sibling sweep's best rules.
7. Tables under `plots/data_analysis/evaluation/rl_manager_two_worlds/`.

## Status: both guards pass, three seeds running

Both blockers are cleared. `auto/manager-common-pool-reward` merged cleanly;
`auto/free-punishment-fix` merged with the single predicted docstring conflict
and its own test file passes on the merged tree. Both pre-launch guards pass
(notes 22-24). Jobs **30401560** (s42), **30401561** (s43), **30401562**
(s44), ~6 h wall clock.

## Successor

1. Take the common-pool mode from `auto/manager-common-pool-reward`, flip
   `REWARD_MODE`, regenerate.
2. Resolve D1 — fixed, or explicitly accepted with the validity-conditioned
   diagnostic as a first-class output.
3. Submit the three runs. They fit one wave, but the association's 8 job
   slots are shared with sibling experiments (note 16).
4. Read `notes/autoresearch_log/rule-based-manager-sweep.md` for the
   comparison points before writing the results table.

## Results

No training run has been made. The only job submitted is the cost pilot.

| date | run | steps | wall clock | outcome |
|---|---|---|---|---|
| 2026-09-21 | `rl_new_clones_pilot` (job 30400864) | 40 | 4:37 total, 3:46 in loop | `COMPLETED`; 5.65 s/step, 5.20 s/step late-stage; projects to ~5.8-6.3 h for 4000 steps (note 15) |

The pilot is a timing measurement. Its policy is not read and its reward mode
is the one being replaced, so nothing behavioural is recorded from it.

The behavioural table below is the one the experiment exists to fill, per seed:

| seed | punish rate | mean p | mean p given p>0 | mean p at `contribution_valid=False` | common good | verdict |
|---|---|---|---|---|---|---|
| — | not run | | | | | |

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

12. **Scope revision: arm OLD dropped, no self-play, three seeds against the
    corrected clones.** The maintainer judged arm OLD too hard to replicate
    faithfully, and note 7 is part of why: its switch artifact carries the
    pre-#123 anchoring, so the only faithful version of that arm contains a
    one-round lag that would sit in one arm and not the other. Dropping it
    also removes the two recovered artifacts of note 1 from the branch — the
    recipe that found them stays in note 1, so they are recoverable if the
    comparison is ever revived. Self-play (note 10) was declined and is not
    built. The cross-evaluation collapses accordingly: three learned managers
    plus the artificial punisher, the zero floor and the sibling sweep's best
    rules, all in one world.

13. **Rebased onto `auto/rl-manager-timeout-view` (PR #205); its fix replaced
    this branch's.** Both branches independently fixed D5, and the rebase
    merged them *textually* rather than flagging a conflict, leaving two
    served-state layers in `run_batch` — a live example of why a duplicated
    fix is worse than a missing one. #205's version is the better of the two
    and is what survives: it binds `state` to the served view once after
    `reset()` and `step()`, so the manager, the replay copy and the opponent
    are covered at one point, and it keeps the raw state under the name
    `recorded` for the metrics, which are the run's record of what the game
    charged. This branch's own served-state change is gone; only the opponent
    dispatch remains in `rl_manager.py`.

14. **Cost pilot (job 30400864).** Submitted from
    `AI_REMOTE_DIR=~/repros/ai-runs/rl-two-worlds` as
    `configs/training/rl_manager/rl_new_clones_pilot.yml`: 40 update steps,
    eval every 10, otherwise byte-identical to the real configs. It confirmed
    the parts that had never been exercised together — the frontier
    contribution, valid and joint-exodus switch models load, and the batched
    linear punisher loads and serves as the opponent inside the real rollout,
    which is the first time `linear_opponent.py` has run against the actual
    env rather than the parity test.

15. **Cost, measured (job 30400864, A100, `COMPLETED`).** 40 update steps ran
    in **3 min 46 s** of training loop — **5.65 s/step** averaged, **5.20
    s/step** at the late-stage rate once the first few warm-up steps are past.
    Total job elapsed **4 min 37 s**, so fixed overhead (venv, `module load
    cuda`, wandb init, loading four models, the final save/load round-trip and
    the parquet write) is **~51 s**. `MaxRSS` 5.9 GB against the template's
    16 GB request. The saved policy is **6.7 MB**, so all three fit in git-lfs
    without a second thought.

    **Projection for one real run (4000 steps): ~5.8-6.3 h.** 4000 x 5.20 s =
    5 h 47 m at the late rate, 4000 x 5.65 s = 6 h 17 m if the warm-up rate is
    charged throughout; the real configs use `eval_period: 20` against the
    pilot's 10, so they run half as many on-policy eval rollouts per step and
    the true figure sits at or below the lower end.

    Three runs in parallel: **~6 h wall clock, ~18 A100-hours total.** Well
    inside the `gpu` partition's 24 h limit and the association's 8 job slots.

    Two things follow. **Three seeds is comfortably affordable and so is
    more** — at ~6 A100-hours each, five seeds would cost 30 A100-hours and
    still fit one wave. Given that the seed spread is meant to be a
    first-class result and three points make a poor spread estimate, five is
    worth considering. And **the original six-run two-arm design was never
    implausible** either; the `--time=20:00:00` in
    `scripts/manager/run_training.sh` is roughly 3x pessimistic, which costs
    queue priority on a backfill scheduler. Lowering it to ~10 h would start
    these jobs sooner. Not changed here: it is a shared template and other
    experiments' manager runs may not be this size.

16. **Queue contention is the real constraint, not GPU-hours.** The
    association `mpib_gpu` allows 8 concurrent jobs and at submission time 6
    were already queued by sibling experiments, with one unrelated job
    running. Three runs fit, but not instantly and not alongside an arbitrary
    number of siblings. Worth coordinating rather than assuming.

17. **Manager artifacts survive a re-sync.** `train_cluster.sh` adds
    `--exclude='artifacts/manager/'` when `AI_REMOTE_DIR` points at an
    isolated dir, so `rsync --delete` does not remove trained checkpoints.
    A `--sync-only` to ship a code change is therefore safe with runs' output
    sitting on the cluster — the hazard the `--no-sync` rule guards against
    does not apply to this directory.

18. **Guard 1 passes exactly: the reward is the common pool.** Merged
    `auto/manager-common-pool-reward` (clean, no conflicts) and flipped
    `REWARD_MODE`. `scripts/rl_two_worlds/launch_guards.py` drives the real
    env from the real config and checks `env.reward` three ways over 24 rounds
    x 64 episodes x 2 groups, with the manager forced to punish the maximum on
    every cell so the reward sits far from 0 and the check is not satisfied
    trivially:

    | check | max abs residual |
    |---|---|
    | vs `1.6*sum(c) - sum(p)` recomputed from the env's state | **0.0** (exact) |
    | vs `common_good * n_valid`, a different code path | 1.5e-05 (float32) |
    | vs the same rollout under `reward_mode='sum'` | 240.0 — the modes genuinely differ |

    The third row is what makes the first meaningful: flipping a named
    constant is exactly the kind of change that can silently not happen, and a
    `sum` rollout deviating from the pool by up to 240 shows the flip is live.
    The `common_good` cross-check matters for the same reason — it is produced
    by `update_common_good`, not by the reward path, so the agreement is a
    confirmation rather than a restatement.

19. **Guard 2 fails before the fix — which is what makes it a guard.** The
    same run, on the unfixed tree, found the lever wide open: of 1,659
    timed-out cells observed being served to the contribution, validity and
    switch models, **only 1,207 carried punishment 0**. The rest carried real
    punishment, 194 of them the maximum 30, and `prev_punishment` shows the
    same population one round later — the channel the contribution model
    actually reads. Recorded as
    `plots/data_analysis/evaluation/rl_manager_two_worlds/guards_before_fix.json`.

    A fourth number in the same file reads the defect from the accounting side
    rather than the serving side: the pool computed *without* zeroing invalid
    cells differs from the env's reward by up to 31.2, i.e. the env does zero
    them. That is precisely the asymmetry D1 describes — the punishment is not
    charged to the manager and is still delivered to the players. Under a
    common-pool reward, where every other punishment point costs 1 directly,
    that makes the timed-out cells the only free action in the space.

    The guard is therefore known to be capable of failing, and the post-fix
    run is a real test rather than a formality.

20. **Cross-evaluation and measurement written ahead of the runs.**
    `configs/simulation/manager_testing/24_rl_new_clones_cross_eval.yml` scores
    everything in the training world slot for slot, so a difference between two
    rows is a difference between managers and not between worlds. The
    artificial punisher appears twice over — as the opponent in group 1 of
    every pairing, and through `lin_punisher_self` as a group-0 manager scored
    exactly as the learned ones are, because the success criterion is whether a
    learned policy beats that clone and a baseline that only ever appears as an
    opponent cannot be beaten on the same axes.
    `scripts/rl_two_worlds/measure.py` turns its `per_round.parquet` into the
    tables, reading the human reference through the suite's canonical frame so
    a manager timeout stays a NaN rather than collapsing into "punished 0" —
    pooling those would deflate every human punishment statistic by the 4.2% of
    rows with no manager input.

21. **One measurement does not come from the simulation, and needs its own
    probe.** Item 2 of the plan — punishment conditioned on
    `contribution_valid` — cannot be read off `per_round.parquet`: that file
    carries `punishment`, `common_good`, `contribution`, `agent_group` and
    nothing about validity. It has to come from the env, as the guard already
    does. The guard forces maximum punishment, which is right for testing the
    lever but wrong for measuring a policy, so the trained-policy version is a
    separate run of the same machinery with the checkpoint's greedy action in
    place of the forced one. Written once the checkpoints exist, rather than
    guessed at now.

22. **Both fixes merged; the free-punishment merge conflicted exactly where
    its own log predicted.** `auto/free-punishment-fix` at `799e96a`: one
    conflicted file, `environment.py`, one conflicted hunk, the two docstrings
    at the top of `punish()`. Resolved by keeping both halves and changing no
    code, and the merged body was then read back against the block that
    branch's section 4 predicted — the zeroing first, then the accounting,
    then `compute_reward_per_group` on the already-charged state. The order
    matters and is idempotent either way: `th.where(valid, p, 0)` applied
    twice is itself, so the reward cannot double-count the correction.

    That branch warned that an earlier `merge-tree` check had reported *no*
    conflict and silently dropped its change. The cheap check against that is
    its own test file, so it was run: `test_free_punishment.py` plus
    `test_manager_reward.py`, `test_rl_manager_timeout_view.py` and
    `test_linear_opponent.py` — **34 passed** on Raven against the merged
    tree. Four of those fail loudly if the zeroing is lost.

23. **Guard 2 failed once more before it passed, and the fault was the
    guard's.** On the merged tree the current-round channel came back clean
    immediately — all 1,659 timed-out cells served `punishment` 0 — but
    `prev_punishment` still showed 55 cells at 30. That is not the lever: the
    guard was masking `prev_punishment` with round *t*'s validity when that
    channel carries round *t-1*'s punishment. A player who gave input at t-1
    was punishable then, the punishment was charged, and it is correct for it
    to remain visible at t even though they have since timed out. Masking a
    lagged channel with a current mask flags correct behaviour as a defect.
    Corrected to `prev_contribution_valid` (with round 0's default excluded,
    as `served_state` does), and the same run then reads **627 of 627
    previously-timed-out cells at 0**.

    Worth stating because the failure mode is symmetrical with the defect
    itself: the whole D1 family came from applying a value at the wrong
    round, and the first attempt to police it made the same class of mistake
    in the opposite direction.

24. **Both guards pass; the three seeds are launched.** Evidence:
    `plots/data_analysis/evaluation/rl_manager_two_worlds/guards_{before,after}_fix.json`.

    | guard | before the fix | after |
    |---|---|---|
    | reward == `1.6*sum(c) - sum(p)`, max residual | 0.0 | **0.0** |
    | vs `common_good * n_valid` (different code path) | 1.5e-05 | **1.5e-05** |
    | vs `reward_mode='sum'` (must differ) | 240.0 | **240.0** |
    | timed-out cells served `punishment` 0 | 1,207 / 1,659 | **1,659 / 1,659** |
    | of which served the maximum 30 | 194 | **0** |
    | previously-timed-out cells served `prev_punishment` 0 | — | **627 / 627** |

    The guard forces the maximum punishment on every group-0 cell, so it tests
    the mechanism rather than a policy; the before/after contrast on the
    identical rollout is what shows the path is genuinely exercised and that
    the zeros are not the trivial kind. Jobs **30401560** (s42), **30401561**
    (s43), **30401562** (s44).

25. **The rule sweep answers the control question before my runs land, and it
    changes what a quiet policy would mean.** `auto/rule-based-manager-sweep`
    (PR #207), pooling three seeds: `never` earns 99.63 common good, the
    artificial punisher 111.06, `thr9_p10` ("punish 10 whenever a player
    contributed 9 or less") 123.79, `prop10` 136.04. Fifteen of eighteen
    punishing rules beat `never` with intervals excluding zero, and the
    arithmetic says why — the pool pays 1.6 per contribution unit and charges
    1 per punishment point, so punishment pays above 0.625 units bought per
    point and the rules buy 0.83 to 1.47.

    So the branch of my control that said "the world is telling us punishment
    does not pay" is now **closed in this world**: it does pay, decisively.
    If the learned managers converge on a near-zero policy, the remaining
    reading is training failure, not a correct read of the environment —
    roughly 25 common good left on the table against `prop10`, against a
    seed-to-seed standard deviation of 0.86 to 10.9. I will not soften that.

    One caveat on comparability, from the sweep's own section 3.0: its common
    good sums both groups with the same manager on both sides, whereas my
    cross-evaluation puts the learned manager in group 0 against the clone in
    group 1. The like-for-like baseline is the sibling's competing arm (rules
    against the clone and against a never-punish rival, one group each), and
    my runs should be read against that rather than against the table above.

26. **All three runs completed; the cost projection held.** Jobs 30401560 /
    30401561 / 30401562, elapsed **06:04:57 / 05:59:06 / 06:02:44**, against
    the 5 h 47 m – 6 h 17 m projected from the 40-step pilot. Three 6.7 MB
    checkpoints.

27. **The value function had not converged at 4000 steps.** Greedy-eval
    `q_mean` is still climbing inside the last quarter of training in every
    seed — +0.56 (s42), +2.70 (s43), +2.80 (s44) between the first and second
    halves of the final 1,000 update steps. The behavioural quantities are
    steadier over that window (mean punishment 1.81 / 1.08 / 1.80, mean
    contribution 7.48 / 8.24 / 8.45), so the policies are not wandering, but
    nothing here should be called a converged optimum. Reported because the
    step count was inherited from `03_2g8a_sum.yml` and never revisited, and
    because a longer run is the cheapest experiment available at ~6 h.

28. **Punishment conditioned on validity — the required output.** From
    `scripts/rl_two_worlds/validity_conditioned.py`, greedy policy, 128
    episodes, in the training configuration;
    `validity_s4{2,3,4}.json`. Two quantities, because the fix changed what
    they mean: *realised* is what the game charged and every model saw;
    *intended* is the argmax the policy actually picked, before the env
    zeroed it.

    | seed | intended, gave input | intended, timed out | realised, timed out |
    |---|---|---|---|
    | 42 | mean 1.756, 38.7% > 0 | mean 0.551, 11.0% > 0 | **0.000, max 0** |
    | 43 | mean 0.987, 9.8% > 0 | mean 4.788, 30.6% > 0 | **0.000, max 0** |
    | 44 | mean 0.914, 45.0% > 0 | mean 0.618, 17.3% > 0 | **0.000, max 0** |

    The regression check passes on all three: nothing reaches an artificial
    human at a timed-out cell. The *intended* column is the behavioural
    reading and it is not uniform — s42 and s44 aim **less** at timed-out
    cells than at ordinary ones, while s43 aims four times **more** there
    (4.79 against 0.99). With the env zeroing those cells they are a
    don't-care region, so a policy can park anything there at no cost and
    s43's number is most likely an unconstrained artefact rather than a
    learned preference. It is exactly the signature D1 predicted, though, and
    it is the reason this measurement stays a standing output: had the lever
    still been open, s43 would have been exploiting it.

29. **The opponent clone punishes timed-out players about half the time —
    because the ceiling punisher has no validity feature.** Measured on the
    same rollouts: the artificial punisher's intended punishment at timed-out
    cells is mean 3.72 / 3.83 / 3.96 with 47–60% above zero, against 1.55–1.69
    and 27–30% at cells where the player gave input. That is not a defect in
    the run — the env zeroes it — but it is worth recording, and it has a
    clean cause. `punishment_multinomial_ceiling_severity_copula`, the
    artifact this experiment was told to use, selects
    `contribution`, `contribution_max`, `prev_contribution`,
    `prev_punishment`, `round_number`, `is_first` and **not**
    `contribution_valid`. A timed-out player is served contribution 0, so the
    ceiling punisher cannot tell them from a genuine zero contributor and
    punishes them like one. The frontier's
    `punishment_multinomial_timeout_severity_copula` does carry that feature
    and, per PR #208's step 1, aims at one such cell in 19,200. So the
    baseline in this experiment behaves differently at those cells from the
    baseline in the rule sweep, which used the timeout punisher — a
    comparability caveat for anyone reading the two side by side.
