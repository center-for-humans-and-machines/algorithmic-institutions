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

## Status: setup done and priced; the three runs are NOT started

Only the cost pilot has run. The three real configs carry a
`NOT YET RUN -- BLOCKED` banner and are held on two things:

1. **The free-punishment defect (review D1) is unfixed.** Under a common-pool
   reward this stops being a curiosity: punishment costs the manager
   everywhere *except* on a timed-out cell, so a free lever strictly dominates
   a paid one and a value-maximising learner has every reason to find it. That
   is the exact failure mode that would invalidate the headline question. Fix
   it first, or start with the diagnostic — measurement-plan item 2 is a
   required output either way.
2. **The common-pool reward mode is not on this branch yet.**
   `auto/manager-common-pool-reward` was not on the remote at the time of
   writing. `REWARD_MODE` in the generator is one line.

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
