"""Emit the RL-manager training configs for the new-clones experiment.

Originally a two-arm comparison; arm OLD was dropped by the maintainer as too
hard to replicate faithfully (its switch artifact carries the pre-#123
anchoring, see notes/autoresearch_log/rl-manager-two-worlds.md note 7). What
remains is one world -- the current corrected clones -- and one question: does
a manager trained against them punish at all, what does its policy look like,
and does it beat the artificial punisher.

Two arms are emitted, in the same world and over the same three seeds, and
they differ in `reward_mode` and nothing else:

    POOL       rl_new_clones_s{42,43,44}            reward_mode: common_pool
    PER-CAPITA rl_new_clones_percapita_s{42,43,44}  reward_mode:
                                                    common_pool_per_capita

Within an arm the runs differ only in `seed`, and between the arms only in
`reward_mode`, `job_id` and `output_dir`. Both claims are mechanical because
the configs come out of this file rather than being hand-edited -- which is
the whole value of the per-capita arm, since it exists to be a clean control
on the reward and on nothing else.

    python scripts/rl_two_worlds/make_configs.py
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT_DIR = os.path.join(ROOT, "configs/training/rl_manager")

# 42-44 are the three finished runs. 45 and 46 were added by the exploration
# comparison (notes/autoresearch_log/rl-manager-annealed-local.md) so that the
# unmodified behaviour policy has a five-seed control that pairs seed for seed
# with every arm. Regenerating leaves 42-44 byte-identical.
SEEDS = (42, 43, 44, 45, 46)

# The current frontier stack: the contribution trunk with the per-group virtual
# node, the direct stimulus skip and the stamped herding copula; the
# joint-exodus switch model; the multinomial punisher with the ceiling
# indicator and its severity copula, conditioned on round t's contribution.
# Paths read off the sim configs, not guessed.
CONTRIBUTION_MODEL = (
    "artifacts/artificial_humans/"
    "group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/"
    "model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
)
SWITCH_MODEL = (
    "artifacts/artificial_humans/switch_joint_exodus/"
    "model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt"
)
# Also the baseline the learned policies are scored against, not only the
# opponent: it is this project's clone of a human manager.
OPPONENT_PUNISHER = (
    "artifacts/baselines/punishment_multinomial_ceiling_severity_copula.joblib"
)
# Plumbing, not a slot.
VALID_MODEL = "artifacts/artificial_humans/raven_script_22/model/rnn_False__dataset_full.pt"  # noqa: E501

# The real runs must train on the common pool -- what a real manager was
# actually paid -- rather than on `sum` (the sum of contributor payoffs, which
# prices punishment ~5x too high and is ~63% headcount by variance; review
# S1). The mode is being added on auto/manager-common-pool-reward and is not on
# this branch yet, so `sum` stands here only so the cost pilot can run: wall
# clock does not depend on the reward. Flip this one constant when that branch
# lands, and regenerate.
REWARD_MODE = "common_pool"

# The control arm's mode. Same pool, divided by the number of players in the
# group who gave an input -- the share one member actually receives. See
# PER_CAPITA_BANNER for why, and `REWARD_MODES` in
# src/aimanager/manager/environment.py for the divisor.
REWARD_MODE_PER_CAPITA = "common_pool_per_capita"

BLOCKED_BANNER = """\
# Cleared to run. Both pre-launch guards pass on the merged tree
# (scripts/rl_two_worlds/launch_guards.py; evidence in
# plots/data_analysis/evaluation/rl_manager_two_worlds/guards_after_fix.json):
#   * the reward is the common pool -- env.reward equals 1.6*sum(c) - sum(p)
#     to 0.0, agrees with common_good * n_valid through a different code path,
#     and differs from reward_mode='sum' by up to 240, which is what shows the
#     mode really changed rather than the constant merely being renamed;
#   * the free punishment lever is closed -- with the manager forced to punish
#     the maximum on every cell, all 1,659 timed-out cells are served
#     punishment 0 and all 627 previously-timed-out cells prev_punishment 0.
#     Before auto/free-punishment-fix the same run served 194 of them a 30.
# See notes/autoresearch_log/rl-manager-two-worlds.md."""

PER_CAPITA_BANNER = """\
# THE CONTROL ARM on the reward, and nothing else. Same world, same seeds,
# same hyperparameters as rl_new_clones_s{{42,43,44}} (jobs 30401560/61/62);
# `reward_mode`, `job_id` and `output_dir` are the only keys that differ.
#
# Those runs pay the manager its own group's UNDIVIDED common pool,
# 1.6*sum(c) - sum(p). That is faithful to the experiment but it pays for
# headcount: a bigger group is a bigger pool. All three seeds were learning
# to punish less, gain members and grow the pool while contribution per
# member stayed flat, so it cannot be told whether the manager learned
# something about cooperation or simply learned to collect people.
#
# This arm pays the PER-CAPITA share of that same pool instead:
#
#     reward = (1.6*sum(c) - sum(p)) / n_valid
#
# where the divisor `n_valid` is `count_valid_per_group` -- the number of
# players in the group who gave an input this round, NOT the group's
# membership. The two differ whenever a member times out. That divisor is
# the game's own rule (reports/basics.md: the pool "is splitted equally
# between the contributors"; `payoff = 20 - c - p + pool/n_valid` reproduces
# the human payoff column to 7.1e-15 over 19,166 rows), and it comes from
# the env's single `share_pool_per_group`, the same divisor the `common_good`
# state field is built from. A timed-out member contributed 0, was punished 0
# and took no share, so they move neither numerator nor divisor and the
# reward does not see them; dividing by the membership instead would invent a
# headcount penalty the game never charged.
#
# To first order this is headcount-neutral: a member contributing at the
# group average adds about as much to the numerator as to the divisor. If the
# punish-less behaviour survives the change it is about cooperation. If it
# disappears it was about collecting people.
#
# Cleared to run. All three pre-launch guards pass
# (scripts/rl_two_worlds/launch_guards.py; evidence in
# plots/data_analysis/evaluation/rl_manager_percapita/guards.json):
#   * the free punishment lever is closed -- with the manager forced to
#     punish the maximum on every cell, all {timeout_cells} timed-out cells
#     are served punishment 0 and all {prev_timeout_cells} previously
#     timed-out cells prev_punishment 0;
#   * the reward is exactly that pool over that divisor -- max residual
#     {g1_own} against this file's own division of the env's state tensors,
#     and {g1_cg} against the `common_good` state field, which
#     `update_common_good` produces and the reward path never touches, so
#     the agreement is a confirmation and not a restatement;
#   * it was divided, not relabelled -- on the identical trajectory it
#     differs from reward_mode='common_pool' by up to {gap} points
#     ({gap_mean} on average), `reward * n_valid` returns the pool to
#     {inverse}, and on a controlled rollout where every group holds
#     {headcount} members and nobody times out the ratio
#     common_pool / per_capita is the single value {headcount}.0. A renamed
#     constant would have given 1.
# See notes/autoresearch_log/rl-manager-percapita-reward.md."""

# Read off the committed guard evidence,
# plots/data_analysis/evaluation/rl_manager_percapita/guards.json, and quoted
# in PER_CAPITA_BANNER so the config states what was actually measured rather
# than what was hoped for. Regenerate after any guard re-run.
GUARD_EVIDENCE = {
    "timeout_cells": "1,659",
    "prev_timeout_cells": "627",
    "g1_own": "0.0, exact",
    "g1_cg": "1.9e-06, float32",
    "gap": "224.0",
    "gap_mean": "46.4",
    "inverse": "7.6e-06",
    "headcount": "4",
}

PILOT_BANNER = """\
# COST PILOT -- a wall-clock measurement, not a result. 40 update steps, run
# only to price a full 4000-step run before committing three A100-days. Its
# reward mode and the unfixed free-punishment defect do not affect timing, and
# nothing it produces is science. Do not read its policy."""

TEMPLATE = """\
# AUTOGENERATED by scripts/rl_two_worlds/make_configs.py -- do not hand-edit.
#
{banner}
#
# RL manager vs the current corrected clones ({job_id}).
# Every key below except `seed`, `job_id` and `output_dir` is byte-identical
# across the three runs, and the hyperparameters are those of
# configs/training/rl_manager/03_2g8a_sum.yml -- nothing is tuned.

artificial_humans: {contribution_model}
artificial_humans_valid: {valid_model}
switch_model: {switch_model}
opponent_manager: {opponent_punisher}
artificial_humans_model: "graph"
job_id: {job_id}
seed: {seed}

manager_args:
  opt_args:
      lr: 2.e-4
  gamma: 0.98
  eps: 0.1
  target_update_freq: 1000
  model_args:
      hidden_size: 100
      add_rnn: true
      add_edge_model: true
      add_global_model: False
      x_encoding:
        - name: contribution
          n_levels: 21
          encoding: numeric
        - name: prev_punishment
          n_levels: 31
          encoding: numeric
        - etype: bool
          name: contribution_valid
        - etype: bool
          name: in_group
      b_encoding:
        - name: round_number
          n_levels: 32
          encoding: onehot

replay_memory_args:
  n_episodes: 100

n_update_steps: {n_update_steps}
eval_period: {eval_period}
training_batch_size: 1

env_args:
  n_groups: 2
  n_agents: 8
  agent_groups: [0, 0, 0, 0, 1, 1, 1, 1]
  rl_group_id: 0
  switch_every: 4
  n_contributions: 21
  n_punishments: 31
  n_rounds: 24
  batch_size: 1000
  reward_mode: {reward_mode}

device: "cuda"
output_dir: "artifacts/manager/{job_id}"
basedir: "."
"""


def write(job_id, seed, n_update_steps, eval_period, banner, reward_mode):
    text = TEMPLATE.format(
        banner=banner,
        job_id=job_id,
        contribution_model=CONTRIBUTION_MODEL,
        switch_model=SWITCH_MODEL,
        opponent_punisher=OPPONENT_PUNISHER,
        valid_model=VALID_MODEL,
        seed=seed,
        n_update_steps=n_update_steps,
        eval_period=eval_period,
        reward_mode=reward_mode,
    )
    path = os.path.join(OUT_DIR, f"{job_id}.yml")
    with open(path, "w") as f:
        f.write(text)
    print(path)


def main():
    for seed in SEEDS:
        write(f"rl_new_clones_s{seed}", seed, 4000, 20, BLOCKED_BANNER, REWARD_MODE)
    write("rl_new_clones_pilot", 42, 40, 10, PILOT_BANNER, REWARD_MODE)
    for seed in SEEDS:
        write(
            f"rl_new_clones_percapita_s{seed}",
            seed,
            4000,
            20,
            PER_CAPITA_BANNER.format(**GUARD_EVIDENCE),
            REWARD_MODE_PER_CAPITA,
        )


if __name__ == "__main__":
    main()
