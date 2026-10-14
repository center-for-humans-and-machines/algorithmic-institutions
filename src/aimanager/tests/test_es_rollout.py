"""Rollout-level guarantees of the evolution-strategies arm, against the real
environment and the real artifacts.

Three claims the arm rests on, and none of them is safe to assume:

  1. the evaluation this arm runs is the evaluation the other three arms run
     -- a single-member population rollout must reproduce
     `rl_manager.run_batch(..., on_policy=True)` value for value;
  2. the evaluation scores the MEAN parameter vector, not a population
     member, even while the members hold perturbed weights;
  3. a generation reproduces exactly from its seed.

Raven only, and slow: the artificial humans are real GNNs. Everything runs on
the CPU over a handful of episodes, which is enough because all three claims
are exact rather than statistical.
"""

import os
import random

import numpy as np
import pytest
import torch as th
import yaml

from aimanager.artificial_humans import AH_MODELS
from aimanager.es_manager import (
    PopulationRollout,
    build_members,
    flat_params,
    set_flat_params_,
)
from aimanager.manager.environment import ArtificialHumanEnv
from aimanager.manager.linear_opponent import load_opponent
from aimanager.manager.manager import ArtificalManager
from aimanager.rl_manager import run_batch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
CONFIG = os.path.join(ROOT, "configs/training/rl_manager/rl_es_pilot.yml")
SHARED_METRICS = [
    "punishment",
    "opp_punishment",
    "contribution",
    "common_good",
    "contributor_payoff",
    "group_payoff",
    "group_payoff_sum",
    "opp_sum_payoff",
    "next_reward",
    "rl_end_group_size",
    "opp_end_group_size",
]


def seed_all(seed):
    th.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@pytest.fixture(scope="module")
def world():
    """The real env and the real opponent, tiny: 4 episodes, 4 rounds."""
    if not os.path.exists(CONFIG):
        pytest.skip("ES pilot config missing")
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    basedir = os.path.join(ROOT, cfg["basedir"])
    for key in (
        "artificial_humans",
        "artificial_humans_valid",
        "switch_model",
        "opponent_manager",
    ):
        if not os.path.exists(os.path.join(basedir, cfg[key])):
            pytest.skip(f"artifact missing: {cfg[key]}")

    device = th.device("cpu")
    kind = AH_MODELS[cfg["artificial_humans_model"]]

    def load(key):
        return kind.load(os.path.join(basedir, cfg[key]), device=device).to(device)

    env_args = dict(cfg["env_args"])
    rl_group_id = env_args.pop("rl_group_id")
    env_args.update(batch_size=4, n_rounds=4)
    seed_all(0)
    env = ArtificialHumanEnv(
        artifical_humans=load("artificial_humans"),
        artifical_humans_valid=load("artificial_humans_valid"),
        artifical_humans_switch=load("switch_model"),
        device=device,
        **env_args,
    )
    opponent = load_opponent(
        os.path.join(basedir, cfg["opponent_manager"]),
        n_groups=env.n_groups,
        device=device,
    ).to(device)
    return cfg, env, opponent, rl_group_id, device


def make_manager(cfg, env, device, with_optimiser=True):
    ma = cfg["manager_args"]
    extra = (
        dict(
            model_args=ma["model_args"],
            opt_args=ma["opt_args"],
            gamma=ma["gamma"],
            target_update_freq=ma["target_update_freq"],
            eps=ma["eps"],
        )
        if with_optimiser
        else {}
    )
    return ArtificalManager(
        n_contributions=env.n_contributions,
        n_punishments=env.n_punishments,
        n_groups=env.n_groups,
        default_values=env.default_values,
        device=device,
        **extra,
    )


def test_single_member_rollout_reproduces_run_batch(world):
    """Claim 1: this arm's evaluation is the other arms' evaluation."""
    cfg, env, opponent, rl_group_id, device = world
    seed_all(7)
    manager = make_manager(cfg, env, device)
    manager.policy_model.eval()
    theta = flat_params(manager.policy_model).clone()

    seed_all(99)
    reference = run_batch(
        manager,
        env,
        replay_mem=None,
        on_policy=True,
        update_step=3,
        opponent_manager=opponent,
        rl_group_id=rl_group_id,
    )
    assert th.equal(flat_params(manager.policy_model), theta)

    rollout = PopulationRollout(env, opponent, rl_group_id, device)
    seed_all(99)
    got, episode_return, shape = rollout.run(
        [manager], update_step=3, sampling="greedy", collect_shape=True
    )

    assert len(got) == len(reference) == env.n_rounds
    for g, r in zip(got, reference):
        assert g["round_number"] == r["round_number"]
        assert g["update_step"] == r["update_step"] == 3
        assert g["sampling"] == r["sampling"] == "greedy"
        for k in SHARED_METRICS:
            assert g[k] == pytest.approx(r[k], abs=0, rel=0), k
    # The episode return is the undiscounted sum of the per-round rewards the
    # same rollout reported.
    assert float(episode_return.mean()) == pytest.approx(
        sum(r["next_reward"] for r in reference), rel=1e-5
    )
    # and the shape read covers the RL group's agent-rounds, no more
    total = sum(r["n"] for r in shape.rows("greedy", 3, False) if r["subset"] == "all")
    assert 0 < total <= env.batch_size * env.n_agents * env.n_rounds


def test_the_evaluation_scores_theta_not_a_population_member(world):
    """Claim 2: perturbed members never leak into the evaluation rollout."""
    cfg, env, opponent, rl_group_id, device = world
    seed_all(7)
    base = make_manager(cfg, env, device)
    base.policy_model.eval()
    theta = flat_params(base.policy_model).clone()

    members = build_members(
        base.policy_model,
        dict(
            n_contributions=env.n_contributions,
            n_punishments=env.n_punishments,
            n_groups=env.n_groups,
            default_values=env.default_values,
        ),
        2,
        device,
    )
    # A perturbation large enough to move the argmax on every cell.
    for i, m in enumerate(members):
        set_flat_params_(m.policy_model, theta + (3.0 if i == 0 else -3.0))

    rollout = PopulationRollout(env, opponent, rl_group_id, device)
    seed_all(5)
    behaviour, _, _ = rollout.run(members, 0, "es-population")
    seed_all(5)
    evaluation, _, _ = rollout.run([base], 0, "greedy")

    assert th.equal(flat_params(base.policy_model), theta)
    mean_behaviour = np.mean([m["punishment"] for m in behaviour])
    mean_eval = np.mean([m["punishment"] for m in evaluation])
    assert mean_behaviour != mean_eval


def test_a_generation_reproduces_from_its_seed(world):
    """Claim 3."""
    cfg, env, opponent, rl_group_id, device = world
    seed_all(7)
    base = make_manager(cfg, env, device)
    base.policy_model.eval()
    kwargs = dict(
        n_contributions=env.n_contributions,
        n_punishments=env.n_punishments,
        n_groups=env.n_groups,
        default_values=env.default_values,
    )
    rollout = PopulationRollout(env, opponent, rl_group_id, device)

    def generation(seed):
        members = build_members(base.policy_model, kwargs, 2, device)
        g = th.Generator(device=device).manual_seed(1234)
        eps = th.randn((2, flat_params(base.policy_model).numel()), generator=g)
        theta = flat_params(base.policy_model)
        for p, m in enumerate(members):
            set_flat_params_(m.policy_model, theta + 0.02 * eps[p])
        seed_all(seed)
        metrics, returns, _ = rollout.run(members, 0, "es-population")
        return metrics, returns

    a_m, a_r = generation(21)
    b_m, b_r = generation(21)
    c_m, c_r = generation(22)
    assert th.equal(a_r, b_r)
    for x, y in zip(a_m, b_m):
        assert x == y
    assert not th.equal(a_r, c_r)


def test_the_population_partitions_the_batch_exactly(world):
    cfg, env, opponent, rl_group_id, device = world
    seed_all(7)
    base = make_manager(cfg, env, device)
    rollout = PopulationRollout(env, opponent, rl_group_id, device)
    members = build_members(
        base.policy_model,
        dict(
            n_contributions=env.n_contributions,
            n_punishments=env.n_punishments,
            n_groups=env.n_groups,
            default_values=env.default_values,
        ),
        3,
        device,
    )
    with pytest.raises(AssertionError, match="divisible"):
        rollout.run(members, 0, "es-population")
