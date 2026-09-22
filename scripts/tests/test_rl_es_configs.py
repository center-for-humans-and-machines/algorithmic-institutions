"""The evolution-strategies arm's configs must differ from the DQN reference
in exactly one thing: how the behaviour policy is produced.

Runs locally -- yaml only, no torch.
"""

import os

import pytest
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CONFIG_DIR = os.path.join(ROOT, "configs/training/rl_manager")
REFERENCE = os.path.join(CONFIG_DIR, "rl_new_clones_s42.yml")
SEEDS = (42, 43, 44, 45, 46)
SHARED_KEYS = (
    "artificial_humans",
    "artificial_humans_valid",
    "switch_model",
    "opponent_manager",
    "artificial_humans_model",
    "manager_args",
    "env_args",
    "device",
    "basedir",
)


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def reference():
    return load(REFERENCE)


@pytest.fixture(scope="module")
def es_configs():
    return {
        seed: load(os.path.join(CONFIG_DIR, f"rl_es_s{seed}.yml")) for seed in SEEDS
    }


def test_all_five_seeds_exist(es_configs):
    assert sorted(es_configs) == list(SEEDS)


def test_models_env_and_architecture_are_the_reference_config_verbatim(
    reference, es_configs
):
    for seed, cfg in es_configs.items():
        for key in SHARED_KEYS:
            assert cfg[key] == reference[key], f"seed {seed}, key {key}"


def test_the_reward_is_the_common_pool(es_configs):
    for cfg in es_configs.values():
        assert cfg["env_args"]["reward_mode"] == "common_pool"


def test_env_matches_the_contract(es_configs):
    want = dict(
        n_groups=2,
        n_agents=8,
        agent_groups=[0, 0, 0, 0, 1, 1, 1, 1],
        rl_group_id=0,
        switch_every=4,
        n_contributions=21,
        n_punishments=31,
        n_rounds=24,
        batch_size=1000,
    )
    for cfg in es_configs.values():
        for key, value in want.items():
            assert cfg["env_args"][key] == value, key


def test_only_the_seed_the_job_id_and_the_output_dir_vary(es_configs):
    varying = {"seed", "job_id", "output_dir"}
    base = es_configs[42]
    for seed, cfg in es_configs.items():
        assert set(cfg) == set(base)
        for key in cfg:
            if key in varying:
                continue
            assert cfg[key] == base[key], f"seed {seed}, key {key}"
        assert cfg["seed"] == seed
        assert cfg["job_id"] == f"rl_es_s{seed}"


def test_the_episode_budget_matches_the_dqn_arm(reference, es_configs):
    """Equal environment episodes, not update steps.

    One generation scores the whole population inside one env rollout of
    `batch_size` episodes, which is what one DQN update step consumes, so the
    budgets match exactly when `n_generations == n_update_steps`.
    """
    for cfg in es_configs.values():
        assert cfg["n_generations"] == reference["n_update_steps"]
        assert cfg["env_args"]["batch_size"] == reference["env_args"]["batch_size"]
        assert cfg["eval_period"] == reference["eval_period"]


def test_the_population_partitions_the_batch(es_configs):
    for cfg in es_configs.values():
        population = cfg["es_args"]["population_size"]
        assert cfg["env_args"]["batch_size"] % population == 0
        # mirrored sampling needs pairs
        assert population % 2 == 0


def test_fitness_shaping_is_on(es_configs):
    """Rank shaping is load-bearing; turning it off is a decision, not a
    default, so the config has to say so out loud."""
    for cfg in es_configs.values():
        assert cfg["es_args"]["fitness_shaping"] == "centered_rank"
        assert cfg["es_args"]["mirrored"] is True


def test_the_pilot_is_short_and_not_a_seed_run(es_configs):
    pilot = load(os.path.join(CONFIG_DIR, "rl_es_pilot.yml"))
    assert pilot["n_generations"] < 100
    assert pilot["job_id"] == "rl_es_pilot"
    assert pilot["job_id"] not in {c["job_id"] for c in es_configs.values()}
