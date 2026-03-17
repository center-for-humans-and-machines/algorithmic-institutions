import torch as th

from aimanager.manager.environment import ArtificialHumanEnv


class MockArtificialHuman:
    def __init__(self, default_values):
        self.default_values = default_values

    def predict(self, state, reset_rnn, edge_index):
        return (th.ones((state["contribution"].shape), dtype=th.long),)


class MockArtificialHumanValid:
    def __init__(self):
        pass

    def predict(self, state, reset_rnn, edge_index):
        return (th.ones((state["contribution"].shape), dtype=th.bool),)


def test_multi_group_env():
    test_punish = th.tensor([[[1], [0], [1], [0], [1], [0]]])
    default_values = {
        "punishment": 1,
        "contribution": 1,
        "payoffs": 1,
        "contribution_valid": True,
        "common_good": 1,
        "round_number": 1,
        "player_id": 1,
    }
    artifical_humans = MockArtificialHuman(default_values=default_values)
    artifical_humans_valid = MockArtificialHumanValid()

    env = ArtificialHumanEnv(
        artifical_humans=artifical_humans,
        artifical_humans_valid=artifical_humans_valid,
        batch_size=1,
        n_agents=6,
        n_contributions=3,
        n_punishments=3,
        n_rounds=3,
        n_groups=2,
        device="cpu",
        default_values={
            "punishment": 0,
            "contribution": 0,
            "round_number": 0,
            "is_first": False,
            "contribution_valid": False,
            "punishment_valid": False,
            "common_good": 0,
            "contributor_payoff": 0,
            "manager_payoff": 0,
            "reward": 0,
        },
        reward_formula="group_payoff",
    )

    env.update_groups(th.tensor([[0, 0, 0, 1, 1, 1]]))

    env.punish(test_punish)

    old_state = env.state.copy()

    all_common_good = []
    all_reward = []
    all_contribution = []
    all_punishment = []

    all_contribution.append(env.contribution)
    all_punishment.append(env.punishment)
    all_common_good.append(env.common_good)

    # Test step
    state, reward, done = env.step()
    env.punish(test_punish)
    all_common_good.append(env.common_good)

    all_reward.append(env.reward)
    all_contribution.append(env.contribution)
    all_punishment.append(env.punishment)

    assert th.allclose(env.punishment, test_punish)
    assert th.allclose(env.prev_punishment, old_state["punishment"])
    assert th.allclose(env.prev_contribution, old_state["contribution"])
    assert th.allclose(env.round_number, th.tensor([[[1], [1], [1], [1], [1], [1]]]))
    assert th.allclose(env.agent_group, th.tensor([[[0], [0], [0], [1], [1], [1]]]))
    assert th.allclose(
        env.common_good,
        th.tensor(
            [
                [[(3 * 1.6 - th.sum(test_punish[0, :3])) / 3]] * 3
                + [[(3 * 1.6 - th.sum(test_punish[0, 3:])) / 3]] * 3,
            ]
        ),
    )
    assert th.allclose(
        env.group_payoff,
        th.tensor(
            [
                [
                    [
                        (
                            th.sum(env.common_good[0, :3])
                            - th.sum(test_punish[0, :3])
                            + 19 * 3
                        )
                        / 3
                    ],
                    [
                        (
                            th.sum(env.common_good[0, 3:])
                            - th.sum(test_punish[0, 3:])
                            + 19 * 3
                        )
                        / 3
                    ],
                ]
            ]
        ),
    )

    # Finish the game
    while not done:
        state, reward, done = env.step()
        all_reward.append(env.reward)
        if not done:
            all_common_good.append(env.common_good)
            all_contribution.append(env.contribution)
            all_punishment.append(env.punishment)
        env.punish(test_punish)

    # Test cumulative common good invariant
    all_common_good = th.cat(all_common_good, axis=-1)
    all_contribution = th.cat(all_contribution, axis=-1)
    all_punishment = th.cat(all_punishment, axis=-1)

    assert th.allclose(
        all_common_good.sum(dim=1),
        all_contribution.sum(dim=1) * 1.6 - all_punishment.sum(dim=1),
    )

    # Terminal reward is -avg_prev_punishment_per_group / 32
    avg_prev_p = env.compute_average_per_group(
        th.where(env.prev_contribution_valid, env.prev_punishment, 0)
    )
    assert th.allclose(reward, -avg_prev_p / 32)

    assert done


def test_artificial_human_env():
    # create mock artifical humans
    default_values = {
        "punishment": 1,
        "contribution": 1,
        "payoffs": 1,
        "contribution_valid": True,
        "common_good": 1,
        "round_number": 1,
        "player_id": 1,
    }

    artifical_humans = MockArtificialHuman(default_values=default_values)
    artifical_humans_valid = MockArtificialHumanValid()

    env = ArtificialHumanEnv(
        artifical_humans=artifical_humans,
        artifical_humans_valid=artifical_humans_valid,
        batch_size=2,
        n_agents=3,
        n_contributions=3,
        n_punishments=3,
        n_rounds=2,
        device="cpu",
        default_values={
            "punishment": 0,
            "contribution": 0,
            "round_number": 0,
            "is_first": False,
            "contribution_valid": False,
            "punishment_valid": False,
            "common_good": 0,
            "contributor_payoff": 0,
            "manager_payoff": 0,
            "reward": 0,
        },
    )

    # Test if the environment state is initialized correctly
    assert env.batch_size == 2
    assert env.n_agents == 3
    assert env.n_contributions == 3
    assert env.n_punishments == 3
    assert env.n_rounds == 2
    assert env.device == "cpu"
    assert env.default_values == {
        "punishment": 0,
        "contribution": 0,
        "round_number": 0,
        "is_first": False,
        "contribution_valid": False,
        "punishment_valid": False,
        "common_good": 0,
        "contributor_payoff": 0,
        "manager_payoff": 0,
        "reward": 0,
    }

    # Test if the state is initialized correctly
    assert env.state["punishment"].shape == (2, 3, 1)
    assert env.state["contribution"].shape == (2, 3, 1)
    assert env.state["round_number"].shape == (2, 3, 1)
    assert env.state["is_first"].shape == (2, 3, 1)
    assert env.state["contribution_valid"].shape == (2, 3, 1)
    assert env.state["punishment_valid"].shape == (2, 3, 1)
    assert env.state["common_good"].shape == (2, 3, 1)
    assert env.state["contributor_payoff"].shape == (2, 3, 1)
    assert env.state["reward"].shape == (2, 1, 1)
    assert env.state["agent_group"].shape == (2, 3, 1)

    # Test if the contribution works correctly
    assert th.allclose(env.contribution, th.tensor([[[1], [1], [1]], [[1], [1], [1]]]))
    assert th.allclose(
        env.contribution_valid,
        th.tensor([[[True], [True], [True]], [[True], [True], [True]]]),
    )

    # Test if the punishment works correctly
    env.punish(th.tensor([[[1], [2], [0]], [[0], [2], [1]]]))
    assert th.allclose(env.punishment, th.tensor([[[1], [2], [0]], [[0], [2], [1]]]))
    assert th.allclose(
        env.punishment_valid,
        th.tensor([[[True], [True], [True]], [[True], [True], [True]]]),
    )
    assert th.allclose(
        env.common_good.sum(), env.contribution.sum() * 1.6 - env.punishment.sum()
    )

    # Test round number
    assert th.allclose(env.round_number, th.tensor([[[0], [0], [0]], [[0], [0], [0]]]))

    # Copy the state
    old_state = env.state.copy()

    all_common_good = []
    all_reward = []
    all_contribution = []
    all_punishment = []

    all_contribution.append(env.contribution)
    all_punishment.append(env.punishment)
    all_common_good.append(env.common_good)

    # Test step
    state, reward, done = env.step()
    env.punish(th.tensor([[[1], [0], [1]], [[1], [0], [1]]]))
    all_common_good.append(env.common_good)
    all_reward.append(env.reward)
    all_contribution.append(env.contribution)
    all_punishment.append(env.punishment)
    assert th.allclose(env.punishment, th.tensor([[[1], [0], [1]], [[1], [0], [1]]]))
    assert th.allclose(env.prev_punishment, old_state["punishment"])
    assert th.allclose(env.prev_contribution, old_state["contribution"])
    assert th.allclose(env.round_number, th.tensor([[[1], [1], [1]], [[1], [1], [1]]]))

    # Test if reward is calculated correctly (same-round data, per-group)
    prev_c = env.prev_contribution.float()
    prev_p = env.prev_punishment.float()
    n = prev_c.shape[1]
    cg = (prev_c.sum(dim=1, keepdim=True) * 1.6 - prev_p.sum(dim=1, keepdim=True)) / n
    expected = (20 - prev_c - prev_p + cg).mean(dim=1, keepdim=True) / 32
    assert th.allclose(env.reward, expected)

    # Finish the game
    while not done:
        state, reward, done = env.step()
        all_reward.append(env.reward)
        if not done:
            all_common_good.append(env.common_good)
            all_contribution.append(env.contribution)
            all_punishment.append(env.punishment)
        env.punish(th.tensor([[[1], [0], [1]], [[1], [0], [1]]]))

    # Test if the game is finished
    assert done

    # Test final rewards (terminal: -avg_prev_punishment / 32)
    expected_terminal = -env.prev_punishment.float().mean(dim=1, keepdim=True) / 32
    assert th.allclose(reward, expected_terminal)

    # Test common good invariant
    all_common_good = th.cat(all_common_good, axis=-1)
    all_contribution = th.cat(all_contribution, axis=-1)
    all_punishment = th.cat(all_punishment, axis=-1)

    assert th.allclose(
        all_common_good.sum(dim=1),
        all_contribution.sum(dim=1) * 1.6 - all_punishment.sum(dim=1),
    )


def _make_env(reward_formula, n_rounds=3):
    """Helper to create a single-group env for reward formula tests."""
    default_values = {
        "punishment": 1,
        "contribution": 1,
        "payoffs": 1,
        "contribution_valid": True,
        "common_good": 1,
        "round_number": 1,
        "player_id": 1,
    }
    return ArtificialHumanEnv(
        artifical_humans=MockArtificialHuman(default_values),
        artifical_humans_valid=MockArtificialHumanValid(),
        batch_size=1,
        n_agents=3,
        n_contributions=3,
        n_punishments=3,
        n_rounds=n_rounds,
        device="cpu",
        default_values={
            "punishment": 0,
            "contribution": 0,
            "round_number": 0,
            "is_first": False,
            "contribution_valid": False,
            "punishment_valid": False,
            "common_good": 0,
            "contributor_payoff": 0,
            "manager_payoff": 0,
            "reward": 0,
        },
        reward_formula=reward_formula,
    )


def test_group_payoff_round_non_terminal():
    """group_payoff_round reward equals group_payoff computed during punish."""
    env = _make_env("group_payoff_round", n_rounds=3)
    punishment = th.tensor([[[1], [0], [2]]])

    env.punish(punishment)
    group_payoff_after_punish = env.group_payoff.clone()

    state, reward, done = env.step()
    assert not done
    # Reward must equal group_payoff computed during punish()
    assert th.allclose(reward, group_payoff_after_punish)


def test_group_payoff_round_terminal():
    """group_payoff_round terminal reward uses group_payoff, not punishment."""
    env = _make_env("group_payoff_round", n_rounds=2)
    punishment = th.tensor([[[1], [0], [2]]])

    # Round 0
    env.punish(punishment)
    state, reward, done = env.step()
    assert not done

    # Round 1 (terminal)
    env.punish(punishment)
    group_payoff_after_punish = env.group_payoff.clone()

    state, reward, done = env.step()
    assert done
    # Terminal reward should be group_payoff, NOT -avg_punishment/32
    assert th.allclose(reward, group_payoff_after_punish)


def test_group_payoff_uses_same_round_data():
    """group_payoff reward uses prev_* (same-round), not next-round contrib."""
    env = _make_env("group_payoff", n_rounds=3)
    punishment = th.tensor([[[1], [0], [2]]])

    env.punish(punishment)
    # Record same-round data before step advances it
    contrib = env.contribution.clone()
    punish = env.punishment.clone()
    valid = env.contribution_valid.clone()

    state, reward, done = env.step()
    assert not done

    # After step, prev_* should hold the same-round values
    assert th.allclose(env.prev_contribution, contrib)
    assert th.allclose(env.prev_punishment, punish)
    assert th.allclose(env.prev_contribution_valid, valid)

    # Manually compute expected reward from same-round data
    # Common good is a group aggregate: (sum_c * 1.6 - sum_p) / n_valid
    n_valid = valid.sum(dim=1, keepdim=True).float()
    sum_c = (contrib * valid).sum(dim=1, keepdim=True).float()
    sum_p = (punish * valid).sum(dim=1, keepdim=True).float()
    common_good = (sum_c * 1.6 - sum_p) / n_valid
    # Average payoff per group: mean over agents of (20 - c - p + cg)
    payoff_per_agent = 20 - contrib.float() - punish.float() + common_good
    expected_reward = payoff_per_agent.mean(dim=1, keepdim=True) / 32
    assert th.allclose(reward, expected_reward, atol=1e-5)


def test_group_payoff_terminal():
    """group_payoff terminal reward uses -avg_punishment/32."""
    env = _make_env("group_payoff", n_rounds=2)
    punishment = th.tensor([[[1], [0], [2]]])

    # Round 0
    env.punish(punishment)
    state, reward, done = env.step()
    assert not done

    # Round 1 (terminal)
    env.punish(punishment)
    state, reward, done = env.step()
    assert done
    expected = -punishment.float().mean() / 32
    assert th.allclose(reward[0, 0, 0], expected)
