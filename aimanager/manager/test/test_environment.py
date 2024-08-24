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
    assert env.state["manager_payoff"].shape == (2, 3, 1)
    assert env.state["reward"].shape == (2, 3, 1)
    assert env.state["group"].shape == (2, 3, 1)

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

    # Test if reward is calculated correctly
    assert th.allclose(env.reward * 32, env.contribution * 1.6 - env.prev_punishment)

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

    # Test final rewards
    assert th.allclose(reward * 32, -env.punishment.float())

    # Test cummulative rewards
    all_reward = th.cat(all_reward, axis=-1)
    all_common_good = th.cat(all_common_good, axis=-1)
    all_contribution = th.cat(all_contribution, axis=-1)
    all_punishment = th.cat(all_punishment, axis=-1)

    assert th.allclose(
        all_reward[:, :, :-1] * 32,
        all_contribution[:, :, 1:] * 1.6 - all_punishment[:, :, :-1],
    )

    assert th.allclose(
        all_common_good.sum(dim=1),
        all_contribution.sum(dim=1) * 1.6 - all_punishment.sum(dim=1),
    )

    assert th.allclose(
        all_reward[:, :, -1] * 32,
        -all_punishment[:, :, -1].float(),
    )

    # the first contribution do not enter into the reward, as the
    # the manager has not influence on those

    assert th.allclose(
        all_common_good.sum() - all_contribution[:, :, 0].sum() * 1.6,
        all_reward.sum() * 32,
    )
