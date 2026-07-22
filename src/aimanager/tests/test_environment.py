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
        reward_mode="avg",
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
    last_group_payoff = None
    while not done:
        last_group_payoff = env.group_payoff.clone()
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

    # Terminal reward equals group_payoff from the last punish()
    assert th.allclose(reward, last_group_payoff)

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
        reward_mode="avg",
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
    group_payoff_before_step = env.group_payoff.clone()
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

    # Reward equals group_payoff computed during the preceding punish()
    assert th.allclose(reward, group_payoff_before_step)

    # Finish the game
    last_group_payoff = None
    while not done:
        last_group_payoff = env.group_payoff.clone()
        state, reward, done = env.step()
        all_reward.append(env.reward)
        if not done:
            all_common_good.append(env.common_good)
            all_contribution.append(env.contribution)
            all_punishment.append(env.punishment)
        env.punish(th.tensor([[[1], [0], [1]], [[1], [0], [1]]]))

    # Test if the game is finished
    assert done

    # Terminal reward equals group_payoff from the last punish()
    assert th.allclose(reward, last_group_payoff)

    # Test common good invariant
    all_common_good = th.cat(all_common_good, axis=-1)
    all_contribution = th.cat(all_contribution, axis=-1)
    all_punishment = th.cat(all_punishment, axis=-1)

    assert th.allclose(
        all_common_good.sum(dim=1),
        all_contribution.sum(dim=1) * 1.6 - all_punishment.sum(dim=1),
    )


def _make_env(n_rounds=3):
    """Helper to create a single-group env for reward tests."""
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
        reward_mode="avg",
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


def test_reward_non_terminal():
    """Reward equals group_payoff computed during punish."""
    env = _make_env(n_rounds=3)
    punishment = th.tensor([[[1], [0], [2]]])

    env.punish(punishment)
    group_payoff_after_punish = env.group_payoff.clone()

    state, reward, done = env.step()
    assert not done
    # Reward must equal group_payoff computed during punish()
    assert th.allclose(reward, group_payoff_after_punish)


def test_reward_terminal():
    """Terminal reward uses group_payoff."""
    env = _make_env(n_rounds=2)
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


def _make_two_group_env(reward_mode):
    """6-agent env with asymmetric groups [4, 2] so sum != avg."""
    default_values = {
        "punishment": 1,
        "contribution": 1,
        "payoffs": 1,
        "contribution_valid": True,
        "common_good": 1,
        "round_number": 1,
        "player_id": 1,
    }
    env = ArtificialHumanEnv(
        artifical_humans=MockArtificialHuman(default_values),
        artifical_humans_valid=MockArtificialHumanValid(),
        batch_size=1,
        n_agents=6,
        n_contributions=3,
        n_punishments=3,
        n_rounds=2,
        n_groups=2,
        device="cpu",
        agent_groups=[0, 0, 0, 0, 1, 1],
        reward_mode=reward_mode,
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
    return env


def test_reward_mode_sum_vs_avg():
    """sum and avg paths log both fields; reward selects per mode."""
    punishment = th.tensor([[[1], [0], [2], [1], [0], [2]]])

    # group 0 (agents 0..3): 4 agents, contribution=1 each, punish=[1,0,2,1]
    # group 1 (agents 4..5): 2 agents, contribution=1 each, punish=[0,2]
    # common_good_g0 = (4*1.6 - 4) / 4 = 0.6
    # common_good_g1 = (2*1.6 - 2) / 2 = 0.6
    # contributor_payoff_i = 19 - p_i + cg
    # group_0 payoffs: [18.6, 19.6, 17.6, 18.6] -> sum 74.4, avg 18.6
    # group_1 payoffs: [19.6, 17.6]             -> sum 37.2, avg 18.6
    expected_sum = th.tensor([[[74.4], [37.2]]])
    expected_avg = th.tensor([[[18.6], [18.6]]])

    for mode, expected_reward in (("avg", expected_avg), ("sum", expected_sum)):
        env = _make_two_group_env(reward_mode=mode)
        env.punish(punishment)

        # Both fields logged regardless of mode
        assert th.allclose(
            env.group_payoff, expected_avg
        ), f"avg field mismatch in mode={mode}"
        assert th.allclose(
            env.group_payoff_sum, expected_sum
        ), f"sum field mismatch in mode={mode}"

        # Reward picks the right field
        _, reward, _ = env.step()
        assert th.allclose(
            reward, expected_reward
        ), f"reward in mode={mode} expected {expected_reward}, got {reward}"


def test_reward_mode_invalid_raises():
    import pytest

    with pytest.raises(ValueError, match="reward_mode"):
        _make_two_group_env(reward_mode="median")


class _RecordingSwitch:
    """Switch predictor stub: records what it sees per call, flips agent 0."""

    def __init__(self):
        self.calls = []

    def predict(self, state, reset_rnn, edge_index):
        self.calls.append(
            {
                "round": int(state["round_number"][0, 0, 0]),
                "pun": int(state["punishment"][0, 0, 0]),
                "group0": int(state["agent_group"][0, 0, 0]),
                "reset": bool(reset_rnn),
            }
        )
        ds = th.zeros_like(state["contribution"], dtype=th.bool)
        ds[:, 0, :] = True
        return ds, None


def test_switch_predictor_end_of_round_anchoring():
    """#123: the switch predictor is queried at the END of round s -- after
    punish(), with round-s outcomes in the CURRENT state keys and membership
    still pre-decision -- and the change applies at s+1. It runs every round
    (RNN warm-up); only decision-round outputs flip groups, and the episode's
    last round is never a decision."""
    default_values = {
        "punishment": 1,
        "contribution": 1,
        "payoffs": 1,
        "contribution_valid": True,
        "common_good": 1,
        "round_number": 1,
        "player_id": 1,
    }
    switch = _RecordingSwitch()
    env = ArtificialHumanEnv(
        artifical_humans=MockArtificialHuman(default_values),
        artifical_humans_valid=MockArtificialHumanValid(),
        artifical_humans_switch=switch,
        switch_every=2,
        batch_size=1,
        n_agents=4,
        n_contributions=3,
        n_punishments=3,
        n_rounds=6,
        n_groups=2,
        device="cpu",
        agent_groups=[0, 0, 1, 1],
        reward_mode="avg",
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

    group_of_a0 = [int(env.state["agent_group"][0, 0, 0])]
    done = False
    r = 0
    while not done:
        env.punish(th.full((1, 4, 1), r % 3, dtype=th.int64))
        _, _, done = env.step()
        if not done:
            group_of_a0.append(int(env.state["agent_group"][0, 0, 0]))
        r += 1

    # one call at the end of every round; RNN reset only on the first
    assert [c["round"] for c in switch.calls] == [0, 1, 2, 3, 4, 5]
    assert [c["reset"] for c in switch.calls] == [True] + [False] * 5
    # each call saw the JUST-PLAYED round's punishment (current key, not prev)
    assert [c["pun"] for c in switch.calls] == [r % 3 for r in range(6)]
    # and the pre-decision membership of that round
    assert [c["group0"] for c in switch.calls] == group_of_a0
    # decisions at s=1,3 -> agent 0 flips at arrivals 2 and 4; s=5 is the
    # last round (no round 6), so its output is never applied
    assert group_of_a0 == [0, 0, 1, 1, 0, 0]
