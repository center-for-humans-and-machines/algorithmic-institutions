"""A punishment aimed at a timed-out player is charged as 0, as in the game.

Runs on Raven (the env needs a working torch; one test imports PyG)."""

import numpy as np
import torch as th

from aimanager.manager.environment import ArtificialHumanEnv

AGENT_GROUP = [0, 0, 0, 0, 1, 1, 1, 1]
DEFAULTS = {
    "punishment": 0,
    "contribution": 9,
    "contribution_valid": False,
    "punishment_valid": False,
    "recorded": False,
    "common_good": 12.333333333333334,
    "agent_group": 0,
    "does_switch": False,
    "own_grp_prev_mean_contr": 9,
}
# agent 0 times out, agent 1 genuinely contributes 0, so a cell the manager
# may punish sits next to one it may not in every assertion
C_ROUND = [7, 0, 20, 3, 11, 5, 14, 2]
VALID = [False] + [True] * 7
N_ROUNDS = 4
FREE = 30  # what the manager aims at the timed-out player
PAID = 5  # what it aims at everyone else, the genuine zero included


class _Fixed:
    """Artificial human stand-in returning the same draw every round."""

    autoregressive = False
    default_values = DEFAULTS

    def __init__(self, row, dtype=th.int64):
        self.row, self.dtype = row, dtype

    def predict(self, state, **_):
        return th.tensor(self.row, dtype=self.dtype).reshape(1, 8, 1), None


def _env():
    return ArtificialHumanEnv(
        artifical_humans=_Fixed(C_ROUND),
        artifical_humans_valid=_Fixed(VALID, dtype=th.bool),
        artifical_humans_switch=None,
        switch_every=4,
        batch_size=1,
        n_agents=8,
        n_contributions=21,
        n_punishments=31,
        n_rounds=N_ROUNDS,
        device=th.device("cpu"),
        n_groups=2,
        agent_groups=AGENT_GROUP,
        default_values=DEFAULTS,
    )


def _action(aimed_at_timeout=FREE):
    a = th.full((1, 8, 1), PAID, dtype=th.int64)
    a[0, 0, 0] = aimed_at_timeout
    return a


def _col(state, key):
    return state[key].reshape(-1).numpy()


def _played():
    """One round played with the manager aiming FREE at the timed-out agent."""
    env = _env()
    env.reset()
    state = env.punish(_action())
    return env, state


def test_the_recorded_punishment_on_a_timeout_is_zero():
    """The recorded punishment on a timed-out player is 0, as in the human data."""
    _, state = _played()
    p = _col(state, "punishment")
    assert p[0] == 0
    np.testing.assert_array_equal(p[1:], [PAID] * 7)


def test_the_switch_model_is_served_zero_this_round():
    """The switch model reads round t's `punishment`."""
    env, _ = _played()
    served = _col(env.state, "punishment")
    assert served[0] == 0
    np.testing.assert_array_equal(served[1:], [PAID] * 7)


def test_the_contribution_model_is_served_zero_next_round():
    """The contribution model reads it as `prev_punishment` next round."""
    env, _ = _played()
    env.step()
    served = _col(env.state, "prev_punishment")
    assert served[0] == 0
    np.testing.assert_array_equal(served[1:], [PAID] * 7)


def test_the_punisher_record_carries_the_charged_value():
    """The punisher's round record carries the charged value, not the action."""
    from aimanager.manager.api_manager import create_data
    from aimanager.simulation.simulate import add_punishments, make_round

    env, state = _played()
    charged = state["punishment"].reshape(-1).tolist()
    rd = make_round(
        state["contribution"].squeeze().tolist(),
        0,
        ["m"] * 8,
        0,
        agent_group=AGENT_GROUP,
        contribution_valid=state["contribution_valid"].reshape(-1).tolist(),
    )
    first = add_punishments(rd, charged)
    env.step()
    second = make_round(
        env.state["contribution"].squeeze().tolist(),
        1,
        ["m"] * 8,
        0,
        agent_group=AGENT_GROUP,
        contribution_valid=env.state["contribution_valid"].reshape(-1).tolist(),
    )
    data = create_data([first, second], ["m"], DEFAULTS)
    prev = data["prev_punishment"][0, :, -1].reshape(-1).numpy()
    assert prev[0] == 0
    np.testing.assert_array_equal(prev[1:], [PAID] * 7)


def test_a_punishment_on_a_genuine_zero_is_untouched():
    """A punishment on a player who chose to contribute 0 is still charged."""
    env = _env()
    env.reset()
    a = th.zeros((1, 8, 1), dtype=th.int64)
    a[0, 1, 0] = FREE
    state = env.punish(a)
    assert _col(state, "punishment")[1] == FREE
    assert _col(env.state, "punishment")[1] == FREE


def test_the_reward_is_unchanged():
    """Common good and group payoff do not depend on it (accounting zeroed it)."""
    both = []
    for aimed in (0, FREE):
        env = _env()
        env.reset()
        state = env.punish(_action(aimed))
        both.append(
            (
                float(state["common_good"].reshape(-1)[0]),
                float(env.group_payoff_sum.reshape(-1)[0]),
            )
        )
    assert both[0] == both[1]
    # group 0: agents 1..3 valid, contributions 0 + 20 + 3, each punished PAID
    expected = (1.6 * 23 - 3 * PAID) / 3
    assert abs(both[0][0] - expected) < 1e-6


def test_round_zero_prev_punishment_default_is_untouched():
    """Round 0's `prev_punishment` keeps the dataset default."""
    env = _env()
    env.reset()
    prev = _col(env.state, "prev_punishment")
    np.testing.assert_array_equal(prev, [DEFAULTS["punishment"]] * 8)


def test_the_timed_out_contribution_is_left_to_timeout_contribution():
    """The timed-out contribution is left to `timeout_contribution`."""
    _, state = _played()
    assert _col(state, "contribution")[0] == DEFAULTS["contribution"]
