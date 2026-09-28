"""A punishment aimed at a player who gave no input must be charged as 0.

Fourth and last instance of one defect. The real game never charged, and
never showed, a punishment on a player who timed out: all 560
`player_no_input` rows in `experiments/2group_8agent_50ep.csv` carry
`punishment == 0.0` exactly. The env's accounting already agreed --
`compute_common_good_per_group` and `compute_payoff_per_group` zero the
invalid cell themselves -- so the action cost the manager nothing, while
`punish()` stored the raw value and `step()` copied it into
`prev_punishment`, the contribution model's only channel from the manager.
Free deterrence on a cell every model can identify, and a lever a
cost-bearing learner would find.

These tests pin the realised value at every seat that reads it:

  * the env's own record (`self.state["punishment"]`), which `Memory.add`
    turns into `per_round.parquet` -- the one recorded value this fix
    deliberately changes, because the human data carries the same zero;
  * the switch model's channel, round t's `punishment`;
  * the contribution model's channel, round t-1's `prev_punishment`;
  * the punisher's own `prev_punishment`, which `api_manager.create_data`
    shifts out of the round record `simulate.py` keeps;

plus the guards that say nothing else moved: a real punishment on a player
who genuinely contributed 0, the common good and group payoff (unchanged,
because the accounting already zeroed the cell), round 0's default lag, and
the imputed contribution in the recorded output, which stays imputed.

All but the `create_data` test are plain torch and run locally; that one
imports torch_geometric through `api_manager`, so Raven
(`scripts/remote_test.sh`).
"""

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
    """What `Memory.add` writes into `per_round.parquet`. This is the one
    recorded value the fix changes, and it changes it toward the human data:
    `convert.load_human` keeps a timed-out player's row and it carries 0."""
    _, state = _played()
    p = _col(state, "punishment")
    assert p[0] == 0
    np.testing.assert_array_equal(p[1:], [PAID] * 7)


def test_the_switch_model_is_served_zero_this_round():
    """The switch model reads round t's own `punishment`."""
    env, _ = _played()
    served = _col(env.served_state(), "punishment")
    assert served[0] == 0
    np.testing.assert_array_equal(served[1:], [PAID] * 7)


def test_the_contribution_model_is_served_zero_next_round():
    """`prev_punishment` is the contribution model's only channel from the
    manager -- the place the free lever actually acted."""
    env, _ = _played()
    env.step()
    served = _col(env.served_state(), "prev_punishment")
    assert served[0] == 0
    np.testing.assert_array_equal(served[1:], [PAID] * 7)


def test_the_punisher_record_carries_the_charged_value():
    """The seam `simulate.py` uses: the round record the punisher's own
    `prev_punishment` feature is shifted out of is built from the env's
    realised punishment, not from the raw action."""
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
    """Agent 1 contributed 0 by choice; the manager may punish that, the
    game charged it, and the fix must not reach it."""
    env = _env()
    env.reset()
    a = th.zeros((1, 8, 1), dtype=th.int64)
    a[0, 1, 0] = FREE
    state = env.punish(a)
    assert _col(state, "punishment")[1] == FREE
    assert _col(env.served_state(), "punishment")[1] == FREE


def test_the_reward_is_unchanged():
    """The guard on the dependency argument: the accounting already zeroed
    the invalid cell, so the common good and the group payoff sum are the
    same before and after, and the same whatever the manager aims there."""
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
    # and it is the value the env's own accounting produces for group 0
    # (agents 1..3 valid, contributions 0 + 20 + 3, punished PAID each)
    expected = (1.6 * 23 - 3 * PAID) / 3
    assert abs(both[0][0] - expected) < 1e-6


def test_round_zero_prev_punishment_default_is_untouched():
    """Round 0 has no previous round to have been punished in; its `prev_*`
    cells carry the dataset default, as create_torch_data's shift() puts
    them there in training."""
    env = _env()
    env.reset()
    prev = _col(env.served_state(), "prev_punishment")
    np.testing.assert_array_equal(prev, [DEFAULTS["punishment"]] * 8)


def test_the_recorded_contribution_still_carries_the_imputed_value():
    """The boundary of the parent's fix, which this one does not move: the
    env records the imputed contribution for a timed-out player and only the
    *served* view corrects it. Recording 0 there would push ~2% of the scored
    rows to a hard zero against human rows the suite drops."""
    env, state = _played()
    assert _col(state, "contribution")[0] == DEFAULTS["contribution"]
    assert _col(env.served_state(), "contribution")[0] == 0
