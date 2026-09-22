"""What the RL manager is rewarded for, and for which round.

`reports/basics.md` states the rule: the contributors are paid in proportion
to their private accounts and "The manager is receiving a payout
proportionally to the common pool" -- 1.6 * sum(contributions) minus the
punishments the manager dealt. That pool is the `common_good` column of
`experiments/2group_8agent_50ep.csv` and the identity reproduces it to a
maximum residual of 2.8e-14 over all 4,512 human group-rounds.

These tests pin, in plain torch (no PyG, so they run locally):

  * `reward_mode: common_pool` -- equal to the group's pool on hand-built
    rounds with a timed-out player, an empty group and a reshuffle;
  * `reward_mode: common_pool_per_capita` -- that pool divided by the
    players who gave an input, which is the share the game hands one member.
    The divisor is the *valid* headcount, not the membership, so the tests
    pin both the plain case (where they coincide) and the timed-out case
    (where they do not), and they pin the ratio to `common_pool` on the same
    round: it must be exactly that headcount, which is what separates a real
    division from a renamed constant;
  * the reward's *round*: the reward for acting at round s is round s's
    outcome, not round s+1's. It is computed in `punish()`, where the action
    resolves, so a later reordering of `step()` cannot shift it silently;
  * the timed-out player's payoff (review finding D2): the real game paid
    them, so `sum` and `avg` must pay them too, and the tests state the size
    of that correction rather than only its direction.
"""

import numpy as np
import pytest
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

# One distinct contribution vector per round, so a reward that is a round out
# is a different number and not a coincidence.
C_ROUNDS = [
    [7, 0, 20, 3, 11, 5, 14, 2],
    [4, 1, 18, 6, 9, 1, 20, 8],
    [12, 2, 5, 19, 0, 17, 3, 6],
    [1, 14, 9, 2, 20, 4, 11, 7],
    [6, 8, 13, 5, 2, 16, 0, 10],
]
# ... and one distinct punishment vector per round.
P_ROUNDS = [
    [0, 3, 1, 0, 2, 0, 5, 1],
    [2, 0, 0, 4, 1, 6, 0, 0],
    [1, 1, 7, 0, 0, 2, 3, 0],
    [0, 5, 2, 1, 4, 0, 1, 2],
    [3, 0, 0, 2, 0, 1, 6, 0],
]
ALL_VALID = [True] * 8
# agent 0 times out, agent 1 genuinely contributes 0
ONE_TIMEOUT = [False] + [True] * 7


class _Fixed:
    """Artificial human stand-in returning a fixed draw per call.

    Call k is round k: `update_contribution` runs once inside `reset()` (round
    0) and once per `step()` thereafter. Short lists repeat their last row.
    """

    autoregressive = False
    default_values = DEFAULTS

    def __init__(self, values, dtype=th.int64):
        self.values = values
        self.dtype = dtype
        self.calls = 0

    def predict(self, state, **_):
        row = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return th.tensor(row, dtype=self.dtype).reshape(1, 8, 1), None


def _env(
    reward_mode="common_pool",
    valid=ALL_VALID,
    agent_groups=AGENT_GROUP,
    switch=None,
    n_rounds=5,
):
    """A 1-episode, 8-agent, 2-group env driven by the fixed draws above."""
    ah_switch = _Fixed(switch, dtype=th.bool) if switch is not None else None
    return ArtificialHumanEnv(
        artifical_humans=_Fixed(C_ROUNDS),
        artifical_humans_valid=_Fixed([valid], dtype=th.bool),
        artifical_humans_switch=ah_switch,
        switch_every=4,
        batch_size=1,
        n_agents=8,
        n_contributions=21,
        n_punishments=31,
        n_rounds=n_rounds,
        device=th.device("cpu"),
        n_groups=2,
        agent_groups=agent_groups,
        default_values=DEFAULTS,
        reward_mode=reward_mode,
    )


def _punish(round_index):
    return th.tensor(P_ROUNDS[round_index], dtype=th.int64).reshape(1, 8, 1)


def _pool(round_index, members, valid=ALL_VALID):
    """1.6 * sum(c) - sum(p) over `members`, zeroing the timed-out cells --
    the accounting the human data satisfies."""
    c = sum(C_ROUNDS[round_index][a] for a in members if valid[a])
    p = sum(P_ROUNDS[round_index][a] for a in members if valid[a])
    return 1.6 * c - p


def _reward(env, group):
    return env.state["reward"][0, group, 0].item()


# --------------------------------------------------------------------------- #
# the new mode computes the common pool
# --------------------------------------------------------------------------- #
def test_common_pool_reward_equals_the_group_pool():
    """Both groups, a plain round: the reward is 1.6 * sum(c) - sum(p) over
    the group's members, not a payoff sum."""
    env = _env()
    env.punish(_punish(0))
    assert _reward(env, 0) == pytest.approx(_pool(0, range(4)))
    assert _reward(env, 1) == pytest.approx(_pool(0, range(4, 8)))
    # and it is emphatically not the payoff-sum reward
    assert _reward(env, 0) != pytest.approx(
        env.state["group_payoff_sum"][0, 0, 0].item()
    )


def test_common_pool_reward_with_a_timed_out_player():
    """A player who gave no input contributed nothing and was punished
    nothing -- all 560 timed-out human rows carry contribution == 0 and
    punishment == 0 -- so neither their imputed contribution nor the
    punishment aimed at them may enter the pool."""
    env = _env(valid=ONE_TIMEOUT)
    env.punish(_punish(0))
    expected = _pool(0, range(4), valid=ONE_TIMEOUT)
    assert _reward(env, 0) == pytest.approx(expected)
    # the raw state still carries the imputed contribution for agent 0 ...
    assert env.state["contribution"][0, 0, 0].item() == DEFAULTS["contribution"]
    # ... and the punishment aimed at them; neither reaches the reward
    assert env.state["punishment"][0, 0, 0].item() == P_ROUNDS[0][0]
    assert _reward(env, 0) != pytest.approx(_pool(0, range(4)))
    # the other group is untouched
    assert _reward(env, 1) == pytest.approx(_pool(0, range(4, 8)))


def test_common_pool_reward_of_an_empty_group_is_zero():
    """Empty groups are not hypothetical: 288 of the 4,800 human group-rounds
    are empty because all eight players merged into one group."""
    env = _env(agent_groups=[0] * 8)
    env.punish(_punish(0))
    assert _reward(env, 0) == pytest.approx(_pool(0, range(8)))
    assert _reward(env, 1) == 0.0


def test_common_pool_reward_follows_a_reshuffle():
    """The reward for round s is computed on the membership in force during
    round s. Agent 4 moves to group 0 entering round 4 (switch_every = 4, the
    decision taken at round 3), so round 4's pools are over {0..4} / {5,6,7}.
    """
    # every agent stays put except agent 4, which flips at the round-3 decision
    stay = [False] * 8
    flip = [False] * 4 + [True] + [False] * 3
    env = _env(switch=[stay, stay, stay, flip, stay])

    for r in range(4):
        env.punish(_punish(r))
        env.step()
    assert env.state["agent_group"].reshape(-1).tolist() == [0, 0, 0, 0, 0, 1, 1, 1]

    env.punish(_punish(4))
    assert _reward(env, 0) == pytest.approx(_pool(4, range(5)))
    assert _reward(env, 1) == pytest.approx(_pool(4, range(5, 8)))
    # the pre-reshuffle split would have given a different number
    assert _reward(env, 0) != pytest.approx(_pool(4, range(4)))


# --------------------------------------------------------------------------- #
# the reward belongs to the round the action was taken in
# --------------------------------------------------------------------------- #
def test_reward_is_the_acting_rounds_outcome():
    """The reward handed back with the transition out of round s is round s's
    pool. This is the test that fails if anyone ever shifts the reward by a
    round: `step()` advances the round number and draws round s+1's
    contributions before it returns, and round s+1's pool is a different
    number for every round here.
    """
    env = _env()
    for r in range(4):
        env.punish(_punish(r))
        _, reward, _ = env.step()
        got = reward[0, 0, 0].item()
        assert got == pytest.approx(_pool(r, range(4))), f"round {r}"
        # the next round's contributions are already in the state by now ...
        np.testing.assert_array_equal(
            env.state["contribution"].reshape(-1).numpy(), C_ROUNDS[r + 1]
        )
        # ... and the reward is emphatically not computed from them
        assert got != pytest.approx(_pool(r + 1, range(4)))


def test_reward_is_settled_by_punish_not_by_step():
    """`punish()` is where the manager's action resolves, so it is where the
    reward is computed. `step()` only hands it on -- which is what makes the
    round correct by construction rather than by the order of the updates."""
    env = _env()
    env.punish(_punish(0))
    settled = env.state["reward"].clone()
    assert settled[0, 0, 0].item() == pytest.approx(_pool(0, range(4)))
    _, reward, _ = env.step()
    assert th.equal(reward, settled)


# --------------------------------------------------------------------------- #
# the payoff modes: unchanged where everyone plays, corrected where they don't
# --------------------------------------------------------------------------- #
def test_sum_mode_is_unchanged_when_everyone_plays():
    """`sum` = 20 * n_valid + 0.6 * sum(c) - 2 * sum(p), the closed form that
    reproduces the human valid-payoff sum to 5.7e-14. With no timeout the
    payoff fix changes nothing, so this value is the pre-fix value too."""
    env = _env(reward_mode="sum")
    env.punish(_punish(0))
    c = sum(C_ROUNDS[0][:4])
    p = sum(P_ROUNDS[0][:4])
    assert _reward(env, 0) == pytest.approx(20 * 4 + 0.6 * c - 2 * p)


def test_sum_mode_pays_the_timed_out_player():
    """Review finding D2. The real game paid a player who gave no input their
    endowment and their share of the pool -- `payoff = 20 - 0 - 0 +
    pool/n_valid` holds exactly on all 526 such human rows (mean 33.94, never
    0). The env used to score them 0 and drop them from the group total.
    """
    env = _env(reward_mode="sum", valid=ONE_TIMEOUT)
    env.punish(_punish(0))

    pool = _pool(0, range(4), valid=ONE_TIMEOUT)
    n_valid = 3
    dropped = 20 - 0 - 0 + pool / n_valid  # what the pre-fix env discarded
    pre_fix = 20 * n_valid + 0.6 * sum(C_ROUNDS[0][1:4]) - 2 * sum(P_ROUNDS[0][1:4])

    assert _reward(env, 0) == pytest.approx(pre_fix + dropped)
    assert dropped == pytest.approx(20 + pool / n_valid)
    assert env.state["contributor_payoff"][0, 0, 0].item() == pytest.approx(dropped)
    # the timed-out player is paid, and not out of the punished player's pocket
    assert _reward(env, 0) > pre_fix


def test_avg_mode_divides_by_the_group_membership():
    """Once every member has a payoff, the group's mean payoff is over its
    members. The timed-out player is in both the numerator and the divisor."""
    env = _env(reward_mode="avg", valid=ONE_TIMEOUT)
    env.punish(_punish(0))

    pool = _pool(0, range(4), valid=ONE_TIMEOUT)
    share = pool / 3
    payoffs = [20 + share] + [
        20 - C_ROUNDS[0][a] - P_ROUNDS[0][a] + share for a in range(1, 4)
    ]
    assert _reward(env, 0) == pytest.approx(sum(payoffs) / 4)
    # dividing by the valid headcount instead would be a mean of nothing
    assert _reward(env, 0) != pytest.approx(sum(payoffs) / 3)


def test_all_players_timed_out_pays_the_endowment():
    """The 34 human group-rounds where everyone timed out pay exactly 20.0 per
    player, with common_good 0. The env used to pay 0."""
    env = _env(reward_mode="sum", valid=[False] * 8)
    env.punish(_punish(0))
    assert _reward(env, 0) == pytest.approx(20.0 * 4)
    assert env.state["common_good"][0, 0, 0].item() == 0.0
    env_pool = _env(reward_mode="common_pool", valid=[False] * 8)
    env_pool.punish(_punish(0))
    assert _reward(env_pool, 0) == 0.0


# --------------------------------------------------------------------------- #
# the per-capita mode divides that pool the way the game divides it
# --------------------------------------------------------------------------- #
def test_per_capita_reward_is_the_pool_over_the_valid_headcount():
    """`reports/basics.md`: "the common pool is splitted equally between the
    contributors". With all four members valid the divisor is 4."""
    env = _env(reward_mode="common_pool_per_capita")
    env.punish(_punish(0))
    assert _reward(env, 0) == pytest.approx(_pool(0, range(4)) / 4)
    assert _reward(env, 1) == pytest.approx(_pool(0, range(4, 8)) / 4)
    # emphatically not the undivided pool
    assert _reward(env, 0) != pytest.approx(_pool(0, range(4)))


def test_per_capita_reward_divides_by_the_valid_players_not_the_members():
    """The divisor is `count_valid_per_group`, and it parts company with the
    group's membership the moment somebody times out. Group 0 still holds
    four members but only three of them gave an input, so the share is over
    3 -- the game's own rule, `payoff = 20 - c - p + pool/n_valid`."""
    env = _env(reward_mode="common_pool_per_capita", valid=ONE_TIMEOUT)
    env.punish(_punish(0))

    pool = _pool(0, range(4), valid=ONE_TIMEOUT)
    assert _reward(env, 0) == pytest.approx(pool / 3)
    # dividing by the four members instead would be a different number
    assert _reward(env, 0) != pytest.approx(pool / 4)
    # the untouched group still divides by its four valid players
    assert _reward(env, 1) == pytest.approx(_pool(0, range(4, 8)) / 4)


def test_per_capita_reward_is_the_common_good_state_field():
    """The reward is produced by the same `share_pool_per_group` that fills
    the `common_good` state field, so the two must agree on the nose. This is
    the cross-path check the launch guard makes at scale, pinned here on a
    hand-built round.

    Note which `common_good` this is: the env's state field, the per-capita
    share. The column of the same name in the human CSV is the *undivided*
    pool, which is why the mode is not named after it.
    """
    env = _env(reward_mode="common_pool_per_capita", valid=ONE_TIMEOUT)
    env.punish(_punish(0))
    for group, members in ((0, range(4)), (1, range(4, 8))):
        agent = next(a for a in members)
        assert _reward(env, group) == pytest.approx(
            env.state["common_good"][0, agent, 0].item()
        )


def test_per_capita_is_the_pool_divided_not_the_pool_renamed():
    """Same rounds, both modes: the ratio is exactly the valid headcount and
    never 1. A mode that merely relabelled `common_pool` would give 1."""
    env_pool = _env()
    env_share = _env(reward_mode="common_pool_per_capita")
    for r in range(4):
        env_pool.punish(_punish(r))
        env_share.punish(_punish(r))
        for group in (0, 1):
            pool, share = _reward(env_pool, group), _reward(env_share, group)
            assert share == pytest.approx(pool / 4)
            assert pool / share == pytest.approx(4.0)
            assert abs(pool - share) > 1.0
        env_pool.step()
        env_share.step()


def test_per_capita_reward_of_an_empty_group_is_zero():
    """No members, no valid players, nothing to share. Same 288 empty human
    group-rounds as the pool mode's test."""
    env = _env(reward_mode="common_pool_per_capita", agent_groups=[0] * 8)
    env.punish(_punish(0))
    assert _reward(env, 0) == pytest.approx(_pool(0, range(8)) / 8)
    assert _reward(env, 1) == 0.0


def test_per_capita_reward_is_zero_when_everyone_timed_out():
    """The pool is 0 and there is nobody to divide it between; the env must
    not divide by zero. Matches the 34 such human group-rounds."""
    env = _env(reward_mode="common_pool_per_capita", valid=[False] * 8)
    env.punish(_punish(0))
    assert _reward(env, 0) == 0.0
    assert _reward(env, 1) == 0.0


def test_per_capita_reward_follows_a_reshuffle_and_divides_by_the_new_size():
    """The point of the mode, on the one round where it bites. Agent 4 moves
    into group 0 entering round 4, so group 0's pool grows -- and under
    `common_pool` that alone would raise the manager's reward. Here the
    divisor grows with it, from 4 to 5, so gaining a member is close to
    neutral rather than a payout.
    """
    stay = [False] * 8
    flip = [False] * 4 + [True] + [False] * 3
    env = _env(reward_mode="common_pool_per_capita", switch=[stay] * 3 + [flip, stay])

    for r in range(4):
        env.punish(_punish(r))
        env.step()
    assert env.state["agent_group"].reshape(-1).tolist() == [0, 0, 0, 0, 0, 1, 1, 1]

    env.punish(_punish(4))
    assert _reward(env, 0) == pytest.approx(_pool(4, range(5)) / 5)
    assert _reward(env, 1) == pytest.approx(_pool(4, range(5, 8)) / 3)
    # dividing by the pre-reshuffle headcount would be a different number
    assert _reward(env, 0) != pytest.approx(_pool(4, range(5)) / 4)


def test_unknown_reward_mode_is_rejected():
    with pytest.raises(ValueError, match="reward_mode"):
        _env(reward_mode="common_good")
