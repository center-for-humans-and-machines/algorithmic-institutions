"""The reinforcement-learning manager must see the recorded 0 for a timeout.

Third and last instance of one defect. A player who gave no input contributed
nothing -- that is what the game charged and what everyone saw -- but
`environment.update_contribution` overwrites those cells with the imputed
default (9) before the state is passed on. `auto/punisher-timeout-feature`
corrected the two punisher serving paths; `auto/sim-timeout-imputation`
corrected the contribution and switch models' through
`ArtificialHumanEnv.served_state`. The RL path was left out: `reset()`,
`punish()` and `step()` return `self.state`, the raw one, which is what
`rl_manager.run_batch` handed to the manager, to the replay buffer the TD
update trains on, and to the fixed opponent punisher.

These tests mirror `test_sim_timeout_serving.py` and pin all three seats at
the same table, plus the guards that say nothing else moved:

  * what `manager.get_action` is served, with the validity flag consistent;
  * what lands in the replay `Memory`, i.e. what `manager.update` trains on;
  * what the opponent punisher is served;
  * round 0's `prev_*` defaults, which the fix must not touch;
  * the env's own state, its recorded metrics and its common good, which stay
    on the imputed value exactly as `per_round.parquet` does.

`rl_manager` imports torch_geometric, so the rollout tests need Raven
(`scripts/remote_test.sh`); `test_env_return_values_are_unchanged` is plain
torch and runs locally.
"""

import numpy as np
import torch as th

from aimanager.manager.environment import ArtificialHumanEnv
from aimanager.manager.memory import Memory

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
# agent 0 times out, agent 1 genuinely contributes 0; the same draw every
# round, so a round index never has to be reasoned about
C_ROUND = [7, 0, 20, 3, 11, 5, 14, 2]
VALID = [False] + [True] * 7
N_ROUNDS = 4
# the manager punishes 1 everywhere, the opponent 0; group 0 is the RL group
RL_PUNISHMENT = 1
REPLAY_KEYS = [
    "agent_group",
    "contribution",
    "contribution_valid",
    "prev_punishment",
    "punishment",
    "round_number",
]


class _Fixed:
    """Artificial human stand-in returning the same draw every round."""

    autoregressive = False
    default_values = DEFAULTS

    def __init__(self, row, dtype=th.int64):
        self.row = row
        self.dtype = dtype

    def predict(self, state, **_):
        return th.tensor(self.row, dtype=self.dtype).reshape(1, 8, 1), None


class _RecordingModel:
    """Opponent punisher stand-in that keeps every state it was served."""

    def __init__(self):
        self.seen = []

    def predict(self, state, **_):
        self.seen.append({k: v.clone() for k, v in state.items() if th.is_tensor(v)})
        return th.zeros((1, 8, 1), dtype=th.int64), None


class _RecordingManager:
    """Manager stand-in: keeps what it was served and plays a constant."""

    def __init__(self):
        self.seen = []

    def get_action(self, state, first=False, greedy=False):
        self.seen.append({k: v.clone() for k, v in state.items() if th.is_tensor(v)})
        return (
            th.full((1, 8, 1), RL_PUNISHMENT, dtype=th.int64),
            th.zeros((1, 8, 1, 31), dtype=th.float),
        )


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


def _rollout():
    """One real `rl_manager.run_batch` over the stubbed stack."""
    from aimanager import rl_manager

    rl_manager.replay_keys = list(REPLAY_KEYS)
    env, manager, opponent = _env(), _RecordingManager(), _RecordingModel()
    replay = Memory(device=th.device("cpu"), n_episodes=1, n_episode_steps=N_ROUNDS)
    metrics = rl_manager.run_batch(
        manager,
        env,
        replay,
        on_policy=True,
        update_step=0,
        opponent_manager=opponent,
        rl_group_id=0,
    )
    return env, manager, opponent, replay, metrics


def _col(state, key):
    return state[key].reshape(-1).numpy()


def test_manager_sees_zero_for_a_timeout():
    """Round t's own contribution, the manager's only contribution channel:
    0 for the timeout with the flag saying so, 0 for the genuine zero with
    the flag clear, nobody else disturbed."""
    _, manager, _, _, _ = _rollout()
    seen = manager.seen[0]
    c, cv = _col(seen, "contribution"), _col(seen, "contribution_valid")
    assert c[0] == 0 and not cv[0]  # timed out: recorded 0, flag says so
    assert c[1] == 0 and cv[1]  # genuine zero: unchanged, flag set
    np.testing.assert_array_equal(c[2:], C_ROUND[2:])
    np.testing.assert_array_equal(cv[2:], [True] * 6)


def test_replay_buffer_stores_the_recorded_zero():
    """What `manager.update` trains on. The online view and the replayed
    view must agree -- disagreeing would be worse than either alone."""
    _, _, _, replay, _ = _rollout()
    c = replay.memory["contribution"][0, 0, :, 0].numpy()
    cv = replay.memory["contribution_valid"][0, 0, :, 0].numpy()
    assert c[0] == 0 and not cv[0]
    assert c[1] == 0 and cv[1]
    np.testing.assert_array_equal(c[2:], C_ROUND[2:])


def test_opponent_manager_sees_zero_for_a_timeout():
    """The fixed opponent punisher is served from the same state and is the
    one model in the loop with a `contribution_valid` feature built for this
    population; it must not be handed the imputed value either."""
    _, _, opponent, _, _ = _rollout()
    c = _col(opponent.seen[0], "contribution")
    assert c[0] == 0
    assert c[1] == 0
    np.testing.assert_array_equal(c[2:], C_ROUND[2:])


def test_round_zero_defaults_are_untouched():
    """Round 0 has no previous round to have timed out; its `prev_*` cells
    carry the dataset default, exactly as create_torch_data's shift() puts
    them there in training. The fix must not touch them."""
    _, manager, _, _, _ = _rollout()
    prev = _col(manager.seen[0], "prev_contribution")
    np.testing.assert_array_equal(prev, [DEFAULTS["contribution"]] * 8)


def test_recorded_metrics_and_env_state_keep_the_imputed_value():
    """Only what a model is *served* changes. The env's own state and the
    metrics `run_batch` records -- the record of what the game charged, the
    same role `per_round.parquet` plays in the simulation -- keep the
    imputed value, so no recorded output moves."""
    env, _, _, _, metrics = _rollout()
    assert env.state["contribution"].reshape(-1)[0].item() == DEFAULTS["contribution"]
    # the RL group is agents 0..3; its mean still carries the imputed 9
    expected = (DEFAULTS["contribution"] + C_ROUND[1] + C_ROUND[2] + C_ROUND[3]) / 4
    assert abs(metrics[0]["contribution"] - expected) < 1e-6


def test_common_good_is_unchanged_by_the_fix():
    """The env's accounting zeroes invalid contributions -- and the
    punishment on them -- on its own; serving the recorded value must not
    double-count or shift it."""
    env, _, _, _, metrics = _rollout()
    # group 0: agents 1..3 valid, contributions 0 + 20 + 3, punished 1 each
    expected = (1.6 * 23 - 3 * RL_PUNISHMENT) / 3
    cg = env.compute_common_good_per_group(
        th.tensor(C_ROUND, dtype=th.int64).reshape(1, 8, 1),
        th.full((1, 8, 1), RL_PUNISHMENT, dtype=th.int64),
        th.tensor(VALID, dtype=th.bool).reshape(1, 8, 1),
    )
    assert abs(cg[0, 0, 0].item() - expected) < 1e-6
    assert abs(metrics[0]["common_good"] - expected) < 1e-6


def test_env_return_values_are_unchanged():
    """The guard on the boundary of the fix: `reset`, `punish` and `step`
    still hand back `self.state`, so `simulate.py` -- which records their
    return value into `per_round.parquet` -- is untouched. Plain torch."""
    env = _env()
    assert env.reset() is env.state
    assert env.punish(th.zeros((1, 8, 1), dtype=th.int64)) is env.state
    state, _, _ = env.step()
    assert state is env.state
    assert state["contribution"].reshape(-1)[0].item() == DEFAULTS["contribution"]
