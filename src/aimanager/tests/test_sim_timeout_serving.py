"""The contribution and switch models must see the recorded 0 for a timeout.

A player who gave no input contributed nothing -- that is what the game
charged and what everyone saw -- and it is what the training data stores.
`environment.update_contribution` overwrites those cells with the imputed
default (9) before the state is passed on, so a model reads a value the game
never used. `auto/punisher-timeout-feature` corrected the two punisher
serving paths; these tests pin the remaining two, in the same shape:

  * the env contract (`ArtificialHumanEnv.served_state`): the switch model
    gets 0 for round t, the contribution model gets 0 for round t-1, a
    genuine zero is unchanged, round 0's default lag is untouched and the
    env's own recorded state still carries the imputed value -- plain torch,
    runs locally;
  * the linear adapter's env-driven path -- plain torch, runs locally;
  * the GNN input `Encoder` on the served state -- imports torch_scatter, so
    Raven (`scripts/remote_test.sh`).
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
# agent 0 times out, agent 1 genuinely contributes 0
C_ROUND = [[7, 0, 20, 3, 11, 5, 14, 2], [4, 0, 18, 6, 9, 1, 20, 8]]
VALID = [False] + [True] * 7


class _Recorder:
    """Artificial human stand-in that returns a fixed draw and keeps every
    state it was served, so a test can read what reached the model."""

    autoregressive = False
    default_values = DEFAULTS

    def __init__(self, values, dtype=th.int64):
        self.values = values
        self.dtype = dtype
        self.seen = []

    def predict(self, state, **_):
        self.seen.append({k: v.clone() for k, v in state.items() if th.is_tensor(v)})
        row = self.values[min(len(self.seen) - 1, len(self.values) - 1)]
        return th.tensor(row, dtype=self.dtype).reshape(1, 8, 1), None


def _env(switch=True):
    ah = _Recorder(C_ROUND)
    ah_valid = _Recorder([VALID], dtype=th.bool)
    ah_switch = _Recorder([[False] * 8], dtype=th.bool) if switch else None
    env = ArtificialHumanEnv(
        artifical_humans=ah,
        artifical_humans_valid=ah_valid,
        artifical_humans_switch=ah_switch,
        switch_every=4,
        batch_size=1,
        n_agents=8,
        n_contributions=21,
        n_punishments=31,
        n_rounds=4,
        device=th.device("cpu"),
        n_groups=2,
        agent_groups=AGENT_GROUP,
        default_values=DEFAULTS,
    )
    return env, ah, ah_valid, ah_switch


def _played(env):
    """Play round 0 through to the start of round 1."""
    env.punish(th.zeros((1, 8, 1), dtype=th.int64))
    env.step()


def _col(state, key):
    return state[key].reshape(-1).numpy()


def test_switch_model_sees_zero_for_a_timeout():
    """Round t's own contribution: 0 for the timeout, 0 for the genuine zero,
    everyone else untouched."""
    env, _, _, ah_switch = _env()
    _played(env)
    seen = ah_switch.seen[0]
    c, cv = _col(seen, "contribution"), _col(seen, "contribution_valid")
    assert c[0] == 0 and not cv[0]  # timed out: recorded 0, flag says so
    assert c[1] == 0 and cv[1]  # genuine zero: unchanged, flag clear
    np.testing.assert_array_equal(c[2:], C_ROUND[0][2:])


def test_contribution_model_sees_zero_for_the_previous_round_timeout():
    """Round t-1's contribution, the prev-anchored contributor's input."""
    env, ah, _, _ = _env()
    _played(env)
    seen = ah.seen[1]  # the call for round 1
    prev = _col(seen, "prev_contribution")
    assert prev[0] == 0  # timed out in round 0
    assert prev[1] == 0  # genuine zero in round 0
    np.testing.assert_array_equal(prev[2:], C_ROUND[0][2:])


def test_round_zero_lag_keeps_the_dataset_default():
    """Round 0 has no previous round to have timed out; its `prev_*` cells
    carry the dataset default, exactly as create_torch_data's shift() puts
    them there in training. The fix must not touch them."""
    _, ah, _, _ = _env()
    prev = _col(ah.seen[0], "prev_contribution")
    np.testing.assert_array_equal(prev, [DEFAULTS["contribution"]] * 8)


def test_env_state_and_recorded_output_keep_the_imputed_value():
    """Only what a model is *served* changes. The env's own state -- which is
    what Memory records into per_round.parquet and what make_round hands the
    manager -- is untouched, so the evaluation's inputs are unchanged."""
    env, _, _, _ = _env()
    assert env.state["contribution"].reshape(-1)[0].item() == DEFAULTS["contribution"]
    served = env.served_state()
    assert served["contribution"].reshape(-1)[0].item() == 0
    assert env.state["contribution"].reshape(-1)[0].item() == DEFAULTS["contribution"]


def test_common_good_is_unchanged_by_the_fix():
    """The env's accounting zeroes invalid contributions on its own; serving
    the recorded value must not double-count or shift it."""
    env, _, _, _ = _env()
    cg = env.compute_common_good_per_group(
        env.state["contribution"],
        th.zeros((1, 8, 1), dtype=th.int64),
        env.state["contribution_valid"],
    )
    # group 0: agents 1..3 valid, contributions 0 + 20 + 3 -> 1.6 * 23 / 3
    assert abs(cg[0, 0, 0].item() - 1.6 * 23 / 3) < 1e-6


# --------------------------------------------------------------------------- #
# the linear adapter's env-driven path
# --------------------------------------------------------------------------- #
class _RecordingEstimator:
    def __init__(self, n_levels):
        self.classes_ = np.arange(n_levels)
        self.seen = []

    def predict_proba(self, X):
        self.seen.append(np.asarray(X, dtype=float).copy())
        return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))


class _IdentityScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)


def _linear(target, features, n_levels):
    from aimanager.simulation.linear_ah import LinearAHAdapter

    bundle = {
        "model": "multinomial",
        "estimator": _RecordingEstimator(n_levels),
        "scaler": _IdentityScaler(),
        "features": features,
        "target": target,
        "n_levels": n_levels,
        "default_values": DEFAULTS,
        "switch_every": 4,
    }
    return LinearAHAdapter(
        bundle, n_agents=8, n_contributions=21, device=th.device("cpu")
    )


def test_linear_contribution_model_sees_zero_for_the_previous_round_timeout():
    env, _, _, _ = _env(switch=False)
    ah = _linear("contribution", ["prev_contribution"], 21)
    ah.predict(env.served_state())  # round 0
    _played(env)
    ah.predict(env.served_state())  # round 1
    X = ah.estimator.seen[-1]
    np.testing.assert_array_equal(X[:, 0], [0.0, 0.0] + C_ROUND[0][2:])


def test_linear_switch_model_sees_zero_and_the_validity_flag():
    env, _, _, _ = _env(switch=False)
    ah = _linear("does_switch", ["contribution", "contribution_valid"], 2)
    ah.predict(env.served_state())
    X = ah.estimator.seen[-1]
    assert X[0, 0] == 0.0 and X[0, 1] == 0.0  # timeout: value 0, flag says so
    assert X[1, 0] == 0.0 and X[1, 1] == 1.0  # genuine zero, flag set
    np.testing.assert_array_equal(X[2:, 0], C_ROUND[0][2:])
    np.testing.assert_array_equal(X[2:, 1], [1.0] * 6)


# --------------------------------------------------------------------------- #
# the GNN input encoder (needs torch_scatter -> Raven)
# --------------------------------------------------------------------------- #
def test_gnn_contribution_encoder_sees_zero_for_a_timeout():
    """The frontier contributor's own encoding of `prev_contribution`
    (numeric, 21 levels) over the served state: the timed-out agent's lag
    encodes as 0.0, not as the default's 9/20."""
    from aimanager.generic.encoder import Encoder

    env, _, _, _ = _env(switch=False)
    _played(env)
    enc = Encoder(
        [{"name": "prev_contribution", "n_levels": 21, "encoding": "numeric"}],
        refrence="contribution",
    )
    x = enc(**env.served_state()).reshape(8, -1).numpy()
    raw = enc(**env.state).reshape(8, -1).numpy()
    assert x[0, 0] == 0.0  # served the recorded 0
    assert abs(raw[0, 0] - 9 / 20) < 1e-6  # what the raw state still carries
    np.testing.assert_allclose(x[1:, 0], raw[1:, 0])  # nobody else moves


def test_gnn_switch_encoder_sees_zero_for_a_timeout():
    from aimanager.generic.encoder import Encoder

    env, _, _, _ = _env(switch=False)
    enc = Encoder(
        [{"name": "contribution", "n_levels": 21, "encoding": "numeric"}],
        refrence="does_switch",
    )
    x = enc(**env.served_state()).reshape(8, -1).numpy()
    assert x[0, 0] == 0.0
    np.testing.assert_allclose(x[2:, 0], np.array(C_ROUND[0][2:]) / 20)
