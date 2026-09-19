"""The contributor must see whether IT gave the whole endowment last round.

`prev_contribution_max = I(c_{t-1} = 20)` is the lagged counterpart of the
punisher's `contribution_max`. Three things have to hold for it to be a legal,
usable contributor feature:

  * the lag is a real lag -- round 0 reads False and round t reads round t-1;
  * masking the contribution target does not touch it (it is built from
    `prev_contribution`, which `mask_data` never masks), so it cannot leak
    c_t into a model that predicts c_t;
  * the simulation's `ArtificialHumanEnv` derives exactly the same tensor the
    training path does, round by round -- otherwise the trained feature means
    something different in the closed loop.

The masking and `Encoder` tests import PyG (torch_scatter) and so need Raven
(`scripts/remote_test.sh`); the data-path and environment tests run locally.
"""

import torch as th

from aimanager.generic.data import MAX_CONTRIBUTION
from aimanager.manager.environment import ArtificialHumanEnv

# round 0 / 1 / 2 contributions for four agents; agent 0 is at the ceiling in
# round 0 only, agent 3 in rounds 1 and 2, so a lag error is visible.
C = [[20, 5, 0, 13], [7, 19, 11, 20], [2, 20, 20, 20]]
DEFAULTS = {
    "punishment": 0,
    "contribution": 9,
    "round_number": 0,
    "is_first": False,
    "contribution_valid": False,
    "punishment_valid": False,
    "common_good": 0.0,
    "contributor_payoff": 0.0,
    "reward": 0.0,
    "agent_group": 0,
    "does_switch": False,
    "switch_mask": False,
}


def _frame():
    import pandas as pd

    rows = []
    for t, cs in enumerate(C):
        for a, c in enumerate(cs):
            rows.append(
                dict(
                    episode_id=0,
                    round_number=t,
                    player_id=a,
                    global_group_id="g",
                    group_id=0,
                    player_no_input=0,
                    manager_no_input=0,
                    contribution=float(c),
                    punishment=0.0,
                    common_good=4.0,
                )
            )
    return pd.DataFrame(rows)


def test_training_path_lags_the_ceiling_indicator():
    from aimanager.generic.data import create_torch_data

    data, defaults, _ = create_torch_data(_frame())
    got = data["prev_contribution_max"]
    assert got.dtype == th.bool
    assert defaults["contribution"] != MAX_CONTRIBUTION  # round 0 must be False
    assert not got[0, :, 0].any()
    for t in (1, 2):
        assert got[0, :, t].tolist() == [c == MAX_CONTRIBUTION for c in C[t - 1]]
    # and it is genuinely the lag of the unlagged indicator, not a copy of it
    assert data["contribution_max"][0, :, 0].tolist() == [
        c == MAX_CONTRIBUTION for c in C[0]
    ]


def test_masking_the_contribution_target_leaves_the_indicator_intact():
    """`apply_mask_pattern` blanks `contribution` for the predicted agents.
    `prev_contribution_max` comes off `prev_contribution` and must survive --
    the same guarantee `prev_contribution` itself has."""
    import numpy as np

    from aimanager.artificial_humans.train import apply_mask_pattern
    from aimanager.generic.data import create_torch_data

    data, defaults, _ = create_torch_data(_frame())
    before = data["prev_contribution_max"].clone()
    pattern = th.tensor(np.array([[True, True, False, False]]))
    masked = apply_mask_pattern(
        {k: v.clone() for k, v in data.items()},
        pattern,
        "contribution",
        "contribution_valid",
        defaults,
    )
    assert th.equal(masked["prev_contribution_max"], before)
    assert th.equal(masked["prev_contribution"], data["prev_contribution"])
    # the target did get masked (into `contribution_masked`, the tensor the
    # autoregressive head reads), so the check above is not vacuous -- and
    # there is no `prev_contribution_masked` for the indicator to shadow
    assert not th.equal(masked["contribution_masked"], data["contribution"])
    assert "prev_contribution_masked" not in masked


class _ScriptedHuman:
    """Returns the scripted round's contributions and records what the
    contributor was shown when it was asked."""

    def __init__(self):
        self.default_values = DEFAULTS
        self.t = 0
        self.seen = []

    def predict(self, state, reset_rnn, edge_index):
        self.seen.append(state["prev_contribution_max"].squeeze(-1)[0].tolist())
        c = th.tensor(C[self.t], dtype=th.int64).reshape(1, len(C[0]), 1)
        self.t += 1
        return (c,)


def test_simulation_state_matches_the_training_lag():
    ah = _ScriptedHuman()
    env = ArtificialHumanEnv(
        artifical_humans=ah,
        batch_size=1,
        n_agents=len(C[0]),
        n_contributions=21,
        n_punishments=31,
        n_rounds=len(C),
        n_groups=1,
        device="cpu",
        default_values=DEFAULTS,
    )
    for t in range(len(C) - 1):
        assert env.state["contribution_max"].squeeze(-1)[0].tolist() == [
            c == MAX_CONTRIBUTION for c in C[t]
        ]
        env.punish(th.zeros((1, len(C[0]), 1), dtype=th.int64))
        env.step()
    # what the contributor was shown at each decision: False at round 0, then
    # the previous round's ceiling flags -- identical to the training tensor
    assert ah.seen[0] == [False] * len(C[0])
    for t in range(1, len(C)):
        assert ah.seen[t] == [c == MAX_CONTRIBUTION for c in C[t - 1]]


def test_encoder_reads_the_indicator_as_one_bool_channel():
    from aimanager.generic.data import create_torch_data
    from aimanager.generic.encoder import Encoder

    data, _, _ = create_torch_data(_frame())
    enc = Encoder(
        [{"etype": "bool", "name": "prev_contribution_max"}], refrence="contribution"
    )
    x = enc(**data)
    assert tuple(x.shape) == (1, len(C[0]), len(C), 1)
    assert x[0, :, 1, 0].tolist() == [float(c == MAX_CONTRIBUTION) for c in C[0]]
