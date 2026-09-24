"""`timeout_contribution`: the contribution recorded for a timed-out player.

Runs on Raven (the env needs a working torch; two tests import PyG)."""

import pytest
import torch as th

from aimanager.manager.environment import ArtificialHumanEnv

N_AGENTS = 4
PREDICTED = 15  # what the stub contribution model always predicts
DEFAULT = 9  # the stub dataset default (median) contribution


class _Contribution:
    default_values = {
        "punishment": 0,
        "contribution": DEFAULT,
        "contribution_valid": False,
        "punishment_valid": False,
        "common_good": 0,
    }

    def predict(self, state, reset_rnn, edge_index):
        return (th.full(state["contribution"].shape, PREDICTED, dtype=th.long),)


class _Valid:
    """Agent 0 times out; everyone else gives input."""

    def predict(self, state, reset_rnn, edge_index):
        valid = th.ones(state["contribution"].shape, dtype=th.bool)
        valid[:, 0] = False
        return (valid,)


def _env(**kwargs):
    ah = _Contribution()
    return ArtificialHumanEnv(
        artifical_humans=ah,
        artifical_humans_valid=_Valid(),
        batch_size=1,
        n_agents=N_AGENTS,
        n_contributions=21,
        n_punishments=31,
        n_rounds=3,
        n_groups=2,
        device="cpu",
        default_values={
            **ah.default_values,
            "round_number": 0,
            "is_first": False,
            "contributor_payoff": 0,
            "reward": 0,
        },
        **kwargs,
    )


def _contribution(env):
    return env.state["contribution"].reshape(-1).tolist()


@pytest.mark.parametrize("value", [0, 7, 20])
def test_env_records_the_configured_value_for_a_timeout(value):
    env = _env(timeout_contribution=value)
    assert _contribution(env) == [value] + [PREDICTED] * (N_AGENTS - 1)
    assert env.state["contribution_valid"].reshape(-1).tolist() == [
        False,
        True,
        True,
        True,
    ]


def test_default_and_absent_key_keep_the_dataset_default():
    assert _contribution(_env(timeout_contribution="default"))[0] == DEFAULT
    assert _contribution(_env())[0] == DEFAULT


def test_configured_value_persists_into_the_next_round():
    env = _env(timeout_contribution=0)
    env.punish(th.zeros((1, N_AGENTS, 1), dtype=th.int64))
    env.step()
    assert env.state["prev_contribution"].reshape(-1)[0].item() == 0
    assert _contribution(env)[0] == 0


def test_accounting_ignores_the_timed_out_contribution():
    # common good counts valid players only
    goods = []
    for value in (0, 20):
        env = _env(timeout_contribution=value)
        env.punish(th.zeros((1, N_AGENTS, 1), dtype=th.int64))
        goods.append(env.state["common_good"].clone())
    assert th.equal(goods[0], goods[1])


@pytest.mark.parametrize("bad", [-1, 21, 9.0, True, "median", None])
def test_invalid_values_are_rejected(bad):
    with pytest.raises(ValueError, match="timeout_contribution"):
        _env(timeout_contribution=bad)


def test_make_round_carries_the_env_validity_flag():
    from aimanager.simulation.simulate import make_round

    r = make_round(
        [0, 15, 15, 15],
        2,
        ["m"] * 4,
        0,
        agent_group=[0, 0, 1, 1],
        contribution_valid=[False, True, True, True],
    )
    assert r["contribution_valid"] == [False, True, True, True]
    assert make_round([0, None], 2, ["m"] * 2, 0)["contribution_valid"] == [
        True,
        False,
    ]


def test_gnn_punisher_reads_the_recorded_value_of_a_timed_out_player():
    from aimanager.manager.api_manager import create_data

    def rnd(t, c, valid):
        return {
            "contribution": c,
            "contribution_valid": valid,
            "punishment": [0] * len(c),
            "punishment_valid": [True] * len(c),
            "group": ["m"] * len(c),
            "agent_group": [0] * len(c),
            "round": t,
        }

    rounds = [
        rnd(0, [12, 12], [True, True]),
        rnd(1, [0, 12], [False, True]),  # agent 0 timed out, recorded 0
    ]
    defaults = {
        "punishment": 0,
        "contribution": DEFAULT,
        "contribution_valid": False,
        "punishment_valid": False,
        "agent_group": 0,
    }
    data = create_data(rounds, ["m"], defaults)
    contribution = data["contribution"]
    # the recorded 0, not the default 9
    assert contribution.reshape(-1, 2)[:, -1].tolist()[0] == 0
    assert data["contribution_valid"].reshape(-1, 2)[:, -1].tolist()[0] is False
