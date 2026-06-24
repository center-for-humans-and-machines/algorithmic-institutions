import os
import tempfile

import pytest
import torch as th

from aimanager.generic.graph import (
    EdgeEncoder,
    GraphNetwork,
    SameGroupEdgeEncoder,
)
from aimanager.manager.environment import ArtificialHumanEnv


@pytest.fixture
def switch_scenario():
    """4 nodes, 2 rounds, one mid-episode switch.

    agent_group is (N, n_rounds): node 1 moves group 0 -> 1 at round 1.
        node0: [0, 0]   node1: [0, 1]   node2: [1, 1]   node3: [1, 1]
    edge_index = all directed i != j pairs (the fully-connected graph).
    """
    agent_group = th.tensor([[0, 0], [0, 1], [1, 1], [1, 1]], dtype=th.int64)
    pairs = [(i, j) for i in range(4) for j in range(4) if i != j]
    edge_index = th.tensor(pairs, dtype=th.int64).T  # (2, 12)
    return agent_group, edge_index, pairs


def test_same_group_edge_encoder_values(switch_scenario):
    agent_group, edge_index, pairs = switch_scenario
    enc = SameGroupEdgeEncoder()
    assert enc.size == 1
    out = enc(edge_index=edge_index, agent_group=agent_group)  # (E, n_rounds, 1)
    assert out.shape == (12, 2, 1)
    assert out.dtype == th.float

    same = out.squeeze(-1)  # (E, n_rounds)
    idx = {p: k for k, p in enumerate(pairs)}
    # (0,1): same group at r0, split at r1 (node 1 switched)
    assert th.equal(same[idx[(0, 1)]], th.tensor([1.0, 0.0]))
    # (1,2): split at r0, same at r1 (node 1 joined node 2's group)
    assert th.equal(same[idx[(1, 2)]], th.tensor([0.0, 1.0]))
    # (2,3): same group both rounds
    assert th.equal(same[idx[(2, 3)]], th.tensor([1.0, 1.0]))
    # (0,2): different group both rounds
    assert th.equal(same[idx[(0, 2)]], th.tensor([0.0, 0.0]))


def test_same_group_edge_encoder_rejects_non_bool():
    with pytest.raises(AssertionError):
        SameGroupEdgeEncoder(etype="int")


def test_edge_encoder_empty_is_zero_width(switch_scenario):
    _, edge_index, _ = switch_scenario
    enc = EdgeEncoder([], refrence="contribution")
    assert enc.size == 0
    out = enc(edge_index=edge_index, n_rounds=2)
    assert out.shape == (12, 2, 0)
    assert out.dtype == th.float


def test_edge_encoder_dispatches_same_group(switch_scenario):
    agent_group, edge_index, _ = switch_scenario
    enc = EdgeEncoder(
        [{"name": "same_group", "etype": "bool"}], refrence="contribution"
    )
    assert enc.size == 1
    out = enc(edge_index=edge_index, n_rounds=2, agent_group=agent_group)
    expected = SameGroupEdgeEncoder()(edge_index=edge_index, agent_group=agent_group)
    assert th.equal(out, expected)


def _make_model(edge_encoding):
    model = GraphNetwork(
        y_levels=21,
        y_name="contribution",
        hidden_size=4,
        add_rnn=True,
        add_edge_model=True,
        add_global_model=False,
        x_encoding=[
            {"name": "prev_contribution", "n_levels": 21, "encoding": "numeric"}
        ],
        edge_encoding=edge_encoding,
        default_values={},
    )
    return model.to("cpu")


def _make_data(n_batch=2, n_player=4, n_rounds=3):
    th.manual_seed(0)
    shape = (n_batch, n_player, n_rounds)
    return {
        "contribution": th.randint(0, 21, shape),
        "prev_contribution": th.randint(0, 21, shape),
        "agent_group": th.randint(0, 2, shape),
    }


def test_forward_with_same_group_runs():
    """End-to-end: encode populates a (E, n_rounds, 1) edge_attr and forward
    runs (exercises the op2 empty-edge_attr fix)."""
    model = _make_model([{"name": "same_group", "etype": "bool"}])
    data = _make_data()
    encoded = model.encode(data, y_encode=True, device="cpu")
    n_edges = 2 * 4 * 3  # n_batch * n_player * (n_player - 1)
    assert encoded["edge_attr"].shape == (n_edges, 3, 1)
    out = model(encoded)
    assert out.shape == (2 * 4, 3, 21)  # (N, n_rounds, y_levels)


def test_forward_without_edge_encoding_is_backward_compatible():
    """No edge_encoding -> empty (E, n_rounds, 0) edge_attr, model still runs."""
    model = _make_model([])
    assert model.edge_encoding == []
    assert model.edge_encoder.size == 0
    data = _make_data()
    encoded = model.encode(data, y_encode=True, device="cpu")
    n_edges = 2 * 4 * 3
    assert encoded["edge_attr"].shape == (n_edges, 3, 0)
    out = model(encoded)
    assert out.shape == (2 * 4, 3, 21)


class _OnesContribution:
    """Stand-in contribution AH: lets the env step without a real GNN."""

    def __init__(self):
        self.default_values = {"contribution": 0}

    def predict(self, state, reset_rnn, edge_index):
        return (th.ones_like(state["contribution"]),)


class _FlipAgent0Switch:
    """Switch predictor that always flips agent 0's group."""

    def predict(self, state, reset_rnn, edge_index):
        ds = th.zeros_like(state["contribution"], dtype=th.bool)
        ds[:, 0, :] = True
        return ds, None


def _edge_attr_seen_by_model(env, edge_encoder):
    """Reproduce exactly what GraphNetwork.encode does: same_group from the
    env's current agent_group + the (fully-connected) batch_edge_index."""
    ag = env.state["agent_group"].flatten(0, 1)  # (N, n_rounds)
    return edge_encoder(
        edge_index=env.batch_edge_index, n_rounds=ag.shape[1], agent_group=ag
    )


def test_env_recomputes_same_group_after_switch():
    """In sim, when an agent switches groups the same_group edge feature the
    contribution model receives must update to the new membership."""
    env = ArtificialHumanEnv(
        artifical_humans=_OnesContribution(),
        artifical_humans_valid=None,
        artifical_humans_switch=_FlipAgent0Switch(),
        switch_every=4,
        batch_size=1,
        n_agents=8,
        n_contributions=3,
        n_punishments=3,
        n_rounds=5,
        n_groups=2,
        device="cpu",
        reward_mode="avg",
        agent_groups=[0, 0, 0, 0, 1, 1, 1, 1],
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
    ee = EdgeEncoder([{"name": "same_group", "etype": "bool"}], refrence="contribution")

    env.reset()
    assert env.state["agent_group"].flatten().tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    before = _edge_attr_seen_by_model(env, ee)

    # real switch path: flips agent 0 from group 0 -> 1 and syncs state
    env.update_groups_from_switch_predictor()
    assert env.state["agent_group"].flatten().tolist() == [1, 0, 0, 0, 1, 1, 1, 1]
    after = _edge_attr_seen_by_model(env, ee)

    row, col = env.batch_edge_index
    e01 = int(((row == 0) & (col == 1)).nonzero()[0])  # agent0 - a group-0 peer
    e04 = int(((row == 0) & (col == 4)).nonzero()[0])  # agent0 - a group-1 peer
    # before: agent0 in g0 -> same as agent1, different from agent4
    assert before[e01, 0, 0] == 1.0 and before[e04, 0, 0] == 0.0
    # after the switch: agent0 in g1 -> different from agent1, same as agent4
    assert after[e01, 0, 0] == 0.0 and after[e04, 0, 0] == 1.0
    assert not th.equal(before, after)


def test_save_load_round_trips_edge_encoding():
    model = _make_model([{"name": "same_group", "etype": "bool"}])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        loaded = GraphNetwork.load(path, device="cpu")
        assert loaded.edge_encoding == [{"name": "same_group", "etype": "bool"}]
        assert loaded.edge_encoder.size == 1

        # legacy checkpoint without the key still loads (defaults to no edges)
        saved = th.load(path, map_location="cpu")
        del saved["edge_encoding"]
        legacy_path = os.path.join(d, "legacy.pt")
        th.save(saved, legacy_path)
        legacy = GraphNetwork.load(legacy_path, device="cpu")
        assert legacy.edge_encoding == []
        assert legacy.edge_encoder.size == 0
