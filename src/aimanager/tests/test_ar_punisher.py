"""Tests for the autoregressive (``ar_punishment``) punisher.

Covers the gated edge encoder itself, the leakage guarantees it is supposed
to give (no other sub-group, no undecided source, no self-loop, no
current-round contribution), train/sim parity of the tensors that reach the
encoder, and the ``predict_autoreg`` loop with an RNN.

Mask polarity throughout: ``autoreg_mask`` True means *undecided* (still to
be predicted), matching ``apply_mask_pattern`` and ``predict_autoreg``.

PyG-dependent -- run on the Raven cluster (``scripts/remote_test.sh``).
"""

import os
import tempfile

import numpy as np
import pytest
import torch as th

from aimanager.artificial_humans.train import apply_mask_pattern
from aimanager.generic.graph import ARPunishmentEdgeEncoder, GraphNetwork

N_PUNISHMENTS = 31
DEFAULT_VALUES = {"punishment": 0}
EDGE_ENCODING = [{"name": "ar_punishment", "n_levels": N_PUNISHMENTS}]


def _make_model(seed=0, hidden_size=6):
    """The real punisher architecture (rnn_edge_50ep_doubled) plus AR edges."""
    th.manual_seed(seed)
    model = GraphNetwork(
        y_levels=N_PUNISHMENTS,
        y_name="punishment",
        autoregressive=True,
        hidden_size=hidden_size,
        add_rnn=True,
        add_edge_model=True,
        add_global_model=False,
        x_encoding=[
            {"name": "prev_contribution", "n_levels": 21, "encoding": "numeric"},
            {"name": "prev_punishment", "n_levels": 31, "encoding": "numeric"},
            {"name": "is_first", "etype": "bool"},
        ],
        edge_encoding=EDGE_ENCODING,
        default_values=DEFAULT_VALUES,
    )
    model.eval()
    return model.to("cpu")


def _make_data(groups, n_batch=1, n_rounds=2, seed=0):
    """Batch/agent/round data dict for the AR punisher.

    ``groups`` is a per-node list of per-round sub-group labels. Punishments
    are all >= 1 so masking to the default (0) is always visible.
    """
    agent_group = th.tensor(groups, dtype=th.int64)
    assert agent_group.shape[1] == n_rounds
    n_nodes = agent_group.shape[0]
    shape = (n_batch, n_nodes, n_rounds)
    th.manual_seed(seed)
    data = {
        "contribution": th.randint(0, 21, shape),
        "prev_contribution": th.randint(0, 21, shape),
        "punishment": th.randint(1, N_PUNISHMENTS, shape),
        "prev_punishment": th.randint(0, N_PUNISHMENTS, shape),
        "is_first": th.zeros(shape, dtype=th.bool),
        "punishment_valid": th.ones(shape, dtype=th.bool),
        "agent_group": agent_group.unsqueeze(0).repeat(n_batch, 1, 1),
    }
    data["is_first"][:, :, 0] = True
    # default state: nothing decided yet (all undecided)
    data["punishment_masked"] = data["punishment"].clone()
    data["autoreg_mask"] = th.ones(shape, dtype=th.bool)
    return data


def _clone(data):
    return {k: v.clone() for k, v in data.items()}


def _decide(data, decided):
    """Flip the given agents to decided (autoreg_mask False) in place."""
    data["autoreg_mask"][:, list(decided)] = False
    return data


def _encode(model, data, edge_index):
    with th.no_grad():
        return model.encode(data, y_encode=False, edge_index=edge_index, device="cpu")


def _logits(model, data, edge_index):
    with th.no_grad():
        return model(_encode(model, data, edge_index))


def _edge_index(model, data):
    n_batch, n_nodes, _ = data["punishment"].shape
    return model.create_fully_connected(n_nodes, n_batch=n_batch)


def _record_encode(model):
    """Wrap ``model.encode`` to capture every AR step's inputs and output."""
    records = []
    original = model.encode

    def spy(data, **kwargs):
        encoded = original(data, **kwargs)
        records.append(
            {
                "punishment_masked": data["punishment_masked"].clone(),
                "autoreg_mask": data["autoreg_mask"].clone(),
                "encoded": encoded,
            }
        )
        return encoded

    model.encode = spy
    return records


@pytest.fixture
def switch_scenario():
    """4 nodes, 2 rounds, one mid-episode switch, fully-connected graph.

    node0: [0, 0]   node1: [0, 1]   node2: [1, 1]   node3: [1, 1]
    """
    agent_group = th.tensor([[0, 0], [0, 1], [1, 1], [1, 1]], dtype=th.int64)
    pairs = [(i, j) for i in range(4) for j in range(4) if i != j]
    edge_index = th.tensor(pairs, dtype=th.int64).T  # (2, 12)
    return agent_group, edge_index, pairs


def test_ar_punishment_encoder_values(switch_scenario):
    """Gate = same per-round group AND decided source; value = p / 30."""
    agent_group, edge_index, pairs = switch_scenario
    #                    round0  round1
    punishment_masked = th.tensor([[3, 6], [0, 15], [30, 30], [10, 20]], dtype=th.int64)
    undecided = th.tensor(
        [
            [False, False],  # node0 decided in both rounds
            [False, True],  # node1 decided at r0, undecided at r1
            [True, True],  # node2 never decided
            [False, False],  # node3 decided in both rounds
        ]
    )
    enc = ARPunishmentEdgeEncoder(n_levels=N_PUNISHMENTS)
    assert enc.size == 2
    out = enc(
        edge_index=edge_index,
        agent_group=agent_group,
        punishment_masked=punishment_masked,
        autoreg_mask=undecided,
    )
    assert out.shape == (12, 2, 2)
    assert out.dtype == th.float

    idx = {p: k for k, p in enumerate(pairs)}
    gate, value = out[..., 0], out[..., 1]

    # (0, 1): same group at r0, split at r1 because node1 switched
    assert th.equal(gate[idx[(0, 1)]], th.tensor([1.0, 0.0]))
    assert th.allclose(value[idx[(0, 1)]], th.tensor([3 / 30, 0.0]))
    # (1, 0): node1 decided at r0 with level 0 -> gate on, value 0
    assert th.equal(gate[idx[(1, 0)]], th.tensor([1.0, 0.0]))
    assert th.allclose(value[idx[(1, 0)]], th.tensor([0.0, 0.0]))
    # (1, 2): split at r0; same group at r1 but node1 is undecided there
    assert th.equal(gate[idx[(1, 2)]], th.tensor([0.0, 0.0]))
    # (3, 2): groupmates in both rounds, decided source -> p / 30
    assert th.equal(gate[idx[(3, 2)]], th.tensor([1.0, 1.0]))
    assert th.allclose(value[idx[(3, 2)]], th.tensor([10 / 30, 20 / 30]))
    # (2, 3): source node2 is undecided -> gate off, max level suppressed
    assert th.equal(gate[idx[(2, 3)]], th.tensor([0.0, 0.0]))
    assert th.equal(value[idx[(2, 3)]], th.tensor([0.0, 0.0]))
    # (0, 2): different groups in both rounds
    assert th.equal(gate[idx[(0, 2)]], th.tensor([0.0, 0.0]))
    # the value channel is gated everywhere
    assert th.all(value[gate == 0.0] == 0.0)


def test_other_sub_group_punishments_do_not_leak():
    """Perturbing the other sub-group's decided punishments leaves the
    observed group's logits bit-identical."""
    model = _make_model()
    groups = [[0, 0]] * 3 + [[1, 1]] * 3
    data = _make_data(groups)
    edge_index = _edge_index(model, data)
    observed, other = [0, 1, 2], [3, 4, 5]
    # the other group has already been drawn, ours has not
    _decide(data, other)

    before = _logits(model, data, edge_index)
    perturbed = _clone(data)
    perturbed["punishment_masked"][:, other, -1] = (
        perturbed["punishment_masked"][:, other, -1] + 7
    ) % N_PUNISHMENTS
    after = _logits(model, perturbed, edge_index)

    assert th.equal(before[observed], after[observed])
    # the perturbation is real: the other group's own edge features changed
    assert not th.equal(
        _encode(model, data, edge_index)["edge_attr"],
        _encode(model, perturbed, edge_index)["edge_attr"],
    )


def test_undecided_source_punishments_do_not_leak():
    """A source whose autoreg_mask is True contributes nothing at all."""
    model = _make_model()
    data = _make_data([[0, 0]] * 4)
    edge_index = _edge_index(model, data)
    _decide(data, [0, 1, 3])  # node2 stays undecided

    before = _logits(model, data, edge_index)
    perturbed = _clone(data)
    perturbed["punishment_masked"][:, 2] = (
        perturbed["punishment_masked"][:, 2] + 11
    ) % N_PUNISHMENTS
    after = _logits(model, perturbed, edge_index)

    assert th.equal(before, after)
    assert th.equal(
        _encode(model, data, edge_index)["edge_attr"],
        _encode(model, perturbed, edge_index)["edge_attr"],
    )


def test_no_self_leak():
    """An agent's own punishment never reaches its own logits: the fully
    connected graph has no self-loops."""
    model = _make_model()
    data = _make_data([[0, 0]] * 4)
    edge_index = _edge_index(model, data)
    _decide(data, [0, 1, 2, 3])
    row, col = edge_index
    assert not bool((row == col).any())  # no self-loops by construction

    before = _logits(model, data, edge_index)
    perturbed = _clone(data)
    perturbed["punishment_masked"][:, 1, -1] = (
        perturbed["punishment_masked"][:, 1, -1] + 5
    ) % N_PUNISHMENTS
    after = _logits(model, perturbed, edge_index)

    assert th.equal(before[1], after[1])  # node1's own logits are unchanged
    assert not th.equal(before, after)  # its groupmates do see the change


def test_current_round_contribution_does_not_leak():
    """Features are prev_-anchored: the current round's contribution is
    invisible to the punisher."""
    model = _make_model()
    data = _make_data([[0, 0]] * 4)
    edge_index = _edge_index(model, data)
    _decide(data, [0, 2])

    before = _logits(model, data, edge_index)
    perturbed = _clone(data)
    perturbed["contribution"][:, :, -1] = (perturbed["contribution"][:, :, -1] + 9) % 21
    after = _logits(model, perturbed, edge_index)

    assert not th.equal(data["contribution"], perturbed["contribution"])
    assert th.equal(before, after)


def test_train_sim_parity_of_encoder_inputs():
    """At a fixed reveal set, the edge features the training path builds via
    apply_mask_pattern equal those predict_autoreg builds at the same step."""
    model = _make_model()
    base = _make_data([[0, 0], [0, 0], [1, 1], [1, 1]])
    n_batch, n_nodes, _ = base["punishment"].shape
    edge_index = _edge_index(model, base)

    records = _record_encode(model)
    seed = 7
    np.random.seed(seed)
    th.manual_seed(seed)
    y_pred, _ = model.predict(_clone(base), sample=True, edge_index=edge_index)

    np.random.seed(seed)
    order = np.random.permutation(np.arange(n_nodes))
    assert len(records) == n_nodes
    # step 0: nothing is decided, not even the agent being drawn
    assert bool(records[0]["autoreg_mask"].all())

    step = 2
    decided = th.tensor(order[:step].copy(), dtype=th.int64)
    undecided = th.tensor(order[step:].copy(), dtype=th.int64)

    # training path: the revealed agents carry their drawn level as truth,
    # the agents still to be predicted are the mask pattern
    train_data = _clone(base)
    train_data["punishment"][:, decided, -1] = y_pred[:, decided, -1]
    mask_pattern = th.zeros((n_batch, n_nodes), dtype=th.bool)
    mask_pattern[:, undecided] = True
    train_data = apply_mask_pattern(
        train_data, mask_pattern, "punishment", "punishment_valid", DEFAULT_VALUES
    )

    ar = records[step]
    assert th.equal(train_data["autoreg_mask"], ar["autoreg_mask"])
    # the masked targets genuinely differ outside the gate (training zeroes
    # the undecided agents' whole history, the sim keeps it) ...
    assert not th.equal(train_data["punishment_masked"], ar["punishment_masked"])
    # ... but the gate makes the encoder input identical
    train_encoded = _encode(model, train_data, edge_index)
    assert th.equal(train_encoded["edge_attr"], ar["encoded"]["edge_attr"])
    assert th.equal(train_encoded["x"], ar["encoded"]["x"])


def test_predict_autoreg_with_rnn_shapes_and_determinism():
    """predict_autoreg runs on an RNN model, matches predict_independent's
    shapes and is deterministic under fixed numpy/torch seeds."""
    model = _make_model()
    data = _make_data([[0, 0]] * 4, n_batch=2)
    n_batch, n_nodes, n_rounds = data["punishment"].shape
    edge_index = _edge_index(model, data)
    assert model.rnn_n is not None
    assert model.autoregressive

    np.random.seed(3)
    th.manual_seed(3)
    pred_a, proba_a = model.predict(_clone(data), sample=True, edge_index=edge_index)
    np.random.seed(3)
    th.manual_seed(3)
    pred_b, proba_b = model.predict(_clone(data), sample=True, edge_index=edge_index)

    assert pred_a.shape == (n_batch, n_nodes, n_rounds)
    assert proba_a.shape == (n_batch, n_nodes, n_rounds, N_PUNISHMENTS)
    assert th.equal(pred_a, pred_b)
    assert th.equal(proba_a, proba_b)
    assert bool(((pred_a >= 0) & (pred_a < N_PUNISHMENTS)).all())

    ind_pred, ind_proba = model.predict_independent(
        _clone(data), sample=True, edge_index=edge_index
    )
    assert ind_pred.shape == pred_a.shape
    assert ind_proba.shape == proba_a.shape


def test_predict_autoreg_conditions_on_earlier_draws():
    """A later agent's distribution depends on the level drawn for an
    earlier-revealed groupmate."""
    model = _make_model()
    base = _make_data([[0, 0]] * 4)
    n_nodes = base["punishment"].shape[1]
    edge_index = _edge_index(model, base)

    records = _record_encode(model)
    seed = 11
    np.random.seed(seed)
    th.manual_seed(seed)
    model.predict(_clone(base), sample=True, edge_index=edge_index)
    np.random.seed(seed)
    order = np.random.permutation(np.arange(n_nodes))
    first, second = int(order[0]), int(order[1])

    # step 1 state: exactly the first agent is decided
    step1 = records[1]
    assert not bool(step1["autoreg_mask"][:, first].any())
    others = [i for i in range(n_nodes) if i != first]
    assert bool(step1["autoreg_mask"][:, others].all())

    with th.no_grad():
        logits_a = model(step1["encoded"])
    conditioned = _clone(base)
    conditioned["punishment_masked"] = step1["punishment_masked"].clone()
    conditioned["autoreg_mask"] = step1["autoreg_mask"].clone()
    # rebuilding the step faithfully reproduces the AR step's logits
    assert th.equal(logits_a, _logits(model, conditioned, edge_index))
    conditioned["punishment_masked"][:, first, -1] = (
        conditioned["punishment_masked"][:, first, -1] + 13
    ) % N_PUNISHMENTS
    logits_b = _logits(model, conditioned, edge_index)

    assert not th.equal(logits_a[second, -1], logits_b[second, -1])


def test_save_load_round_trips_ar_punisher():
    """save/load preserves the edge encoding and the autoregressive flag and
    reproduces predictions exactly."""
    model = _make_model()
    data = _make_data([[0, 0], [0, 0], [1, 1], [1, 1]])
    edge_index = _edge_index(model, data)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ar_punisher.pt")
        model.save(path)
        loaded = GraphNetwork.load(path, device="cpu")

    assert loaded.edge_encoding == EDGE_ENCODING
    assert loaded.edge_encoder.size == 2
    assert loaded.autoregressive is True
    assert loaded.y_name == "punishment"

    np.random.seed(5)
    th.manual_seed(5)
    pred_a, proba_a = model.predict(_clone(data), sample=False, edge_index=edge_index)
    np.random.seed(5)
    th.manual_seed(5)
    pred_b, proba_b = loaded.predict(_clone(data), sample=False, edge_index=edge_index)

    assert th.equal(pred_a, pred_b)
    assert th.equal(proba_a, proba_b)
