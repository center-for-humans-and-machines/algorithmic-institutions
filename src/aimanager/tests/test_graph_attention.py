import os
import tempfile

import torch as th
from torch_geometric.nn import MetaLayer

from aimanager.generic.graph import (
    AttentionMetaLayer,
    EdgeAttention,
    EdgeModel,
    GraphNetwork,
    NodeModel,
    SameGroupEdgeEncoder,
)

N_NODES = 4
N_ROUNDS = 2
X_FEATURES = 3
EDGE_FEATURES = 1
U_FEATURES = 2
HIDDEN = 6


def _graph():
    """One fully-connected 4-node graph, 2 rounds, same_group as edge feature.

    Groups are [0, 0, 1, 1], so every node sees a mix of same- and
    other-group incoming edges.
    """
    th.manual_seed(0)
    pairs = [(i, j) for i in range(N_NODES) for j in range(N_NODES) if i != j]
    edge_index = th.tensor(pairs, dtype=th.int64).T
    agent_group = th.tensor([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=th.int64)
    edge_attr = SameGroupEdgeEncoder()(edge_index=edge_index, agent_group=agent_group)
    x = th.randn(N_NODES, N_ROUNDS, X_FEATURES)
    u = th.randn(1, N_ROUNDS, U_FEATURES)
    batch = th.zeros(N_NODES, dtype=th.int64)
    return x, edge_index, edge_attr, u, batch


def _make_model(use_attention, edge_encoding):
    model = GraphNetwork(
        y_levels=21,
        y_name="contribution",
        hidden_size=4,
        add_rnn=True,
        add_edge_model=True,
        add_global_model=False,
        use_attention=use_attention,
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


def _logits(model, data):
    model.eval()
    with th.no_grad():
        return model(model.encode(data, y_encode=True, device="cpu"))


def test_node_model_without_weight_is_scatter_mean():
    """The back-compat hinge: no edge_weight -> the uniform mean, exactly."""
    x, edge_index, edge_attr, u, batch = _graph()
    node_model = NodeModel(X_FEATURES, EDGE_FEATURES, U_FEATURES, out_features=5)
    out = node_model(x, edge_index, edge_attr, u, batch)

    col = edge_index[1]
    agg = th.stack([edge_attr[col == n].mean(dim=0) for n in range(N_NODES)])
    expected = node_model.node_mlp(th.cat([x, agg, u[batch]], dim=-1))
    assert th.allclose(out, expected, atol=1e-6)

    # a uniform 1/deg weight reproduces the same aggregation
    deg = th.bincount(col, minlength=N_NODES).float()
    weight = (1.0 / deg[col]).reshape(-1, 1, 1).expand(-1, N_ROUNDS, 1)
    weighted = node_model(x, edge_index, edge_attr, u, batch, edge_weight=weight)
    assert th.allclose(weighted, out, atol=1e-6)


def test_default_graph_network_keeps_metalayer():
    model = _make_model(False, [])
    assert not model.use_attention
    assert isinstance(model.op1, MetaLayer)
    assert not isinstance(model.op1, AttentionMetaLayer)


def test_attention_weights_sum_to_one_per_destination():
    x, edge_index, edge_attr, u, batch = _graph()
    att = EdgeAttention(X_FEATURES, EDGE_FEATURES, U_FEATURES)
    row, col = edge_index
    alpha = att(x[row], x[col], edge_attr, u, batch[row], col, N_NODES)
    assert alpha.shape == (edge_index.shape[1], N_ROUNDS, 1)
    sums = th.stack([alpha[col == n].sum(dim=0) for n in range(N_NODES)])
    assert th.allclose(sums, th.ones_like(sums), atol=1e-6)


def test_zeroed_score_reproduces_the_metalayer():
    x, edge_index, edge_attr, u, batch = _graph()
    edge_model = EdgeModel(X_FEATURES, EDGE_FEATURES, U_FEATURES, out_features=HIDDEN)
    node_model = NodeModel(X_FEATURES, HIDDEN, U_FEATURES, out_features=5)
    att = EdgeAttention(X_FEATURES, EDGE_FEATURES, U_FEATURES)
    th.nn.init.zeros_(att.score.weight)
    th.nn.init.zeros_(att.score.bias)

    plain = MetaLayer(edge_model, node_model, None)
    attentive = AttentionMetaLayer(edge_model, node_model, None, att)
    x_plain, edge_plain, _ = plain(x, edge_index, edge_attr, u, batch)
    x_att, edge_att, _ = attentive(x, edge_index, edge_attr, u, batch)
    assert th.allclose(x_att, x_plain, atol=1e-6)
    assert th.allclose(edge_att, edge_plain, atol=1e-6)


def test_attention_responds_to_same_group():
    x, edge_index, edge_attr, u, batch = _graph()
    att = EdgeAttention(X_FEATURES, EDGE_FEATURES, U_FEATURES)
    th.nn.init.zeros_(att.score.weight)
    th.nn.init.zeros_(att.score.bias)
    with th.no_grad():
        # score input layout: [src, dest, edge_attr, u] -> the edge slot
        att.score.weight[0, 2 * X_FEATURES] = 5.0

    row, col = edge_index
    alpha = att(x[row], x[col], edge_attr, u, batch[row], col, N_NODES)
    flipped = att(x[row], x[col], 1.0 - edge_attr, u, batch[row], col, N_NODES)
    assert not th.allclose(alpha, flipped, atol=1e-4)


def test_attention_model_save_load_round_trip():
    model = _make_model(True, [{"name": "same_group", "etype": "bool"}])
    assert isinstance(model.op1, AttentionMetaLayer)
    data = _make_data()
    before = _logits(model, data)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        loaded = GraphNetwork.load(path, device="cpu")
    assert loaded.use_attention
    assert isinstance(loaded.op1, AttentionMetaLayer)
    assert isinstance(loaded.op1.edge_attention, EdgeAttention)
    assert th.allclose(_logits(loaded, data), before, atol=1e-6)


def test_legacy_checkpoint_without_use_attention_loads():
    """Artifacts saved before the flag existed omit the key entirely."""
    model = _make_model(False, [])
    data = _make_data()
    before = _logits(model, data)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        saved = th.load(path, map_location="cpu")
        assert "use_attention" in saved
        del saved["use_attention"]
        legacy_path = os.path.join(d, "legacy.pt")
        th.save(saved, legacy_path)
        legacy = GraphNetwork.load(legacy_path, device="cpu")
    assert not legacy.use_attention
    assert isinstance(legacy.op1, MetaLayer)
    assert th.allclose(_logits(legacy, data), before, atol=1e-6)
