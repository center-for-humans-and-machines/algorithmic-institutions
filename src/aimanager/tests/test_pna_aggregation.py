"""PNA-style multi-aggregator at op1's peer-message aggregation (step 3 of
notes/autoresearch_log/contribution-pna-aggregation.md).

Runs in BOTH environments, and every test records which one it ran in (the
``pyg_environment`` property, and ``test_the_environment_is_what_it_reports``
asserts the stand-ins were installed exactly when the real packages are
absent):

    local   PYTHONPATH=$PWD/src pytest src/aimanager/tests/test_pna_aggregation.py
    Raven   scripts/remote_test.sh -- -k pna_aggregation

WHAT IS AT STAKE. The change is an architecture change on a slot whose every
other artifact was trained before it existed. Two things therefore have to be
true at once: the new path must aggregate what it claims to aggregate, and the
old path must be untouched to the bit -- a legacy `.pt` unpickles a
``NodeModel`` with no ``aggregators`` attribute at all, and step 13 of the plan
gates the whole experiment on a control simulation reproducing the parent's
`per_round.parquet` sha256. The legacy tests below (b/c) are that gate in
miniature; the correctness tests (a) are what licenses reading the new
aggregators as mean/max/min/std of the incoming messages.

THE STAND-INS ARE NOT PLAIN REDUCTIONS. `torch_scatter.scatter_max` and
`scatter_min` return a ``(values, argmax)`` PAIR -- graph.py's wrappers take
``[0]`` -- and `scatter_std` takes the ``unbiased`` keyword graph.py passes as
``False``. A stand-in returning a bare tensor, or rejecting the keyword, fails
on the unwrapping rather than on the behaviour under test. The three stand-ins
below reproduce the real signatures, and `_scatter_std` reproduces
torch_scatter's own formula (sum of squared deviations over a clamped count,
with its ``+ 1e-6``) so the local and Raven numbers agree to float32.
"""

import importlib
import importlib.util
import os
import sys
import tempfile
import types

import pytest
import torch as th


# --------------------------------------------------------------------------- #
# PyG stand-ins (macOS only)
# --------------------------------------------------------------------------- #
def _scatter_mean(src, index, dim=0, dim_size=None):
    assert dim == 0, "the stand-in only implements dim=0, which is all graph.py uses"
    index = index.reshape(-1).to(th.int64)
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() else 0
    out = th.zeros((dim_size, *src.shape[1:]), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    count = th.zeros(dim_size, dtype=src.dtype, device=src.device)
    count.index_add_(0, index, th.ones_like(index, dtype=src.dtype))
    shape = (dim_size,) + (1,) * (src.dim() - 1)
    return out / count.reshape(shape).clamp(min=1.0)


def _scatter_extreme(src, index, dim, dim_size, largest):
    """`scatter_max` / `scatter_min`: a (values, argmax) PAIR, not a tensor.

    Empty neighbourhoods keep torch_scatter's zero fill (it initialises `out`
    to the reduction's identity and masks that sentinel back to 0 when it
    allocated `out` itself); see the isolated-node test below.
    """
    assert dim == 0, "the stand-in only implements dim=0, which is all graph.py uses"
    index = index.reshape(-1).to(th.int64)
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() else 0
    values = th.zeros((dim_size, *src.shape[1:]), dtype=src.dtype, device=src.device)
    arg = th.full(values.shape, src.size(0), dtype=th.int64, device=src.device)
    rows = th.arange(src.size(0), device=src.device)
    for node in range(dim_size):
        mask = index == node
        if not bool(mask.any()):
            continue
        picked = src[mask]
        value, where = picked.max(dim=0) if largest else picked.min(dim=0)
        values[node] = value
        arg[node] = rows[mask][where]
    return values, arg


def _scatter_max(src, index, dim=0, dim_size=None):
    return _scatter_extreme(src, index, dim, dim_size, True)


def _scatter_min(src, index, dim=0, dim_size=None):
    return _scatter_extreme(src, index, dim, dim_size, False)


def _scatter_std(src, index, dim=0, dim_size=None, unbiased=True):
    """torch_scatter's own std, `unbiased` keyword included: graph.py passes
    ``unbiased=False`` so a degree-1 node reduces to 0 rather than dividing by
    zero."""
    assert dim == 0, "the stand-in only implements dim=0, which is all graph.py uses"
    index = index.reshape(-1).to(th.int64)
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() else 0
    shape = (dim_size,) + (1,) * (src.dim() - 1)
    count = th.zeros(dim_size, dtype=src.dtype, device=src.device)
    count.index_add_(0, index, th.ones_like(index, dtype=src.dtype))
    count = count.reshape(shape).clamp(min=1.0)
    total = th.zeros((dim_size, *src.shape[1:]), dtype=src.dtype, device=src.device)
    total.index_add_(0, index, src)
    deviation = src - (total / count)[index]
    out = th.zeros_like(total)
    out.index_add_(0, index, deviation * deviation)
    if unbiased:
        count = (count - 1.0).clamp(min=1.0)
    return (out / (count + 1e-6)).sqrt()


class _MetaLayer(th.nn.Module):
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None):
        row, col = edge_index
        if self.edge_model is not None:
            edge_attr = self.edge_model(
                x[row], x[col], edge_attr, u, batch if batch is None else batch[row]
            )
        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, batch)
        if self.global_model is not None:
            u = self.global_model(x, edge_index, edge_attr, u, batch)
        return x, edge_attr, u


def _install_pyg_stand_ins():
    try:
        importlib.import_module("torch_scatter")
        importlib.import_module("torch_geometric.nn")
        return False
    except ImportError:
        pass
    scatter = types.ModuleType("torch_scatter")
    scatter.scatter_mean = _scatter_mean
    scatter.scatter_max = _scatter_max
    scatter.scatter_min = _scatter_min
    scatter.scatter_std = _scatter_std
    sys.modules.setdefault("torch_scatter", scatter)
    geometric = types.ModuleType("torch_geometric")
    geometric_nn = types.ModuleType("torch_geometric.nn")
    geometric_nn.MetaLayer = _MetaLayer
    geometric.nn = geometric_nn
    sys.modules.setdefault("torch_geometric", geometric)
    sys.modules.setdefault("torch_geometric.nn", geometric_nn)
    return True


STAND_INS = _install_pyg_stand_ins()
ENVIRONMENT = "stand-ins (no PyG)" if STAND_INS else "real PyG"

from aimanager.generic import graph as graph_module  # noqa: E402
from aimanager.generic.graph import (  # noqa: E402
    AGGREGATORS,
    GraphNetwork,
    NodeModel,
)

SEED = 20260915
N_AGENTS = 8
N_LEVELS = 21
HIDDEN = 20
N_ROUNDS = 4
N_BATCH = 3
GROUPS = [0, 0, 0, 0, 1, 1, 1, 1]
ALL_FOUR = ["mean", "max", "min", "std"]
SAME_GROUP = [{"name": "same_group", "etype": "bool"}]
# The M0 trunk's node features (configs/training/artificial_humans/
# contribution/group_switching_contribution_50ep.yml): two numerics and a
# 2-level onehot, so x_features is 4 -- the plan's "3 + 80" miscounts the
# onehot as one column; the width asserted below is the computed one.
X_ENCODING = [
    {"name": "prev_contribution", "n_levels": 21, "encoding": "numeric"},
    {"name": "prev_punishment", "n_levels": 31, "encoding": "numeric"},
    {"name": "agent_group", "n_levels": 2, "encoding": "onehot"},
]
X_FEATURES = 4
# Population std over 7 messages is a sum of squared deviations divided by a
# count, against torch's own two-pass std: float32 agreement, not bit
# identity. max/min are compared exactly.
TOL = dict(rtol=1e-5, atol=1e-6)


@pytest.fixture(autouse=True)
def _report_environment(record_property):
    record_property("pyg_environment", ENVIRONMENT)


def make_model(
    seed=SEED,
    hidden_size=HIDDEN,
    edge_encoding=(),
    add_edge_model=True,
    **kwargs,
):
    """M0-shaped contribution model (21 levels, RNN + edge model, no global)."""
    th.manual_seed(seed)
    model = GraphNetwork(
        y_levels=N_LEVELS,
        y_name="contribution",
        hidden_size=hidden_size,
        add_rnn=True,
        add_edge_model=add_edge_model,
        add_global_model=False,
        x_encoding=X_ENCODING,
        edge_encoding=list(edge_encoding),
        default_values={"contribution": 0},
        **kwargs,
    )
    return model.to("cpu")


def make_data(n_batch=N_BATCH, n_rounds=N_ROUNDS, seed=7):
    """Simulation-shaped state, (n_batch, n_agents, n_rounds)."""
    th.manual_seed(seed)
    shape = (n_batch, N_AGENTS, n_rounds)
    agent_group = th.tensor(GROUPS, dtype=th.int64).reshape(1, N_AGENTS, 1)
    return {
        "contribution": th.zeros(shape, dtype=th.int64),
        "prev_contribution": th.randint(0, 21, shape),
        "prev_punishment": th.randint(0, 31, shape),
        "agent_group": agent_group.expand(shape).contiguous(),
        "round_number": th.full(shape, 0, dtype=th.int64),
    }


def logits(model, data=None, edge_index=None):
    data = make_data() if data is None else data
    encoded = model.encode(data, y_encode=False, edge_index=edge_index, device="cpu")
    model.eval()
    return model(encoded, reset_rnn=True)


def node_model_inputs(n_batch=N_BATCH, n_rounds=N_ROUNDS, edge_features=5, seed=3):
    """x / edge_attr / u / batch of the shapes `GraphNetwork.forward` hands the
    node model, with the production edge builder (`create_fully_connected` is
    the ONLY edge builder on both the training and simulation paths)."""
    th.manual_seed(seed)
    edge_index = make_model().create_fully_connected(N_AGENTS, n_batch=n_batch)
    n_nodes = n_batch * N_AGENTS
    x = th.randn(n_nodes, n_rounds, X_FEATURES)
    edge_attr = th.randn(edge_index.shape[1], n_rounds, edge_features)
    u = th.randn(n_batch, n_rounds, 0)
    batch = th.tensor([i for i in range(n_batch) for _ in range(N_AGENTS)])
    return x, edge_index, edge_attr, u, batch


def brute_force(name, edge_attr, col, n_nodes):
    """Per-node Python loop over the INCOMING edges -- the reference the
    scatter reductions are checked against."""
    out = []
    for node in range(n_nodes):
        rows = edge_attr[col == node]
        if rows.shape[0] == 0:
            out.append(th.zeros(edge_attr.shape[1:], dtype=edge_attr.dtype))
        elif name == "mean":
            out.append(rows.mean(dim=0))
        elif name == "max":
            out.append(rows.max(dim=0).values)
        elif name == "min":
            out.append(rows.min(dim=0).values)
        elif name == "std":
            out.append(rows.std(dim=0, unbiased=False))
        else:
            raise ValueError(name)
    return th.stack(out, dim=0)


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
def test_the_environment_is_what_it_reports():
    """The stand-ins are installed exactly when the real packages are absent,
    and `graph.py` is bound to whichever set is live."""
    if STAND_INS:
        assert graph_module.scatter_mean is _scatter_mean
        assert graph_module.scatter_max is _scatter_max
        assert graph_module.scatter_min is _scatter_min
        assert graph_module.scatter_std is _scatter_std
    else:
        assert graph_module.scatter_mean.__module__.startswith("torch_scatter")
        assert importlib.import_module("torch_scatter") is not None


# --------------------------------------------------------------------------- #
# (a) correctness per aggregator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL_FOUR)
def test_each_aggregator_matches_a_brute_force_loop(name):
    _, edge_index, edge_attr, _, _ = node_model_inputs()
    _, col = edge_index
    n_nodes = N_BATCH * N_AGENTS
    got = AGGREGATORS[name](edge_attr, col, dim=0, dim_size=n_nodes)
    want = brute_force(name, edge_attr, col, n_nodes)
    assert got.shape == want.shape, f"{name} in {ENVIRONMENT}"
    if name in ("max", "min"):
        assert th.equal(got, want), f"{name} in {ENVIRONMENT}"
    else:
        assert th.allclose(got, want, **TOL), f"{name} in {ENVIRONMENT}"


def test_the_aggregator_table_is_the_four_declared_reductions():
    assert sorted(AGGREGATORS) == sorted(ALL_FOUR)


def test_node_model_concatenates_the_aggregators_in_config_order():
    """The node MLP's message block is the four brute-force reductions
    concatenated in the order the config lists them -- order included, since a
    permutation is silent in every width check."""
    x, edge_index, edge_attr, u, batch = node_model_inputs()
    _, col = edge_index
    n_nodes = x.shape[0]
    node_model = NodeModel(
        x_features=X_FEATURES,
        edge_features=edge_attr.shape[-1],
        u_features=0,
        out_features=HIDDEN,
        aggregators=ALL_FOUR,
    )
    got = node_model(x, edge_index, edge_attr, u, batch)
    messages = th.cat(
        [brute_force(name, edge_attr, col, n_nodes) for name in ALL_FOUR], dim=-1
    )
    want = node_model.node_mlp(th.cat([x, messages, u[batch]], dim=-1))
    assert th.allclose(got, want, **TOL), ENVIRONMENT


# --------------------------------------------------------------------------- #
# (b) legacy bit-identity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delete_attribute", [False, True])
def test_a_node_model_without_aggregators_is_the_legacy_mean(delete_attribute):
    """`delete_attribute=True` is the unpickled pre-change artifact: the
    instance has no ``aggregators`` in its ``__dict__`` at all, so `hasattr` is
    false and only the `getattr` default keeps it running."""
    x, edge_index, edge_attr, u, batch = node_model_inputs()
    node_model = NodeModel(
        x_features=X_FEATURES,
        edge_features=edge_attr.shape[-1],
        u_features=0,
        out_features=HIDDEN,
    )
    assert node_model.node_mlp.in_features == X_FEATURES + edge_attr.shape[-1]
    if delete_attribute:
        del node_model.__dict__["aggregators"]
        assert not hasattr(node_model, "aggregators")
    got = node_model(x, edge_index, edge_attr, u, batch)
    _, col = edge_index
    legacy = graph_module.scatter_mean(edge_attr, col, dim=0, dim_size=x.size(0))
    want = node_model.node_mlp(th.cat([x, legacy, u[batch]], dim=-1))
    assert th.equal(got, want), ENVIRONMENT


def _sibling_module():
    """`test_contribution_copula_graph.py`'s M0-shaped constructor, written
    before this change: the reference for 'a legacy GraphNetwork is what it
    always was'."""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_contribution_copula_graph.py",
    )
    spec = importlib.util.spec_from_file_location("_pna_sibling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_legacy_graph_network_is_bit_identical_to_before_the_change():
    sibling = _sibling_module()
    before = sibling.make_model(seed=SEED)  # the call as it was written
    after = sibling.make_model(seed=SEED, aggregators=None)  # the new keyword
    assert before.aggregators is None
    assert before.op1.node_model.aggregators is None
    before_sd, after_sd = before.state_dict(), after.state_dict()
    assert list(before_sd) == list(after_sd)
    for key, value in before_sd.items():
        assert th.equal(value, after_sd[key]), f"{key} in {ENVIRONMENT}"
    # The node MLP's input width is the pre-change one: x + 1 * edge + u.
    hidden = before.op1.node_model.node_mlp[0].out_features
    assert before.op1.node_model.node_mlp[0].in_features == X_FEATURES + hidden


def test_a_legacy_artifact_without_the_key_loads_as_the_single_mean():
    """A `.pt` saved before this change carries no ``aggregators`` key and its
    pickled `NodeModel` carries no such attribute. It must load, and predict,
    exactly as it does today."""
    model = make_model()
    data = make_data()
    expected = logits(model, data)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "legacy.pt")
        model.save(path)
        saved = th.load(path)
        assert saved["aggregators"] is None
        del saved["aggregators"]  # pre-change artifact: the key is absent
        del saved["op1"].node_model.__dict__["aggregators"]
        loaded = GraphNetwork(**saved)
    loaded = loaded.to("cpu")
    assert loaded.aggregators is None
    assert th.equal(logits(loaded, data), expected), ENVIRONMENT


# --------------------------------------------------------------------------- #
# (c) a single mean is the legacy path
# --------------------------------------------------------------------------- #
def test_a_single_mean_aggregator_equals_the_legacy_output():
    legacy = make_model()
    single = make_model(aggregators=["mean"])
    assert single.aggregators == ["mean"]
    # Same widths, same RNG stream: the weights are identical, not merely
    # loadable into one another.
    legacy_sd, single_sd = legacy.state_dict(), single.state_dict()
    assert list(legacy_sd) == list(single_sd)
    for key, value in legacy_sd.items():
        assert th.equal(value, single_sd[key]), key
    data = make_data()
    assert th.equal(logits(legacy, data), logits(single, data)), ENVIRONMENT


# --------------------------------------------------------------------------- #
# (d) save / load round trip
# --------------------------------------------------------------------------- #
def test_save_load_round_trips_the_aggregators():
    model = make_model(aggregators=ALL_FOUR)
    data = make_data()
    expected = logits(model, data)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pna.pt")
        model.save(path)
        assert th.load(path)["aggregators"] == ALL_FOUR
        loaded = GraphNetwork.load(path, device="cpu")
    assert loaded.aggregators == ALL_FOUR
    assert loaded.op1.node_model.aggregators == ALL_FOUR
    assert th.equal(logits(loaded, data), expected), ENVIRONMENT


@pytest.mark.parametrize(
    "saved_with, claimed",
    [
        (ALL_FOUR, ["mean"]),
        (ALL_FOUR, None),
        (None, ["mean", "max"]),
    ],
)
def test_the_load_path_rejects_an_aggregator_disagreement(saved_with, claimed):
    """An artifact cannot claim one aggregation and run another: the node
    MLP's input width is baked into the saved weights."""
    model = make_model(aggregators=saved_with)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pt")
        model.save(path)
        saved = th.load(path)
    saved["aggregators"] = claimed
    with pytest.raises(AssertionError, match="aggregates with"):
        GraphNetwork(**saved)


# --------------------------------------------------------------------------- #
# (e) four aggregators: widths and a live forward
# --------------------------------------------------------------------------- #
def test_four_aggregators_widen_op1_only():
    model = make_model(aggregators=ALL_FOUR)
    op1_in = model.op1.node_model.node_mlp[0].in_features
    # x (2 numerics + a 2-level onehot) + 4 aggregators x hidden.
    assert op1_in == X_FEATURES + 4 * HIDDEN == 84
    assert model.op1.node_model.node_mlp[0].out_features == HIDDEN
    # op2 reads the post-RNN node embedding and no edge features at all; it
    # stays on the default single mean.
    assert model.op2.node_model.aggregators is None
    assert model.op2.node_model.node_mlp.in_features == HIDDEN
    assert model.op2.node_model.node_mlp.out_features == N_LEVELS
    legacy_in = make_model().op1.node_model.node_mlp[0].in_features
    assert legacy_in == X_FEATURES + HIDDEN


@pytest.mark.parametrize("edge_encoding", [(), tuple(SAME_GROUP)])
def test_the_forward_runs_on_m0_shaped_data(edge_encoding):
    """Both arms of the plan: arm A (no edge features) and arm B (the
    `same_group` edge bit)."""
    model = make_model(aggregators=ALL_FOUR, edge_encoding=edge_encoding)
    assert model.edge_encoder.size == (1 if edge_encoding else 0)
    data = make_data()
    out = logits(model, data)
    assert out.shape == (N_BATCH * N_AGENTS, N_ROUNDS, N_LEVELS)
    assert th.isfinite(out).all(), ENVIRONMENT
    # The aggregators change the output; they are not decorative.
    legacy_out = logits(make_model(edge_encoding=edge_encoding), data)
    assert not th.allclose(out, legacy_out)


# --------------------------------------------------------------------------- #
# constructor validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "aggregators, kwargs",
    [
        (["mean", "mean"], {}),
        ([], {}),
        (["nope"], {}),
        ("mean", {}),  # a bare string is iterable, and would be silently wrong
        (ALL_FOUR, {"add_edge_model": False}),
    ],
)
def test_the_constructor_rejects_illegal_aggregators(aggregators, kwargs):
    with pytest.raises(AssertionError):
        make_model(aggregators=aggregators, **kwargs)


# --------------------------------------------------------------------------- #
# documentation: the empty neighbourhood
# --------------------------------------------------------------------------- #
def test_an_isolated_node_documents_the_empty_neighbourhood_fill():
    """NOT a live case, deliberately documented.

    `create_fully_connected` (graph.py, and `artificial_humans/train.py`) is
    the only edge builder on the training and the simulation path, so every
    node always has exactly ``n_nodes - 1`` incoming edges and no aggregator
    ever sees an empty neighbourhood. If a future change makes the topology
    sparse -- per-group aggregation, a dropped edge, a singleton group -- this
    is the answer, already written down: torch_scatter fills an unreferenced
    output row with ZERO for all four reductions (it allocates `out` at the
    reduction's identity and masks that sentinel back to 0), so an isolated
    node's messages read as "a peer at exactly 0", which is a legal
    contribution and therefore silent. Handle it there; do not read this test
    as permission to rely on it.

    The zero fill is not a guess: on Raven's torch_scatter 2.0.9 an
    unreferenced row of `scatter_max` / `scatter_min` comes back as 0.0 with
    its argmax entry set to ``src.size(0)``, and `scatter_mean` /
    `scatter_std` divide a zero sum by a count clamped to 1 (both are plain
    Python in the installed package).
    """
    edge_attr = th.randn(4, N_ROUNDS, 5)
    # Every edge points at node 1 or 2; node 0 receives nothing.
    col = th.tensor([1, 1, 2, 2])
    for name in ALL_FOUR:
        out = AGGREGATORS[name](edge_attr, col, dim=0, dim_size=3)
        assert th.equal(
            out[0], th.zeros_like(out[0])
        ), f"{name} fills an empty neighbourhood with {out[0]} in {ENVIRONMENT}"
