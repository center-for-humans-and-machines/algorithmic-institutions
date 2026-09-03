"""Tests for the DETACH between the trunk and the joint exodus head.

Runs locally on macOS with plain pytest::

    PYTHONPATH=$PWD/src python -m pytest tests/switch/test_joint_exodus_detach.py -q

The cut itself lives in ``joint_exodus.JointExodusHead.forward`` (plan step 2);
what is under test here is its consequence for the training objective of plan
step 5, which is the only place the cut can be observed at all.

Stand-ins for ``torch_scatter`` / ``torch_geometric.nn`` are installed only
when the real packages are missing -- the same discipline as
tests/switch/test_joint_exodus_graph.py and test_joint_exodus_loss.py -- so on
Raven this file exercises the real PyG. ``STAND_INS`` lists what THIS file
installed, so it is empty both on Raven and whenever a sibling suite in the
same pytest process imported first. Every assertion here is an INVARIANCE (head-on
trunk gradients equal head-off trunk gradients) or an EXCLUSION (the joint
term reaches the head and nothing above it), both evaluated with the same
message-passing implementation on either side, so a stand-in cannot
manufacture a pass.

The claim under test: the joint loss is a readout objective. It trains the
head's own MLP with full gradient and leaves the message-passing layers, the
RNNs and the encoders with exactly the gradient they would have had with no
head at all -- which is what makes the candidate the base model's trunk plus a
head, and any score movement the mechanism's.

Context: notes/autoresearch_log/switch-joint-exodus-gmlp.md.
"""

import importlib
import sys
import types

import torch as th


# --------------------------------------------------------------------------- #
# stand-ins (macOS only)
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


class _MetaLayer(th.nn.Module):
    """``torch_geometric.nn.MetaLayer``: edge model, then node model, then
    global model, each fed the outputs of the previous one."""

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


class _Tqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self._iterable = [] if iterable is None else iterable

    def __iter__(self):
        return iter(self._iterable)

    def set_postfix(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _install_stand_ins():
    installed = []
    try:
        importlib.import_module("torch_scatter")
        importlib.import_module("torch_geometric.nn")
    except ImportError:
        scatter = types.ModuleType("torch_scatter")
        scatter.scatter_mean = _scatter_mean
        sys.modules.setdefault("torch_scatter", scatter)
        geometric = types.ModuleType("torch_geometric")
        geometric_nn = types.ModuleType("torch_geometric.nn")
        geometric_nn.MetaLayer = _MetaLayer
        geometric.nn = geometric_nn
        sys.modules.setdefault("torch_geometric", geometric)
        sys.modules.setdefault("torch_geometric.nn", geometric_nn)
        installed.append("torch_geometric")
    try:
        importlib.import_module("tqdm")
    except ImportError:
        tqdm_mod = types.ModuleType("tqdm")
        tqdm_mod.tqdm = _Tqdm
        sys.modules.setdefault("tqdm", tqdm_mod)
        installed.append("tqdm")
    return installed


STAND_INS = _install_stand_ins()

from aimanager.artificial_humans.train import (  # noqa: E402
    compute_batch_loss,
    joint_exodus_loss,
)
from aimanager.generic.graph import GraphNetwork  # noqa: E402
from aimanager.generic.joint_exodus import SIZE_NORM, pool_by_group  # noqa: E402

N_AGENTS = 8
N_ROUNDS = 24
HEAD_PREFIX = "joint_exodus_head."


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_model(seed=0, **kwargs):
    th.manual_seed(seed)
    model = GraphNetwork(
        y_levels=2,
        y_name="does_switch",
        hidden_size=4,
        add_rnn=True,
        add_edge_model=True,
        add_global_model=False,
        x_encoding=[
            {"name": "agent_group", "n_levels": 2, "encoding": "numeric"},
            {"name": "round_number", "n_levels": N_ROUNDS, "encoding": "numeric"},
        ],
        edge_encoding=[],
        default_values={"does_switch": 0},
        **kwargs,
    )
    return model.to("cpu")


def make_state(groups, switches, valid, n_rounds=2, round_number=3):
    n_batch, n_player = len(groups), len(groups[0])
    shape = (n_batch, n_player, n_rounds)

    def spread(rows, dtype):
        return (
            th.tensor(rows, dtype=dtype)
            .reshape(n_batch, n_player, 1)
            .expand(shape)
            .contiguous()
        )

    return {
        "does_switch": spread(switches, th.bool),
        "switch_valid": spread(valid, th.bool),
        "agent_group": spread(groups, th.int64),
        "round_number": th.full(shape, round_number, dtype=th.int64),
    }


def encode(model, state, n_batch):
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=n_batch)
    return model.encode(state, mask="switch_valid", edge_index=edge_index, device="cpu")


#: A batch with two complete pairs, so the joint term is non-empty and the
#: per-agent term sees both labels.
GROUPS = [[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]]
SWITCH = [[1, 0, 0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0, 1, 0]]
VALID = [[True] * 8, [True] * 8]


def training_step(model, l1_entropy=0.0):
    """One optimiser-shaped step: forward, loss, backward. No ``step()``, so
    the gradients stay on the parameters for inspection."""
    state = make_state(GROUPS, SWITCH, VALID)
    loss_fn = th.nn.CrossEntropyLoss(reduction="none")
    model.zero_grad()
    loss, components = compute_batch_loss(
        model, encode(model, state, len(GROUPS)), loss_fn, l1_entropy
    )
    loss.backward()
    return components


def split_parameters(model):
    trunk, head = {}, {}
    for name, param in model.named_parameters():
        (head if name.startswith(HEAD_PREFIX) else trunk)[name] = param
    return trunk, head


# --------------------------------------------------------------------------- #
# 1. the point of the cut: the trunk does not feel the joint term
# --------------------------------------------------------------------------- #
def test_trunk_gradients_are_bitwise_identical_with_the_head_on_and_off():
    """Two identically seeded models -- one with the joint head, one without --
    take one training step on the same batch. Every parameter the two share
    must come out of ``backward()`` with the SAME gradient, bit for bit: that
    is what "the trunk is optimised by the per-agent loss alone" means
    operationally, and it is why the candidate's trunk is the base model's
    trunk rather than a re-fitted one."""
    off = make_model(seed=31)
    on = make_model(seed=31, joint_exodus=True)

    off_trunk, off_head = split_parameters(off)
    on_trunk, on_head = split_parameters(on)
    assert not off_head, "the head-off model must carry no head parameters"
    assert on_head, "the head-on model must carry head parameters"
    assert set(on_trunk) == set(off_trunk)
    # the head is constructed last, so the shared weights start identical --
    # without this the gradient comparison would be meaningless
    for name, param in on_trunk.items():
        assert th.equal(param, off_trunk[name]), name

    off_components = training_step(off)
    on_components = training_step(on)
    assert off_components["joint"] is None
    assert on_components["joint"] is not None and on_components["n_joint"] == 4

    for name, param in on_trunk.items():
        ref = off_trunk[name].grad
        assert (param.grad is None) == (ref is None), name
        if ref is not None:
            assert th.equal(param.grad, ref), name

    # ... and the head itself is trained, with real gradient on every one of
    # its own tensors: detached input, fully attached parameters.
    assert len(on_head) == 4  # two Linear layers, weight + bias
    for name, param in on_head.items():
        assert param.grad is not None, name
        assert bool(th.isfinite(param.grad).all()), name
        assert float(param.grad.abs().sum()) > 0.0, name


def test_the_trunk_gradient_is_the_per_agent_gradient_alone():
    """The same claim stated against the model's own per-agent term rather than
    against a second model: with the head on, backward-ing the TOTAL loss and
    backward-ing only the per-agent component leave the trunk in the same
    state. Anything the joint term contributed to a trunk parameter would show
    up as a difference here."""
    state = make_state(GROUPS, SWITCH, VALID)
    loss_fn = th.nn.CrossEntropyLoss(reduction="none")

    total_model = make_model(seed=37, joint_exodus=True)
    total, components = compute_batch_loss(
        total_model, encode(total_model, state, len(GROUPS)), loss_fn, 0.0
    )
    total.backward()

    agent_model = make_model(seed=37, joint_exodus=True)
    encoded = encode(agent_model, state, len(GROUPS))
    y_logit = agent_model(encoded).flatten(end_dim=-2)
    y_true = encoded["y_enc"].flatten(end_dim=-2)
    node_mask = encoded["mask"].flatten()
    agent = (loss_fn(y_logit, y_true) * node_mask).sum() / node_mask.sum()
    agent.backward()

    assert abs(float(total) - (components["agent"] + components["joint"])) < 1e-6
    assert components["joint"] > 0.0
    total_trunk, _ = split_parameters(total_model)
    agent_trunk, _ = split_parameters(agent_model)
    for name, param in total_trunk.items():
        ref = agent_trunk[name].grad
        assert (param.grad is None) == (ref is None), name
        if ref is not None:
            assert th.equal(param.grad, ref), name


def test_the_joint_term_alone_moves_the_head_and_nothing_above_it():
    """The exclusion, isolated: backward ONLY the joint cross-entropy. Every
    trunk parameter must come back with no gradient at all, while the head's
    parameters get a real one. Without the detach this test fails on the first
    trunk tensor -- it is the one that has teeth."""
    model = make_model(seed=41, joint_exodus=True)
    state = make_state(GROUPS, SWITCH, VALID)
    encoded = encode(model, state, len(GROUPS))
    model.zero_grad()

    _, joint = model(encoded, True, True)
    joint_loss, n_cells = joint_exodus_loss(joint, encoded)
    assert n_cells == 4
    assert joint_loss.requires_grad, "the joint term must still be differentiable"
    joint_loss.backward()

    trunk, head = split_parameters(model)
    for name, param in trunk.items():
        assert param.grad is None or float(param.grad.abs().sum()) == 0.0, name
    for name, param in head.items():
        assert param.grad is not None and float(param.grad.abs().sum()) > 0.0, name


def test_the_cut_is_exactly_the_pooled_embedding():
    """Where the cut is, and what it costs the forward pass -- nothing.

    Intercept the head's own input (the post-RNN node embeddings) and the
    input its MLP is actually handed. The pooled block of that input must
    still hold the true pooled embeddings, VALUE for value: detaching moves
    no number, so the fitted joint and the step-6 sampling are the same
    objects they would be with the head attached. And that input must carry
    no gradient at all -- with the sizes and the round being integer
    constants, detaching the embedding leaves the MLP a constant input, so
    the head's parameters are the only differentiable thing left in it."""
    model = make_model(seed=43, joint_exodus=True)
    head = model.joint_exodus_head
    seen = {}

    class CaptureHead(th.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.embed_size = inner.embed_size
            self.round_norm = inner.round_norm

            class CaptureMLP(th.nn.Module):
                def __init__(self, mlp):
                    super().__init__()
                    self.mlp = mlp

                def forward(self, features):
                    seen["features"] = features
                    return self.mlp(features)

            inner.mlp = CaptureMLP(inner.mlp)

        def forward(self, x, **kwargs):
            seen["x"] = x
            return self.inner(x, **kwargs)

    model.joint_exodus_head = CaptureHead(head)
    encoded = encode(model, make_state(GROUPS, SWITCH, VALID), len(GROUPS))
    _, joint = model(encoded, True, True)

    x, features = seen["x"], seen["features"]
    assert x.requires_grad, "the head reads a trunk tensor that carries gradient"

    # the pooled block is the head's embedding input; the tail is the two
    # normalised sizes and the round, which were never differentiable
    n_pooled = head.n_groups * head.embed_size
    pooled, counts = pool_by_group(
        x,
        encoded["agent_group"],
        encoded["batch"],
        n_batch=len(GROUPS),
        mask=encoded["mask"],
    )
    assert th.equal(features[..., :n_pooled], pooled.flatten(-2, -1).detach())
    assert th.equal(
        features[..., n_pooled : n_pooled + head.n_groups],
        counts.round().to(th.int64).to(x.dtype) / SIZE_NORM,
    )

    # the whole MLP input is a constant -- there is no path back to the trunk
    assert not features.requires_grad, "the joint head still reaches the trunk"
    # while the head's output is differentiable, through its own weights alone
    assert joint[0].requires_grad
