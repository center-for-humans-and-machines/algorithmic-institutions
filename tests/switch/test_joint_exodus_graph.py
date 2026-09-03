"""Tests for the joint exodus GATE in ``GraphNetwork`` (plan step 4).

Runs locally on macOS with plain pytest::

    PYTHONPATH=$PWD/src python -m pytest tests/switch/test_joint_exodus_graph.py -q

``aimanager.generic.graph`` imports ``torch_scatter`` and
``torch_geometric.nn``, which are Linux-only in this project (see the
``sys_platform`` markers in pyproject.toml). Every assertion in this file is an
INVARIANCE -- head off equals the pre-change code, head on equals head off,
saved equals loaded -- evaluated on both sides with the same message-passing
implementation, so a stand-in for those two symbols cannot manufacture a pass.
The stand-ins are installed only when the real packages are missing, so on
Raven this file exercises the real PyG (``STAND_INS`` reports which).

Why the invariances matter more here than in the two sibling suites:
``GraphNetwork`` is NOT the switch model's private class. It is also the class
behind the contribution and punishment GNNs and behind this stack's own
``valid_model`` artifact, and every one of those artifacts predates this
change. A head-off behaviour difference would silently corrupt a slot this
experiment does not even study.

Numerics of the head itself live in tests/switch/test_joint_exodus.py, the
selection sampler in tests/switch/test_conditional_bernoulli.py. Context:
notes/autoresearch_log/switch-joint-exodus-gmlp.md.
"""

import importlib
import importlib.util
import os
import subprocess
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


def _install_pyg_stand_ins():
    """Install the two stand-ins ONLY if the real packages are absent.

    Returns True when stand-ins were installed (macOS), False when the real
    PyG is present (Raven). Exposed as ``STAND_INS`` so a reader of the test
    output can tell which implementation was exercised.
    """
    try:
        importlib.import_module("torch_scatter")
        importlib.import_module("torch_geometric.nn")
        return False
    except ImportError:
        pass
    scatter = types.ModuleType("torch_scatter")
    scatter.scatter_mean = _scatter_mean
    sys.modules.setdefault("torch_scatter", scatter)
    geometric = types.ModuleType("torch_geometric")
    geometric_nn = types.ModuleType("torch_geometric.nn")
    geometric_nn.MetaLayer = _MetaLayer
    geometric.nn = geometric_nn
    sys.modules.setdefault("torch_geometric", geometric)
    sys.modules.setdefault("torch_geometric.nn", geometric_nn)
    return True


STAND_INS = _install_pyg_stand_ins()

from aimanager.generic.graph import GraphNetwork  # noqa: E402
from aimanager.generic.joint_exodus import (  # noqa: E402
    MAX_GROUP_SIZE,
    JointExodusHead,
    joint_count_mask,
)

SEED = 20260903
N_AGENTS = 8
GROUPS = [0, 0, 0, 1, 1, 1, 1, 1]
GRID = MAX_GROUP_SIZE + 1
HIDDEN = 4

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPH_REL_PATH = "src/aimanager/generic/graph.py"


# --------------------------------------------------------------------------- #
# the pre-change baseline, taken from git rather than restated here
# --------------------------------------------------------------------------- #
def _git(*args):
    return subprocess.run(
        ("git", "-C", REPO_ROOT) + args,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _load_baseline_graph_module():
    """Import the last committed ``graph.py`` that predates this change.

    The pre-change sampling path is recovered from git -- the newest commit
    touching ``graph.py`` whose blob has no ``joint_exodus`` in it -- rather
    than restated in this file, so the identity check cannot drift from what
    the code actually used to do, and does not depend on any pinned sha
    surviving a rebase. Returns ``None`` when git or the history is not
    available (e.g. a source tree rsynced to the cluster without ``.git``);
    ``test_head_off_matches_the_pre_change_expression`` then still covers the
    same identity without git.
    """
    try:
        revs = _git("log", "--format=%H", "--", GRAPH_REL_PATH).split()
    except (OSError, subprocess.CalledProcessError):
        return None
    for rev in revs:
        try:
            source = _git("show", f"{rev}:{GRAPH_REL_PATH}")
        except subprocess.CalledProcessError:
            continue
        if "joint_exodus" in source:
            continue
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "graph_baseline.py")
            with open(path, "w") as fh:
                fh.write(source)
            spec = importlib.util.spec_from_file_location(
                "aimanager_graph_baseline", path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        module._baseline_rev = rev
        return module
    return None


BASELINE = _load_baseline_graph_module()
needs_baseline = pytest.mark.skipif(
    BASELINE is None,
    reason="pre-change graph.py not recoverable from git in this checkout",
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def make_model(seed=0, cls=None, **kwargs):
    """Small switch-shaped model; ``cls`` lets the baseline class stand in."""
    cls = GraphNetwork if cls is None else cls
    th.manual_seed(seed)
    model = cls(
        y_levels=2,
        y_name="does_switch",
        hidden_size=HIDDEN,
        add_rnn=True,
        add_edge_model=True,
        add_global_model=False,
        x_encoding=[
            {"name": "prev_contribution", "n_levels": 21, "encoding": "numeric"}
        ],
        edge_encoding=[],
        default_values={"does_switch": 0},
        **kwargs,
    )
    return model.to("cpu")


def make_data(n_batch=2, n_rounds=1, round_number=0, groups=GROUPS, seed=1):
    th.manual_seed(seed)
    shape = (n_batch, N_AGENTS, n_rounds)
    agent_group = th.tensor(groups, dtype=th.int64).reshape(1, N_AGENTS, 1)
    return {
        "does_switch": th.zeros(shape, dtype=th.int64),
        "prev_contribution": th.randint(0, 21, shape),
        "agent_group": agent_group.expand(shape).contiguous(),
        "round_number": th.full(shape, round_number, dtype=th.int64),
    }


def legacy_predict(model, data, edge_index=None, reset_rnn=True):
    """``predict_independent(sample=True)`` written out as the pre-change code
    expressed it, so the git-free identity check leans on neither the class
    under test nor git history."""
    n_batch, n_nodes, _ = data[model.y_name].shape
    if edge_index is None:
        edge_index = model.create_fully_connected(n_nodes, n_batch=n_batch)
    encoded = model.encode(
        data, y_encode=False, edge_index=edge_index, device=model.device
    )
    model.eval()
    y_logit = model(encoded, reset_rnn)
    proba = th.nn.functional.softmax(y_logit, dim=-1)
    dec = th.multinomial(proba.reshape(-1, proba.shape[-1]), 1)
    y_pred = dec.reshape(proba.shape[:-1])
    return tuple(t.reshape((n_batch, n_nodes, *t.shape[1:])) for t in (y_pred, proba))


def run_seeded(fn):
    """Run ``fn`` from a fixed RNG state and report the state it leaves.

    The returned state is the whole generator state, not a sample of it, so an
    extra or missing draw anywhere inside ``fn`` shows up.
    """
    th.manual_seed(SEED)
    out = fn()
    return out, th.get_rng_state().clone()


# --------------------------------------------------------------------------- #
# 1. head off: the gate is invisible
# --------------------------------------------------------------------------- #
def test_head_is_off_by_default():
    model = make_model()
    assert model.joint_exodus is False
    assert model.joint_exodus_head is None
    assert model.joint_exodus_switch_every is None
    data = make_data()
    encoded = model.encode(data, y_encode=False, device="cpu")
    # nothing extra is carried into the encoded state
    assert "agent_group" not in encoded
    assert "round_number" not in encoded


@needs_baseline
def test_head_off_encodes_exactly_the_pre_change_keys():
    """Not "no new keys I remembered to check" -- the exact key set the
    pre-change class produces, and the same tensors."""
    model = make_model(seed=7)
    baseline = make_model(seed=7, cls=BASELINE.GraphNetwork)
    data = make_data()
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=2)
    new = model.encode(data, y_encode=False, edge_index=edge_index, device="cpu")
    ref = baseline.encode(data, y_encode=False, edge_index=edge_index, device="cpu")
    assert set(new) == set(ref)
    for key, value in ref.items():
        assert th.equal(new[key], value), key


@needs_baseline
def test_head_off_sampling_is_bitwise_identical_to_the_pre_change_class():
    """The load-bearing back-compat gate: same weights out of the same seed,
    same prediction, same probabilities, and the global RNG left in exactly
    the same state."""
    model = make_model(seed=7)
    baseline = make_model(seed=7, cls=BASELINE.GraphNetwork)

    ref_state = baseline.state_dict()
    new_state = model.state_dict()
    assert set(new_state) == set(ref_state)
    for key, value in ref_state.items():
        assert th.equal(new_state[key], value), key

    data = make_data()
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=2)
    (ref_pred, ref_proba), ref_rng = run_seeded(
        lambda: baseline.predict_independent(data, sample=True, edge_index=edge_index)
    )
    (new_pred, new_proba), new_rng = run_seeded(
        lambda: model.predict_independent(data, sample=True, edge_index=edge_index)
    )
    assert th.equal(new_pred, ref_pred)
    assert th.equal(new_proba, ref_proba)
    assert th.equal(new_rng, ref_rng)


def test_head_off_matches_the_pre_change_expression():
    """The same identity without git, so it also holds where the history is
    not available."""
    model = make_model(seed=11)
    data = make_data()
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=2)
    (ref_pred, ref_proba), ref_rng = run_seeded(
        lambda: legacy_predict(model, data, edge_index=edge_index)
    )
    (new_pred, new_proba), new_rng = run_seeded(
        lambda: model.predict_independent(data, sample=True, edge_index=edge_index)
    )
    assert th.equal(new_pred, ref_pred)
    assert th.equal(new_proba, ref_proba)
    assert th.equal(new_rng, ref_rng)


def test_return_joint_is_none_when_the_head_is_off():
    model = make_model()
    data = make_data()
    encoded = model.encode(data, y_encode=False, device="cpu")
    model.eval()
    logit, joint = model(encoded, True, True)
    assert joint is None
    assert th.equal(logit, model(encoded))


def test_head_off_needs_neither_membership_nor_the_round():
    """A contribution / punishment artifact's state carries neither key; with
    the head off ``encode`` must not start asking for them."""
    model = make_model()
    data = make_data()
    del data["agent_group"]
    del data["round_number"]
    encoded = model.encode(data, y_encode=False, device="cpu")
    assert "x" in encoded


# --------------------------------------------------------------------------- #
# 2. head on: the trunk is untouched
# --------------------------------------------------------------------------- #
def test_head_on_shares_the_trunk_state_dict_and_logits():
    """The head is constructed LAST, so every shared parameter is initialised
    from exactly the RNG state it saw before the head existed; and the head
    branches off the post-RNN embeddings and feeds nothing back, so the
    per-agent logits are bit-identical."""
    off = make_model(seed=3)
    on = make_model(seed=3, joint_exodus=True, joint_exodus_switch_every=4)

    off_state = off.state_dict()
    on_state = on.state_dict()
    assert set(off_state).issubset(set(on_state))
    assert set(on_state) - set(off_state), "the head contributes no parameters"
    for key, value in off_state.items():
        assert th.equal(value, on_state[key]), key

    data = make_data()
    edge_index = on.create_fully_connected(N_AGENTS, n_batch=2)
    enc_off = off.encode(data, y_encode=False, edge_index=edge_index, device="cpu")
    enc_on = on.encode(data, y_encode=False, edge_index=edge_index, device="cpu")
    off.eval()
    on.eval()

    logit_off = off(enc_off)
    logit_on = on(enc_on)
    assert th.equal(logit_off, logit_on)

    # the joint return path returns the very same per-agent logits
    logit_on_joint, joint = on(enc_on, True, True)
    assert th.equal(logit_on_joint, logit_on)
    assert joint is not None


def test_head_on_leaves_sampling_alone_at_this_step():
    """Step 4 is the gate only: ``predict_independent`` is not touched, so
    even with the head on, sampling is still the pre-change expression --
    bit-for-bit and RNG-for-RNG. The joint draw arrives in step 6."""
    on = make_model(seed=3, joint_exodus=True, joint_exodus_switch_every=4)
    data = make_data()
    edge_index = on.create_fully_connected(N_AGENTS, n_batch=2)
    (ref_pred, _), ref_rng = run_seeded(
        lambda: legacy_predict(on, data, edge_index=edge_index)
    )
    (new_pred, _), new_rng = run_seeded(
        lambda: on.predict_independent(data, sample=True, edge_index=edge_index)
    )
    assert th.equal(new_pred, ref_pred)
    assert th.equal(new_rng, ref_rng)


def test_the_head_reads_the_post_rnn_width():
    """``embed_size`` is the post-RNN node width, not the input encoding's
    width and not the readout's. Confirmed against the RNN's own output rather
    than against a remembered number."""
    on = make_model(seed=3, joint_exodus=True)
    assert on.x_encoder.size != HIDDEN, "the test would be vacuous"
    assert on.joint_exodus_head.embed_size == HIDDEN

    widths = []
    handle = on.rnn_n.register_forward_hook(
        lambda _module, _inp, out: widths.append(out[0].shape[-1])
    )
    try:
        data = make_data()
        encoded = on.encode(data, y_encode=False, device="cpu")
        on.eval()
        on(encoded, True, True)
    finally:
        handle.remove()
    assert widths == [HIDDEN]
    assert on.joint_exodus_head.embed_size == widths[0]


# --------------------------------------------------------------------------- #
# 3. the head's output through the real encode path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "groups,k0,k1",
    [
        (GROUPS, 3, 5),
        ([0] * N_AGENTS, 8, 0),
        ([1] * N_AGENTS, 0, 8),
    ],
)
def test_joint_output_is_a_valid_masked_distribution(groups, k0, k1):
    model = make_model(seed=5, joint_exodus=True)
    data = make_data(n_batch=2, round_number=7, groups=groups)
    encoded = model.encode(data, y_encode=False, device="cpu")
    model.eval()
    _, (log_prob, k) = model(encoded, True, True)
    proba = log_prob.exp()

    assert log_prob.shape == (2, 1, GRID, GRID)
    assert th.all(k[..., 0] == k0) and th.all(k[..., 1] == k1)
    assert th.allclose(proba.sum((-2, -1)), th.ones(2, 1))
    valid = joint_count_mask(k)
    assert th.all(proba[~valid] == 0.0)
    assert not th.isnan(proba).any()


def test_decider_mask_shrinks_the_support():
    model = make_model(seed=5, joint_exodus=True)
    data = make_data(n_batch=1, round_number=7)
    encoded = model.encode(data, y_encode=False, device="cpu")
    mask = th.ones(N_AGENTS, 1, dtype=th.bool)
    mask[0] = False  # one group-0 member's selection timed out
    model.eval()
    _, (log_prob, k) = model(encoded, True, True, mask)
    assert th.all(k[..., 0] == 2) and th.all(k[..., 1] == 5)
    assert th.all(log_prob.exp()[0, 0, 3:] == 0.0)


@pytest.mark.parametrize("key", ["agent_group", "round_number"])
def test_encode_requires_the_head_s_inputs_when_it_is_on(key):
    model = make_model(seed=5, joint_exodus=True)
    data = make_data()
    del data[key]
    with pytest.raises(AssertionError, match=key):
        model.encode(data, y_encode=False, device="cpu")


# --------------------------------------------------------------------------- #
# 4. save / load, including the back-compat gate
# --------------------------------------------------------------------------- #
def test_save_load_round_trips_the_head():
    model = make_model(seed=5, joint_exodus=True, joint_exodus_switch_every=4)
    data = make_data(round_number=7)
    encoded = model.encode(data, y_encode=False, device="cpu")
    model.eval()
    _, (ref_log_prob, ref_k) = model(encoded, True, True)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        loaded = GraphNetwork.load(path, device="cpu")

        assert loaded.joint_exodus is True
        assert loaded.joint_exodus_switch_every == 4
        assert isinstance(loaded.joint_exodus_head, JointExodusHead)
        loaded.eval()
        enc = loaded.encode(data, y_encode=False, device="cpu")
        _, (log_prob, k) = loaded(enc, True, True)
        assert th.equal(log_prob, ref_log_prob)
        assert th.equal(k, ref_k)


def test_artifact_without_the_new_keys_loads_and_behaves_as_today():
    """The back-compat gate: every GNN artifact on this stack -- the switch
    base model, the contribution and punishment models, ``valid_model`` --
    was saved BEFORE this change and carries none of the three new keys. Such
    an artifact must load with the head absent and sample bit-identically to
    the pre-change path."""
    model = make_model(seed=5)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        saved = th.load(path, map_location="cpu")
        for key in (
            "joint_exodus",
            "joint_exodus_head",
            "joint_exodus_switch_every",
        ):
            assert key in saved
            del saved[key]
        legacy_path = os.path.join(d, "legacy.pt")
        th.save(saved, legacy_path)
        legacy = GraphNetwork.load(legacy_path, device="cpu")

    assert legacy.joint_exodus is False
    assert legacy.joint_exodus_head is None
    assert legacy.joint_exodus_switch_every is None

    data = make_data()
    edge_index = legacy.create_fully_connected(N_AGENTS, n_batch=2)
    (ref_pred, ref_proba), ref_rng = run_seeded(
        lambda: legacy_predict(legacy, data, edge_index=edge_index)
    )
    (new_pred, new_proba), new_rng = run_seeded(
        lambda: legacy.predict_independent(data, sample=True, edge_index=edge_index)
    )
    assert th.equal(new_pred, ref_pred)
    assert th.equal(new_proba, ref_proba)
    assert th.equal(new_rng, ref_rng)


# --------------------------------------------------------------------------- #
# 5. guards
# --------------------------------------------------------------------------- #
def test_head_is_only_defined_for_the_switch_head():
    with pytest.raises(AssertionError, match="does_switch"):
        th.manual_seed(0)
        GraphNetwork(
            y_levels=21,
            y_name="contribution",
            hidden_size=HIDDEN,
            x_encoding=[],
            default_values={},
            joint_exodus=True,
        )


def test_rejects_a_non_bool_gate():
    with pytest.raises(AssertionError, match="joint_exodus must be a bool"):
        make_model(joint_exodus=1)


def test_gate_and_head_must_agree():
    head = JointExodusHead(embed_size=HIDDEN, hidden_size=HIDDEN)
    with pytest.raises(AssertionError, match="disagree"):
        make_model(joint_exodus=False, joint_exodus_head=head)


@pytest.mark.parametrize("every", [0, -4, 2.0, True])
def test_rejects_a_bad_switch_every(every):
    with pytest.raises(AssertionError, match="positive int"):
        make_model(joint_exodus=True, joint_exodus_switch_every=every)


def test_switch_every_without_the_head_is_rejected():
    with pytest.raises(AssertionError, match="only meaningful"):
        make_model(joint_exodus_switch_every=4)
