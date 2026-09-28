"""Tests for the immediate-stimulus skip GATE in `GraphNetwork` (plan step 2 of
contribution-punishment-response): a second route from THIS round's post-`op1`
embedding to the `op2` readout, bypassing the per-agent RNN, so what I just
gave and what I just got for it reach my next decision without surviving the
update gate of a drifted recurrent state.

Runs locally on macOS with plain pytest:
    PYTHONPATH=$PWD/src <venv python> -m pytest -q tests/skip/test_stimulus_skip.py

Modelled on tests/vnode/test_group_vnode_graph.py, from which the PyG
stand-in machinery below is taken verbatim: `aimanager.generic.graph` imports
`torch_scatter` and `torch_geometric.nn`, Linux-only in this project (see the
`sys_platform` markers in pyproject.toml). The stand-ins are installed only
when the real packages are missing, so on Raven this file exercises the real
PyG. Every assertion that could be manufactured by a stand-in is an
INVARIANCE -- off equals the pre-change module, saved equals loaded, chunked
equals whole -- evaluated on both sides with the same message-passing
implementation.

Gate (a) additionally needs the `graph.py` that existed immediately BEFORE
this change (commit `6cc3bae`, "Step 0: the trunk's punishment response is
right, the loop loses it"; its child `efdb906` is the change under test). That
file is fetched with `git show` at test time -- never checked into this
worktree, never checked out over it -- written to a throwaway temp file and
imported under its own module name, so the identity check does not lean on the
code under test.

The fetch is attempted at import time but is NOT allowed to fail collection of
this module: on an isolated remote dir the worktree's `.git` is a pointer file
naming a local macOS path that does not exist there, so `git show` exits
non-zero. Any failure to fetch is recorded in `LEGACY_UNAVAILABLE_REASON` and
only the tests that actually need the pre-change module skip, with that reason
named explicitly; every other gate in this file imports and runs unchanged.

Context: notes/autoresearch_log/contribution-punishment-response.md, plan
step 2.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import types

import pytest
import torch as th

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRE_CHANGE_REV = "6cc3bae"
PRE_CHANGE_PATH = "src/aimanager/generic/graph.py"


# --------------------------------------------------------------------------- #
# PyG stand-ins (macOS only) -- verbatim from tests/vnode/test_group_vnode_graph.py
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
    """`torch_geometric.nn.MetaLayer`: edge model, then node model, then
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


def _load_pre_change_graph_module():
    """Fetch the `graph.py` that existed immediately before this change via
    `git show` and import it under its own module name -- never checked out
    over anything in this worktree, so gate (a) never leans on the code under
    test.

    Returns `(module, reason)`. On success `module` is the imported legacy
    module and `reason` is `None`. On any failure to run `git show` itself
    (non-zero exit, e.g. no usable git repo on an isolated remote dir; git not
    installed; ...) `module` is `None` and `reason` names the cause -- this
    function does not raise for that case, so importing this test module never
    fails collection. A fetch that DOES succeed but returns the wrong content
    (empty, or mentioning `stimulus_skip`) still raises: that signals a bug in
    the fetch itself, not an environment limitation.
    """
    try:
        result = subprocess.run(
            ["git", "show", f"{PRE_CHANGE_REV}:{PRE_CHANGE_PATH}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (getattr(exc, "stderr", None) or str(exc)).strip()
        reason = (
            "pre-change graph.py unavailable: `git show` failed -- " f"{detail or exc}"
        )
        return None, reason

    source = result.stdout
    assert source, "git show returned nothing for the pre-change graph.py"
    assert "stimulus_skip" not in source, (
        "the pre-change revision unexpectedly mentions stimulus_skip -- "
        "the wrong commit was fetched"
    )
    tmp_dir = tempfile.mkdtemp(prefix="legacy_graph_pre_skip_")
    path = os.path.join(tmp_dir, "legacy_graph_pre_skip.py")
    with open(path, "w") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location("legacy_graph_pre_skip", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, None


LEGACY, LEGACY_UNAVAILABLE_REASON = _load_pre_change_graph_module()

SEED = 20260916
N_AGENTS = 8
N_LEVELS = 21
HIDDEN = 4
VNODE_HIDDEN = 6  # deliberately != HIDDEN, so "grows by exactly hidden_size" bites
GROUPS = [0, 0, 0, 1, 1, 1, 1, 1]
COPULA_RHO = 0.0435568043640977
COPULA_PHI = 1.0
COPULA_SWITCH_EVERY = 1


def _model_kwargs():
    """Contribution-shaped, matching tests/vnode/test_group_vnode_graph.py.
    `u_encoding` is empty and `add_global_model` is off, so `u_features == 0`
    and `NodeModel`'s input is exactly the concatenated node embedding -- the
    premise the layout gate below checks explicitly rather than assumes."""
    return dict(
        y_levels=N_LEVELS,
        y_name="contribution",
        hidden_size=HIDDEN,
        add_rnn=True,
        add_edge_model=True,
        add_global_model=False,
        x_encoding=[
            {"name": "prev_contribution", "n_levels": 21, "encoding": "numeric"},
            {"name": "prev_punishment", "n_levels": 31, "encoding": "numeric"},
        ],
        edge_encoding=[],
        default_values={"contribution": 0, "punishment": 0},
    )


def make_model(seed=SEED, module=None, **kwargs):
    """`kwargs` OVERRIDE the shared defaults, so a gate can flip one field
    (e.g. `add_rnn`) without restating the whole configuration."""
    gn = GraphNetwork if module is None else module.GraphNetwork
    model_kwargs = {**_model_kwargs(), **kwargs}
    th.manual_seed(seed)
    model = gn(**model_kwargs)
    return model.to("cpu")


def make_data(n_batch=2, n_rounds=1, round_number=0, groups=GROUPS, seed=1):
    th.manual_seed(seed)
    shape = (n_batch, N_AGENTS, n_rounds)
    agent_group = th.tensor(groups, dtype=th.int64).reshape(1, N_AGENTS, 1)
    return {
        "contribution": th.zeros(shape, dtype=th.int64),
        "prev_contribution": th.randint(0, 21, shape),
        "prev_punishment": th.randint(0, 31, shape),
        "agent_group": agent_group.expand(shape).contiguous(),
        "round_number": th.full(shape, round_number, dtype=th.int64),
    }


def _slice_round(data, r):
    return {k: v[:, :, r : r + 1] for k, v in data.items()}


def _make_carry_data(n_batch, n_rounds, seed=11):
    """Multi-round data with a mid-episode group switch in one episode only,
    so the vnode's arrival pickup is exercised across a chunk boundary and the
    per-round parity claim is not a constant-input triviality."""
    th.manual_seed(seed)
    shape = (n_batch, N_AGENTS, n_rounds)
    round_number = (
        th.arange(n_rounds).reshape(1, 1, n_rounds).expand(shape).contiguous()
    )
    groups_b0 = [0, 0, 0, 0, 1, 1, 1, 1]
    groups_b1 = [0, 0, 1, 1, 1, 0, 0, 1]
    agent_group = th.stack(
        [
            th.tensor(g, dtype=th.int64)
            .reshape(N_AGENTS, 1)
            .expand(N_AGENTS, n_rounds)
            .clone()
            for g in (groups_b0, groups_b1)
        ],
        dim=0,
    )
    assert agent_group.shape == shape
    # episode 0, agent 1: group 0 -> 1 at round 12.
    agent_group[0, 1, 12:] = 1
    return {
        "contribution": th.zeros(shape, dtype=th.int64),
        "prev_contribution": th.randint(0, 21, shape),
        "prev_punishment": th.randint(0, 31, shape),
        "agent_group": agent_group,
        "round_number": round_number,
    }


def _op2_in_features(model):
    return model.op2.node_model.node_mlp.in_features


def _op2_weight(model):
    return model.op2.node_model.node_mlp.weight


def _forward(model, data, edge_index=None):
    model.eval()
    encoded = model.encode(data, y_encode=False, edge_index=edge_index, device="cpu")
    return model(encoded)


# --------------------------------------------------------------------------- #
# (a) off is bit-identical
# --------------------------------------------------------------------------- #
def test_off_matches_the_flag_absent_bit_for_bit():
    """`stimulus_skip=False` passed explicitly and the kwarg absent altogether
    are the same model: every state_dict entry torch.equal, and the same
    forward output. Construction draws no extra RNG with the flag off."""
    absent = make_model()
    off = make_model(stimulus_skip=False)

    assert absent.stimulus_skip is False
    assert off.stimulus_skip is False

    absent_state = absent.state_dict()
    off_state = off.state_dict()
    assert set(absent_state) == set(off_state)
    for key, value in absent_state.items():
        assert th.equal(value, off_state[key]), key

    data = make_data(n_batch=2, round_number=3)
    assert th.equal(_forward(absent, data), _forward(off, data))


def test_off_matches_the_pre_change_module_bit_for_bit():
    """A model built with the skip off (the default) has every parameter --
    explicitly including op1 / rnn_n / op2 -- torch.equal to one built from
    the graph.py that existed immediately before this change, under the same
    seed, and produces the same forward output."""
    if LEGACY is None:
        pytest.skip(LEGACY_UNAVAILABLE_REASON)
    new_model = make_model()
    legacy_model = make_model(module=LEGACY)

    assert new_model.stimulus_skip is False
    assert not hasattr(legacy_model, "stimulus_skip")

    new_state = new_model.state_dict()
    legacy_state = legacy_model.state_dict()
    assert set(new_state) == set(legacy_state)
    for key, value in legacy_state.items():
        assert th.equal(value, new_state[key]), key

    for prefix in ("op1.", "rnn_n.", "op2."):
        matched = [k for k in legacy_state if k.startswith(prefix)]
        assert matched, f"no parameters matched prefix {prefix!r}"
        for key in matched:
            assert th.equal(legacy_state[key], new_state[key]), key

    data = make_data(n_batch=2, round_number=3)
    assert th.equal(_forward(new_model, data), _forward(legacy_model, data))


def test_off_with_the_vnode_on_matches_the_pre_change_module():
    """The parent's own configuration -- the group virtual node on, the skip
    off -- is likewise untouched: the change must not perturb the frontier
    stack it is stacked on."""
    if LEGACY is None:
        pytest.skip(LEGACY_UNAVAILABLE_REASON)
    new_model = make_model(group_vnode=True, group_vnode_hidden=VNODE_HIDDEN)
    legacy_model = make_model(
        module=LEGACY, group_vnode=True, group_vnode_hidden=VNODE_HIDDEN
    )
    new_state = new_model.state_dict()
    legacy_state = legacy_model.state_dict()
    assert set(new_state) == set(legacy_state)
    for key, value in legacy_state.items():
        assert th.equal(value, new_state[key]), key

    data = make_data(n_batch=2, round_number=3)
    assert th.equal(_forward(new_model, data), _forward(legacy_model, data))


# --------------------------------------------------------------------------- #
# (b) on: the width and the gradient
# --------------------------------------------------------------------------- #
def test_op2_width_grows_by_exactly_hidden_size_additively_with_the_vnode():
    """All four flag combinations. The skip adds exactly `hidden_size` and the
    vnode exactly `group_vnode_hidden`, and the two are additive -- with
    `group_vnode_hidden != hidden_size` so the two contributions cannot be
    confused for one another."""
    neither = make_model()
    skip_only = make_model(stimulus_skip=True)
    vnode_only = make_model(group_vnode=True, group_vnode_hidden=VNODE_HIDDEN)
    both = make_model(
        stimulus_skip=True, group_vnode=True, group_vnode_hidden=VNODE_HIDDEN
    )

    base = _op2_in_features(neither)
    assert base == HIDDEN  # post-RNN embedding alone; u_features == 0
    assert _op2_in_features(skip_only) == base + HIDDEN
    assert _op2_in_features(vnode_only) == base + VNODE_HIDDEN
    assert _op2_in_features(both) == base + VNODE_HIDDEN + HIDDEN

    # the weight matrices agree with the declared in_features
    for model in (neither, skip_only, vnode_only, both):
        assert _op2_weight(model).shape[1] == _op2_in_features(model)

    # nothing ELSE is widened: op1 and rnn_n are identical across all four
    ref_state = neither.state_dict()
    for model in (skip_only, vnode_only, both):
        state = model.state_dict()
        for prefix in ("op1.", "rnn_n."):
            matched = [k for k in ref_state if k.startswith(prefix)]
            assert matched, f"no parameters matched prefix {prefix!r}"
            for key in matched:
                assert th.equal(ref_state[key], state[key]), key


def test_the_skips_slice_of_op2_receives_the_post_op1_embedding():
    """The documented layout is `[post-RNN embedding | group state | post-op1
    embedding]`, so the skip occupies the LAST `hidden_size` columns of op2's
    input. Checked directly: a forward pre-hook captures what `node_mlp`
    actually receives and its tail is compared against op1's output, recomputed
    here from the encoded state."""
    model = make_model(
        stimulus_skip=True, group_vnode=True, group_vnode_hidden=VNODE_HIDDEN
    )
    # the premise of "the last columns": NodeModel concatenates
    # [x | pooled edge_attr | u[batch]], and op2 sees zero of each of the
    # latter two, so its input IS the concatenated node embedding.
    assert model.u_encoder.size == 0
    assert model.op2.edge_model is None

    data = make_data(n_batch=2, round_number=3)
    model.eval()
    encoded = model.encode(data, y_encode=False, device="cpu")
    post_op1, _, _ = model.op1(
        encoded["x"],
        encoded["edge_index"],
        encoded["edge_attr"],
        encoded["u"],
        encoded["batch"],
    )

    captured = {}

    def hook(_module, args):
        captured["input"] = args[0]

    handle = model.op2.node_model.node_mlp.register_forward_pre_hook(hook)
    try:
        model(encoded)
    finally:
        handle.remove()

    op2_input = captured["input"]
    assert op2_input.shape[-1] == _op2_in_features(model)
    assert th.equal(op2_input[..., -HIDDEN:], post_op1)
    # and the head of the input is NOT the same tensor -- the post-RNN
    # embedding differs from the post-op1 one, which is the whole point.
    assert not th.equal(op2_input[..., :HIDDEN], post_op1)


def test_gradient_reaches_the_skips_slice_of_op2():
    """Gradient actually flows into the last `hidden_size` columns of op2's
    weight -- the skip's slice -- and into op1 through the skip. A widened
    layer whose extra columns never receive gradient would train as the
    unmodified model."""
    model = make_model(
        stimulus_skip=True, group_vnode=True, group_vnode_hidden=VNODE_HIDDEN
    )
    model.train()
    data = make_data(n_batch=2, n_rounds=3, round_number=0)
    encoded = model.encode(data, y_encode=False, device="cpu")
    logit = model(encoded)

    th.manual_seed(SEED + 2)
    target = th.randint(0, N_LEVELS, logit.shape[:-1]).reshape(-1)
    loss = th.nn.functional.cross_entropy(logit.reshape(-1, N_LEVELS), target)
    loss.backward()

    grad = _op2_weight(model).grad
    assert grad is not None
    skip_grad = grad[:, -HIDDEN:]
    assert th.isfinite(skip_grad).all()
    assert skip_grad.abs().max() > 0, "no gradient reaches the skip's slice of op2"
    # the pre-existing slices still receive gradient too
    assert grad[:, :HIDDEN].abs().max() > 0
    assert grad[:, HIDDEN : HIDDEN + VNODE_HIDDEN].abs().max() > 0


def test_forward_differs_between_the_skip_on_and_off():
    """Under the same seed the trunks coincide (op2 aside) but the outputs must
    not: a flag that changed nothing observable would be untestable downstream."""
    on = make_model(stimulus_skip=True)
    off = make_model()
    data = make_data(n_batch=2, round_number=5)
    logit_on = _forward(on, data)
    logit_off = _forward(off, data)
    assert logit_on.shape == logit_off.shape
    assert not th.equal(logit_on, logit_off)

    # op1 and rnn_n are built BEFORE op2 is widened, so under the same seed
    # they are torch.equal between on and off -- only op2 differs.
    on_state = on.state_dict()
    off_state = off.state_dict()
    for prefix in ("op1.", "rnn_n."):
        for key in [k for k in off_state if k.startswith(prefix)]:
            assert th.equal(off_state[key], on_state[key]), key
    assert (
        off_state["op2.node_model.node_mlp.weight"].shape
        != on_state["op2.node_model.node_mlp.weight"].shape
    )


# --------------------------------------------------------------------------- #
# (c) the simulation's calling convention
# --------------------------------------------------------------------------- #
def test_per_round_calls_reproduce_the_single_call():
    """24 single-round `predict_independent` calls, `reset_rnn` only at round
    0, reproduce one 24-round call -- the property the 24 per-round calls in
    `simulation/simulate.py` depend on. Both flags on, with the copula stamped,
    so the skip is shown to co-exist with the vnode's h0 carry and the copula's
    AR(1) z carry."""
    n_batch, n_rounds = 2, 24
    kwargs = dict(
        stimulus_skip=True,
        group_vnode=True,
        group_vnode_hidden=VNODE_HIDDEN,
        copula_rho=COPULA_RHO,
        copula_phi=COPULA_PHI,
        copula_switch_every=COPULA_SWITCH_EVERY,
    )
    model_full = make_model(**kwargs)
    model_chunks = make_model(**kwargs)
    # same weights under construction, verified rather than trusted: the
    # parity claim is meaningless if the two models disagree before a round
    # is fed.
    assert set(model_full.state_dict()) == set(model_chunks.state_dict())
    for key, value in model_full.state_dict().items():
        assert th.equal(value, model_chunks.state_dict()[key]), key

    data = _make_carry_data(n_batch, n_rounds)
    edge_index = model_full.create_fully_connected(N_AGENTS, n_batch=n_batch)

    th.manual_seed(SEED + 1)
    pred_full, proba_full = model_full.predict_independent(
        data, sample=True, reset_rnn=True, edge_index=edge_index
    )

    th.manual_seed(SEED + 1)
    pred_chunks = th.empty_like(pred_full)
    proba_chunks = th.empty_like(proba_full)
    for r in range(n_rounds):
        p, pr = model_chunks.predict_independent(
            _slice_round(data, r),
            sample=True,
            reset_rnn=(r == 0),
            edge_index=edge_index,
        )
        pred_chunks[:, :, r] = p[:, :, 0]
        proba_chunks[:, :, r] = pr[:, :, 0]

    proba_diff = (proba_chunks - proba_full).abs().max().item()
    # 1e-6 is the tolerance the sibling vnode suite uses: GRU batching takes a
    # different BLAS code path for a 24-round call than for 24 one-round calls.
    assert proba_diff < 1e-6, proba_diff
    assert th.equal(pred_chunks, pred_full)


def test_per_round_forward_reproduces_the_single_call_deterministically():
    """The same property on the raw forward, with no sampler in the way, so a
    parity failure cannot be masked by the multinomial draw."""
    n_batch, n_rounds = 2, 24
    model = make_model(
        stimulus_skip=True, group_vnode=True, group_vnode_hidden=VNODE_HIDDEN
    )
    model.eval()
    data = _make_carry_data(n_batch, n_rounds, seed=13)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=n_batch)

    full = model(
        model.encode(data, y_encode=False, edge_index=edge_index, device="cpu")
    )
    chunks = th.empty_like(full)
    for r in range(n_rounds):
        enc = model.encode(
            _slice_round(data, r),
            y_encode=False,
            edge_index=edge_index,
            device="cpu",
        )
        chunks[:, r] = model(enc, reset_rnn=(r == 0))[:, 0]

    diff = (chunks - full).abs().max().item()
    assert diff < 1e-6, diff


# --------------------------------------------------------------------------- #
# (d) save / load round-trip
# --------------------------------------------------------------------------- #
def test_save_load_round_trips_the_flag_and_the_forward():
    """A flag that trains but does not survive `load` would silently simulate a
    plain model, so the reloaded model must report the flag AND reproduce the
    original's forward bit-for-bit."""
    model = make_model(
        stimulus_skip=True, group_vnode=True, group_vnode_hidden=VNODE_HIDDEN
    )
    data = make_data(n_batch=2, round_number=3)
    ref_logit = _forward(model, data)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        saved = th.load(path, map_location="cpu")
        assert saved["stimulus_skip"] is True
        loaded = GraphNetwork.load(path, device="cpu")

    assert loaded.stimulus_skip is True
    assert _op2_in_features(loaded) == _op2_in_features(model)

    ref_state = model.state_dict()
    loaded_state = loaded.state_dict()
    assert set(ref_state) == set(loaded_state)
    for key, value in ref_state.items():
        assert th.equal(value, loaded_state[key]), key

    assert th.equal(_forward(loaded, data), ref_logit)


def test_legacy_artifact_without_the_key_loads_with_the_skip_absent():
    """An artifact saved BEFORE this change carries no `stimulus_skip` key. It
    must load with the skip off and forward exactly as the model that saved
    it -- the back-compat guarantee step 1 claims for today's trained
    artifacts."""
    model = make_model(group_vnode=True, group_vnode_hidden=VNODE_HIDDEN)
    data = make_data(n_batch=2, round_number=3)
    ref_logit = _forward(model, data)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        saved = th.load(path, map_location="cpu")
        assert "stimulus_skip" in saved
        del saved["stimulus_skip"]
        legacy_path = os.path.join(d, "legacy.pt")
        th.save(saved, legacy_path)
        legacy = GraphNetwork.load(legacy_path, device="cpu")

    assert legacy.stimulus_skip is False
    assert _op2_in_features(legacy) == HIDDEN + VNODE_HIDDEN
    assert th.equal(_forward(legacy, data), ref_logit)


# --------------------------------------------------------------------------- #
# (e) the add_rnn guard
# --------------------------------------------------------------------------- #
def test_stimulus_skip_without_the_rnn_raises():
    """With no per-agent RNN there is no memory for the skip to bypass and the
    readout would receive the same embedding twice, so the combination is
    rejected at construction. The guard is deliberate -- do not remove it."""
    with pytest.raises(AssertionError, match="stimulus_skip requires the per-agent"):
        make_model(stimulus_skip=True, add_rnn=False)

    # the control: no RNN and no skip still builds, so the guard fires on the
    # combination and not on `add_rnn=False` alone.
    control = make_model(add_rnn=False)
    assert control.rnn_n is None
    assert control.stimulus_skip is False


def test_non_bool_stimulus_skip_raises():
    with pytest.raises(AssertionError, match="stimulus_skip must be a bool"):
        make_model(stimulus_skip=1)
