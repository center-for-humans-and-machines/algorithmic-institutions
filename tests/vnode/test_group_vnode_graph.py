"""Tests for the per-group virtual node GATE in `GraphNetwork` (plan step 4 of
contribution-group-vnode).

Runs locally on macOS with plain pytest:
    PYTHONPATH=$PWD/src <venv python> -m pytest -q tests/vnode/test_group_vnode_graph.py

Modelled on tests/switch/test_joint_exodus_graph.py: `aimanager.generic.graph`
imports `torch_scatter` and `torch_geometric.nn`, Linux-only in this project
(see the `sys_platform` markers in pyproject.toml). Every assertion here is an
INVARIANCE -- off equals the pre-change module, saved equals loaded, on
shares the off model's trunk -- evaluated on both sides with the same
message-passing implementation, so a stand-in for those two symbols cannot
manufacture a pass. The stand-ins are installed only when the real packages
are missing, so on Raven this file exercises the real PyG.

Gate (a) additionally needs the `graph.py` that existed immediately BEFORE
this change (commit `7b440ee`, "Wire the per-group virtual node into
GraphNetwork"; its parent `1b87df8` carries the module but nothing wired in).
That file is fetched with `git show` at test time -- never checked into this
worktree, never checked out over it -- written to a throwaway temp file and
imported under its own module name, so the identity check does not lean on
the code under test.

The fetch is attempted at import time but is NOT allowed to fail collection
of this module: on an isolated remote dir the worktree's `.git` is a pointer
file to a path that does not exist there, so `git show` exits non-zero. Any
failure to fetch is recorded in `LEGACY_UNAVAILABLE_REASON` and only the two
tests that actually need the pre-change module skip, with that reason named
explicitly; every other gate in this file imports and runs unchanged.

Numerics of the node itself live in tests/vnode/test_group_vnode.py.
Context: notes/autoresearch_log/contribution-group-vnode.md, plan step 4.
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
PRE_CHANGE_REV = "7b440ee^"
PRE_CHANGE_PATH = "src/aimanager/generic/graph.py"


# --------------------------------------------------------------------------- #
# PyG stand-ins (macOS only) -- verbatim from tests/switch/test_joint_exodus_graph.py
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
from aimanager.generic.group_vnode import GroupVirtualNode  # noqa: E402


def _load_pre_change_graph_module():
    """Fetch the `graph.py` that existed immediately before this change via
    `git show` and import it under its own module name -- never checked out
    over anything in this worktree, so gate (a) never leans on the code
    under test.

    Returns `(module, reason)`. On success `module` is the imported legacy
    module and `reason` is `None`. On any failure to run `git show` itself
    (non-zero exit, e.g. no usable git repo on an isolated remote dir; git
    not installed; ...) `module` is `None` and `reason` names the cause --
    this function does not raise for that case, so importing this test
    module never fails collection. A fetch that DOES succeed but returns the
    wrong content (empty, or mentioning `group_vnode`) still raises: that
    signals a bug in the fetch itself, not an environment limitation.
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
    assert "group_vnode" not in source, (
        "the pre-change revision unexpectedly mentions group_vnode -- "
        "the wrong commit was fetched"
    )
    tmp_dir = tempfile.mkdtemp(prefix="legacy_graph_pre_vnode_")
    path = os.path.join(tmp_dir, "legacy_graph_pre_vnode.py")
    with open(path, "w") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location("legacy_graph_pre_vnode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, None


LEGACY, LEGACY_UNAVAILABLE_REASON = _load_pre_change_graph_module()

SEED = 20260915
N_AGENTS = 8
N_LEVELS = 21
GROUPS = [0, 0, 0, 1, 1, 1, 1, 1]
COPULA_RHO = 0.06958238086256316
COPULA_PHI = 1.0
COPULA_SWITCH_EVERY = 1


def _model_kwargs():
    """Contribution-shaped, matching test_joint_exodus_graph.py's make_model.
    `agent_group` is deliberately NOT in x_encoding: with it absent, the
    trunk has no pathway to group membership except through the mechanism
    under test, which is exactly what gate (d)'s isolation needs."""
    return dict(
        y_levels=N_LEVELS,
        y_name="contribution",
        hidden_size=4,
        add_rnn=True,
        add_edge_model=True,
        add_global_model=False,
        x_encoding=[
            {"name": "prev_contribution", "n_levels": 21, "encoding": "numeric"}
        ],
        edge_encoding=[],
        default_values={"contribution": 0},
    )


def make_model(seed=0, module=None, **kwargs):
    gn = GraphNetwork if module is None else module.GraphNetwork
    th.manual_seed(seed)
    model = gn(**_model_kwargs(), **kwargs)
    return model.to("cpu")


def make_data(n_batch=2, n_rounds=1, round_number=0, groups=GROUPS, seed=1):
    th.manual_seed(seed)
    shape = (n_batch, N_AGENTS, n_rounds)
    agent_group = th.tensor(groups, dtype=th.int64).reshape(1, N_AGENTS, 1)
    return {
        "contribution": th.zeros(shape, dtype=th.int64),
        "prev_contribution": th.randint(0, 21, shape),
        "agent_group": agent_group.expand(shape).contiguous(),
        "round_number": th.full(shape, round_number, dtype=th.int64),
    }


def legacy_predict(model, data, edge_index=None, reset_rnn=True):
    """`predict_independent(sample=True)` as it was BEFORE this change, so
    the identity check does not lean on the code under test."""
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
    th.manual_seed(SEED)
    out = fn()
    return out, th.randn(5)


def _slice_round(data, r):
    return {k: v[:, :, r : r + 1] for k, v in data.items()}


def _make_copula_carry_data(n_batch, n_rounds, seed=11):
    """24-round data with a mid-episode switch in one episode only, so the
    node's arrival pickup is exercised across a chunk boundary and not just
    within a single call."""
    th.manual_seed(seed)
    shape = (n_batch, N_AGENTS, n_rounds)
    prev_contribution = th.randint(0, 21, shape)
    round_number = (
        th.arange(n_rounds).reshape(1, 1, n_rounds).expand(shape).contiguous()
    )
    groups_b0 = [0, 0, 0, 0, 1, 1, 1, 1]
    groups_b1 = [0, 0, 1, 1, 1, 0, 0, 1]
    agent_group = th.stack(
        [
            th.tensor(groups_b0, dtype=th.int64)
            .reshape(N_AGENTS, 1)
            .expand(N_AGENTS, n_rounds)
            .clone(),
            th.tensor(groups_b1, dtype=th.int64)
            .reshape(N_AGENTS, 1)
            .expand(N_AGENTS, n_rounds)
            .clone(),
        ],
        dim=0,
    )
    assert agent_group.shape == shape
    # episode 0, agent 1: group 0 -> 1 at round 12.
    agent_group[0, 1, 12:] = 1
    return {
        "contribution": th.zeros(shape, dtype=th.int64),
        "prev_contribution": prev_contribution,
        "agent_group": agent_group,
        "round_number": round_number,
    }


# --------------------------------------------------------------------------- #
# (a) off by default
# --------------------------------------------------------------------------- #
def test_off_by_default_matches_the_pre_change_module_bit_for_bit():
    """A model built with group_vnode off (the default) has every parameter
    -- explicitly including op1 / rnn_n / op2 -- torch.equal to one built
    from the graph.py that existed immediately before this change, under the
    same seed: construction draws no extra RNG with the flag off."""
    if LEGACY is None:
        pytest.skip(LEGACY_UNAVAILABLE_REASON)
    new_model = make_model(seed=SEED)
    legacy_model = make_model(seed=SEED, module=LEGACY)

    assert new_model.group_vnode is False
    assert new_model.group_vnode_module is None

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


def test_off_by_default_sampling_matches_the_pre_change_module():
    """predict_independent(sample=True) matches the pre-change module in
    both VALUES and RNG CONSUMPTION -- the licence step 13's control
    comparison rests on."""
    if LEGACY is None:
        pytest.skip(LEGACY_UNAVAILABLE_REASON)
    new_model = make_model(seed=SEED)
    legacy_model = make_model(seed=SEED, module=LEGACY)
    data = make_data(n_batch=2)
    edge_index = new_model.create_fully_connected(N_AGENTS, n_batch=2)

    (legacy_pred, legacy_proba), legacy_rng = run_seeded(
        lambda: legacy_model.predict_independent(
            data, sample=True, edge_index=edge_index
        )
    )
    (new_pred, new_proba), new_rng = run_seeded(
        lambda: new_model.predict_independent(data, sample=True, edge_index=edge_index)
    )
    assert th.equal(new_pred, legacy_pred)
    assert th.equal(new_proba, legacy_proba)
    assert th.equal(new_rng, legacy_rng)


def test_off_by_default_matches_legacy_predict_reimplementation():
    """Cross-check against the from-scratch reimplementation every sibling
    graph-gate suite uses, so gate (a) does not rest on the pre-change
    module alone."""
    model = make_model(seed=SEED)
    data = make_data(n_batch=2)
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


# --------------------------------------------------------------------------- #
# (b) legacy artifact
# --------------------------------------------------------------------------- #
def test_legacy_artifact_without_the_new_keys_loads_off_and_samples_bit_identically():
    """An artifact saved BEFORE this change carries none of the three new
    keys, and must load with the node absent and sample bit-identically to
    the pre-change path."""
    model = make_model(seed=SEED)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        saved = th.load(path, map_location="cpu")
        for key in ("group_vnode", "group_vnode_module", "group_vnode_hidden"):
            assert key in saved
            del saved[key]
        legacy_path = os.path.join(d, "legacy.pt")
        th.save(saved, legacy_path)
        legacy = GraphNetwork.load(legacy_path, device="cpu")

    assert legacy.group_vnode is False
    assert legacy.group_vnode_module is None

    data = make_data(n_batch=2)
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
# (c) round-trip with the flag on
# --------------------------------------------------------------------------- #
def test_save_load_round_trips_the_module_and_hidden_size():
    model = make_model(seed=SEED, group_vnode=True, group_vnode_hidden=6)
    data = make_data(n_batch=2, round_number=3)
    encoded = model.encode(data, y_encode=False, device="cpu")
    model.eval()
    ref_logit = model(encoded)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        loaded = GraphNetwork.load(path, device="cpu")

    assert loaded.group_vnode is True
    assert loaded.group_vnode_hidden == 6
    assert isinstance(loaded.group_vnode_module, GroupVirtualNode)

    ref_state = model.group_vnode_module.state_dict()
    loaded_state = loaded.group_vnode_module.state_dict()
    assert set(ref_state) == set(loaded_state)
    for key, value in ref_state.items():
        assert th.equal(value, loaded_state[key]), key

    loaded.eval()
    enc = loaded.encode(data, y_encode=False, device="cpu")
    logit = loaded(enc)
    assert th.equal(logit, ref_logit)


# --------------------------------------------------------------------------- #
# (d) on differs from off, for the right reason
# --------------------------------------------------------------------------- #
def test_on_shares_the_off_models_pre_readout_trunk_under_the_same_seed():
    """op1 and rnn_n are built BEFORE op2 is widened for the node, so under
    the same seed they are torch.equal between on and off regardless of the
    flag -- only op2 itself (widened to receive the group state) and
    anything built after it are expected to differ."""
    off = make_model(seed=SEED)
    on = make_model(seed=SEED, group_vnode=True)
    off_state = off.state_dict()
    on_state = on.state_dict()
    for prefix in ("op1.", "rnn_n."):
        matched = [k for k in off_state if k.startswith(prefix)]
        assert matched, f"no parameters matched prefix {prefix!r}"
        for key in matched:
            assert th.equal(off_state[key], on_state[key]), key
    # op2 itself is a different shape once the node is on -- widened to take
    # the concatenated group state -- so it is deliberately NOT compared here.
    assert off_state["op2.node_model.node_mlp.weight"].shape != (
        on_state["op2.node_model.node_mlp.weight"].shape
    )
    assert "group_vnode_module.gru.weight_ih_l0" in on_state
    assert "group_vnode_module.gru.weight_ih_l0" not in off_state


def test_forward_differs_between_on_and_off():
    on = make_model(seed=SEED, group_vnode=True)
    off = make_model(seed=SEED)
    data = make_data(n_batch=2, round_number=5)
    on.eval()
    off.eval()
    logit_on = on(on.encode(data, y_encode=False, device="cpu"))
    logit_off = off(off.encode(data, y_encode=False, device="cpu"))
    assert logit_on.shape == logit_off.shape
    assert not th.equal(logit_on, logit_off)


def test_shuffling_agent_group_across_episodes_moves_the_on_model_only():
    """The group-blindness contrast PR #176 note 9 measured directly. With
    `agent_group` absent from x_encoding (see _model_kwargs), the off model
    has NO pathway to it at all, so permuting which episode gets which group
    assignment must leave its output bit-for-bit unchanged; the on model
    reads its own group's state through the virtual node, so the same
    permutation must move its output."""
    on = make_model(seed=SEED, group_vnode=True)
    off = make_model(seed=SEED)
    on.eval()
    off.eval()

    groups_a = [0, 0, 0, 1, 1, 1, 1, 1]
    groups_b = [0, 1, 1, 1, 1, 1, 1, 0]

    def two_episode_data(g0, g1):
        th.manual_seed(7)
        shape = (2, N_AGENTS, 1)
        prev = th.randint(0, 21, shape)
        agent_group = th.stack(
            [th.tensor(g0, dtype=th.int64), th.tensor(g1, dtype=th.int64)], dim=0
        ).unsqueeze(-1)
        return {
            "contribution": th.zeros(shape, dtype=th.int64),
            "prev_contribution": prev,
            "agent_group": agent_group,
            "round_number": th.full(shape, 5, dtype=th.int64),
        }

    original = two_episode_data(groups_a, groups_b)
    shuffled = two_episode_data(groups_b, groups_a)  # swapped across episodes
    # the swap changes only agent_group, nothing else
    assert th.equal(original["prev_contribution"], shuffled["prev_contribution"])
    assert not th.equal(original["agent_group"], shuffled["agent_group"])

    edge_index = on.create_fully_connected(N_AGENTS, n_batch=2)
    logit_on_orig = on(
        on.encode(original, y_encode=False, edge_index=edge_index, device="cpu")
    )
    logit_on_shuf = on(
        on.encode(shuffled, y_encode=False, edge_index=edge_index, device="cpu")
    )
    logit_off_orig = off(
        off.encode(original, y_encode=False, edge_index=edge_index, device="cpu")
    )
    logit_off_shuf = off(
        off.encode(shuffled, y_encode=False, edge_index=edge_index, device="cpu")
    )

    assert not th.equal(logit_on_orig, logit_on_shuf)
    assert th.equal(logit_off_orig, logit_off_shuf)


# --------------------------------------------------------------------------- #
# (e) per-round carry, with the copula fields stamped
# --------------------------------------------------------------------------- #
def test_per_round_calls_reproduce_the_single_call_with_copula_stamped():
    """24 single-round predict_independent calls, reset_rnn only at round 0,
    reproduce one 24-round call -- the vnode's h0 carry AND the copula's
    AR(1) z carry together, so _predict_encoded_copula and the node are
    shown to co-exist. copula_switch_every=1: every round is a decision
    round for the contribution slot."""
    n_batch, n_rounds = 2, 24
    model_full = make_model(
        seed=SEED,
        group_vnode=True,
        copula_rho=COPULA_RHO,
        copula_phi=COPULA_PHI,
        copula_switch_every=COPULA_SWITCH_EVERY,
    )
    model_chunks = make_model(
        seed=SEED,
        group_vnode=True,
        copula_rho=COPULA_RHO,
        copula_phi=COPULA_PHI,
        copula_switch_every=COPULA_SWITCH_EVERY,
    )
    # same weights under construction, verified explicitly rather than
    # trusted: the parity claim below is meaningless if the two models
    # disagree before a single round is fed.
    assert set(model_full.state_dict()) == set(model_chunks.state_dict())
    for key, value in model_full.state_dict().items():
        assert th.equal(value, model_chunks.state_dict()[key]), key

    data = _make_copula_carry_data(n_batch, n_rounds)
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
    # the module-level per-round parity test (test_group_vnode.py) tolerates
    # 1e-6 from GRU batching taking a different BLAS code path; the achieved
    # margin here is ~1.5e-8, two orders tighter.
    assert proba_diff < 1e-6, proba_diff
    assert th.equal(pred_chunks, pred_full)


# --------------------------------------------------------------------------- #
# (f) missing agent_group
# --------------------------------------------------------------------------- #
def test_missing_agent_group_raises_on_the_on_model_only():
    on = make_model(seed=SEED, group_vnode=True)
    off = make_model(seed=SEED)
    data = make_data(n_batch=2)

    # encode()-level: the raw state never carries agent_group at all.
    data_without = dict(data)
    del data_without["agent_group"]
    with pytest.raises(AssertionError, match="agent_group"):
        on.encode(data_without, y_encode=False, device="cpu")
    off.encode(data_without, y_encode=False, device="cpu")  # must not raise

    # forward()-level: a HAND-ASSEMBLED encoded state (built by the off
    # model's own encode(), which never carries the key), not produced via
    # the on model's self.encode() -- this is the assert the diff added
    # specifically so a hand-assembled state raises a readable
    # AssertionError rather than a raw KeyError.
    edge_index = on.create_fully_connected(N_AGENTS, n_batch=2)
    encoded_without_group = off.encode(
        data, y_encode=False, edge_index=edge_index, device="cpu"
    )
    assert "agent_group" not in encoded_without_group
    on.eval()
    off.eval()
    with pytest.raises(AssertionError, match="agent_group"):
        on(encoded_without_group)
    off(encoded_without_group)  # must not raise
