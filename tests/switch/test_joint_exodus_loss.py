"""Tests for the joint exodus TRAINING OBJECTIVE (plan step 5).

Runs locally on macOS with plain pytest::

    PYTHONPATH=$PWD/src python -m pytest tests/switch/test_joint_exodus_loss.py -q

``aimanager.artificial_humans.train`` pulls in ``torch_scatter``,
``torch_geometric.nn`` (via ``AH_MODELS``) and ``tqdm``, none of which are
installed on macOS in this project. Stand-ins are installed ONLY when the real
packages are missing -- the same discipline as
tests/switch/test_joint_exodus_graph.py -- so on Raven this file exercises the
real PyG. ``STAND_INS`` lists what THIS file installed, so it is empty both on
Raven and whenever a sibling suite in the same pytest process imported first;
what it never does is report a stand-in that is not there. The stand-ins
implement message passing and a progress bar; they cannot manufacture a pass
on the count arithmetic, which is pure torch either way.

Context: notes/autoresearch_log/switch-joint-exodus-gmlp.md.
"""

import importlib
import math
import sys
import types

import pytest
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
    DROP_INCOMPLETE_PAIRS,
    compute_batch_loss,
    joint_exodus_counts,
    joint_exodus_loss,
)
from aimanager.generic.graph import GraphNetwork  # noqa: E402
from aimanager.generic.joint_exodus import (  # noqa: E402
    MAX_GROUP_SIZE,
    masked_joint_log_prob,
)

N_AGENTS = 8
GRID = MAX_GROUP_SIZE + 1
N_ROUNDS = 24
#: The two size encodings the head runs -- see JointExodusHead.SIZE_ENCODINGS.
SIZE_ENCODINGS = ("numeric", "onehot")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def as_nodes(rows, n_rounds=1):
    """(n_batch, n_player) python lists -> (N, R) tensor, R copies per round."""
    flat = [v for graph in rows for v in graph]
    return th.tensor(flat, dtype=th.int64).reshape(-1, 1).expand(-1, n_rounds)


def batch_index(n_batch, n_player=N_AGENTS):
    return th.tensor([b for b in range(n_batch) for _ in range(n_player)])


def counts_of(groups, switches, valid=None, n_batch=1):
    """Hand-worked wrapper: single round, one or more graphs."""
    if valid is None:
        valid = [[True] * len(g) for g in groups]
    m, k = joint_exodus_counts(
        as_nodes(switches).to(th.bool),
        as_nodes(valid).to(th.bool),
        as_nodes(groups),
        batch_index(n_batch, len(groups[0])),
        n_batch=n_batch,
    )
    return m[:, 0], k[:, 0]


def fake_joint(m, k, logits=None):
    """A head output whose grid is uniform over the valid cells, so the
    negative log-likelihood of ANY observed pair is exactly
    log((k_0 + 1) * (k_1 + 1)) and the loss is hand-computable."""
    if logits is None:
        logits = th.zeros((*k.shape[:-1], GRID, GRID))
    log_prob, _ = masked_joint_log_prob(logits, k)
    return log_prob, k


def fake_batch(y, mask, agent_group, n_batch, n_player=N_AGENTS):
    return {
        "y": y,
        "mask": mask,
        "agent_group": agent_group,
        "batch": batch_index(n_batch, n_player),
    }


def make_model(seed=0, **kwargs):
    """A switch-shaped model on the real config's group/round features."""
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


def make_state(groups, switches, valid, n_rounds=1, round_number=3):
    """(n_batch, n_player, n_rounds) state dict, the shape train.py feeds
    ``model.encode``."""
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


def legacy_batch_loss(model, batch_data, loss_fn, l1_entropy):
    """The per-agent loss EXACTLY as train.py computed it before this change
    (copied verbatim from the parent revision), so the invariance check does
    not lean on the code under test."""
    y_logit = model(batch_data).flatten(end_dim=-2)
    y_pred = y_logit.softmax(-1)
    y_true = batch_data["y_enc"].flatten(end_dim=-2)
    mask = batch_data["mask"].flatten()
    loss = loss_fn(y_logit, y_true) + (y_pred * y_pred.log()).sum(-1) * l1_entropy
    return (loss * mask).sum() / mask.sum()


# --------------------------------------------------------------------------- #
# 1. the count pair, by hand
# --------------------------------------------------------------------------- #
def test_count_pair_hand_worked():
    #        agent:   0  1  2 | 3  4  5  6  7
    groups = [[0, 0, 0, 1, 1, 1, 1, 1]]
    switch = [[1, 0, 0, 0, 1, 1, 0, 0]]
    m, k = counts_of(groups, switch)
    assert k.tolist() == [[3, 5]]
    assert m.tolist() == [[1, 2]]


def test_count_pair_is_indexed_by_group_label_not_by_size():
    """Order-canonical in the LABEL: mirroring the labels transposes the pair
    rather than leaving it alone, which is what makes the flip-doubled data
    symmetrise the head instead of fighting it."""
    groups = [[0, 0, 0, 1, 1, 1, 1, 1]]
    switch = [[1, 0, 0, 0, 1, 1, 0, 0]]
    flipped = [[1, 1, 1, 0, 0, 0, 0, 0]]
    m, k = counts_of(groups, switch)
    m_f, k_f = counts_of(flipped, switch)
    assert k_f.tolist() == [[5, 3]]
    assert m_f.tolist() == [[2, 1]]
    assert th.equal(m_f.flip(-1), m) and th.equal(k_f.flip(-1), k)


def test_count_pair_with_an_invalid_decider():
    """A timed-out decider leaves BOTH its group's k and, if it moved, its m --
    the pair counts valid deciders and valid leavers, nothing else."""
    groups = [[0, 0, 0, 1, 1, 1, 1, 1]]
    switch = [[1, 0, 0, 0, 1, 1, 0, 0]]

    # agent 0 (group 0, a leaver) times out
    valid = [[False, True, True, True, True, True, True, True]]
    m, k = counts_of(groups, switch, valid)
    assert k.tolist() == [[2, 5]]
    assert m.tolist() == [[0, 2]]

    # agent 4 (group 1, a leaver) times out
    valid = [[True, True, True, True, False, True, True, True]]
    m, k = counts_of(groups, switch, valid)
    assert k.tolist() == [[3, 4]]
    assert m.tolist() == [[1, 1]]

    # agent 6 (group 1, a stayer) times out: k shrinks, m does not
    valid = [[True, True, True, True, True, True, False, True]]
    m, k = counts_of(groups, switch, valid)
    assert k.tolist() == [[3, 4]]
    assert m.tolist() == [[1, 2]]


def test_count_pair_on_a_fully_merged_round():
    """After a full merge one group is empty. Its k and m are 0, the other
    group carries all 8, and nothing is NaN -- a state the simulation reaches
    for real."""
    groups = [[1] * 8]
    switch = [[1, 1, 0, 0, 0, 0, 0, 0]]
    m, k = counts_of(groups, switch)
    assert k.tolist() == [[0, 8]]
    assert m.tolist() == [[0, 2]]

    # the whole (merged) group leaving is m == k == 8 on one side
    switch = [[1] * 8]
    m, k = counts_of(groups, switch)
    assert k.tolist() == [[0, 8]]
    assert m.tolist() == [[0, 8]]


def test_count_pair_across_several_graphs_in_a_batch():
    groups = [[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], [1] * 8]
    switch = [
        [1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
    ]
    m, k = counts_of(groups, switch, n_batch=3)
    assert k.tolist() == [[3, 5], [4, 4], [0, 8]]
    assert m.tolist() == [[1, 2], [0, 0], [0, 3]]


def test_leavers_never_exceed_valid_deciders():
    th.manual_seed(11)
    for _ in range(50):
        groups = th.randint(0, 2, (1, N_AGENTS)).tolist()
        switch = th.randint(0, 2, (1, N_AGENTS)).tolist()
        valid = th.randint(0, 2, (1, N_AGENTS)).bool().tolist()
        m, k = counts_of(groups, switch, valid)
        assert bool((m <= k).all())
        assert int(k.sum()) == sum(valid[0])


# --------------------------------------------------------------------------- #
# 2. decision rounds are derived, not hardcoded
# --------------------------------------------------------------------------- #
def decision_rounds(switch_every, n_rounds=N_ROUNDS):
    """``generic/data.py``'s rule: the round played right before each arrival,
    and never the episode's last round (no arrival follows it)."""
    return [
        r for r in range(n_rounds) if (r + 1) % switch_every == 0 and r < n_rounds - 1
    ]


def masked_over_rounds(switch_every, groups, switches, n_rounds=N_ROUNDS):
    rounds = decision_rounds(switch_every, n_rounds)
    n_player = len(groups[0])
    valid = th.zeros((n_player, n_rounds), dtype=th.bool)
    valid[:, rounds] = True
    y = as_nodes(switches, n_rounds).to(th.bool) & valid
    m, k = joint_exodus_counts(
        y,
        valid,
        as_nodes(groups, n_rounds),
        batch_index(1, n_player),
        n_batch=1,
    )
    return m, k, rounds


def test_decision_rounds_follow_switch_every():
    groups = [[0, 0, 0, 1, 1, 1, 1, 1]]
    switch = [[1, 0, 0, 0, 1, 1, 0, 0]]

    m, k, rounds = masked_over_rounds(4, groups, switch)
    assert rounds == [3, 7, 11, 15, 19]  # the config's switch_every: 4
    live = th.nonzero(k.sum(-1)[0] > 0).flatten().tolist()
    assert live == rounds
    assert k[0, 3].tolist() == [3, 5] and m[0, 3].tolist() == [1, 2]
    assert k[0, 0].tolist() == [0, 0] and m[0, 0].tolist() == [0, 0]

    # a different cadence moves the selection with it -- nothing is hardcoded
    m, k, rounds = masked_over_rounds(6, groups, switch)
    assert rounds == [5, 11, 17]
    assert th.nonzero(k.sum(-1)[0] > 0).flatten().tolist() == rounds


def test_loss_selects_exactly_the_decision_rounds():
    groups = [[0, 0, 0, 1, 1, 1, 1, 1]]
    switch = [[1, 0, 0, 0, 1, 1, 0, 0]]
    rounds = decision_rounds(4)
    valid = th.zeros((N_AGENTS, N_ROUNDS), dtype=th.bool)
    valid[:, rounds] = True
    y = as_nodes(switch, N_ROUNDS).to(th.bool) & valid
    agent_group = as_nodes(groups, N_ROUNDS)
    batch_data = fake_batch(y, valid, agent_group, n_batch=1)
    m, k = joint_exodus_counts(y, valid, agent_group, batch_data["batch"], n_batch=1)

    loss, n_cells = joint_exodus_loss(fake_joint(m, k), batch_data)
    assert n_cells == len(rounds) == 5
    # uniform over the valid grid -> -log p = log((k_0 + 1) * (k_1 + 1))
    assert abs(float(loss) - math.log(4 * 6)) < 1e-6


# --------------------------------------------------------------------------- #
# 3. the ruling: incomplete pairs are dropped
# --------------------------------------------------------------------------- #
def one_round_batch(groups, switches, valid):
    n_batch, n_player = len(groups), len(groups[0])
    y = as_nodes(switches).to(th.bool)
    mask = as_nodes(valid).to(th.bool)
    agent_group = as_nodes(groups)
    batch_data = fake_batch(y, mask, agent_group, n_batch, n_player)
    m, k = joint_exodus_counts(
        y, mask, agent_group, batch_data["batch"], n_batch=n_batch
    )
    return batch_data, m, k


def test_incomplete_pairs_are_dropped_by_default():
    assert DROP_INCOMPLETE_PAIRS is True
    groups = [[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]]
    switch = [[1, 0, 0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 1, 1, 0, 0]]
    # graph 1 is complete (k sums to 8); graph 2 has a timeout (k sums to 7)
    valid = [[True] * 8, [False] + [True] * 7]
    batch_data, m, k = one_round_batch(groups, switch, valid)
    assert k.sum(-1).flatten().tolist() == [8, 7]

    loss, n_cells = joint_exodus_loss(fake_joint(m, k), batch_data)
    assert n_cells == 1
    assert abs(float(loss) - math.log(4 * 6)) < 1e-6

    carried, n_carried = joint_exodus_loss(
        fake_joint(m, k), batch_data, drop_incomplete_pairs=False
    )
    assert n_carried == 2
    expected = (math.log(4 * 6) + math.log(3 * 6)) / 2
    assert abs(float(carried) - expected) < 1e-6


def test_a_fully_merged_pair_is_kept_when_every_decider_is_valid():
    """k = (0, 8) still sums to 8, so a merged round is a COMPLETE pair and
    survives the drop -- exactly the state the simulation produces."""
    groups = [[1] * 8]
    switch = [[1, 1, 0, 0, 0, 0, 0, 0]]
    valid = [[True] * 8]
    batch_data, m, k = one_round_batch(groups, switch, valid)
    assert k.tolist() == [[[0, 8]]]
    loss, n_cells = joint_exodus_loss(fake_joint(m, k), batch_data)
    assert n_cells == 1
    assert abs(float(loss) - math.log(1 * 9)) < 1e-6


def test_a_batch_with_no_usable_pair_yields_a_finite_zero():
    """Every decider timed out: no cell survives, the term must not be NaN and
    must still be differentiable."""
    groups = [[0, 0, 0, 1, 1, 1, 1, 1]]
    switch = [[0] * 8]
    valid = [[False] * 8]
    batch_data, m, k = one_round_batch(groups, switch, valid)
    logits = th.zeros((1, 1, GRID, GRID), requires_grad=True)
    loss, n_cells = joint_exodus_loss(fake_joint(m, k, logits), batch_data)
    assert n_cells == 0
    assert float(loss) == 0.0
    loss.backward()
    assert bool(th.isfinite(logits.grad).all())


def test_the_observed_pair_is_never_on_a_masked_cell():
    """m_g <= k_g by construction, so the gathered log-probability is always
    finite -- the reason the loss can select cells rather than multiply by a
    0/1 mask."""
    th.manual_seed(5)
    for _ in range(30):
        groups = th.randint(0, 2, (2, N_AGENTS)).tolist()
        switch = th.randint(0, 2, (2, N_AGENTS)).tolist()
        valid = th.randint(0, 2, (2, N_AGENTS)).bool().tolist()
        batch_data, m, k = one_round_batch(groups, switch, valid)
        log_prob, _ = fake_joint(m, k, th.randn(2, 1, GRID, GRID))
        flat = (m[..., 0] * GRID + m[..., 1]).unsqueeze(-1)
        picked = log_prob.flatten(-2, -1).gather(-1, flat)
        assert bool(th.isfinite(picked).all())
        # a pair one past the support IS masked, so the check has teeth
        over = ((k[..., 0] + 1).clamp(max=GRID - 1) * GRID + k[..., 1]).unsqueeze(-1)
        beyond = log_prob.flatten(-2, -1).gather(-1, over)
        assert bool(th.isinf(beyond[k[..., 0] < MAX_GROUP_SIZE].unsqueeze(-1)).all())


# --------------------------------------------------------------------------- #
# 4. the loss scores what the head emits
# --------------------------------------------------------------------------- #
def test_training_counts_match_the_heads_own_pooling():
    """The correspondence the objective rests on: the k the loss derives from
    ``data.py``-shaped labels is the k the head derived from its own pooling
    inside ``forward``, on the same membership and the same validity mask."""
    groups = [[0, 0, 0, 1, 1, 1, 1, 1], [1] * 8, [0, 0, 0, 0, 1, 1, 1, 1]]
    switch = [[1, 0, 0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], [0] * 8]
    valid = [[True] * 8, [True] * 7 + [False], [True] * 8]
    model = make_model(seed=7, joint_exodus=True)
    state = make_state(groups, switch, valid, n_rounds=2)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=3)
    encoded = model.encode(
        state, mask="switch_valid", edge_index=edge_index, device="cpu"
    )
    _, joint = model(encoded, True, True)
    log_prob, k_head = joint

    m, k = joint_exodus_counts(
        encoded["y"],
        encoded["mask"],
        encoded["agent_group"],
        encoded["batch"],
        n_batch=3,
    )
    assert th.equal(k, k_head)
    assert k[:, 0].tolist() == [[3, 5], [0, 7], [4, 4]]
    assert m[:, 0].tolist() == [[1, 2], [0, 2], [0, 0]]
    # and the real head's grid is a proper distribution on the valid support
    prob = log_prob.exp()
    assert th.allclose(prob.flatten(-2, -1).sum(-1), th.ones(3, 2), atol=1e-6)

    # the loss consumes the encoded state directly, and drops graph 2 (k = 7)
    _, n_cells = joint_exodus_loss(joint, encoded)
    assert n_cells == 4  # 2 complete graphs x 2 rounds


def test_a_mismatched_count_pair_is_caught():
    """The assert is the guard: hand the loss a k that did not come from the
    head's pooling and it must refuse rather than silently train on it."""
    groups = [[0, 0, 0, 1, 1, 1, 1, 1]]
    switch = [[1, 0, 0, 0, 1, 1, 0, 0]]
    valid = [[True] * 8]
    batch_data, m, k = one_round_batch(groups, switch, valid)
    wrong = fake_joint(m, k.flip(-1))
    try:
        joint_exodus_loss(wrong, batch_data)
    except AssertionError as err:
        assert "disagree" in str(err)
    else:
        raise AssertionError("a mismatched count pair was not caught")


# --------------------------------------------------------------------------- #
# 5. the per-agent loss is untouched when the head is off
# --------------------------------------------------------------------------- #
def encode_for_loss(model, state, n_batch):
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=n_batch)
    return model.encode(state, mask="switch_valid", edge_index=edge_index, device="cpu")


def test_head_off_reproduces_the_legacy_loss_bit_for_bit():
    groups = [[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]]
    switch = [[1, 0, 0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0, 1, 0]]
    valid = [[True] * 8, [False] + [True] * 7]
    state = make_state(groups, switch, valid, n_rounds=3)
    loss_fn = th.nn.CrossEntropyLoss(reduction="none")

    ref_model = make_model(seed=13)
    new_model = make_model(seed=13)
    ref = legacy_batch_loss(
        ref_model, encode_for_loss(ref_model, state, 2), loss_fn, 0.0
    )
    new, components = compute_batch_loss(
        new_model, encode_for_loss(new_model, state, 2), loss_fn, 0.0
    )
    assert th.equal(new, ref)
    assert components["joint"] is None and components["n_joint"] == 0
    assert components["agent"] == ref.item()

    # ... and the gradients it hands the optimiser are identical too
    ref.backward()
    new.backward()
    ref_grads = dict(ref_model.named_parameters())
    for name, param in new_model.named_parameters():
        assert th.equal(param.grad, ref_grads[name].grad), name


def test_head_off_reproduces_the_legacy_loss_with_entropy_regularisation():
    """``l1_entropy`` is 0 in the shipped config; check the other branch anyway
    so the extraction is not only tested on the term it zeroes out."""
    groups = [[0, 0, 0, 1, 1, 1, 1, 1]]
    switch = [[1, 0, 0, 0, 1, 1, 0, 0]]
    valid = [[True] * 8]
    state = make_state(groups, switch, valid, n_rounds=2)
    loss_fn = th.nn.CrossEntropyLoss(reduction="none")
    ref_model = make_model(seed=17)
    new_model = make_model(seed=17)
    ref = legacy_batch_loss(
        ref_model, encode_for_loss(ref_model, state, 1), loss_fn, 0.25
    )
    new, _ = compute_batch_loss(
        new_model, encode_for_loss(new_model, state, 1), loss_fn, 0.25
    )
    assert th.equal(new, ref)


@pytest.mark.parametrize("size_encoding", SIZE_ENCODINGS)
def test_head_on_keeps_the_per_agent_component_identical(size_encoding):
    """The head branches off the post-RNN embeddings and feeds nothing back, so
    at equal trunk weights the per-agent component is the legacy number; only
    the total carries the extra term. Holds under either size encoding: the
    encoding only changes the head's own input width, not the trunk."""
    groups = [[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]]
    switch = [[1, 0, 0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0, 1, 0]]
    valid = [[True] * 8, [True] * 8]
    state = make_state(groups, switch, valid, n_rounds=2)
    loss_fn = th.nn.CrossEntropyLoss(reduction="none")

    off = make_model(seed=23)
    on = make_model(
        seed=23, joint_exodus=True, joint_exodus_size_encoding=size_encoding
    )
    ref = legacy_batch_loss(off, encode_for_loss(off, state, 2), loss_fn, 0.0)
    total, components = compute_batch_loss(
        on, encode_for_loss(on, state, 2), loss_fn, 0.0
    )
    assert components["agent"] == ref.item()
    assert components["joint"] is not None
    assert components["n_joint"] == 4  # 2 complete graphs x 2 rounds
    assert abs(float(total) - (components["agent"] + components["joint"])) < 1e-6


# --------------------------------------------------------------------------- #
# 6. end to end: the objective actually optimises
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size_encoding", SIZE_ENCODINGS)
def test_the_joint_objective_trains_on_a_synthetic_batch(size_encoding):
    """A real GraphNetwork with the head on, a real optimiser, real backward
    passes: the joint component must fall and nothing may go non-finite. Holds
    under either size encoding: the wider onehot readout is still a plain MLP
    fitted by gradient descent."""
    groups = [[0, 0, 0, 1, 1, 1, 1, 1]] * 4
    # the same state always produces the same pair -- a learnable target
    switch = [[1, 0, 0, 0, 1, 1, 0, 0]] * 4
    valid = [[True] * 8] * 4
    state = make_state(groups, switch, valid, n_rounds=2)
    loss_fn = th.nn.CrossEntropyLoss(reduction="none")
    model = make_model(
        seed=29, joint_exodus=True, joint_exodus_size_encoding=size_encoding
    )
    optimizer = th.optim.Adam(model.parameters(), lr=5e-3)

    history = []
    for _ in range(200):
        optimizer.zero_grad()
        encoded = encode_for_loss(model, state, 4)
        loss, components = compute_batch_loss(model, encoded, loss_fn, 0.0)
        assert math.isfinite(float(loss))
        assert components["n_joint"] == 8  # 4 complete graphs x 2 rounds
        loss.backward()
        for param in model.parameters():
            assert param.grad is None or bool(th.isfinite(param.grad).all())
        optimizer.step()
        history.append(components["joint"])

    # the target pair is (1, 2) on a k = (3, 5) grid; uniform over that grid
    # is log 24 = 3.178 nats, and the head must end far below it
    assert all(math.isfinite(h) for h in history)
    assert history[-1] == min(history), history[-5:]
    assert history[-1] < 1.0, history[-5:]

    # and the fitted head puts its mass on the observed pair
    encoded = encode_for_loss(model, state, 4)
    _, joint = model(encoded, True, True)
    log_prob, k = joint
    assert k[0, 0].tolist() == [3, 5]
    assert float(log_prob[0, 0, 1, 2].exp()) > 0.8
    # the mass that is left sits on the valid support, not off it
    assert abs(float(log_prob[0, 0].exp().sum()) - 1.0) < 1e-5
    assert float(log_prob[0, 0, :4, :6].exp().sum()) > 1.0 - 1e-5
