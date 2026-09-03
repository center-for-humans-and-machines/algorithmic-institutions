"""Unit tests for the joint exodus head's numerics: the masked joint over the
count pair ``(m_0, m_1)``, the label-canonical group pooling that feeds it,
and the head end to end (support, feature convention, determinism,
differentiability).

Every value that is checked is derived independently here -- a plain Python
group mean, a Python log-sum-exp over the valid cells -- rather than restated
from the implementation, and the structural claims are written as
invariances (transposition under the label flip, batch separation, exact
zeros off the support) so a wrong implementation cannot pass by agreement.

Invariants and rationale:
``notes/autoresearch_log/switch-joint-exodus-gmlp.md``.

Local test (CPU torch, no PyG):
    .venv/bin/python -m pytest tests/switch/test_joint_exodus.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch as th

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # tests/switch -> repo root
# this checkout's src must win over any installed/editable aimanager
sys.path.insert(0, str(ROOT / "src"))

from aimanager.generic.joint_exodus import (  # noqa: E402
    MAX_GROUP_SIZE,
    ROUND_NORM,
    SIZE_NORM,
    JointExodusHead,
    joint_count_mask,
    masked_joint_log_prob,
    pool_by_group,
)

SEED = 20260903
N_AGENTS = 8
GRID = MAX_GROUP_SIZE + 1


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_nodes(n_batch, n_rounds, n_features, groups, seed=SEED):
    """Return (x, agent_group, batch) shaped as `GraphNetwork.encode` flattens.

    `groups` is a per-agent label list, broadcast over rounds and batch.
    """
    th.manual_seed(seed)
    n_agents = len(groups)
    n = n_batch * n_agents
    x = th.randn(n, n_rounds, n_features, dtype=th.float64)
    agent_group = (
        th.tensor(groups, dtype=th.int64)
        .repeat(n_batch)
        .reshape(n, 1)
        .expand(n, n_rounds)
    )
    batch = th.arange(n_batch, dtype=th.int64).repeat_interleave(n_agents)
    return x, agent_group.contiguous(), batch


def make_head(embed_size=5, hidden_size=7, seed=SEED):
    th.manual_seed(seed)
    return JointExodusHead(embed_size=embed_size, hidden_size=hidden_size).double()


def python_masked_log_prob(logits, k0, k1):
    """The masked joint, recomputed in plain Python: renormalise over the
    cells with `m_0 <= k_0` and `m_1 <= k_1` and nothing else."""
    cells = [
        (m0, m1) for m0 in range(GRID) for m1 in range(GRID) if m0 <= k0 and m1 <= k1
    ]
    values = [float(logits[m0][m1]) for m0, m1 in cells]
    top = max(values)
    norm = top + math.log(sum(math.exp(v - top) for v in values))
    out = [[-math.inf] * GRID for _ in range(GRID)]
    for (m0, m1), v in zip(cells, values):
        out[m0][m1] = v - norm
    return out


# --------------------------------------------------------------------------- #
# 1. the masked softmax
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "k0,k1",
    [(4, 4), (3, 5), (8, 0), (0, 8), (0, 0), (1, 7), (2, 6), (8, 8)],
)
def test_masked_softmax_sums_to_one_over_valid_cells(k0, k1):
    th.manual_seed(SEED)
    logits = th.randn(GRID, GRID, dtype=th.float64)
    k = th.tensor([k0, k1], dtype=th.int64)
    log_prob, valid = masked_joint_log_prob(logits, k)
    proba = log_prob.exp()

    assert valid.sum().item() == (k0 + 1) * (k1 + 1)
    assert proba[valid].sum().item() == pytest.approx(1.0, abs=1e-12)
    assert proba.sum().item() == pytest.approx(1.0, abs=1e-12)

    # the whole grid against an independent Python renormalisation
    expected = python_masked_log_prob(logits.tolist(), k0, k1)
    for m0 in range(GRID):
        for m1 in range(GRID):
            got = float(log_prob[m0, m1])
            if math.isinf(expected[m0][m1]):
                assert math.isinf(got) and got < 0
            else:
                assert got == pytest.approx(expected[m0][m1], abs=1e-12)


@pytest.mark.parametrize("k0,k1", [(4, 4), (8, 0), (0, 8), (0, 0)])
def test_masked_cells_have_exactly_zero_probability(k0, k1):
    """-inf before the softmax, so masked cells are zero bit-for-bit, not a
    small residual from a large finite penalty."""
    th.manual_seed(SEED)
    logits = th.randn(GRID, GRID, dtype=th.float64) * 10.0
    k = th.tensor([k0, k1], dtype=th.int64)
    log_prob, valid = masked_joint_log_prob(logits, k)
    proba = log_prob.exp()

    assert th.all(proba[~valid] == 0.0)
    assert th.all(th.isinf(log_prob[~valid]) & (log_prob[~valid] < 0))
    assert th.isfinite(log_prob[valid]).all()


def test_masked_softmax_ignores_logits_off_the_support():
    """A cell outside the support cannot influence the surviving cells --
    the invariance that separates masking BEFORE the softmax from a penalty
    added after it."""
    th.manual_seed(SEED)
    logits = th.randn(GRID, GRID, dtype=th.float64)
    k = th.tensor([3, 5], dtype=th.int64)
    reference, _ = masked_joint_log_prob(logits, k)

    perturbed = logits.clone()
    perturbed[4:, :] += 50.0
    perturbed[:, 6:] -= 50.0
    moved, _ = masked_joint_log_prob(perturbed, k)

    assert th.equal(reference[:4, :6], moved[:4, :6])


def test_degenerate_all_in_one_group_is_a_valid_distribution():
    """k = (8, 0): the whole population merged into group 0, so every column
    but m_1 = 0 is masked -- a real simulation state, not a corner case."""
    th.manual_seed(SEED)
    logits = th.randn(GRID, GRID, dtype=th.float64)
    k = th.tensor([MAX_GROUP_SIZE, 0], dtype=th.int64)
    log_prob, valid = masked_joint_log_prob(logits, k)
    proba = log_prob.exp()

    assert valid[:, 0].all()
    assert not valid[:, 1:].any()
    assert th.isfinite(proba).all()
    assert not th.isnan(proba).any()
    assert proba[:, 0].sum().item() == pytest.approx(1.0, abs=1e-12)
    # every leaver count 0..8 out of group 0 remains reachable
    assert th.all(proba[:, 0] > 0.0)


def test_degenerate_mirror_is_the_transpose():
    """k = (0, 8) is the label flip of k = (8, 0): on transposed logits the
    joint must be the exact transpose, which is the symmetry the flip-doubled
    training data relies on."""
    th.manual_seed(SEED)
    logits = th.randn(GRID, GRID, dtype=th.float64)
    a, _ = masked_joint_log_prob(logits, th.tensor([MAX_GROUP_SIZE, 0]))
    b, _ = masked_joint_log_prob(
        logits.t().contiguous(), th.tensor([0, MAX_GROUP_SIZE])
    )

    assert th.equal(a, b.t())


def test_empty_grid_is_unreachable():
    """(0, 0) is valid for every non-negative k, so the flattened softmax
    always has at least one finite entry."""
    for k0 in range(GRID):
        for k1 in range(GRID):
            k = th.tensor([k0, k1], dtype=th.int64)
            assert bool(joint_count_mask(k)[0, 0])


def test_mask_rejects_out_of_range_counts():
    with pytest.raises(AssertionError, match="non-negative"):
        joint_count_mask(th.tensor([-1, 3]))
    with pytest.raises(AssertionError, match="max_group_size"):
        joint_count_mask(th.tensor([9, 3]))


def test_masked_softmax_batches_over_leading_dims():
    th.manual_seed(SEED)
    logits = th.randn(3, 4, GRID * GRID, dtype=th.float64)
    k = th.randint(0, GRID, (3, 4, 2))
    log_prob, valid = masked_joint_log_prob(logits, k)
    proba = log_prob.exp()

    assert log_prob.shape == (3, 4, GRID, GRID)
    assert th.allclose(proba.sum((-2, -1)), th.ones(3, 4, dtype=th.float64))
    assert th.all(proba[~valid] == 0.0)

    # each leading cell equals the same call made on its own
    for i in range(3):
        for j in range(4):
            alone, _ = masked_joint_log_prob(logits[i, j], k[i, j])
            assert th.equal(alone, log_prob[i, j])


def test_gradients_stay_finite_through_the_mask():
    th.manual_seed(SEED)
    logits = th.randn(GRID, GRID, dtype=th.float64, requires_grad=True)
    k = th.tensor([MAX_GROUP_SIZE, 0], dtype=th.int64)
    log_prob, valid = masked_joint_log_prob(logits, k)
    # a cross-entropy against one valid cell, the training objective's shape
    loss = -log_prob[3, 0]
    loss.backward()

    assert th.isfinite(logits.grad).all()
    assert th.all(logits.grad[~valid] == 0.0)


# --------------------------------------------------------------------------- #
# 2. group pooling
# --------------------------------------------------------------------------- #
def test_pooling_matches_a_plain_python_group_mean():
    groups = [0, 0, 0, 1, 1, 1, 1, 1]
    x, agent_group, batch = make_nodes(2, 3, 4, groups)
    pooled, counts = pool_by_group(x, agent_group, batch)

    assert pooled.shape == (2, 3, 2, 4)
    assert counts.shape == (2, 3, 2)
    for b in range(2):
        rows = x[b * len(groups) : (b + 1) * len(groups)]
        for g, size in ((0, 3), (1, 5)):
            member = [i for i, gg in enumerate(groups) if gg == g]
            for r in range(3):
                for f in range(4):
                    expected = sum(float(rows[i, r, f]) for i in member) / size
                    assert float(pooled[b, r, g, f]) == pytest.approx(
                        expected, abs=1e-12
                    )
            assert th.all(counts[b, :, g] == size)


def test_pooling_is_canonical_in_the_group_label_not_in_size():
    """The flip-doubled training data mirrors the labels. Under label-order
    pooling the mirrored copy is the exact transpose of the original, which is
    what symmetrises the head; size-order pooling would collapse both copies
    onto the same input."""
    groups = [0, 0, 0, 1, 1, 1, 1, 1]
    flipped = [1 - g for g in groups]
    x, agent_group, batch = make_nodes(1, 2, 4, groups)
    _, agent_group_flipped, _ = make_nodes(1, 2, 4, flipped)

    pooled, counts = pool_by_group(x, agent_group, batch)
    pooled_f, counts_f = pool_by_group(x, agent_group_flipped, batch)

    # group 0 is the size-3 group in one copy and the size-5 group in the
    # other: the head sees the labels swapped, not re-sorted by size
    assert th.all(counts[0, :, 0] == 3) and th.all(counts[0, :, 1] == 5)
    assert th.all(counts_f[0, :, 0] == 5) and th.all(counts_f[0, :, 1] == 3)
    assert th.equal(pooled[0, :, 0], pooled_f[0, :, 1])
    assert th.equal(pooled[0, :, 1], pooled_f[0, :, 0])


def test_pooling_ignores_node_order_within_a_group():
    """Membership comes from `agent_group` alone, so a permutation of the
    nodes cannot move the pooled cells."""
    groups = [0, 1, 0, 1, 0, 1, 0, 1]
    x, agent_group, batch = make_nodes(1, 2, 3, groups)
    perm = th.tensor([5, 0, 7, 2, 1, 6, 3, 4])

    pooled, counts = pool_by_group(x, agent_group, batch)
    permuted, counts_p = pool_by_group(x[perm], agent_group[perm], batch)

    assert th.allclose(pooled, permuted)
    assert th.equal(counts, counts_p)


def test_empty_group_pools_to_zero_without_nan():
    """A full merge leaves a group of size 0 -- a state the simulation reaches
    (mean larger-group size 6.088 in the human data, size 8 at 14.4%)."""
    groups = [0] * N_AGENTS
    x, agent_group, batch = make_nodes(2, 3, 4, groups)
    pooled, counts = pool_by_group(x, agent_group, batch)

    assert th.all(counts[:, :, 0] == N_AGENTS)
    assert th.all(counts[:, :, 1] == 0)
    assert th.all(pooled[:, :, 1] == 0.0)
    assert not th.isnan(pooled).any()


def test_pooling_respects_the_decider_mask():
    """k_g counts valid DECIDERS, not nominal members: 109 of 2,000 human
    decision rows fail `switch_valid`."""
    groups = [0, 0, 0, 0, 1, 1, 1, 1]
    x, agent_group, batch = make_nodes(1, 1, 3, groups)
    mask = th.ones(N_AGENTS, 1, dtype=th.bool)
    mask[0] = False  # one group-0 member timed out
    pooled, counts = pool_by_group(x, agent_group, batch, mask=mask)

    assert counts[0, 0, 0].item() == 3
    assert counts[0, 0, 1].item() == 4
    for f in range(3):
        expected = sum(float(x[i, 0, f]) for i in (1, 2, 3)) / 3
        assert float(pooled[0, 0, 0, f]) == pytest.approx(expected, abs=1e-12)


def test_all_deciders_masked_out_pools_to_zero_without_nan():
    """A group whose every member is invalid is indistinguishable from an
    empty one: count 0, zero vector, no NaN."""
    groups = [0, 0, 0, 0, 1, 1, 1, 1]
    x, agent_group, batch = make_nodes(1, 2, 3, groups)
    mask = th.ones(N_AGENTS, 2, dtype=th.bool)
    mask[:4] = False
    pooled, counts = pool_by_group(x, agent_group, batch, mask=mask)

    assert th.all(counts[0, :, 0] == 0)
    assert th.all(pooled[0, :, 0] == 0.0)
    assert not th.isnan(pooled).any()


def test_pooling_separates_batch_elements():
    groups = [0, 0, 0, 0, 1, 1, 1, 1]
    x, agent_group, batch = make_nodes(3, 2, 4, groups)
    pooled, _ = pool_by_group(x, agent_group, batch)
    assert not th.allclose(pooled[0], pooled[1])
    for b in range(3):
        rows = x[b * N_AGENTS : (b + 1) * N_AGENTS]
        assert th.allclose(pooled[b, :, 0], rows[:4].mean(dim=0))
        assert th.allclose(pooled[b, :, 1], rows[4:].mean(dim=0))

    # a change confined to one graph stays confined to it
    bumped = x.clone()
    bumped[0] += 100.0
    moved, _ = pool_by_group(bumped, agent_group, batch)
    assert not th.allclose(moved[0], pooled[0])
    assert th.equal(moved[1:], pooled[1:])


# --------------------------------------------------------------------------- #
# 3. the head end to end
# --------------------------------------------------------------------------- #
def head_inputs(groups, n_batch=2, n_rounds=3, round_number=7, embed=5):
    x, agent_group, batch = make_nodes(n_batch, n_rounds, embed, groups)
    rounds = th.arange(round_number, round_number + n_rounds, dtype=th.int64)
    round_tensor = rounds.reshape(1, n_rounds).expand(x.shape[0], n_rounds)
    return x, agent_group, batch, round_tensor.contiguous()


def test_head_emits_a_valid_masked_joint():
    groups = [0, 0, 0, 1, 1, 1, 1, 1]
    x, agent_group, batch, round_number = head_inputs(groups)
    head = make_head()
    log_prob, k = head(
        x, agent_group=agent_group, round_number=round_number, batch=batch
    )
    proba = log_prob.exp()

    assert log_prob.shape == (2, 3, GRID, GRID)
    assert k.shape == (2, 3, 2)
    assert th.all(k[..., 0] == 3) and th.all(k[..., 1] == 5)
    assert th.allclose(proba.sum((-2, -1)), th.ones(2, 3, dtype=th.float64))
    valid = joint_count_mask(k)
    assert th.all(proba[~valid] == 0.0)
    assert th.all(proba[valid] > 0.0)


def test_head_handles_the_fully_merged_state():
    x, agent_group, batch, round_number = head_inputs([0] * N_AGENTS)
    head = make_head()
    log_prob, k = head(
        x, agent_group=agent_group, round_number=round_number, batch=batch
    )
    proba = log_prob.exp()

    assert th.all(k[..., 0] == N_AGENTS) and th.all(k[..., 1] == 0)
    assert th.isfinite(proba).all() and not th.isnan(proba).any()
    assert th.allclose(proba.sum((-2, -1)), th.ones(2, 3, dtype=th.float64))
    assert th.all(proba[..., 1:] == 0.0)


def test_head_support_follows_the_decider_mask():
    """An invalid decider shrinks the support: with one group-0 member out,
    m_0 = 4 becomes unreachable even though the group still has 4 members."""
    groups = [0, 0, 0, 0, 1, 1, 1, 1]
    x, agent_group, batch, round_number = head_inputs(groups, n_batch=1, n_rounds=1)
    head = make_head()
    mask = th.ones(N_AGENTS, 1, dtype=th.bool)
    mask[0] = False

    _, k_full = head(x, agent_group=agent_group, round_number=round_number, batch=batch)
    log_prob, k = head(
        x,
        agent_group=agent_group,
        round_number=round_number,
        batch=batch,
        decider_mask=mask,
    )
    proba = log_prob.exp()

    assert th.all(k_full[..., 0] == 4)
    assert th.all(k[..., 0] == 3) and th.all(k[..., 1] == 4)
    assert th.all(proba[0, 0, 4:, :] == 0.0)
    assert th.all(proba[0, 0, :, 5:] == 0.0)
    assert proba.sum().item() == pytest.approx(1.0, abs=1e-12)


def test_head_feature_normalisation_convention():
    """Sizes enter as k / 8 and the round as r / 23 -- the `IntEncoder`
    numeric convention (`v / (n_levels - 1)`) the model's own `round_number`
    feature already uses."""
    groups = [0, 0, 0, 1, 1, 1, 1, 1]
    x, agent_group, batch, round_number = head_inputs(groups, n_batch=1, n_rounds=2)
    head = make_head()

    captured = {}

    class Capture(th.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, features):
            captured["features"] = features.detach().clone()
            return self.inner(features)

    head.mlp = Capture(head.mlp)
    head(x, agent_group=agent_group, round_number=round_number, batch=batch)

    features = captured["features"]
    assert features.shape == (1, 2, 2 * 5 + 2 + 1)
    pooled, _ = pool_by_group(x, agent_group, batch)
    assert th.allclose(features[..., : 2 * 5], pooled.flatten(-2, -1))
    assert th.allclose(
        features[0, :, 10], th.full((2,), 3.0 / SIZE_NORM, dtype=th.float64)
    )
    assert th.allclose(
        features[0, :, 11], th.full((2,), 5.0 / SIZE_NORM, dtype=th.float64)
    )
    expected_round = th.tensor([7.0, 8.0], dtype=th.float64) / ROUND_NORM
    assert th.allclose(features[0, :, 12], expected_round)
    assert SIZE_NORM == float(MAX_GROUP_SIZE) and ROUND_NORM == 23.0


def test_head_is_deterministic_and_differentiable():
    groups = [0, 0, 0, 0, 1, 1, 1, 1]
    x, agent_group, batch, round_number = head_inputs(groups, n_batch=1, n_rounds=1)
    head = make_head()
    log_prob, k = head(
        x, agent_group=agent_group, round_number=round_number, batch=batch
    )
    again, _ = head(x, agent_group=agent_group, round_number=round_number, batch=batch)
    assert th.equal(log_prob, again)

    (-log_prob[0, 0, 2, 1]).backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads
    assert all(th.isfinite(g).all() for g in grads)
    assert any(np.abs(g.numpy()).sum() > 0 for g in grads)
