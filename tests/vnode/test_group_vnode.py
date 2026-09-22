"""Tests for the per-group virtual node (plan step 3 of
contribution-group-vnode).

Plain pytest, torch only -- no `torch_geometric` / `torch_scatter` -- so this
runs locally on macOS:
    uv run pytest tests/vnode/test_group_vnode.py -q

The `GraphNetwork` gate (default off, save/load back-compat, forward pass
unchanged, the copula co-existing with the node) is covered by
tests/vnode/test_group_vnode_graph.py.
Context: notes/autoresearch_log/contribution-group-vnode.md.
"""

import pytest
import torch as th

from aimanager.generic.group_vnode import GroupVirtualNode, pool_by_group

SEED = 20260915
N_AGENTS = 8
N_GROUPS = 2


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_agent_group(group_lists, n_rounds):
    """(N, R) int64 group labels, constant across rounds within a batch
    element. `group_lists` is one per-agent label list per batch element,
    concatenated batch-major -- the way `GraphNetwork.encode` flattens
    nodes."""
    rows = []
    for groups in group_lists:
        g = th.tensor(groups, dtype=th.int64).reshape(-1, 1)
        rows.append(g.expand(len(groups), n_rounds))
    return th.cat(rows, dim=0).contiguous()


def make_batch(n_batch, n_agents=N_AGENTS):
    return th.arange(n_batch, dtype=th.int64).repeat_interleave(n_agents)


def make_node(embed_size=4, hidden_size=6, seed=SEED, **kwargs):
    th.manual_seed(seed)
    return GroupVirtualNode(embed_size, hidden_size, **kwargs).double()


def make_x(n, n_rounds, n_features, seed=SEED):
    th.manual_seed(seed)
    return th.randn(n, n_rounds, n_features, dtype=th.float64)


# --------------------------------------------------------------------------- #
# 1. output shapes
# --------------------------------------------------------------------------- #
def test_output_shapes():
    n_batch, n_rounds, embed, hidden = 3, 5, 4, 6
    groups = [0, 0, 0, 1, 1, 1, 1, 1]
    agent_group = make_agent_group([groups] * n_batch, n_rounds)
    batch = make_batch(n_batch)
    x = make_x(n_batch * N_AGENTS, n_rounds, embed)
    node = make_node(embed, hidden)

    g_node, h = node(x, agent_group=agent_group, batch=batch)

    assert g_node.shape == (n_batch * N_AGENTS, n_rounds, hidden)
    assert h.shape == (1, n_batch * N_GROUPS, hidden)
    assert th.isfinite(g_node).all()
    assert th.isfinite(h).all()


# --------------------------------------------------------------------------- #
# 2. group isolation
# --------------------------------------------------------------------------- #
def test_group_isolation_perturbing_one_group_leaves_the_other_exact():
    """Two groups fed distinguishable inputs: perturbing only group 1's
    members must leave every group-0 node's output bit-for-bit unchanged,
    round by round, and its slice of `h` unchanged too."""
    n_rounds, embed, hidden = 4, 4, 5
    groups = [0, 0, 0, 1, 1, 1, 1, 1]
    agent_group = make_agent_group([groups], n_rounds)
    batch = make_batch(1)
    x = make_x(N_AGENTS, n_rounds, embed)
    node = make_node(embed, hidden)

    g0, h0 = node(x, agent_group=agent_group, batch=batch)

    th.manual_seed(SEED + 1)
    x_perturbed = x.clone()
    x_perturbed[3:] += th.randn_like(x_perturbed[3:])
    g1, h1 = node(x_perturbed, agent_group=agent_group, batch=batch)

    # group-0 members (indices 0..2) are untouched exactly
    assert th.equal(g0[:3], g1[:3])
    assert th.equal(h0[:, 0:1], h1[:, 0:1])
    # group-1 members (indices 3..7) did change, and so did their cell of h
    assert not th.equal(g0[3:], g1[3:])
    assert not th.equal(h0[:, 1:2], h1[:, 1:2])


# --------------------------------------------------------------------------- #
# 3. arrival pickup
# --------------------------------------------------------------------------- #
def test_arrival_pickup_reads_old_group_then_new_group():
    """Agent 0 switches from group 0 to group 1 at round 3 of 6. Before the
    switch it must read exactly what the other permanent group-0 members read
    at those rounds; from the switch round on it must read exactly what the
    permanent group-1 members read -- and not the other group, in either
    window."""
    n_rounds, embed, hidden = 6, 4, 5
    # agent 0: group 0 for rounds 0-2, group 1 for rounds 3-5
    # agents 1, 2: permanent group 0; agents 3-7: permanent group 1
    schedule = [
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ]
    agent_group = th.tensor(schedule, dtype=th.int64)
    batch = make_batch(1)
    x = make_x(N_AGENTS, n_rounds, embed)
    node = make_node(embed, hidden)

    g_node, h = node(x, agent_group=agent_group, batch=batch)

    # cross-check against an independent re-derivation from the already
    # separately-tested pool_by_group plus the module's own GRU instance, so
    # the assertions below are not just re-testing the gather indexing.
    pooled, counts = pool_by_group(x, agent_group, batch, n_batch=1, n_groups=2)
    seq = th.cat([pooled, (counts / node.size_norm).unsqueeze(-1)], dim=-1)
    seq = seq.permute(0, 2, 1, 3).reshape(1 * 2, n_rounds, embed + 1)
    g_direct, h_direct = node.gru(seq)
    cell = batch.reshape(N_AGENTS, 1) * 2 + agent_group
    expected = th.gather(
        g_direct, 0, cell.unsqueeze(-1).expand(N_AGENTS, n_rounds, hidden)
    )
    assert th.equal(g_node, expected)
    assert th.equal(h, h_direct)

    # before the switch (rounds 0-2): agent 0 matches the permanent group-0
    # members, not the permanent group-1 members
    assert th.equal(g_node[0:1, :3], g_node[1:2, :3])
    assert th.equal(g_node[0:1, :3], g_node[2:3, :3])
    assert not th.equal(g_node[0:1, :3], g_node[3:4, :3])

    # from the switch round on (rounds 3-5): agent 0 matches every permanent
    # group-1 member, not the permanent group-0 members
    for member in range(3, 8):
        assert th.equal(g_node[0:1, 3:], g_node[member : member + 1, 3:])
    assert not th.equal(g_node[0:1, 3:], g_node[1:2, 3:])


# --------------------------------------------------------------------------- #
# 4. emptied group
# --------------------------------------------------------------------------- #
def test_emptied_group_is_finite_and_the_run_completes():
    """All of group 1's members switch into group 0 partway through the
    episode, leaving group 1 with zero members for the remaining rounds. The
    zero-input step must stay finite everywhere in both outputs."""
    n_agents, embed, hidden = 4, 4, 5
    schedule = [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    n_rounds = len(schedule)
    agent_group = th.tensor(schedule, dtype=th.int64).T.contiguous()
    batch = make_batch(1, n_agents)
    x = make_x(n_agents, n_rounds, embed)
    node = make_node(embed, hidden)

    g_node, h = node(x, agent_group=agent_group, batch=batch)

    assert g_node.shape == (n_agents, n_rounds, hidden)
    assert th.isfinite(g_node).all()
    assert th.isfinite(h).all()
    assert not th.isnan(g_node).any()
    assert not th.isnan(h).any()

    # group 1 really is empty for the last three rounds: its pooled input is
    # an exact zero vector with count 0, not merely "small".
    pooled, counts = pool_by_group(x, agent_group, batch, n_batch=1, n_groups=2)
    assert th.all(counts[0, 2:, 1] == 0)
    assert th.all(pooled[0, 2:, 1] == 0.0)


# --------------------------------------------------------------------------- #
# 5. per-round parity -- the train/sim contract
# --------------------------------------------------------------------------- #
def test_per_round_calls_reproduce_the_single_call():
    """24 rounds fed as 24 separate R=1 calls, carrying `h`, must reproduce
    the single R=24 call to 1e-6 -- the contract between training (one
    24-round call) and simulation (`environment.py: update_contribution`
    calling the model once per round). Includes an agent that switches group
    mid-sequence, in one batch element but not the other."""
    n_rounds, embed, hidden, n_batch = 24, 6, 5, 2
    groups_b0 = [0, 0, 0, 0, 1, 1, 1, 1]
    groups_b1 = [0, 0, 1, 1, 1, 0, 0, 1]
    agent_group = make_agent_group([groups_b0, groups_b1], n_rounds).clone()
    # agent 1 in batch element 0 switches group 0 -> 1 at round 12
    agent_group[1, 12:] = 1
    # agent 5 in batch element 1 (index 8 + 5) switches group 0 -> 1 at round 6
    agent_group[13, 6:] = 1
    batch = make_batch(n_batch)
    x = make_x(n_batch * N_AGENTS, n_rounds, embed)
    node = make_node(embed, hidden)

    g_all, h_all = node(
        x, agent_group=agent_group, batch=batch, n_batch=n_batch, h0=None
    )

    h = None
    g_chunks = []
    for r in range(n_rounds):
        g_r, h = node(
            x[:, r : r + 1],
            agent_group=agent_group[:, r : r + 1],
            batch=batch,
            h0=h,
            n_batch=n_batch,
        )
        g_chunks.append(g_r)
    g_seq = th.cat(g_chunks, dim=1)

    assert g_seq.shape == g_all.shape
    g_diff = (g_seq - g_all).abs().max().item()
    h_diff = (h - h_all).abs().max().item()
    assert g_diff < 1e-6, g_diff
    assert h_diff < 1e-6, h_diff


# --------------------------------------------------------------------------- #
# 6. flip-augmentation invariance
# --------------------------------------------------------------------------- #
def test_flip_augmentation_permutes_group_states_output_unchanged():
    """Relabelling 0 <-> 1 (the flip-doubled training data's augmentation)
    permutes which hidden row holds which group's state but must leave every
    node's own output exactly unchanged -- a node still reads its own group,
    whatever that group is now called."""
    n_rounds, embed, hidden = 4, 4, 5
    groups = [0, 0, 0, 1, 1, 1, 1, 1]
    agent_group = make_agent_group([groups], n_rounds)
    flipped = 1 - agent_group
    batch = make_batch(1)
    x = make_x(N_AGENTS, n_rounds, embed)
    node = make_node(embed, hidden)

    g, h = node(x, agent_group=agent_group, batch=batch)
    g_flipped, h_flipped = node(x, agent_group=flipped, batch=batch)

    assert th.equal(g, g_flipped)
    # the hidden rows are swapped, not merely "different"
    assert th.equal(h[:, 0:1], h_flipped[:, 1:2])
    assert th.equal(h[:, 1:2], h_flipped[:, 0:1])


# --------------------------------------------------------------------------- #
# 7. occupancy counts
# --------------------------------------------------------------------------- #
def test_pool_by_group_counts_match_membership():
    groups_b0 = [0, 0, 0, 1, 1, 1, 1, 1]
    groups_b1 = [0, 1, 1, 1, 1, 1, 1, 1]
    n_rounds = 3
    agent_group = make_agent_group([groups_b0, groups_b1], n_rounds)
    batch = make_batch(2)
    x = make_x(2 * N_AGENTS, n_rounds, 4)

    _, counts = pool_by_group(x, agent_group, batch)

    assert counts.shape == (2, n_rounds, 2)
    for b, groups in enumerate((groups_b0, groups_b1)):
        for g in (0, 1):
            expected = groups.count(g)
            assert th.all(counts[b, :, g] == expected)


# --------------------------------------------------------------------------- #
# 8. n_batch and h0 contracts
# --------------------------------------------------------------------------- #
def test_explicit_larger_n_batch_widens_h_but_leaves_node_outputs_unchanged():
    n_rounds, embed, hidden = 3, 4, 5
    groups = [0, 0, 0, 1, 1, 1, 1, 1]
    agent_group = make_agent_group([groups], n_rounds)
    batch = make_batch(1)
    x = make_x(N_AGENTS, n_rounds, embed)
    node = make_node(embed, hidden)

    g_default, h_default = node(x, agent_group=agent_group, batch=batch)
    g_wide, h_wide = node(x, agent_group=agent_group, batch=batch, n_batch=3)

    assert h_default.shape == (1, 1 * N_GROUPS, hidden)
    assert h_wide.shape == (1, 3 * N_GROUPS, hidden)
    # a wider GRU batch takes a different BLAS code path, so the untouched
    # rows agree only to floating-point noise (~1e-16 here), not bit-exactly
    assert th.allclose(g_default, g_wide, atol=1e-12)
    assert th.allclose(h_default, h_wide[:, :N_GROUPS], atol=1e-12)
    # the extra graphs' cells are a real zero-input group state, not NaN
    assert th.isfinite(h_wide[:, N_GROUPS:]).all()
    assert not th.isnan(h_wide).any()


def test_too_small_n_batch_raises():
    n_rounds, embed, hidden = 2, 4, 5
    groups = [0, 0, 0, 1, 1, 1, 1, 1]
    agent_group = make_agent_group([groups, groups], n_rounds)
    batch = make_batch(2)
    x = make_x(2 * N_AGENTS, n_rounds, embed)
    node = make_node(embed, hidden)

    with pytest.raises(AssertionError, match="n_batch"):
        node(x, agent_group=agent_group, batch=batch, n_batch=1)


def test_mismatched_h0_shape_raises():
    n_rounds, embed, hidden = 2, 4, 5
    groups = [0, 0, 0, 1, 1, 1, 1, 1]
    agent_group = make_agent_group([groups], n_rounds)
    batch = make_batch(1)
    x = make_x(N_AGENTS, n_rounds, embed)
    node = make_node(embed, hidden)

    wrong_hidden = th.zeros(1, 1 * N_GROUPS, hidden + 1, dtype=th.float64)
    with pytest.raises(AssertionError, match="h0 must be"):
        node(x, agent_group=agent_group, batch=batch, h0=wrong_hidden)

    wrong_batch = th.zeros(1, (1 * N_GROUPS) + 1, hidden, dtype=th.float64)
    with pytest.raises(AssertionError, match="h0 must be"):
        node(x, agent_group=agent_group, batch=batch, h0=wrong_batch)
