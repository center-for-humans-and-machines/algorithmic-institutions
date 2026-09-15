"""Per-group virtual node: a learned, persistent state for each GROUP.

The trunk this attaches to has no object that *is* a group. ``op1`` pools its
seven incoming edge messages with a group-blind ``scatter_mean``, the edge
index is complete over all 8 agents regardless of membership
(``GraphNetwork.create_fully_connected`` and ``train.create_fully_connected``
both emit every ``i != j`` pair), and ``agent_group`` enters only as a onehot
on the node -- shuffling it costs the frontier contributor nothing in held-out
log-loss. On the human data the peer response is *entirely* own-group-specific
(own-group mean weight 0.278 against a group-blind seven-peer mean of -0.0005
once both enter), so the missing piece is a group the readout can read.

This module supplies exactly that: the virtual node of Gilmer et al. (2017) /
OGB, but one per **group** rather than one per graph, and **recurrent** rather
than recomputed. Each round the members of each group are mean-pooled into a
group input, a GRU whose weights are shared across groups turns the sequence
of those inputs into a persistent group state, and that state is broadcast
back to the group's own members.

Design points that are load-bearing:

* **Pooling happens post-``op1``, injection at the ``op2`` readout.** The node
  reads the message-passing embeddings, and its state is concatenated to each
  member's post-``rnn_n`` embedding just before the readout. The per-agent path
  ``op1 -> rnn_n`` is therefore untouched, and the node is trained *attached*:
  the per-agent loss flows back through the pooling into the trunk. This is a
  trunk change, not a readout head, so the ``JointExodusHead`` detach does not
  apply here -- there is no second loss whose magnitude could swamp the
  per-agent one.
* **Membership is time-varying, and that is the point.** ``agent_group`` is
  read at round ``r`` when the state is broadcast back, never at a fixed round,
  so an agent who switches reads the *arrival* group's state from the switch
  round on (``apply_switch`` runs before ``update_contribution``). Human
  arrivals weight the receiving group's recent history at 0.280; a newcomer
  picking up a state that already encodes what that group has been is the
  mechanism behind that claim.
* **Occupancy is part of the input.** ``k / 8`` is concatenated to the pooled
  vector, on the same normalisation convention the numeric ``IntEncoder``
  already gives the rest of the model (``joint_exodus.SIZE_NORM``): a mean over
  two members and a mean over six are different states of the world, and the
  mean alone cannot tell them apart.
* **An emptied group is a real state, not an error.** Agents can all switch
  away, leaving a group with no members. ``pool_by_group`` gives that cell an
  all-zero vector and count 0 rather than a NaN, its GRU keeps stepping on that
  zero input, and no member reads the resulting state because no member is in
  it. Finiteness is asserted rather than hoped for.
* **The group index is the FASTEST axis of the GRU batch**, so a group's row is
  ``batch_element * n_groups + group`` and the hidden layout
  ``(1, n_batch * n_groups, H)`` does not depend on the number of rounds fed.
  That is the train/sim contract: training feeds all 24 rounds in one call,
  while the simulation calls the model once per round with ``n_rounds = 1``
  carrying the hidden state, and the two must agree.
* **Every member counts.** Pooling takes no validity mask -- membership is not
  validity (PR #173 step 1's ruling); a group's state is what its members are,
  including the ones whose own decision that round is invalid.

``pool_by_group`` is reused unchanged from ``joint_exodus``. Torch only -- no
``torch_geometric`` / ``torch_scatter`` -- so this module imports and is
unit-testable on macOS, mirroring ``generic/joint_exodus.py``.
"""

import torch as th
from torch.nn import GRU

from aimanager.generic.joint_exodus import N_GROUPS, SIZE_NORM, pool_by_group


class GroupVirtualNode(th.nn.Module):
    """A recurrent state per ``(batch element, group label)``.

    Args:
        embed_size: width ``F`` of the node embeddings that are pooled (the
            post-``op1`` node width).
        hidden_size: width ``H`` of the group state, i.e. of what each member
            receives at the readout.
        n_groups: number of group labels; 2 for this game.
        size_norm: divisor for the occupancy count, 8 by the encoder's own
            ``k / (n_levels - 1)`` convention for a size in ``0..8``.

    The GRU's weights are shared across the groups and across the batch: there
    is one group *mechanism*, instantiated once per ``(batch element, group)``
    cell by its own hidden state. Two groups in the same episode therefore
    differ only through what has happened in them, which is what makes the two
    cultures CG measures emergent rather than parameterised.
    """

    # `size_norm` defaults to joint_exodus.SIZE_NORM, which is 8.0: the same
    # k / 8 the joint head already uses, so the two group-level modules read
    # occupancy on one convention.
    def __init__(
        self, embed_size, hidden_size, *, n_groups=N_GROUPS, size_norm=SIZE_NORM
    ):
        super().__init__()
        assert (
            isinstance(embed_size, int) and not isinstance(embed_size, bool)
        ) and embed_size > 0, f"embed_size must be a positive int, got {embed_size!r}"
        assert (
            isinstance(hidden_size, int) and not isinstance(hidden_size, bool)
        ) and hidden_size > 0, f"hidden_size must be a positive int, {hidden_size!r}"
        assert (
            isinstance(n_groups, int) and not isinstance(n_groups, bool)
        ) and n_groups > 0, f"n_groups must be a positive int, got {n_groups!r}"
        assert float(size_norm) > 0, f"size_norm must be positive, got {size_norm!r}"
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_groups = n_groups
        self.size_norm = float(size_norm)
        # +1 for the occupancy scalar appended to the pooled embedding.
        self.gru = GRU(
            input_size=embed_size + 1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x, *, agent_group, batch, h0=None, n_batch=None):
        """Step every group's state over ``R`` rounds and hand it to members.

        Args:
            x: float ``(N, R, F)`` post-``op1`` node embeddings, with
                ``N = n_batch * n_player`` flattened the way
                ``GraphNetwork.encode`` flattens them (batch-major, player
                fastest).
            agent_group: int ``(N, R)`` group label in ``0..n_groups - 1``,
                read at each round -- time-varying by design.
            batch: int ``(N,)`` graph id of each node, values
                ``0..n_batch - 1``.
            h0: optional float ``(1, n_batch * n_groups, H)`` carried hidden
                state. ``None`` starts every group from zeros, which is what
                ``reset_rnn`` means for the per-agent RNNs.
            n_batch: number of graphs. Inferred as ``batch.max() + 1`` when
                omitted -- correct because ``encode`` emits a contiguous
                ``0..n_batch - 1`` with every graph present. Pass it
                explicitly when a caller could hold a batch whose last graph
                has no nodes; it is asserted consistent with ``batch`` either
                way, because an ``n_batch`` that drifts between per-round
                calls would silently re-map every group's hidden row.

        Returns:
            ``(g_node, h)`` -- ``g_node`` float ``(N, R, H)``, the state of the
            group node ``n`` belongs to at round ``r``, and ``h`` float
            ``(1, n_batch * n_groups, H)``, the final hidden state to carry
            into the next round's call.
        """
        assert x.dim() == 3, f"x must be (N, R, F), got {tuple(x.shape)}"
        n, n_rounds, n_features = x.shape
        assert n_features == self.embed_size, (
            f"x has {n_features} features, this node was built for "
            f"{self.embed_size}"
        )
        batch = batch.reshape(-1).to(th.int64)
        assert len(batch) == n, f"batch has {len(batch)} entries for {n} nodes"
        assert (
            agent_group.numel() == n * n_rounds
        ), f"agent_group has {agent_group.numel()} entries for {n} x {n_rounds}"
        agent_group = agent_group.reshape(n, n_rounds).to(th.int64)

        inferred = int(batch.max().item()) + 1 if n else 0
        if n_batch is None:
            n_batch = inferred
        else:
            assert (
                isinstance(n_batch, int) and not isinstance(n_batch, bool)
            ) and n_batch >= inferred, (
                f"n_batch={n_batch!r} is inconsistent with batch, whose "
                f"largest graph id implies at least {inferred}"
            )

        pooled, counts = pool_by_group(
            x, agent_group, batch, n_batch=n_batch, n_groups=self.n_groups
        )
        # Membership is not validity: no mask, so every member of a cell is in
        # its mean and the counts are the membership counts exactly.
        assert (
            int(counts.sum().item()) == n * n_rounds
        ), "pooled counts do not account for every node-round exactly once"

        # (n_batch, R, G, F) + (n_batch, R, G, 1) -> (n_batch, R, G, F + 1).
        seq = th.cat([pooled, (counts / self.size_norm).unsqueeze(-1)], dim=-1)
        # Group FASTEST: row b * G + g, so the hidden layout below is
        # (1, n_batch * G, H) whatever R is, and 24 single-round calls that
        # carry `h` reproduce one 24-round call.
        seq = seq.permute(0, 2, 1, 3).reshape(
            n_batch * self.n_groups, n_rounds, n_features + 1
        )
        # An emptied group is a zero vector with occupancy 0, never a NaN --
        # pool_by_group divides by counts.clamp(min=1). Checked, not assumed.
        assert bool(th.isfinite(seq).all()), "the group input is not finite"

        if h0 is not None:
            assert h0.shape == (
                1,
                n_batch * self.n_groups,
                self.hidden_size,
            ), (
                f"h0 must be (1, {n_batch * self.n_groups}, "
                f"{self.hidden_size}), got {tuple(h0.shape)}"
            )
        g, h = self.gru(seq, h0)

        # g_node[n, r] = g[batch[n] * G + agent_group[n, r], r]. The label is
        # taken at round r, so a switcher picks up the ARRIVAL group's state
        # from the switch round on.
        cell = batch.reshape(n, 1) * self.n_groups + agent_group
        g_node = th.gather(
            g, 0, cell.unsqueeze(-1).expand(n, n_rounds, self.hidden_size)
        )
        return g_node, h
