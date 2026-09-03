"""Joint exodus head: a round-level distribution over the PAIR of leaver
counts ``(m_0, m_1)`` -- how many players leave group 0 and how many leave
group 1 in one decision round.

The per-agent switch head factorises the round as independent Bernoulli draws
and can therefore only express the correlation of its own conditional means.
The human data carries a residual between-group correlation of -0.4676 that
survives conditioning on both group sizes, the round, and both groups' full
round-level observable state (-0.3038); this stack reproduces -0.1987 of it
pooled over decision rounds (see
``notes/autoresearch_log/switch-joint-exodus-gmlp.md`` note 3). A joint
categorical over the pair is the object that represents exactly that.

Design points that are load-bearing:

* **Pooling is order-canonical in the group LABEL, not in size.** The GNN
  trains on the flip-doubled data, in which every game appears twice with the
  group labels mirrored. Pooling group 0 first and group 1 second makes the
  two copies of a game exact transposes of each other on the ``(m_0, m_1)``
  grid, so the doubling symmetrises the head. Pooling by size instead (small
  group first) would collapse both copies onto the same input and hand the
  head a label-asymmetric target.
* **The mask is by ``agent_group``, never by the edge index.** The graph is
  complete over all 8 agents regardless of membership
  (``GraphNetwork.create_fully_connected`` and ``train.create_fully_connected``
  both emit every ``i != j`` pair within a batch element), so the edge index
  carries no group information at all.
* **``k_g`` is the number of valid DECIDERS in group ``g``**, which is not
  always the group's nominal size: 109 of 2,000 human decision rows fail
  ``switch_valid``. The support of the head is tied to the deciders it is
  fitted on.
* **The pooled embedding is DETACHED, so this head is a pure readout.** No
  gradient of the joint loss reaches the message-passing layers, the RNN or
  the encoders; see ``JointExodusHead.forward`` for why.

**Why group size can enter as a one-hot.** Humans empty a group at a rate
that rises with its size only up to four and then hits a hard floor:
``P(full exodus | k)`` is 0.161 / 0.147 / 0.200 / 0.177 for ``k = 1..4``
(n 31 / 34 / 35 / 96 complete-pair cells) and **0 of 119** cells for
``k >= 5`` -- no human group of five or more ever emptied. A single numeric
``k / 8`` scalar can only bend one smooth curve through that hump and that
floor, which biases the head's size response toward monotone; the stack it
was fitted in empties a singleton group in 46.5% of cells against the human
16.1% (``notes/autoresearch_log/switch-exodus-k-onehot.md`` notes 3-5).
``size_encoding="onehot"`` replaces the two scalars by two nine-wide one-hot
codes over ``k in {0..8}``, i.e. nine free intercept vectors per group label,
so the size dependence is free-form. The default stays ``"numeric"`` so a
head pickled before the option existed keeps its 23-wide MLP and samples
bit-identically.

Torch only -- no ``torch_geometric`` / ``torch_scatter`` -- so this module
imports and is unit-testable on macOS.
"""

import torch as th
from torch.nn import Linear as Lin, Sequential as Seq, Tanh

#: Largest group a round can hold; the count grid is ``0..MAX_GROUP_SIZE``.
MAX_GROUP_SIZE = 8

#: The head is defined for the two-group game.
N_GROUPS = 2

# Normalisation convention. ``IntEncoder(encoding="numeric", n_levels=n)``
# maps an integer v to ``th.linspace(0, 1, n)[v] == v / (n - 1)``
# (``generic/encoder.py``). ``round_number`` is configured with
# ``n_levels: 24``, hence the r / 23 the rest of the model already sees. A
# group size lives in ``0..8``, whose ``n_levels=9`` analogue is k / 8. Both
# scalars therefore land in [0, 1] under the encoder's own rule.
ROUND_NORM = 23.0
SIZE_NORM = float(MAX_GROUP_SIZE)


def pool_by_group(x, agent_group, batch, *, n_batch=None, mask=None, n_groups=N_GROUPS):
    """Mean-pool node embeddings per ``(batch element, round, group label)``.

    Args:
        x: float ``(N, R, F)`` node features, ``N = n_batch * n_player``
            flattened the way ``GraphNetwork.encode`` flattens them.
        agent_group: int ``(N, R)`` group label in ``0..n_groups - 1``. This
            is the only source of membership -- the edge index is complete
            over all players and says nothing about who is in which group.
        batch: int ``(N,)`` graph id of each node, values ``0..n_batch - 1``.
        n_batch: number of graphs; inferred from ``batch`` when omitted.
        mask: optional bool/float ``(N, R)`` selecting the nodes that count
            (the valid deciders). ``None`` counts every node.
        n_groups: number of group labels.

    Returns:
        ``(pooled, counts)`` -- ``pooled`` float ``(n_batch, R, n_groups, F)``
        in LABEL order and ``counts`` float ``(n_batch, R, n_groups)``. A cell
        with no member (an emptied group after a full merge, a real state the
        simulation reaches) gets count 0 and an all-zero pooled vector rather
        than a NaN.
    """
    assert x.dim() == 3, f"x must be (N, R, F), got {tuple(x.shape)}"
    n, n_rounds, n_features = x.shape
    batch = batch.reshape(-1).to(th.int64)
    assert len(batch) == n, f"batch has {len(batch)} entries for {n} nodes"
    if n_batch is None:
        n_batch = int(batch.max().item()) + 1 if n else 0
    agent_group = agent_group.reshape(n, n_rounds).to(th.int64)
    if n:
        assert int(agent_group.min().item()) >= 0, "agent_group must be >= 0"
        assert (
            int(agent_group.max().item()) < n_groups
        ), f"agent_group exceeds n_groups={n_groups}"

    if mask is None:
        weight = th.ones((n, n_rounds), dtype=x.dtype, device=x.device)
    else:
        weight = mask.reshape(n, n_rounds).to(x.dtype)

    # Dense cell id over (batch, round, group). Group is the FASTEST axis, so
    # the two groups of a cell sit next to each other and the reshape below
    # keeps them in label order.
    rounds = th.arange(n_rounds, device=x.device).reshape(1, n_rounds)
    cell = (batch.reshape(n, 1) * n_rounds + rounds) * n_groups + agent_group
    flat = cell.reshape(-1)

    n_cells = n_batch * n_rounds * n_groups
    summed = th.zeros((n_cells, n_features), dtype=x.dtype, device=x.device)
    summed.index_add_(0, flat, (x * weight.unsqueeze(-1)).reshape(-1, n_features))
    counts = th.zeros(n_cells, dtype=x.dtype, device=x.device)
    counts.index_add_(0, flat, weight.reshape(-1))

    pooled = summed / counts.clamp(min=1.0).unsqueeze(-1)
    return (
        pooled.reshape(n_batch, n_rounds, n_groups, n_features),
        counts.reshape(n_batch, n_rounds, n_groups),
    )


def joint_count_mask(k, *, max_group_size=MAX_GROUP_SIZE):
    """Bool ``(..., G + 1, G + 1)``, True where ``m_0 <= k_0`` and
    ``m_1 <= k_1``.

    ``k`` is int ``(..., 2)`` valid-decider counts -- not nominal group sizes;
    a member whose decision is invalid cannot leave and does not widen the
    support. ``(0, 0)`` is valid for every non-negative ``k``, so at least one
    cell always survives -- an entirely masked grid is unreachable, which is
    what keeps the softmax below well defined even when a whole row or column
    is masked out (``k_g == 0`` after a full merge).
    """
    assert k.shape[-1] == 2, f"k must be (..., 2), got {tuple(k.shape)}"
    assert bool((k >= 0).all()), "k must be non-negative"
    assert bool(
        (k <= max_group_size).all()
    ), f"k must not exceed max_group_size={max_group_size}"
    m = th.arange(max_group_size + 1, device=k.device)
    ok_0 = m.reshape(-1, 1) <= k[..., 0, None, None]
    ok_1 = m.reshape(1, -1) <= k[..., 1, None, None]
    return ok_0 & ok_1


def masked_joint_log_prob(logits, k, *, max_group_size=MAX_GROUP_SIZE):
    """Log-probabilities over the padded ``(G + 1) x (G + 1)`` count grid.

    Invalid cells are set to ``-inf`` BEFORE a single ``log_softmax`` over the
    flattened grid, so they carry exactly zero probability (``exp(-inf) == 0``
    bit-for-bit, unlike a large finite penalty) and the surviving cells
    renormalise to one on their own.

    Args:
        logits: float ``(..., (G + 1) ** 2)`` or ``(..., G + 1, G + 1)``.
        k: int ``(..., 2)`` valid-decider counts per group.

    Returns:
        ``(log_prob, valid)`` -- ``log_prob`` float ``(..., G + 1, G + 1)``
        with ``-inf`` on masked cells, ``valid`` the bool mask.
    """
    grid = max_group_size + 1
    if logits.shape[-1] == grid * grid and logits.shape[-2:] != (grid, grid):
        logits = logits.reshape(*logits.shape[:-1], grid, grid)
    assert logits.shape[-2:] == (
        grid,
        grid,
    ), f"logits must end in ({grid}, {grid}), got {tuple(logits.shape)}"
    valid = joint_count_mask(k, max_group_size=max_group_size)
    assert valid.shape[:-2] == logits.shape[:-2], (
        f"k shape {tuple(k.shape)} does not line up with logits "
        f"{tuple(logits.shape)}"
    )
    masked = logits.masked_fill(~valid, float("-inf"))
    log_prob = th.log_softmax(masked.flatten(-2, -1), dim=-1)
    return log_prob.reshape(masked.shape), valid


class JointExodusHead(th.nn.Module):
    """Reads post-RNN node embeddings out into a joint over ``(m_0, m_1)``.

    Input to the readout MLP: the two group-pooled embeddings concatenated in
    LABEL order, both valid-decider counts, and the round as ``r / 23`` -- the
    same numeric-encoder convention the model's own ``round_number`` feature
    already uses (see ``ROUND_NORM``). The counts enter either as two scalars
    ``k / 8`` (``size_encoding="numeric"``, the default) or as two nine-wide
    one-hot codes over ``k in {0..8}`` (``size_encoding="onehot"``, feature
    layout ``[pooled (2F) | onehot(k_0) (9) | onehot(k_1) (9) | round (1)]``).
    No new observable enters the model either way; the head only refactorises
    the label distribution.
    """

    #: Accepted values of ``size_encoding``.
    SIZE_ENCODINGS = ("numeric", "onehot")

    def __init__(
        self,
        embed_size,
        hidden_size,
        *,
        max_group_size=MAX_GROUP_SIZE,
        n_groups=N_GROUPS,
        round_norm=ROUND_NORM,
        size_encoding="numeric",
    ):
        super().__init__()
        assert n_groups == N_GROUPS, "the joint exodus grid is defined for 2 groups"
        assert hidden_size is not None, "the joint exodus head needs a hidden_size"
        assert size_encoding in self.SIZE_ENCODINGS, (
            f"size_encoding must be one of {self.SIZE_ENCODINGS}, "
            f"got {size_encoding!r}"
        )
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.max_group_size = max_group_size
        self.n_groups = n_groups
        self.round_norm = float(round_norm)
        self.size_encoding = size_encoding
        self.grid = max_group_size + 1
        # two normalised size scalars, or one nine-wide one-hot code per group
        size_features = n_groups if size_encoding == "numeric" else n_groups * self.grid
        # two pooled vectors + the size block + the normalised round
        in_features = n_groups * embed_size + size_features + 1
        self.mlp = Seq(
            Lin(in_features=in_features, out_features=hidden_size),
            Tanh(),
            Lin(in_features=hidden_size, out_features=self.grid * self.grid),
        )

    def __setstate__(self, state):
        """Unpickle a head saved before ``size_encoding`` existed as numeric.

        ``th.load`` restores a pickled module's ``__dict__`` without ever
        running ``__init__``, so the heads inside artifacts trained before
        this option existed carry no ``size_encoding`` and a size block of
        two scalars. Defaulting it here keeps those heads on their original
        MLP width and bit-identical in the simulation.
        """
        super().__setstate__(state)
        self.__dict__.setdefault("size_encoding", "numeric")

    def forward(
        self,
        x,
        *,
        agent_group,
        round_number,
        batch,
        n_batch=None,
        decider_mask=None,
    ):
        """Args mirror what ``GraphNetwork.forward`` has in hand post-RNN.

        Args:
            x: float ``(N, R, F)`` post-RNN node embeddings.
            agent_group: int ``(N, R)`` group label; the sole membership
                signal, since the graph is complete over all players.
            round_number: int ``(N, R)`` round index (constant within a graph).
            batch: int ``(N,)`` graph id per node.
            decider_mask: optional bool ``(N, R)`` marking valid deciders, so
                that ``k`` counts deciders rather than members.
            n_batch: number of graphs; inferred when omitted.

        Returns:
            ``(log_prob, k)`` -- ``log_prob`` float
            ``(n_batch, R, grid, grid)`` with ``-inf`` on ``m_g > k_g``, and
            ``k`` int ``(n_batch, R, 2)`` valid-decider counts per group.
        """
        pooled, counts = pool_by_group(
            x,
            agent_group,
            batch,
            n_batch=n_batch,
            mask=decider_mask,
            n_groups=self.n_groups,
        )
        # ---------------------------------------------------------------- #
        # THE CUT. The trunk is optimised by the per-agent loss ALONE; this
        # head is fitted as a readout on top of it.
        #
        # Why: the joint term sits at ~2-3 nats against the per-agent term's
        # ~0.5, so attached it dominates the shared trunk's gradient by sheer
        # magnitude, and the cost lands on the per-agent switch model itself.
        # SB is this experiment's declared watch item and the 21-row mean is
        # gate 2, so degrading the per-agent model to buy a better joint fit
        # is the trade this experiment cannot make. Detaching removes it by
        # construction rather than by tuning: the candidate is the base
        # model's trunk plus a joint head, and any score movement is
        # attributable to the mechanism, not to a re-fitted representation.
        #
        # ONLY the embedding is cut. `k` below is an integer count and the
        # round is an integer feature, so neither ever carried gradient, and
        # everything downstream -- this module's whole MLP -- still receives
        # full gradient and trains normally.
        # ---------------------------------------------------------------- #
        pooled = pooled.detach()

        n_batch, n_rounds = pooled.shape[0], pooled.shape[1]
        k = counts.round().to(th.int64)
        # Either encoding of `k` is a function of an integer count and so
        # carries no gradient at all -- the cut above is the same cut.
        if self.size_encoding == "onehot":
            # (..., 2, grid) -> (..., 2 * grid), group 0's codes first, the
            # same LABEL order the pooled block above is concatenated in.
            sizes = th.nn.functional.one_hot(k, self.grid)
            sizes = sizes.flatten(-2, -1).to(x.dtype)
        else:
            sizes = k.to(x.dtype) / SIZE_NORM

        # The round is constant across the agents of a graph, so mean-pooling
        # it over the single group label 0 recovers it exactly, and reuses the
        # scatter above rather than assuming any particular node ordering.
        rounds, _ = pool_by_group(
            round_number.reshape(x.shape[0], n_rounds, 1).to(x.dtype),
            th.zeros_like(round_number),
            batch,
            n_batch=n_batch,
            n_groups=1,
        )
        rounds = rounds.reshape(n_batch, n_rounds, 1) / self.round_norm

        features = th.cat([pooled.flatten(-2, -1), sizes, rounds], dim=-1)
        logits = self.mlp(features)
        log_prob, _ = masked_joint_log_prob(
            logits, k, max_group_size=self.max_group_size
        )
        return log_prob, k
