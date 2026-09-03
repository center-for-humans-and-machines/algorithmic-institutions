"""Conditional-Bernoulli sampler: given per-agent switch probabilities and a
group leaver COUNT ``m``, choose WHICH ``m`` of the group's ``k`` members
leave.

This is the second half of the joint-exodus mechanism
(``notes/autoresearch_log/switch-joint-exodus-gmlp.md``, plan step 3). The
round head (``generic/joint_exodus.py``) decides the pair ``(m_0, m_1)`` of
how many leave each group; this module decides which ones, conditioning on
the existing per-agent head's probabilities ``p_i`` so that the member the
model thinks is most likely to leave is also the one most likely to be
picked. The group-level arrangement comes from the joint head, the selection
inside it stays with the trained per-agent propensities.

The target distribution is the Bernoulli vector conditioned on its sum::

    P(S | sum = m) is proportional to  prod_{i in S} p_i / (1 - p_i)

i.e. a member's presence in the leaving set is weighted by its own odds,
independent of every other member's odds -- the standard "conditional
Bernoulli" / fixed-margin sampling distribution. A group has at most
``MAX_GROUP_SIZE == 8`` members, so the number of size-``m`` subsets is at
most ``C(8, 4) == 70``: cheap enough to enumerate every subset exactly rather
than approximate with a sequential heuristic.

The degenerate cell is worth naming. When ``m == k`` there is exactly ONE
subset of the right size, so no selection happens and the per-agent
propensities are bypassed entirely; the same is true of ``m == 0`` (nobody
leaves) and ``k == 0`` (nobody can). That is a property of conditioning on
the sum, not an implementation shortcut: the protection this module offers is
confined to partial cells, and how much of a round's mass sits in full-exodus
cells is a property of the joint head that feeds it.

Ragged-batch representation. A round's two groups (and, across many episodes
and rounds, many such calls) have differing ``k`` and ``m``. This module
fixes every row's WIDTH at ``MAX_GROUP_SIZE`` and marks which of those 8
slots are real members with a boolean ``mask`` (``None`` meaning "all 8 are
real"), mirroring the ``mask`` convention ``joint_exodus.pool_by_group``
already uses for valid deciders. A row's ``k`` is simply ``mask.sum(-1)`` and
can differ freely from its neighbours; a fully merged / fully emptied group
(``k == 0``) is just an all-``False`` row, not a separate code path. This
keeps every call a single batched tensor op with no Python loop over groups,
and it composes directly with ``joint_exodus.pool_by_group``'s own per-group
masking.

Numerical care. ``p_i`` can sit at ``1e-9`` or ``1 - 1e-9``, where the raw
odds ``p / (1 - p)`` over- or under-flow long before ``log(p) - log1p(-p)``
does. Every subset weight is therefore accumulated as a SUM of log-odds
(never a product of odds), and the normalising sum over subsets is a single
``log_softmax`` -- a numerically stable log-sum-exp -- over the (masked) grid
of ``2 ** MAX_GROUP_SIZE`` subset codes, exactly the ``masked_joint_log_prob``
pattern ``joint_exodus.py`` already uses for its own grid.

Torch only -- no ``torch_geometric`` / ``torch_scatter`` -- so this module
imports and is unit-testable on macOS, mirroring ``generic/joint_exodus.py``.
"""

import torch as th

#: The largest group the two-group game reaches (one group holds everyone).
#: Mirrors ``joint_exodus.MAX_GROUP_SIZE``; kept as an independent constant
#: here so this module has no import-time dependency on ``joint_exodus``.
MAX_GROUP_SIZE = 8

#: Clamp bound for probabilities before taking a log. Far outside the ``1e-9``
#: / ``1 - 1e-9`` range this module is asked to tolerate, so it never softens
#: an intentionally extreme input -- it only keeps an accidental exact 0 or 1
#: from producing a literal ``-inf`` (at ``p == 0``) or ``+inf`` (at
#: ``p == 1``). Applied as a two-sided clamp, so both ends are protected.
_EPS = 1e-12


def _enumerate_subsets(width, device):
    """All ``2 ** width`` subsets of ``width`` slots.

    Returns ``(bits, popcount)``: ``bits`` bool ``(2 ** width, width)``, LSB
    of the subset code first, and ``popcount`` int64 ``(2 ** width,)``, the
    number of members each subset contains. Built fresh on every call --
    ``width`` is at most ``MAX_GROUP_SIZE`` (256 rows), which is cheap enough
    that caching would only add bookkeeping (device/dtype keys) for no
    measurable benefit.
    """
    assert 0 <= width <= 20, (
        f"enumerating 2 ** {width} subsets is not what this sampler is for "
        "(it exists because a group is at most MAX_GROUP_SIZE == 8 members)"
    )
    codes = th.arange(1 << width, device=device)
    shifts = th.arange(width, device=device)
    bits = ((codes.unsqueeze(-1) >> shifts) & 1).to(th.bool)
    return bits, bits.sum(-1).to(th.int64)


def conditional_bernoulli_log_prob(p, m, *, mask=None, max_group_size=MAX_GROUP_SIZE):
    """Log-probability of every subset under ``P(S | sum = m) prop to`` the
    product of the included members' odds.

    Args:
        p: float ``(B, K)``, ``K == max_group_size``. Row ``b``'s per-member
            switch probability; entries at ``mask[b] == False`` are never
            read for their value (see the padding note below).
        m: int ``(B,)``, the required subset size per row.
        mask: optional bool ``(B, K)`` marking the ``k`` real members of each
            row (``None`` means all ``K`` slots are real, i.e. ``k == K``
            for every row). This is how differing group sizes -- including
            ``k == 0``, a fully merged round's emptied group -- are packed
            into one batched call.
        max_group_size: ``K``, asserted to match ``p``'s last dimension.

    Returns:
        ``(log_prob, bits, valid)``:

        * ``log_prob`` float64 ``(B, 2 ** K)``, one row per input row, one
          column per subset CODE (``bits[code]`` is that subset's membership
          vector). ``-inf`` on every subset that either has the wrong size or
          touches a padded slot.
        * ``bits`` bool ``(2 ** K, K)``, shared across the whole batch.
        * ``valid`` bool ``(B, 2 ** K)``, the mask ``log_prob`` was built
          from -- ``True`` iff the subset's size is ``m[b]`` AND it only
          contains slots with ``mask[b] == True``. At least one column is
          always ``True`` per row (``m[b] <= k[b]`` guarantees ``C(k, m) >=
          1``), so ``log_softmax`` below is always over a non-empty support.
    """
    assert p.dim() == 2, f"p must be (B, K), got {tuple(p.shape)}"
    b, width = p.shape
    assert (
        width == max_group_size
    ), f"p's last dim must equal max_group_size={max_group_size}, got {width}"
    m = m.reshape(-1).to(th.int64)
    assert len(m) == b, f"m has {len(m)} entries for {b} rows of p"

    if mask is None:
        mask = th.ones((b, width), dtype=th.bool, device=p.device)
    else:
        mask = mask.reshape(b, width).to(th.bool)
    k = mask.sum(-1)

    assert bool((m >= 0).all()), "m must be non-negative"
    assert bool((m <= k).all()), "m must not exceed a row's number of real members"

    # Padded slots get a NEUTRAL, finite log-odds of exactly 0 (p = 0.5)
    # rather than being driven to +-inf, because the subset-weight sum below
    # is a plain matmul against a 0/1 membership vector: a slot the subset
    # excludes contributes `0 * logit`, and `0 * -inf` is NaN in IEEE
    # arithmetic, not 0. Giving padded slots a finite value sidesteps that
    # trap entirely -- and it is silent when it goes wrong, since a batch
    # only NaNs on the rows that happen to carry padding. Correctness is
    # unaffected because `valid` below throws out every subset that would
    # have included one of them anyway.
    p64 = p.reshape(b, width).to(th.float64)
    p_safe = th.where(mask, p64.clamp(_EPS, 1.0 - _EPS), th.full_like(p64, 0.5))
    logit = th.log(p_safe) - th.log1p(-p_safe)  # log(p / (1 - p)), stable near 0/1

    bits, popcount = _enumerate_subsets(width, p.device)
    bits64 = bits.to(th.float64)

    weight_log = logit @ bits64.T  # (B, 2**K): sum of log-odds in each subset
    touches_padding = (~mask).to(th.float64) @ bits64.T  # > 0 iff S uses a padded slot
    right_size = popcount.reshape(1, -1) == m.reshape(-1, 1)
    valid = (touches_padding == 0.0) & right_size
    assert bool(valid.any(-1).all()), (
        "no subset of the requested size respects a row's mask -- this can "
        "only happen if m > k slipped past the assert above"
    )

    masked = weight_log.masked_fill(~valid, float("-inf"))
    log_prob = th.log_softmax(masked, dim=-1)
    return log_prob, bits, valid


def sample_conditional_bernoulli(p, m, *, mask=None, max_group_size=MAX_GROUP_SIZE):
    """Draw one leaving subset per row from ``conditional_bernoulli_log_prob``.

    Args:
        p, m, mask, max_group_size: see ``conditional_bernoulli_log_prob``.

    Returns:
        bool ``(B, K)``, ``True`` at the members chosen to leave. Always
        exactly ``m[b]`` true entries in row ``b``, and never ``True`` at a
        padded (``mask[b] == False``) slot.

    RNG contract: exactly one ``th.multinomial`` call over the whole batch,
    drawing from torch's global RNG (governed by ``th.manual_seed``, no
    generator argument) -- the same convention ``generic/encoder.py`` uses
    for the legacy independent switch draw (``th.multinomial(arr, 1)``). One
    call for the batch rather than one per row is not only cheaper: it makes
    the number of RNG draws a round consumes exactly predictable, which is
    what lets a run with the joint head switched off stay bitwise identical
    to the base model. This sampler carries no state between calls -- each
    group and round is independent once ``m`` is known -- so a single
    categorical draw over the enumerated subsets is both simplest and exact.
    """
    log_prob, bits, _ = conditional_bernoulli_log_prob(
        p, m, mask=mask, max_group_size=max_group_size
    )
    idx = th.multinomial(log_prob.exp(), 1).reshape(-1)  # draw 1/1
    return bits[idx]
