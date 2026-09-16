"""Tests for the decision-round one-hot encoding in `JointExodusHead`.

Plain pytest, no PyG imports -- runs locally on macOS:
    uv run pytest tests/switch/test_joint_exodus_round_onehot.py -q

Context: `src/aimanager/generic/joint_exodus.py::JointExodusHead`,
`notes/autoresearch_log/switch-round-onehot.md`. With both
`round_onehot_slots` and `round_onehot_every` left `None` the head keeps its
original numeric-round behaviour (covered here as the "flag off" baseline);
with both set, the round enters the readout MLP as a one-hot over the
decision rounds instead of the scalar `r / 23`. Helper conventions
(`make_nodes`, `make_head`, `head_inputs`) mirror
`tests/switch/test_joint_exodus.py` so the two suites read as one family.
"""

import pytest
import torch as th

from aimanager.generic.joint_exodus import (
    JointExodusHead,
    MAX_GROUP_SIZE,
    ROUND_NORM,
    SIZE_NORM,
)

SEED = 20260916
N_AGENTS = 8
GRID = MAX_GROUP_SIZE + 1
GROUPS = [0, 0, 0, 1, 1, 1, 1, 1]

# The 24-round game this experiment targets: cadence 4, 5 decision rounds,
# plus round 23 which satisfies the same modulo but is the final round.
N_ROUNDS_GAME = 24
EVERY = 4
SLOTS = 5
DECISION_ROUNDS = [3, 7, 11, 15, 19, 23]  # (r + 1) % 4 == 0


# --------------------------------------------------------------------------- #
# helpers (mirror tests/switch/test_joint_exodus.py)
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


def make_head(embed_size=5, hidden_size=7, seed=SEED, **onehot):
    th.manual_seed(seed)
    return JointExodusHead(
        embed_size=embed_size, hidden_size=hidden_size, **onehot
    ).double()


def head_inputs(groups, n_batch=2, n_rounds=3, round_number=7, embed=5, seed=SEED):
    x, agent_group, batch = make_nodes(n_batch, n_rounds, embed, groups, seed=seed)
    rounds = th.arange(round_number, round_number + n_rounds, dtype=th.int64)
    round_tensor = rounds.reshape(1, n_rounds).expand(x.shape[0], n_rounds)
    return x, agent_group, batch, round_tensor.contiguous()


def single_round_inputs(round_value, groups=GROUPS, n_batch=1, embed=5, seed=SEED):
    """One batch element, one round, so a single round's slot behaviour can
    be probed in isolation."""
    x, agent_group, batch = make_nodes(n_batch, 1, embed, groups, seed=seed)
    round_tensor = th.full((n_batch * len(groups), 1), round_value, dtype=th.int64)
    return x, agent_group, batch, round_tensor


def zero_non_round_inputs(head, embed, n_size_cols=2):
    """Zero the first MLP layer's weight columns for the pooled embeddings
    and the group sizes, and its bias, so the logits become a pure function
    of the round-encoding block alone -- exercising the real `forward` path
    (the `scatter_`-built one-hot feeding the real `Linear`) rather than a
    hand-rolled re-implementation of the encoding."""
    n_pooled_cols = 2 * embed
    with th.no_grad():
        head.mlp[0].weight[:, : n_pooled_cols + n_size_cols] = 0.0
        head.mlp[0].bias[:] = 0.0


# --------------------------------------------------------------------------- #
# 1. flag off is unchanged
# --------------------------------------------------------------------------- #
def test_flag_off_keeps_the_original_numeric_round_path():
    """With both new kwargs at their `None` default, `in_features` is
    unchanged (`2*embed + 3`: two pooled embeddings, two sizes, one scalar
    round), a head built twice from the same seed is bit-identical under a
    fixed `torch.manual_seed` (pins the numeric path's `log_prob` and `k`
    exactly), and two different rounds -- everything else held fixed --
    produce different logits, which is only possible if the round genuinely
    reaches the MLP as the varying scalar `r / 23` rather than being ignored
    or constant-folded."""
    embed = 5
    head_a = make_head(embed_size=embed)
    assert head_a.round_onehot_slots is None
    assert head_a.round_onehot_every is None
    assert head_a.mlp[0].in_features == 2 * embed + 3

    head_b = make_head(embed_size=embed)  # same seed -> same weights
    x, agent_group, batch, round_number = head_inputs(GROUPS, n_batch=1, n_rounds=1)

    th.manual_seed(0)
    log_prob_a, k_a = head_a(
        x, agent_group=agent_group, round_number=round_number, batch=batch
    )
    th.manual_seed(0)
    log_prob_b, k_b = head_b(
        x, agent_group=agent_group, round_number=round_number, batch=batch
    )
    assert th.equal(log_prob_a, log_prob_b)
    assert th.equal(k_a, k_b)

    # same x/agent_group/batch, two different rounds -> different logits
    x2, agent_group2, batch2, round_5 = single_round_inputs(5)
    _, _, _, round_9 = single_round_inputs(9)
    log_prob_5, _ = head_a(
        x2, agent_group=agent_group2, round_number=round_5, batch=batch2
    )
    log_prob_9, _ = head_a(
        x2, agent_group=agent_group2, round_number=round_9, batch=batch2
    )
    assert not th.allclose(log_prob_5, log_prob_9)


# --------------------------------------------------------------------------- #
# 2. the one-hot fires on exactly the decision rounds
# --------------------------------------------------------------------------- #
def test_onehot_block_is_zero_off_decision_rounds_and_live_on_them():
    """Isolates the round-encoding block by zeroing every OTHER input weight
    of the first MLP layer (the pooled embeddings' and the sizes' columns),
    so the full, real forward pass (`mlp` + the masked softmax) becomes a
    pure, invertible function of the round block alone. Under that surgery:
    every non-decision round must produce IDENTICAL `log_prob` (the block is
    all-zero there, so the round is indistinguishable from any other
    non-decision round), while every decision round must differ from that
    shared all-zero baseline (a live one-hot bit reaches the MLP). This
    exercises `forward`'s actual `scatter_` call and `Linear` layer, not a
    hand-rolled index formula."""
    embed = 5
    head = make_head(
        embed_size=embed, round_onehot_slots=SLOTS, round_onehot_every=EVERY
    )
    zero_non_round_inputs(head, embed)

    def log_prob_for(round_value):
        x, agent_group, batch, round_number = single_round_inputs(round_value)
        log_prob, _ = head(
            x, agent_group=agent_group, round_number=round_number, batch=batch
        )
        return log_prob

    non_decision = [r for r in range(N_ROUNDS_GAME) if r not in DECISION_ROUNDS]
    baseline = log_prob_for(non_decision[0])
    for r in non_decision[1:]:
        assert th.allclose(log_prob_for(r), baseline), (
            f"non-decision round {r} must be indistinguishable from "
            f"{non_decision[0]} once only the round block can move the logits"
        )

    for r in DECISION_ROUNDS:
        assert not th.allclose(log_prob_for(r), baseline), (
            f"decision round {r} must differ from the all-zero, "
            "non-decision baseline"
        )


# --------------------------------------------------------------------------- #
# 3. the five realised decision rounds map to five distinct slots
# --------------------------------------------------------------------------- #
def test_decision_rounds_map_to_a_slot_bijection_and_23_shares_19s_slot():
    """Rounds 3, 7, 11, 15, 19 must occupy five distinct slots (a bijection):
    checked by reading the first MLP layer's OUTPUT (a forward hook on the
    real `Linear`, not a re-derivation of the index formula) after zeroing
    every non-round input column, so two rounds landing in the same slot
    produce a bit-identical hidden vector (a pure column lookup with a fixed
    random weight per slot) and two rounds in different slots produce
    generically distinct vectors. Also pins the documented clamp: round 23
    satisfies `(r + 1) % every == 0` too, so its bit fires, but
    `idx = clamp(round // every, 0, slots - 1)` sends `23 // 4 == 5` down to
    slot 4 -- the SAME slot as round 19. Round 23 is never an actual training
    decision round (`generic/data.py` excludes the last round: it has no
    following arrival round to realise a sampled switch against), so this
    collision is inert in practice but is exactly what the clamp produces if
    the head is ever asked to encode it."""
    embed = 5
    head = make_head(
        embed_size=embed, round_onehot_slots=SLOTS, round_onehot_every=EVERY
    )
    zero_non_round_inputs(head, embed)

    captured = {}

    def hook(module, inp, out):
        captured["h"] = out.detach().clone()

    handle = head.mlp[0].register_forward_hook(hook)

    def hidden_for(round_value):
        x, agent_group, batch, round_number = single_round_inputs(round_value)
        head(x, agent_group=agent_group, round_number=round_number, batch=batch)
        return captured["h"]

    try:
        hiddens = {r: hidden_for(r) for r in [3, 7, 11, 15, 19, 23]}
    finally:
        handle.remove()

    slot_reps = {}
    for r in [3, 7, 11, 15, 19]:
        h = hiddens[r]
        for prior_r, rep in slot_reps.items():
            assert not th.allclose(h, rep), (
                f"round {r} collides with round {prior_r}'s slot -- the five "
                "decision rounds must be a bijection onto five distinct slots"
            )
        slot_reps[r] = h

    assert th.allclose(hiddens[23], hiddens[19]), (
        "round 23 must clamp into round 19's slot (slot 4), per "
        "idx = clamp(round // every, 0, slots - 1)"
    )


# --------------------------------------------------------------------------- #
# 4. shape and validation
# --------------------------------------------------------------------------- #
def test_onehot_shapes_and_constructor_validation():
    embed = 5
    head = make_head(
        embed_size=embed, round_onehot_slots=SLOTS, round_onehot_every=EVERY
    )
    assert head.mlp[0].in_features == 2 * embed + 2 + SLOTS

    x, agent_group, batch, round_number = head_inputs(GROUPS, n_batch=2, n_rounds=3)
    log_prob, k = head(
        x, agent_group=agent_group, round_number=round_number, batch=batch
    )
    assert log_prob.shape == (2, 3, GRID, GRID)
    assert k.shape == (2, 3, 2)

    with pytest.raises(AssertionError):
        JointExodusHead(embed_size=embed, hidden_size=7, round_onehot_slots=5)
    with pytest.raises(AssertionError):
        JointExodusHead(
            embed_size=embed,
            hidden_size=7,
            round_onehot_slots=True,
            round_onehot_every=EVERY,
        )
    with pytest.raises(AssertionError):
        JointExodusHead(
            embed_size=embed,
            hidden_size=7,
            round_onehot_slots=0,
            round_onehot_every=EVERY,
        )


# --------------------------------------------------------------------------- #
# 5. legacy heads (pickled before the change) still forward
# --------------------------------------------------------------------------- #
def test_legacy_head_without_the_new_attributes_still_forwards():
    """Simulates a `JointExodusHead` pickled before this change: built with
    the one-hot OFF (so its weights match the numeric-path shape), then has
    both new attributes removed with `delattr` so `forward`'s
    `getattr(self, ..., None)` reads are the only thing standing between it
    and an `AttributeError`. Must forward successfully and reproduce exactly
    what the same head gave before the strip -- the module's behaviour, not
    just an absence of a crash."""
    head = make_head(embed_size=5)
    x, agent_group, batch, round_number = head_inputs(GROUPS, n_batch=1, n_rounds=2)

    log_prob_before, k_before = head(
        x, agent_group=agent_group, round_number=round_number, batch=batch
    )

    assert hasattr(head, "round_onehot_slots")
    assert hasattr(head, "round_onehot_every")
    delattr(head, "round_onehot_slots")
    delattr(head, "round_onehot_every")

    log_prob_after, k_after = head(
        x, agent_group=agent_group, round_number=round_number, batch=batch
    )
    assert th.equal(log_prob_before, log_prob_after)
    assert th.equal(k_before, k_after)


def test_norm_constants_match_the_documented_convention():
    """Guards the constants the numeric path (test 1) and the docstring
    above both lean on, so a future rename of either is caught here too, not
    only by `tests/switch/test_joint_exodus.py`."""
    assert ROUND_NORM == 23.0
    assert SIZE_NORM == float(MAX_GROUP_SIZE)
