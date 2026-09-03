"""Tests for wiring the joint exodus draw into the simulation (plan step 4).

Runs locally on macOS with plain pytest:

    PYTHONPATH=$PWD/src uv run pytest tests/switch/test_joint_exodus_sampling.py -q

Stand-ins for `torch_scatter` / `torch_geometric.nn` are installed only when the
real packages are missing -- the same discipline as
tests/switch/test_joint_exodus_graph.py, test_joint_exodus_loss.py and
test_joint_exodus_detach.py -- so on Raven this file exercises the real PyG.
Nothing asserted here depends on the message-passing implementation: every
claim is about WHICH categorical draws are made, how many, and how their
outcomes are spent on the two groups. The trunk only supplies the numbers.

What is under test (notes/autoresearch_log/switch-joint-exodus.md, step 4):

* head OFF -- `predict_independent(sample=True)` is bitwise identical to the
  pre-change expression, values AND global RNG state. This is the licence for
  step 7's baseline control, which re-runs the parent's own config and
  requires a bit-identical `per_round.parquet`;
* a NON-decision round with the head ON consumes exactly the same RNG as the
  independent path, because the switch predictor runs every round but its
  output is only used on decision rounds (`manager/environment.py: step`);
* a DECISION round draws a pair (m_0, m_1) and then leaves exactly m_g members
  of group g -- of GROUP g, not of the other one;
* a fully merged round (k = 8, k = 0) is an ordinary cell, not a crash.
"""

import importlib
import os
import sys
import tempfile
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
from aimanager.generic.joint_exodus import MAX_GROUP_SIZE  # noqa: E402

SEED = 20260902
N_AGENTS = 8
GROUPS = [0, 0, 0, 1, 1, 1, 1, 1]
GRID = MAX_GROUP_SIZE + 1
SWITCH_EVERY = 4
# With switch_every 4 the environment takes its decision at round_number 3, 7,
# 11, 15, 19 -- `(r + 1) % 4 == 0`. Rounds 0-2 are held-warm rounds whose
# output is discarded.
DECISION_ROUND = 3
NON_DECISION_ROUND = 2


def make_model(seed=0, **kwargs):
    """Small switch-shaped model, matching test_joint_exodus_graph.py."""
    th.manual_seed(seed)
    model = GraphNetwork(
        y_levels=2,
        y_name="does_switch",
        hidden_size=4,
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
    """`predict_independent(sample=True)` as it was BEFORE step 4, copied
    verbatim, so the head-off identity check does not lean on the code under
    test -- the same device test_joint_exodus_detach.py uses for gradients."""
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


def run_seeded(fn, seed=SEED):
    """Run `fn` from a fixed RNG state and report where it left the RNG."""
    th.manual_seed(seed)
    out = fn()
    return out, th.randn(5)


class CountingMultinomial:
    """Context manager counting `th.multinomial` calls anywhere in the stack
    (`encoder.py`'s decode, the pair draw, the conditional-Bernoulli draw),
    so "how many draws does a round cost" is asserted rather than argued."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        self._real = th.multinomial

        def counting(input, num_samples, *a, **kw):
            self.calls.append(tuple(input.shape))
            return self._real(input, num_samples, *a, **kw)

        th.multinomial = counting
        return self

    def __exit__(self, *exc):
        th.multinomial = self._real
        return False


class PointMassHead(th.nn.Module):
    """Wraps the real head and replaces its joint with a point mass on a
    chosen `(m_0, m_1)`, leaving the valid-decider count `k` untouched.

    The head is a freshly initialised MLP, so on its own it cannot be made to
    emit a chosen pair; but the claim under test is about what the SAMPLER
    does with a pair, not about what the head predicts. Pinning the pair turns
    "the right members left" from a distributional statement into an exact
    one.
    """

    def __init__(self, real, m_0, m_1):
        super().__init__()
        self.real = real
        self.m_0 = m_0
        self.m_1 = m_1
        self.max_group_size = real.max_group_size
        self.n_groups = real.n_groups

    def forward(self, *args, **kwargs):
        log_prob, k = self.real(*args, **kwargs)
        assert bool(
            (k[..., 0] >= self.m_0).all() and (k[..., 1] >= self.m_1).all()
        ), "the pinned pair is not feasible for this membership"
        out = th.full_like(log_prob, float("-inf"))
        out[..., self.m_0, self.m_1] = 0.0
        return out, k


def pin_pair(model, m_0, m_1):
    model.joint_exodus_head = PointMassHead(model.joint_exodus_head, m_0, m_1)
    return model


# --------------------------------------------------------------------------- #
# 1. head OFF -- the pre-change path, bit for bit, including the RNG
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("round_number", [NON_DECISION_ROUND, DECISION_ROUND])
def test_head_off_sampling_is_bitwise_identical_including_rng(round_number):
    """Step 7's baseline control re-runs the parent's own config and requires a
    bit-identical per_round.parquet. The parent's switch artifact carries no
    `joint_exodus` field, so it loads with the head absent -- and then this
    change must be invisible, on decision rounds as much as anywhere else."""
    model = make_model(seed=7)
    data = make_data(round_number=round_number)
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


def test_head_off_costs_exactly_one_categorical_draw():
    model = make_model(seed=7)
    data = make_data(round_number=DECISION_ROUND)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=2)
    th.manual_seed(SEED)
    with CountingMultinomial() as counter:
        model.predict_independent(data, sample=True, edge_index=edge_index)
    assert counter.calls == [(2 * N_AGENTS, 2)]


def test_a_legacy_artifact_has_no_switch_every_and_still_loads():
    """An artifact saved before step 4 carries none of the three joint keys."""
    model = make_model(seed=7)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        saved = th.load(path, map_location="cpu")
        assert saved["joint_exodus_switch_every"] is None
        for key in ("joint_exodus", "joint_exodus_head", "joint_exodus_switch_every"):
            del saved[key]
        legacy_path = os.path.join(d, "legacy.pt")
        th.save(saved, legacy_path)
        legacy = GraphNetwork.load(legacy_path, device="cpu")
    assert legacy.joint_exodus is False
    assert legacy.joint_exodus_head is None
    assert legacy.joint_exodus_switch_every is None


# --------------------------------------------------------------------------- #
# 2. head ON, NON-decision round -- still no extra RNG
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("round_number", [0, 1, 2, 4, 5, 6])
def test_non_decision_round_consumes_no_extra_rng(round_number):
    """The predictor is called every round to keep the GRU warm and its output
    is thrown away off decision rounds. If those calls consumed extra RNG,
    every subsequent draw in the simulation -- contributions included -- would
    shift, and the run would no longer isolate the mechanism."""
    model = make_model(seed=11, joint_exodus=True, joint_exodus_switch_every=4)
    data = make_data(round_number=round_number)
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


def test_non_decision_round_costs_exactly_one_categorical_draw():
    model = make_model(seed=11, joint_exodus=True, joint_exodus_switch_every=4)
    data = make_data(round_number=NON_DECISION_ROUND)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=2)
    th.manual_seed(SEED)
    with CountingMultinomial() as counter:
        model.predict_independent(data, sample=True, edge_index=edge_index)
    assert counter.calls == [(2 * N_AGENTS, 2)]


def test_decision_round_costs_three_categorical_draws():
    """The per-agent draw (kept verbatim so the neutral path IS the old
    expression, then discarded here), the pair draw over the 9 x 9 grid, and
    one batched conditional-Bernoulli draw covering BOTH groups."""
    model = make_model(seed=11, joint_exodus=True, joint_exodus_switch_every=4)
    data = make_data(round_number=DECISION_ROUND)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=2)
    th.manual_seed(SEED)
    with CountingMultinomial() as counter:
        model.predict_independent(data, sample=True, edge_index=edge_index)
    assert counter.calls == [
        (2 * N_AGENTS, 2),  # per-agent, discarded
        (2, GRID * GRID),  # (m_0, m_1) per episode
        (2 * 2, 1 << N_AGENTS),  # both groups of both episodes, one call
    ]


# --------------------------------------------------------------------------- #
# 3. head ON, decision round -- the counts are exactly the drawn pair
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("groups", [GROUPS, [0, 1, 0, 1, 0, 1, 0, 1], [1] + [0] * 7])
def test_switcher_counts_equal_the_drawn_pair(groups):
    """Replay the RNG to recover the pair the sampler actually drew, then check
    the returned decision against it. No distributional assumption: this is an
    exact identity for whatever pair came out."""
    model = make_model(seed=13, joint_exodus=True, joint_exodus_switch_every=4)
    data = make_data(n_batch=3, round_number=DECISION_ROUND, groups=groups)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=3)

    th.manual_seed(SEED)
    pred, _ = model.predict_independent(data, sample=True, edge_index=edge_index)

    # Same seed, same forward (eval, no RNG), same first draw, then the pair.
    th.manual_seed(SEED)
    encoded = model.encode(
        data, y_encode=False, edge_index=edge_index, device=model.device
    )
    model.eval()
    logit, (log_prob, k) = model(encoded, True, True)
    proba = th.nn.functional.softmax(logit, dim=-1)
    model.y_encoder.decode(proba, True)  # the discarded per-agent draw
    cell = th.multinomial(log_prob[:, 0].exp().flatten(-2, -1), 1).reshape(-1)
    m_0, m_1 = th.div(cell, GRID, rounding_mode="floor"), cell % GRID

    group_r = data["agent_group"][:, :, 0]
    switched = pred[:, :, 0].to(th.bool)
    assert th.equal((switched & (group_r == 0)).sum(-1), m_0)
    assert th.equal((switched & (group_r == 1)).sum(-1), m_1)
    # and the pair the head offered was feasible for the membership it saw
    assert th.equal(k[:, 0, 0], (group_r == 0).sum(-1))
    assert th.equal(k[:, 0, 1], (group_r == 1).sum(-1))
    assert bool((m_0 <= k[:, 0, 0]).all() and (m_1 <= k[:, 0, 1]).all())


@pytest.mark.parametrize(
    "groups,m_0,m_1,expected",
    [
        # one member in group 0; the pair says exactly that member leaves
        ([0, 1, 1, 1, 1, 1, 1, 1], 1, 0, [1, 0, 0, 0, 0, 0, 0, 0]),
        # mirrored LABELS at the same positions: now agent 0 must stay
        ([1, 0, 0, 0, 0, 0, 0, 0], 1, 0, None),
        # the whole minority group empties while the majority holds -- the
        # collective exodus this head exists to reproduce
        ([0, 0, 0, 1, 1, 1, 1, 1], 3, 0, [1, 1, 1, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 1, 1, 1, 1, 1], 0, 5, [0, 0, 0, 1, 1, 1, 1, 1]),
    ],
)
def test_the_leavers_are_members_of_the_group_the_count_was_drawn_for(
    groups, m_0, m_1, expected
):
    """Pin the pair, then check WHICH agents left. Transposing the two counts
    -- the silent inversion this step is most exposed to -- fails every row
    below, including the mirrored-label row where the positions are identical
    and only the labels move."""
    model = make_model(seed=17, joint_exodus=True, joint_exodus_switch_every=4)
    pin_pair(model, m_0, m_1)
    data = make_data(n_batch=2, round_number=DECISION_ROUND, groups=groups)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=2)

    th.manual_seed(SEED)
    pred, _ = model.predict_independent(data, sample=True, edge_index=edge_index)
    switched = pred[:, :, 0].to(th.bool)
    group_r = data["agent_group"][:, :, 0]

    assert th.equal((switched & (group_r == 0)).sum(-1), th.full((2,), m_0))
    assert th.equal((switched & (group_r == 1)).sum(-1), th.full((2,), m_1))
    if expected is not None:
        want = th.tensor(expected, dtype=th.bool).expand(2, N_AGENTS)
        assert th.equal(switched, want)
    else:
        # group 0 is agents 1..7 here, so the single leaver is one of those
        assert th.equal(switched.sum(-1), th.full((2,), m_0 + m_1))
        assert not bool(switched[:, 0].any())


def test_no_switchers_when_the_pair_is_zero_zero():
    """(0, 0) is always a valid cell -- the round where nobody moves."""
    model = make_model(seed=19, joint_exodus=True, joint_exodus_switch_every=4)
    pin_pair(model, 0, 0)
    data = make_data(n_batch=2, round_number=DECISION_ROUND)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=2)
    th.manual_seed(SEED)
    pred, _ = model.predict_independent(data, sample=True, edge_index=edge_index)
    assert not bool(pred.any())


# --------------------------------------------------------------------------- #
# 4. the fully merged round -- k = 8 and k = 0
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "groups,k_0,k_1", [([0] * N_AGENTS, N_AGENTS, 0), ([1] * N_AGENTS, 0, N_AGENTS)]
)
def test_a_fully_merged_round_samples_without_error(groups, k_0, k_1):
    """After a full merge one group holds everyone and the other is empty. The
    empty group's row of the count grid is masked away except (0, ...), and the
    conditional-Bernoulli call for it is an all-False membership with m = 0 --
    no division by zero, no indexing into an empty group."""
    model = make_model(seed=23, joint_exodus=True, joint_exodus_switch_every=4)
    data = make_data(n_batch=2, round_number=DECISION_ROUND, groups=groups)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=2)

    th.manual_seed(SEED)
    pred, proba = model.predict_independent(data, sample=True, edge_index=edge_index)
    assert pred.shape == (2, N_AGENTS, 1)
    assert not bool(th.isnan(proba).any())
    group_r = data["agent_group"][:, :, 0]
    switched = pred[:, :, 0].to(th.bool)
    assert bool((switched & (group_r == 0)).sum(-1).max() <= k_0)
    assert bool((switched & (group_r == 1)).sum(-1).max() <= k_1)
    # the empty group can never contribute a leaver
    empty = 1 if k_1 == 0 else 0
    assert not bool((switched & (group_r == empty)).any())


def test_the_whole_group_can_leave_at_once():
    """m_g == k_g on a full group: the 5 -> 8 transition that builds the right
    tail SC is scored on."""
    model = make_model(seed=29, joint_exodus=True, joint_exodus_switch_every=4)
    pin_pair(model, N_AGENTS, 0)
    data = make_data(n_batch=2, round_number=DECISION_ROUND, groups=[0] * N_AGENTS)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=2)
    th.manual_seed(SEED)
    pred, _ = model.predict_independent(data, sample=True, edge_index=edge_index)
    assert bool(pred.all())


# --------------------------------------------------------------------------- #
# 5. determinism, persistence and the guards
# --------------------------------------------------------------------------- #
def test_determinism_under_a_fixed_seed():
    model = make_model(seed=31, joint_exodus=True, joint_exodus_switch_every=4)
    data = make_data(n_batch=4, round_number=DECISION_ROUND)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=4)
    th.manual_seed(SEED)
    first, _ = model.predict_independent(data, sample=True, edge_index=edge_index)
    th.manual_seed(SEED)
    second, _ = model.predict_independent(data, sample=True, edge_index=edge_index)
    assert th.equal(first, second)


def test_switch_every_round_trips_through_save_load():
    model = make_model(seed=31, joint_exodus=True, joint_exodus_switch_every=4)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        loaded = GraphNetwork.load(path, device="cpu")
    assert loaded.joint_exodus_switch_every == 4

    data = make_data(round_number=DECISION_ROUND)
    edge_index = loaded.create_fully_connected(N_AGENTS, n_batch=2)
    th.manual_seed(SEED)
    ref, _ = model.predict_independent(data, sample=True, edge_index=edge_index)
    th.manual_seed(SEED)
    got, _ = loaded.predict_independent(data, sample=True, edge_index=edge_index)
    assert th.equal(got, ref)


def test_sampling_without_switch_every_fails_loudly():
    """The failure mode this guard exists for is silent: falling back to the
    per-agent draw would produce a complete, plausible simulation that does
    not contain the mechanism at all."""
    model = make_model(seed=31, joint_exodus=True)
    assert model.joint_exodus_switch_every is None
    data = make_data(round_number=DECISION_ROUND)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=2)
    with pytest.raises(AssertionError, match="joint_exodus_switch_every"):
        model.predict_independent(data, sample=True, edge_index=edge_index)


def test_sample_false_never_reaches_the_joint_path():
    """`sample=False` is the argmax readout used by evaluation code; it has no
    RNG and must stay the per-agent argmax."""
    model = make_model(seed=31, joint_exodus=True, joint_exodus_switch_every=4)
    data = make_data(round_number=DECISION_ROUND)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=2)
    pred, proba = model.predict_independent(data, sample=False, edge_index=edge_index)
    assert th.equal(pred, proba.argmax(-1))


def test_switch_every_requires_the_head():
    with pytest.raises(AssertionError, match="only meaningful"):
        make_model(joint_exodus=False, joint_exodus_switch_every=4)


@pytest.mark.parametrize("bad", [0, -4, 4.0, True, "4"])
def test_switch_every_must_be_a_positive_int(bad):
    with pytest.raises(AssertionError, match="positive int"):
        make_model(joint_exodus=True, joint_exodus_switch_every=bad)


def test_joint_exodus_and_the_copula_are_mutually_exclusive():
    with pytest.raises(AssertionError, match="alternative switch"):
        make_model(joint_exodus=True, copula_rho=0.3)
