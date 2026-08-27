"""Tests for the severity copula on the autoregressive punisher's sampler.

Pins the invariants ``predict_autoreg``'s copula branch has to hold: the
legacy path (rho absent / 0.0 / ``sample=False``) is bit-identical including
the post-call torch RNG state, the inverse-CDF convention is
``min{a : F(a) >= u}``, the AR conditional marginals survive the copula
untouched, the shared latent correlates groupmates and only groupmates, the
RNG consumption is independent of the group composition, and ``copula_rho``
round-trips through ``save``/``load`` while legacy checkpoints load as
``None``.

Batch elements carry identical features here, so a batch is a set of iid
replications of the same conditional rows -- that is what the marginal and
correlation counts are taken over.

PyG-dependent -- run on the Raven cluster (``scripts/remote_test.sh``).
"""

import math
import os
import tempfile

import numpy as np
import pytest
import torch as th

from aimanager.generic import graph
from aimanager.generic.graph import GraphNetwork

N_PUNISHMENTS = 31
DEFAULT_VALUES = {"punishment": 0}
EDGE_ENCODING = [{"name": "ar_punishment", "n_levels": N_PUNISHMENTS}]
REVEAL_SEED = 7  # pins the numpy reveal permutation across repeated calls
RHO = 0.5

# replication budgets: samples = n_batch * n_calls
MARG_BATCH, MARG_CALLS = 250, 16  # 4000 draws of the first-revealed agent
CORR_BATCH, CORR_CALLS = 500, 8  # 4000 draws of an 8-agent, 2-group round


def _make_model(copula_rho=None, seed=0, hidden_size=6, y_levels=N_PUNISHMENTS):
    """The real punisher architecture (rnn_edge_50ep_doubled) plus AR edges.

    Construction consumes the torch RNG identically whatever ``copula_rho`` is,
    so two calls at one seed differ only in the copula weight.
    """
    th.manual_seed(seed)
    model = GraphNetwork(
        y_levels=y_levels,
        y_name="punishment",
        autoregressive=True,
        copula_rho=copula_rho,
        hidden_size=hidden_size,
        add_rnn=True,
        add_edge_model=True,
        add_global_model=False,
        x_encoding=[
            {"name": "prev_contribution", "n_levels": 21, "encoding": "numeric"},
            {"name": "prev_punishment", "n_levels": 31, "encoding": "numeric"},
            {"name": "is_first", "etype": "bool"},
        ],
        edge_encoding=EDGE_ENCODING,
        default_values=DEFAULT_VALUES,
    )
    model.eval()
    return model.to("cpu")


def _make_data(groups, n_batch=1, n_rounds=2, seed=0):
    """One episode tiled across the batch; ``groups`` is per node, per round."""
    agent_group = th.tensor(groups, dtype=th.int64)
    assert agent_group.shape[1] == n_rounds
    n_nodes = agent_group.shape[0]
    shape = (1, n_nodes, n_rounds)
    th.manual_seed(seed)
    data = {
        "contribution": th.randint(0, 21, shape),
        "prev_contribution": th.randint(0, 21, shape),
        "punishment": th.randint(1, N_PUNISHMENTS, shape),
        "prev_punishment": th.randint(0, N_PUNISHMENTS, shape),
        "is_first": th.zeros(shape, dtype=th.bool),
        "punishment_valid": th.ones(shape, dtype=th.bool),
        "agent_group": agent_group.unsqueeze(0),
    }
    data["is_first"][:, :, 0] = True
    data = {k: v.repeat(n_batch, 1, 1) for k, v in data.items()}
    data["punishment_masked"] = data["punishment"].clone()
    data["autoreg_mask"] = th.ones((n_batch, n_nodes, n_rounds), dtype=th.bool)
    return data


def _clone(data):
    return {k: v.clone() for k, v in data.items()}


def _edge_index(model, data):
    n_batch, n_nodes, _ = data["punishment"].shape
    return model.create_fully_connected(n_nodes, n_batch=n_batch)


def _reveal_order(n_nodes, reveal_seed=REVEAL_SEED):
    np.random.seed(reveal_seed)
    return np.random.permutation(np.arange(n_nodes))


def _predict(model, data, edge_index, sample=True, reveal_seed=REVEAL_SEED):
    np.random.seed(reveal_seed)
    return model.predict(_clone(data), sample=sample, edge_index=edge_index)


def _run(model, data, edge_index, seed, sample=True):
    """One seeded call; returns predictions, probabilities and the post-call
    torch RNG state (the object that would desynchronise if the branch drew
    a different number of times)."""
    th.manual_seed(seed)
    pred, proba = _predict(model, data, edge_index, sample=sample)
    return pred, proba, th.random.get_rng_state()


def _replicate(model, data, edge_index, n_calls, seed):
    """``n_calls`` sampled calls at a pinned reveal order. Returns the round -1
    levels stacked over calls and batch elements, plus the reported
    conditional rows."""
    rows, proba = [], None
    th.manual_seed(seed)
    for _ in range(n_calls):
        pred, proba = _predict(model, data, edge_index)
        rows.append(pred[:, :, -1].numpy())
    return np.concatenate(rows, axis=0), proba


def _ndtri(u):
    """Inverse standard-normal CDF via erfinv (no scipy on the cluster venv)."""
    return float(th.erfinv(th.tensor(2.0 * u - 1.0, dtype=th.float64)) * math.sqrt(2))


def _mean_pair_corr(draws, pairs):
    return float(
        np.mean([np.corrcoef(draws[:, i], draws[:, j])[0, 1] for i, j in pairs])
    )


# --------------------------------------------------------------------------- #
# (a, b) the legacy path is untouched -- values and RNG stream
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("copula_rho", [None, 0.0])
def test_rho_off_is_bit_identical_to_legacy(copula_rho):
    """rho absent or 0.0: same levels, same probabilities, and the stream is
    left exactly where the PR #161 sampler left it."""
    legacy = _make_model()
    model = _make_model(copula_rho=copula_rho)
    data = _make_data([[0, 0], [0, 0], [1, 1], [1, 1]], n_batch=3)
    edge_index = _edge_index(legacy, data)

    pred_a, proba_a, state_a = _run(legacy, data, edge_index, seed=3)
    pred_b, proba_b, state_b = _run(model, data, edge_index, seed=3)

    assert model.copula_rho == copula_rho
    assert th.equal(pred_a, pred_b)
    assert th.equal(proba_a, proba_b)
    assert th.equal(state_a, state_b), "copula gating changed the RNG stream"


def test_sample_false_ignores_rho():
    """``sample=False`` stays on the deterministic argmax at any rho, and
    consumes nothing."""
    legacy = _make_model()
    model = _make_model(copula_rho=RHO)
    data = _make_data([[0, 0], [0, 0], [1, 1], [1, 1]], n_batch=3)
    edge_index = _edge_index(legacy, data)

    th.manual_seed(5)
    before = th.random.get_rng_state()
    pred_a, proba_a, state_a = _run(legacy, data, edge_index, seed=5, sample=False)
    pred_b, proba_b, state_b = _run(model, data, edge_index, seed=5, sample=False)

    assert th.equal(pred_a, pred_b)
    assert th.equal(proba_a, proba_b)
    assert th.equal(state_a, state_b)
    assert th.equal(state_b, before), "the argmax path must not draw"


# --------------------------------------------------------------------------- #
# (c) inverse-CDF convention on a hand-built conditional row
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "u, expected",
    [
        (1e-12, 0),  # far below F(0)
        (0.25, 0),
        (0.5, 0),  # exactly F(0) -> still level 0 (F(a-1) < u <= F(a))
        (0.5 + 1e-9, 1),  # just above F(0) -> next level
        (0.75 - 1e-9, 1),
        (0.75 + 1e-9, 2),  # just above F(1)
        (1.0 - 1e-12, 2),  # top level, no overflow past y_levels - 1
    ],
)
def test_inverse_cdf_convention(monkeypatch, u, expected):
    """The dyadic row [0.5, 0.25, 0.25] has exactly representable cumulative
    boundaries, so the `>=` convention is pinned at u = F(a), not merely near
    it. Driven at a prescribed u: the latent carries ndtri(u)/sqrt(rho) and
    eps is pinned to zero."""
    model = _make_model(copula_rho=RHO, y_levels=3)
    proba = th.tensor([[[0.5, 0.25, 0.25]]], dtype=th.float)  # (1, 1, 3)
    z = th.full((1, 1, 1), _ndtri(u) / math.sqrt(RHO), dtype=th.float64)
    group = th.zeros((1, 1), dtype=th.int64)
    seen = []

    def fake_randn(shape, device=None, dtype=None):
        seen.append(tuple(shape))
        return th.zeros(shape, dtype=dtype)

    monkeypatch.setattr(graph.th, "randn", fake_randn)
    lvl = model._copula_levels(proba, z, group)

    assert seen == [(1, 1)], "expected exactly one eps draw per AR step"
    assert lvl.shape == (1, 1)
    assert lvl.dtype == th.int64
    assert int(lvl[0, 0]) == expected


# --------------------------------------------------------------------------- #
# (d) marginals preserved
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def marginals():
    """First-revealed agent's round -1 levels under the copula and under the
    legacy sampler, plus the conditional row both should reproduce. The first
    agent is drawn with nothing decided, so its row is the same on every call."""
    data = _make_data([[0, 0], [0, 0], [1, 1], [1, 1]], n_batch=MARG_BATCH)
    n_nodes = data["punishment"].shape[1]
    first = int(_reveal_order(n_nodes)[0])
    out = {"first": first, "n": MARG_BATCH * MARG_CALLS}
    for name, rho in (("copula", 0.9), ("legacy", None)):
        model = _make_model(copula_rho=rho)
        edge_index = _edge_index(model, data)
        draws, proba = _replicate(model, data, edge_index, MARG_CALLS, seed=11)
        out[name] = draws[:, first]
        out["P"] = proba[:, first, -1]
    # identical features across the batch -> one conditional row for all of it
    assert th.allclose(out["P"], out["P"][:1].expand_as(out["P"]))
    out["P"] = out["P"][0].numpy().astype(np.float64)
    return out


@pytest.mark.parametrize("source", ["copula", "legacy"])
def test_marginals_match_the_conditional_row(marginals, source):
    """Level frequencies must match the model's own softmax row within 4.5
    binomial SEs; levels with fewer than 20 expected counts are skipped, the
    normal approximation not being usable there."""
    P, draws, n = marginals["P"], marginals[source], marginals["n"]
    assert draws.min() >= 0 and draws.max() < N_PUNISHMENTS
    freq = np.bincount(draws, minlength=N_PUNISHMENTS) / n
    worst = 0.0
    for lvl in range(N_PUNISHMENTS):
        p = P[lvl]
        if p * n < 20:
            continue
        se = math.sqrt(p * (1.0 - p) / n)
        worst = max(worst, abs(freq[lvl] - p) / se)
    assert worst < 4.5, f"{source}: worst deviation {worst:.2f} binomial SEs"


def test_copula_marginal_matches_the_legacy_sampler(marginals):
    """Copula vs multinomial decode: two noisy estimates of one row, so they
    must agree within 4.5 SEs of their difference."""
    P, n = marginals["P"], marginals["n"]
    fc = np.bincount(marginals["copula"], minlength=N_PUNISHMENTS) / n
    fl = np.bincount(marginals["legacy"], minlength=N_PUNISHMENTS) / n
    worst = 0.0
    for lvl in range(N_PUNISHMENTS):
        p = P[lvl]
        if p * n < 20:
            continue
        se = math.sqrt(2.0 * p * (1.0 - p) / n)  # difference of two
        worst = max(worst, abs(fc[lvl] - fl[lvl]) / se)
    assert worst < 4.5, f"worst copula-vs-legacy gap {worst:.2f} SEs"


# --------------------------------------------------------------------------- #
# (e) correlation: within sub-group only
# --------------------------------------------------------------------------- #
WITHIN = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7)]
CROSS = [(0, 4), (0, 5), (1, 6), (2, 7), (3, 4), (3, 7)]


@pytest.fixture(scope="module")
def correlations():
    """8 agents in two fixed sub-groups of 4, round -1 levels, at rho=0.95 and
    on the legacy path."""
    groups = [[0, 0]] * 4 + [[1, 1]] * 4
    data = _make_data(groups, n_batch=CORR_BATCH)
    out = {"n": CORR_BATCH * CORR_CALLS}
    for name, rho in (("copula", 0.95), ("legacy", None)):
        model = _make_model(copula_rho=rho)
        edge_index = _edge_index(model, data)
        out[name], _ = _replicate(model, data, edge_index, CORR_CALLS, seed=13)
    return out


def test_within_group_correlation_induced(correlations):
    """The shared latent has to show up as a strong positive level correlation
    among groupmates -- discretising costs some of rho, never most of it."""
    within = _mean_pair_corr(correlations["copula"], WITHIN)
    assert within > 0.3, f"within-group corr {within:.4f} at rho=0.95"


def test_cross_group_correlation_absent(correlations):
    """Separate latents and a same-group-gated AR channel leave nothing to
    correlate across sub-groups."""
    cross = _mean_pair_corr(correlations["copula"], CROSS)
    within = _mean_pair_corr(correlations["copula"], WITHIN)
    assert abs(cross) < 4.5 / math.sqrt(correlations["n"]), f"cross {cross:.4f}"
    assert within > cross + 0.25


def test_legacy_path_leaves_groupmates_near_independent(correlations):
    """On the legacy path the only coupling left is the AR channel itself,
    which a randomly initialised model barely uses -- bounded generously
    because it is a real, non-zero effect, not sampling noise."""
    within = _mean_pair_corr(correlations["legacy"], WITHIN)
    assert abs(within) < 0.15, f"legacy within-group corr {within:.4f}"


# --------------------------------------------------------------------------- #
# (f) the latent is fresh per call and deterministic under re-seed
# --------------------------------------------------------------------------- #
def test_latent_is_fresh_per_call_and_reproducible():
    model = _make_model(copula_rho=0.9)
    data = _make_data([[0, 0], [0, 0], [1, 1], [1, 1]], n_batch=8)
    edge_index = _edge_index(model, data)
    first = int(_reveal_order(data["punishment"].shape[1])[0])

    th.manual_seed(17)
    call_a = _predict(model, data, edge_index)[0][:, first, -1]
    call_b = _predict(model, data, edge_index)[0][:, first, -1]
    th.manual_seed(17)
    again_a = _predict(model, data, edge_index)[0][:, first, -1]
    again_b = _predict(model, data, edge_index)[0][:, first, -1]

    assert not th.equal(call_a, call_b), "z was reused across calls"
    assert th.equal(call_a, again_a)
    assert th.equal(call_b, again_b)


# --------------------------------------------------------------------------- #
# (g) RNG consumption is independent of the group composition
# --------------------------------------------------------------------------- #
def test_group_composition_does_not_shift_the_stream():
    """One z of node shape plus one eps per AR step: the draw count depends on
    (n_batch, n_nodes, n_rounds) only, so re-partitioning the same agents
    changes the levels but never the stream position."""
    model = _make_model(copula_rho=0.9)
    edge_index = None
    states, preds = [], []
    partitions = [
        [[0, 0]] * 4 + [[1, 1]] * 4,  # two groups of four
        [[0, 0], [1, 1]] * 4,  # interleaved
        [[0, 0]] * 8,  # one group
    ]
    for groups in partitions:
        data = _make_data(groups, n_batch=4)
        if edge_index is None:
            edge_index = _edge_index(model, data)
        pred, _, state = _run(model, data, edge_index, seed=23)
        preds.append(pred)
        states.append(state)

    for state in states[1:]:
        assert th.equal(states[0], state), "draw count depends on composition"
    for pred in preds[1:]:
        assert not th.equal(preds[0], pred), "the partition had no effect"


# --------------------------------------------------------------------------- #
# (h) checkpoint round-trip
# --------------------------------------------------------------------------- #
def test_save_load_round_trips_copula_rho():
    """copula_rho survives save/load and reproduces predictions exactly; a
    checkpoint written before this change loads as None (legacy path)."""
    model = _make_model(copula_rho=0.37)
    data = _make_data([[0, 0], [0, 0], [1, 1], [1, 1]], n_batch=2)
    edge_index = _edge_index(model, data)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ar_copula.pt")
        model.save(path)
        loaded = GraphNetwork.load(path, device="cpu")

        blob = th.load(path, map_location="cpu")
        assert blob["copula_rho"] == 0.37
        del blob["copula_rho"]  # a pre-copula checkpoint has no such key
        legacy_path = os.path.join(tmp, "legacy.pt")
        th.save(blob, legacy_path)
        legacy = GraphNetwork.load(legacy_path, device="cpu")

    assert loaded.copula_rho == 0.37
    assert loaded.autoregressive is True
    assert loaded.edge_encoding == EDGE_ENCODING
    assert legacy.copula_rho is None

    pred_a, _, state_a = _run(model, data, edge_index, seed=29)
    pred_b, _, state_b = _run(loaded, data, edge_index, seed=29)
    assert th.equal(pred_a, pred_b)
    assert th.equal(state_a, state_b)

    # the stripped checkpoint is back on the multinomial decode
    pred_c, _, state_c = _run(_make_model(), data, edge_index, seed=29)
    pred_d, _, state_d = _run(legacy, data, edge_index, seed=29)
    assert th.equal(pred_c, pred_d)
    assert th.equal(state_c, state_d)


# --------------------------------------------------------------------------- #
# (i) the constructor gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rho", [1.0, -0.1, 1.5])
def test_constructor_rejects_rho_outside_the_unit_interval(rho):
    with pytest.raises(AssertionError, match="copula_rho must be None"):
        _make_model(copula_rho=rho)


@pytest.mark.parametrize("rho", [None, 0.0, 0.37, 0.999])
def test_constructor_accepts_valid_rho(rho):
    assert _make_model(copula_rho=rho).copula_rho == rho
