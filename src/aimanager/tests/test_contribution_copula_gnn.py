"""Wiring tests for the contribution copula inside GraphNetwork: induced
within-group correlation at rho > 0 with marginals intact, a bit-identical
legacy path (RNG consumption included) at rho == 0, save/load round-trip of
`copula_rho`, the __init__ gate, the cell construction, and the sim-path
assert. Sampler-level invariants live in tests/copula (pure torch, local);
this file imports graph.py and therefore PyG, so it runs on Raven:

    scripts/remote_test.sh -- src/aimanager/tests/test_contribution_copula_gnn.py -v
"""

import math
import os
import sys
import tempfile

import numpy as np
import pytest
import torch as th

# pytorch geometric meta module has changed place since the committed
# artifacts were saved; the alias lets legacy pickles unpickle (simulate.py
# installs the same one before loading a model)
import torch_geometric.nn.models.meta as meta_module

sys.modules["torch_geometric.nn.meta"] = meta_module

from aimanager.generic.graph import GraphNetwork  # noqa: E402

N_LEVELS = 21
N_AGENTS = 8
GROUPS = [0, 0, 0, 0, 1, 1, 1, 1]
RHO = 0.4
N_DRAW = 400  # predict repeats for the marginal / correlation tests
M0 = (
    "artifacts/artificial_humans/group_switching_contribution_50ep/model/"
    "architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
)
WITHIN = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7)]
CROSS = [(0, 4), (0, 5), (1, 6), (2, 7), (3, 4), (3, 7)]


# --------------------------------------------------------------------------- #
# toy model and data
# --------------------------------------------------------------------------- #
def make_model(y_name="contribution", **over):
    """Smallest workable contribution GNN (same shape as the edge-encoder
    tests'), seeded so the readout is fixed across constructions."""
    th.manual_seed(0)
    kwargs = dict(
        y_levels=N_LEVELS,
        y_name=y_name,
        hidden_size=4,
        add_rnn=True,
        add_edge_model=True,
        add_global_model=False,
        x_encoding=[
            {"name": "prev_contribution", "n_levels": 21, "encoding": "numeric"}
        ],
        edge_encoding=[],
        default_values={},
    )
    kwargs.update(over)
    return GraphNetwork(**kwargs).to("cpu")


def make_data(n_batch=4, n_player=N_AGENTS, n_rounds=2, uniform=True, groups=GROUPS):
    """Toy batch. `uniform` gives every agent the same features, so all rows
    share one marginal and a level correlation can only come from the shared
    latent -- never from shared inputs."""
    shape = (n_batch, n_player, n_rounds)
    if uniform:
        prev = th.full(shape, 9, dtype=th.int64)
    else:
        th.manual_seed(1)
        prev = th.randint(0, 21, shape)
    ag = th.tensor(groups, dtype=th.int64).reshape(1, -1, 1).expand(shape)
    return {
        "contribution": th.zeros(shape, dtype=th.int64),
        "prev_contribution": prev,
        "agent_group": ag.contiguous(),
    }


def legacy_predict(model, data):
    """The contribution draw as committed BEFORE this change, reimplemented
    verbatim from graph.py (encode -> forward -> softmax -> the one-line
    multinomial decode), so the comparison does not depend on the code under
    test."""
    n_batch, n_nodes, n_rounds = data[model.y_name].shape
    edge_index = model.create_fully_connected(n_nodes, n_batch=n_batch)
    encoded = model.encode(
        data, y_encode=False, edge_index=edge_index, device=model.device
    )
    model.eval()
    proba = th.nn.functional.softmax(model(encoded, True), dim=-1)
    dec = th.multinomial(proba.reshape(-1, proba.shape[-1]), 1)
    lvl = dec.reshape(proba.shape[:-1])
    return lvl.reshape((n_batch, n_nodes, *lvl.shape[1:]))


def pair_corr(draws, pairs):
    return float(
        np.mean([np.corrcoef(draws[:, i], draws[:, j])[0, 1] for i, j in pairs])
    )


@pytest.fixture(scope="module")
def sampled():
    """N_DRAW free-running predictions of one fixed batch, at rho=RHO and at
    rho=0, plus the (identical, teacher-forced) probabilities they must
    reproduce. Round 0 only; episodes are independent cells, so they stack."""
    data = make_data()
    out = {}
    for name, rho in (("copula", RHO), ("rho0", 0.0)):
        model = make_model(copula_rho=rho)
        th.manual_seed(11)
        draws = [
            model.predict_independent(data, sample=True)[0][:, :, 0]
            for _ in range(N_DRAW)
        ]
        out[name] = th.stack(draws).reshape(-1, N_AGENTS).numpy()
    model = make_model(copula_rho=RHO)
    # predict_* leaves the graph attached (eval, no no_grad) -> detach to read
    proba = model.predict_independent(data, sample=False)[1].detach()
    out["proba"] = proba[0, :, 0].double()
    return out


# --------------------------------------------------------------------------- #
# (a) the mechanism through predict_independent
# --------------------------------------------------------------------------- #
def test_marginals_match_predicted_proba(sampled):
    """Free-running levels must reproduce the model's own probabilities within
    4 binomial SEs, rho or no rho -- the copula moves dependence only. Levels
    with fewer than 20 expected counts are skipped (normal approximation)."""
    P = sampled["proba"].numpy()
    n = len(sampled["copula"])
    for source in ("copula", "rho0"):
        worst = 0.0
        for agent in range(N_AGENTS):
            freq = np.bincount(sampled[source][:, agent], minlength=N_LEVELS) / n
            for lvl in range(N_LEVELS):
                p = P[agent, lvl]
                if p * n < 20:
                    continue
                se = math.sqrt(p * (1.0 - p) / n)
                worst = max(worst, abs(freq[lvl] - p) / se)
        assert worst < 4.0, f"{source}: worst deviation {worst:.2f} binomial SEs"


def test_within_group_correlation_induced(sampled):
    n = len(sampled["copula"])
    within = pair_corr(sampled["copula"], WITHIN)
    assert within > 6.0 / math.sqrt(n), f"within-group corr {within:.4f}"


def test_cross_group_correlation_absent(sampled):
    n = len(sampled["copula"])
    cross = pair_corr(sampled["copula"], CROSS)
    assert abs(cross) < 4.0 / math.sqrt(n), f"cross-group corr {cross:.4f}"


def test_rho_zero_leaves_agents_independent(sampled):
    n = len(sampled["rho0"])
    for pairs, label in ((WITHIN, "within"), (CROSS, "cross")):
        corr = pair_corr(sampled["rho0"], pairs)
        assert abs(corr) < 4.0 / math.sqrt(n), f"{label} corr {corr:.4f} at rho=0"


def test_within_beats_cross(sampled):
    n = len(sampled["copula"])
    within = pair_corr(sampled["copula"], WITHIN)
    cross = pair_corr(sampled["copula"], CROSS)
    assert within > cross + 5.0 / math.sqrt(n)


# --------------------------------------------------------------------------- #
# (b) the legacy path is bit-identical at rho == 0
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("over", [{}, {"copula_rho": 0.0}])
def test_rho_zero_bit_identical_to_legacy(over):
    """Three consecutive calls under one seed: identical levels AND identical
    RNG consumption (an extra draw would desynchronise call 2)."""
    model = make_model(**over)
    assert model.copula_rho == 0.0
    data = make_data(uniform=False)

    th.manual_seed(42)
    want = [legacy_predict(model, data) for _ in range(3)]
    want_next = th.randn(1).item()  # where the legacy path leaves the stream
    th.manual_seed(42)
    got = [model.predict_independent(data, sample=True)[0] for _ in range(3)]
    got_next = th.randn(1).item()

    for i, (w, g) in enumerate(zip(want, got)):
        assert th.equal(w, g), f"call {i}: {w.tolist()} != {g.tolist()}"
    assert got_next == want_next, "RNG consumption changed"


def test_copula_path_is_reached_at_rho_above_zero():
    """Guard against a vacuous (b): at rho > 0 the levels must NOT match the
    legacy draw under the same seed, or the branch is never taken."""
    model = make_model(copula_rho=RHO)
    data = make_data(uniform=False)
    th.manual_seed(42)
    legacy = legacy_predict(model, data)
    th.manual_seed(42)
    got = model.predict_independent(data, sample=True)[0]
    assert not th.equal(legacy, got)


def test_teacher_forced_probabilities_untouched():
    """sample=False is the training/evaluation path: argmax, no draw, and the
    probabilities identical to a rho-free model's."""
    data = make_data(uniform=False)
    cop = make_model(copula_rho=RHO)
    ind = make_model()
    pred_c, proba_c = cop.predict_independent(data, sample=False)
    pred_i, proba_i = ind.predict_independent(data, sample=False)
    assert th.equal(proba_c, proba_i)
    assert th.equal(pred_c, pred_i)
    assert th.equal(pred_c, proba_c.argmax(-1))


# --------------------------------------------------------------------------- #
# (c) save / load
# --------------------------------------------------------------------------- #
def test_save_load_round_trips_copula_rho():
    model = make_model(copula_rho=RHO)
    data = make_data(uniform=False)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        assert "copula_rho" in th.load(path, map_location="cpu")
        loaded = GraphNetwork.load(path, device="cpu")
        assert loaded.copula_rho == RHO
        assert th.equal(
            loaded.predict_independent(data, sample=False)[1],
            model.predict_independent(data, sample=False)[1],
        )

        # a checkpoint saved before this change has no key at all
        saved = th.load(path, map_location="cpu")
        del saved["copula_rho"]
        legacy_path = os.path.join(d, "legacy.pt")
        th.save(saved, legacy_path)
        legacy = GraphNetwork.load(legacy_path, device="cpu")
        assert legacy.copula_rho == 0.0


def test_committed_m0_loads_at_rho_zero():
    """The reference contribution artifact must load unchanged and stay on the
    independent path until a calibrated copy is stamped."""
    if not os.path.exists(M0):  # pragma: no cover
        pytest.skip(f"{M0} missing")
    model = GraphNetwork.load(M0, device="cpu")
    assert model.copula_rho == 0.0
    assert model.y_name == "contribution"


# --------------------------------------------------------------------------- #
# (d) the __init__ gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rho", [1.0, 1.5, -0.1])
def test_gate_rejects_rho_outside_unit_interval(rho):
    with pytest.raises(AssertionError, match=r"\[0, 1\)"):
        make_model(copula_rho=rho)


@pytest.mark.parametrize("y_name", ["punishment", "does_switch", "contribution_valid"])
def test_gate_rejects_rho_on_other_targets(y_name):
    """The switch, valid and punishment GNNs share this class; a rho must be
    impossible on them."""
    with pytest.raises(AssertionError, match="contribution sampler only"):
        make_model(y_name=y_name, y_levels=2, copula_rho=RHO)


@pytest.mark.parametrize("y_name", ["punishment", "does_switch"])
def test_gate_accepts_other_targets_without_rho(y_name):
    assert make_model(y_name=y_name, y_levels=2).copula_rho == 0.0


@pytest.mark.parametrize("rho", [None, 0.0])
def test_gate_accepts_absent_rho(rho):
    assert make_model(copula_rho=rho).copula_rho == 0.0


# --------------------------------------------------------------------------- #
# (e) cell construction
# --------------------------------------------------------------------------- #
def test_copula_cells_layout_and_sharing():
    """(B*A, T) ids, laid out like the decode input: equal within an
    (episode, round, group), distinct across any two of them."""
    n_batch, n_agents, n_rounds = 2, 4, 3
    ag = th.tensor(
        [
            [[0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]],
            [[0, 0, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1]],
        ]
    )
    model = make_model()
    cells = model.copula_cells(ag, n_batch, n_agents, n_rounds, device="cpu")
    assert cells.shape == (n_batch * n_agents, n_rounds)
    seen = {}
    for b in range(n_batch):
        for t in range(n_rounds):
            for p in range(n_agents):
                key = (b, t, int(ag[b, p, t]))
                cid = int(cells[b * n_agents + p, t])
                assert seen.setdefault(cid, key) == key, "cell id collision"
                for q in range(n_agents):
                    same_cell = cells[b * n_agents + p, t] == cells[b * n_agents + q, t]
                    assert bool(same_cell) == (ag[b, p, t] == ag[b, q, t])


def test_copula_cells_follow_a_switching_agent():
    """agent_group is time-varying: an agent that switches mid-episode joins
    its new group's cell in the round it switches."""
    ag = th.tensor([[[0, 1], [0, 0], [1, 1], [1, 1]]])  # agent 0 switches at t=1
    model = make_model()
    cells = model.copula_cells(ag, 1, 4, 2, device="cpu")
    assert cells[0, 0] == cells[1, 0] and cells[0, 0] != cells[2, 0]
    assert cells[0, 1] == cells[2, 1] and cells[0, 1] != cells[1, 1]


def test_copula_cells_match_the_decode_layout():
    """The ids predict_independent builds must line up with the rows the
    decode call sees: encode flattens (batch, agent), rounds stay on dim 1."""
    data = make_data(n_batch=3, n_rounds=2, uniform=False)
    model = make_model(copula_rho=RHO)
    proba = model.predict_independent(data, sample=False)[1]
    assert proba.shape == (3, N_AGENTS, 2, N_LEVELS)
    # predict_encoded sees proba before predict_independent's reshape
    decode_shape = proba.reshape(3 * N_AGENTS, 2, N_LEVELS).shape[:-1]
    cells = model.copula_cells(data["agent_group"], 3, N_AGENTS, 2, device="cpu")
    assert cells.shape == decode_shape


# --------------------------------------------------------------------------- #
# (f) the sim-path assert
# --------------------------------------------------------------------------- #
def test_missing_agent_group_raises_when_sampling():
    """A calibrated model must never silently lose its correlation: no
    agent_group on the free-running path is an error, not a fallback."""
    model = make_model(copula_rho=RHO)
    data = make_data(uniform=False)
    del data["agent_group"]
    with pytest.raises(AssertionError, match="agent_group"):
        model.predict_independent(data, sample=True)


def test_missing_agent_group_is_fine_when_teacher_forced():
    """Teacher-forced evaluation draws nothing, so it needs no cells."""
    model = make_model(copula_rho=RHO)
    data = make_data(uniform=False)
    del data["agent_group"]
    pred, proba = model.predict_independent(data, sample=False)
    assert th.equal(pred, proba.argmax(-1))
    assert pred.shape == (4, N_AGENTS, 2)


def test_rho_zero_without_agent_group_still_samples():
    """The legacy models keep working on agent_group-free data."""
    model = make_model()
    data = make_data(uniform=False)
    del data["agent_group"]
    pred, _ = model.predict_independent(data, sample=True)
    assert pred.shape == (4, N_AGENTS, 2)
    assert int(pred.min()) >= 0 and int(pred.max()) <= N_LEVELS - 1
