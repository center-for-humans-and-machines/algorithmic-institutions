"""Tests for the CONTRIBUTION head of the herding-copula dispatch in
`GraphNetwork` (plan step 3 of contribution-herding-copula-v2).

Imports torch_geometric via `aimanager.generic.graph`, so it runs on Raven:
    scripts/remote_test.sh -- -k contribution_copula_graph

Sibling of test_switch_copula_graph.py, which owns the same gates for the
2-level `does_switch` head. What is head-specific and therefore re-checked
here: 21 levels instead of 2, and `copula_switch_every=1` -- contributions are
decided EVERY round, so the AR(1) latent must advance on every call rather
than every k-th (the switch head's `k=4` is asserted in the sibling file).
The gate this file shares with it is the LEGACY path: an artifact without
`copula_rho` must sample exactly as before, values and torch RNG consumption
alike. The sampler itself is covered locally by
tests/copula/test_contribution_copula.py.
Context: notes/autoresearch_log/contribution-herding-copula-v2.md.
"""

import os
import tempfile

import numpy as np
import pytest
import torch as th

from aimanager.generic import graph as graph_module
from aimanager.generic.copula import sample_correlated_levels
from aimanager.generic.graph import GraphNetwork
from aimanager.manager.environment import ArtificialHumanEnv

SEED = 20260827
N_AGENTS = 8
N_LEVELS = 21
GROUPS = [0, 0, 0, 0, 1, 1, 1, 1]
# A deliberately large dose, so the induced dependence is measurable at
# N_REPEATS forward passes; the calibrated ~0.07 (PR #149) is exercised at
# large N in tests/copula/test_contribution_copula.py.
RHO = 0.3
PHI = 0.7
N_REPEATS = 3000
# Absolute slack on top of the 5-sigma band, so a level whose predicted
# probability is near zero cannot fail on a single extra count. 2e-3 of
# probability mass is far below any marginal distortion worth catching.
FREQ_SLACK = 2e-3


def make_model(seed=0, **copula):
    """Contribution-shaped model: 21 levels, RNN + edge model, no global, and
    the base artifact's node features (see
    configs/training/artificial_humans/contribution/
    group_switching_contribution_50ep.yml)."""
    th.manual_seed(seed)
    model = GraphNetwork(
        y_levels=N_LEVELS,
        y_name="contribution",
        hidden_size=4,
        add_rnn=True,
        add_edge_model=True,
        add_global_model=False,
        x_encoding=[
            {"name": "prev_contribution", "n_levels": 21, "encoding": "numeric"},
            {"name": "prev_punishment", "n_levels": 31, "encoding": "numeric"},
            {"name": "agent_group", "n_levels": 2, "encoding": "onehot"},
        ],
        edge_encoding=[],
        default_values={"contribution": 0},
        **copula,
    )
    return model.to("cpu")


def paired_models(**copula):
    """A copula model and a legacy model carrying the same weights."""
    copula_model = make_model(**copula)
    legacy_model = make_model()
    legacy_model.load_state_dict(copula_model.state_dict())
    return copula_model, legacy_model


def make_data(n_batch=4, n_rounds=1, round_number=0, seed=1):
    """Simulation-shaped state: one round per call, (n_batch, n_agents, 1)."""
    th.manual_seed(seed)
    shape = (n_batch, N_AGENTS, n_rounds)
    agent_group = th.tensor(GROUPS, dtype=th.int64).reshape(1, N_AGENTS, 1)
    return {
        "contribution": th.zeros(shape, dtype=th.int64),
        "prev_contribution": th.randint(0, 21, shape),
        "prev_punishment": th.randint(0, 31, shape),
        "agent_group": agent_group.expand(shape).contiguous(),
        "round_number": th.full(shape, round_number, dtype=th.int64),
    }


def legacy_predict(model, data, edge_index=None, reset_rnn=True):
    """`predict_independent(sample=True)` as it was BEFORE the copula dispatch,
    reimplemented here so the identity check does not lean on the code under
    test: encode -> forward -> softmax -> one `th.multinomial` over all rows."""
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


class _SamplerSpy:
    """Records what the dispatch hands the sampler and delegates to the real
    one. `graph.py` calls the name it imported, so the patch target is
    `aimanager.generic.graph.sample_correlated_levels`."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.cell_ids = []
        self.z_prev = []
        self.z_cell = []

    def __call__(self, proba, cell_id, rho, z_prev=None, phi=0.0):
        self.cell_ids.append(cell_id.clone())
        self.z_prev.append(None if z_prev is None else z_prev.clone())
        levels, z_cell = sample_correlated_levels(
            proba, cell_id, rho, z_prev=z_prev, phi=phi
        )
        self.z_cell.append(z_cell.clone())
        return levels, z_cell


class _FixedSwitch:
    """Stand-in switch AH: the same agents switch on every decision round, so
    the arrival groups are known without a second GNN."""

    default_values = {"does_switch": 0}

    def __init__(self, switchers):
        self.switchers = switchers

    def predict(self, state, reset_rnn, edge_index):
        does_switch = th.zeros_like(state["contribution"])
        does_switch[:, self.switchers] = 1
        return does_switch, None


def make_env(contribution, batch_size, switch_every=4):
    """The simulation call site: `update_contribution` every round, one round
    per call, with agent 0 switching group on every decision round."""
    return ArtificialHumanEnv(
        artifical_humans=contribution,
        artifical_humans_valid=None,
        artifical_humans_switch=_FixedSwitch([0]),
        switch_every=switch_every,
        batch_size=batch_size,
        n_agents=N_AGENTS,
        n_contributions=N_LEVELS,
        n_punishments=31,
        n_rounds=12,
        n_groups=2,
        device="cpu",
        reward_mode="avg",
        agent_groups=GROUPS,
        default_values={
            "punishment": 0,
            "contribution": 0,
            "round_number": 0,
            "is_first": False,
            "contribution_valid": False,
            "punishment_valid": False,
            "common_good": 0,
            "contributor_payoff": 0,
            "manager_payoff": 0,
            "reward": 0,
        },
    )


def run_seeded(fn):
    """Return (result, next 5 draws) with the RNG seeded before `fn`."""
    th.manual_seed(SEED)
    out = fn()
    return out, th.randn(5)


def expected_cells(agent_group, n_batch):
    """The dense cell id the dispatch must build: batch_index * 2 + group."""
    group = agent_group.reshape(n_batch, N_AGENTS)
    return (th.arange(n_batch).reshape(-1, 1) * 2 + group).reshape(-1)


# --------------------------------------------------------------------------- #
# 1. the legacy path is untouched
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("copula", [{}, {"copula_rho": 0.0}])
def test_default_model_matches_pre_change_path_bitwise(copula):
    """rho absent and rho explicitly 0.0 both keep the 21-level multinomial
    draw: same levels, same probabilities, same post-call RNG state."""
    model = make_model(**copula)
    assert model.copula_rho == 0.0
    assert model.copula_phi == 0.0
    assert model.copula_switch_every is None
    data = make_data(n_batch=2)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=2)

    (ref_pred, ref_proba), ref_rng = run_seeded(
        lambda: legacy_predict(model, data, edge_index=edge_index)
    )
    (new_pred, new_proba), new_rng = run_seeded(
        lambda: model.predict_independent(data, sample=True, edge_index=edge_index)
    )

    assert ref_proba.shape == (2, N_AGENTS, 1, N_LEVELS)
    assert th.equal(new_pred, ref_pred)
    assert th.equal(new_proba, ref_proba)
    # same post-call RNG state -> the legacy path consumes exactly as before
    assert th.equal(new_rng, ref_rng)


# --------------------------------------------------------------------------- #
# 2. the copula path keeps marginals and correlates within the group only
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def copula_draws():
    """`N_REPEATS` sampled rounds from the copula model, plus the 21-level
    marginals of the independent path on the same weights and data."""
    n_batch = 4
    copula_model, legacy_model = paired_models(copula_rho=RHO)
    data = make_data(n_batch=n_batch)
    edge_index = copula_model.create_fully_connected(N_AGENTS, n_batch=n_batch)

    _, proba = legacy_model.predict_independent(
        data, sample=False, edge_index=edge_index
    )
    p = proba.detach().reshape(-1, N_LEVELS).numpy()
    # a row collapsed onto one level would make the correlations vacuous
    assert p.max() < 0.95, p.max()

    th.manual_seed(SEED)
    draws = np.empty((N_REPEATS, n_batch * N_AGENTS), dtype=np.int64)
    for i in range(N_REPEATS):
        y_pred, y_proba = copula_model.predict_independent(
            data, sample=True, edge_index=edge_index
        )
        # the copula must not touch the marginals it inverts
        assert th.equal(y_proba, proba)
        assert y_pred.shape == (n_batch, N_AGENTS, 1)
        assert int(y_pred.min()) >= 0 and int(y_pred.max()) < N_LEVELS
        draws[i] = y_pred.reshape(-1).numpy()
    cell = np.repeat(np.arange(n_batch), N_AGENTS) * 2 + np.tile(GROUPS, n_batch)
    return draws, p, cell


def test_copula_preserves_per_agent_level_marginals(copula_draws):
    """Every one of the 21 per-agent level frequencies stays inside its
    binomial band -- the copula only re-couples the draws, never reweights
    them."""
    draws, p, _ = copula_draws
    freq = np.stack(
        [(draws == level).mean(axis=0) for level in range(N_LEVELS)], axis=1
    )
    assert freq.shape == p.shape
    se = np.sqrt(p * (1.0 - p) / N_REPEATS)
    assert np.all(np.abs(freq - p) < 5.0 * se + FREQ_SLACK), np.max(
        np.abs(freq - p) / se
    )


def test_copula_preserves_per_agent_mean_contribution(copula_draws):
    """The first moment is what the CG spread ratio is built from, so it gets
    its own gate rather than riding on the per-level bands."""
    draws, p, _ = copula_draws
    levels = np.arange(N_LEVELS)
    mean_expected = (p * levels).sum(axis=1)
    var_expected = (p * levels**2).sum(axis=1) - mean_expected**2
    se = np.sqrt(var_expected / N_REPEATS)
    dev = np.abs(draws.mean(axis=0) - mean_expected) / se
    assert np.all(dev < 5.0), dev.max()


def test_copula_correlates_within_cell_only(copula_draws):
    draws, _, cell = copula_draws
    c = np.corrcoef(draws.T.astype(float))
    same = cell[:, None] == cell[None, :]
    off = ~np.eye(len(cell), dtype=bool)
    within = c[same & off].mean()
    across = c[~same].mean()
    assert within > 0.05, within
    assert abs(across) < 0.03, across


# --------------------------------------------------------------------------- #
# 3. save / load
# --------------------------------------------------------------------------- #
def test_save_load_round_trips_copula_fields():
    """`copula_switch_every=1` is the contribution slot's setting, so it is
    the value that has to survive the artifact round trip."""
    model = make_model(copula_rho=RHO, copula_phi=PHI, copula_switch_every=1)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        loaded = GraphNetwork.load(path, device="cpu")
        assert loaded.copula_rho == RHO
        assert loaded.copula_phi == PHI
        assert loaded.copula_switch_every == 1

        # an OLD artifact has none of the three keys: defaults + legacy path
        saved = th.load(path, map_location="cpu")
        for k in ("copula_rho", "copula_phi", "copula_switch_every"):
            del saved[k]
        legacy_path = os.path.join(d, "legacy.pt")
        th.save(saved, legacy_path)
        legacy = GraphNetwork.load(legacy_path, device="cpu")

    assert legacy.copula_rho == 0.0
    assert legacy.copula_phi == 0.0
    assert legacy.copula_switch_every is None

    data = make_data(n_batch=2)
    edge_index = legacy.create_fully_connected(N_AGENTS, n_batch=2)
    (ref_pred, _), ref_rng = run_seeded(
        lambda: legacy_predict(legacy, data, edge_index=edge_index)
    )
    (new_pred, _), new_rng = run_seeded(
        lambda: legacy.predict_independent(data, sample=True, edge_index=edge_index)
    )
    assert th.equal(new_pred, ref_pred)
    assert th.equal(new_rng, ref_rng)


# --------------------------------------------------------------------------- #
# 4. AR(1) latent lifecycle -- every round, not every k-th
# --------------------------------------------------------------------------- #
def test_ar_latent_advances_every_round():
    """At `copula_switch_every=1` every round is a decision round: the latent
    is set after the first call and differs between consecutive calls. (The
    switch head's contrast -- unchanged for three rounds out of four -- is
    test_ar_latent_advances_only_on_decision_rounds in the sibling file.)"""
    n_batch = 2
    model = make_model(copula_rho=RHO, copula_phi=PHI, copula_switch_every=1)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=n_batch)

    th.manual_seed(SEED)
    seen = []
    for r in range(6):
        model.predict_independent(
            make_data(n_batch=n_batch, round_number=r),
            sample=True,
            reset_rnn=(r == 0),
            edge_index=edge_index,
        )
        z = model._copula_z
        assert z is not None, r
        assert z.shape == (2 * n_batch,)
        assert z.dtype == th.float64
        seen.append(z.clone())

    for r in range(1, len(seen)):
        assert not th.equal(seen[r], seen[r - 1]), r


def test_reset_rnn_clears_the_latent(monkeypatch):
    """`reset_rnn` drops the latent with the GRU state. At
    `copula_switch_every=1` the same call immediately redraws it, so the
    clearing is observable as `z_prev=None` reaching the sampler; on the
    persisting rounds `z_prev` is the previous call's `z_cell`."""
    spy = _SamplerSpy()
    monkeypatch.setattr(graph_module, "sample_correlated_levels", spy)
    n_batch = 2
    model = make_model(copula_rho=RHO, copula_phi=PHI, copula_switch_every=1)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=n_batch)

    th.manual_seed(SEED)
    for r in list(range(4)) + [0]:
        model.predict_independent(
            make_data(n_batch=n_batch, round_number=r),
            sample=True,
            reset_rnn=(r == 0),
            edge_index=edge_index,
        )

    assert len(spy.z_prev) == 5
    assert spy.z_prev[0] is None
    assert spy.z_prev[4] is None  # the fresh episode starts from no history
    for r in (1, 2, 3):
        assert spy.z_prev[r] is not None, r
        assert th.equal(spy.z_prev[r], spy.z_cell[r - 1]), r
    assert model._copula_z is not None  # redrawn within the reset round


# --------------------------------------------------------------------------- #
# 5. the unit-root boundary phi = 1.0 -- the adopted persistence (step 8b)
# --------------------------------------------------------------------------- #
def test_unit_phi_is_accepted_and_round_trips():
    """phi = 1.0 (a latent held static for the episode) is a legal dose at
    construction and survives the artifact round trip unchanged."""
    model = make_model(copula_rho=RHO, copula_phi=1.0, copula_switch_every=1)
    assert model.copula_phi == 1.0
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        loaded = GraphNetwork.load(path, device="cpu")
    assert loaded.copula_rho == RHO
    assert loaded.copula_phi == 1.0
    assert loaded.copula_switch_every == 1


def test_env_holds_the_latent_static_at_unit_phi():
    """The simulation call site at the adopted phi: the copula state is drawn
    once at round 0 and then unchanged for the rest of the episode, while
    `reset()` still drops it and redraws a fresh one."""
    batch_size = 3
    contribution = make_model(copula_rho=RHO, copula_phi=1.0, copula_switch_every=1)
    env = make_env(contribution, batch_size)

    th.manual_seed(SEED)
    env.reset()
    z_first = contribution._copula_z.clone()
    assert z_first.shape == (2 * batch_size,)
    for r in range(11):
        env.punish(th.zeros_like(env.state["punishment"]))
        env.step()
        assert th.equal(contribution._copula_z, z_first), r

    env.reset()
    z_second = contribution._copula_z.clone()
    assert not th.equal(z_second, z_first), "reset must redraw the latent"


def test_env_drives_the_contribution_copula_predictor(monkeypatch):
    """The simulation call site: `update_contribution` every round, one round
    per call, cells from the env's POST-switch `agent_group` (`apply_switch`
    runs before `update_contribution` in `step`)."""
    spy = _SamplerSpy()
    monkeypatch.setattr(graph_module, "sample_correlated_levels", spy)
    batch_size = 3
    contribution = make_model(copula_rho=RHO, copula_phi=PHI, copula_switch_every=1)
    env = make_env(contribution, batch_size)

    th.manual_seed(SEED)
    spy.clear()  # drop the draw `__init__`'s own reset() made
    env.reset()
    assert spy.z_prev[0] is None, "reset must not carry the previous episode"
    seen = [contribution._copula_z.clone()]
    groups = [env.state["agent_group"].clone()]
    for _ in range(11):
        env.punish(th.zeros_like(env.state["punishment"]))
        env.step()
        assert contribution._copula_z is not None
        seen.append(contribution._copula_z.clone())
        groups.append(env.state["agent_group"].clone())

    # one sampler call per round, and the latent advances on every one of them
    assert len(spy.cell_ids) == len(seen) == 12
    for r in range(1, len(seen)):
        assert seen[r].shape == (2 * batch_size,)
        assert not th.equal(seen[r], seen[r - 1]), r

    # cells follow the arrival groups: agent 0 leaves group 0 at round 4 and
    # returns at round 8, and the round's cell ids are built from that
    assert int(groups[0][0, 0]) == 0
    assert int(groups[4][0, 0]) == 1
    assert int(groups[8][0, 0]) == 0
    assert th.equal(groups[1], groups[0]), "no switch on a non-arrival round"
    assert not th.equal(groups[4], groups[0])
    for r, (cell_id, group) in enumerate(zip(spy.cell_ids, groups)):
        assert th.equal(cell_id, expected_cells(group, batch_size)), r

    # a fresh episode drops the latent before redrawing it
    spy.clear()
    env.reset()
    assert spy.z_prev[0] is None
    assert contribution._copula_z is not None
