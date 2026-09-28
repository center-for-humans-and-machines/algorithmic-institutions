"""Tests for the herding-copula dispatch in `GraphNetwork` (plan step 14).

Imports torch_geometric via `aimanager.generic.graph`, so it runs on Raven:
    scripts/remote_test.sh -- -k switch_copula_graph

The gate this file owns is the LEGACY path: an artifact without `copula_rho`
must sample exactly as before, values and torch RNG consumption alike. The
sampler itself is covered locally by tests/switch/test_switch_copula.py.
Context: notes/autoresearch_log/switch-herding-copula.md.
"""

import os
import tempfile

import numpy as np
import pytest
import torch as th

from aimanager.generic.graph import GraphNetwork
from aimanager.manager.environment import ArtificialHumanEnv

SEED = 20260812
N_AGENTS = 8
GROUPS = [0, 0, 0, 0, 1, 1, 1, 1]
RHO = 0.3
PHI = 0.7
N_REPEATS = 2000


def make_model(seed=0, **copula):
    """Small switch-shaped model (2 levels, RNN + edge model, no global)."""
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
        "does_switch": th.zeros(shape, dtype=th.int64),
        "prev_contribution": th.randint(0, 21, shape),
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


class _OnesContribution:
    """Stand-in contribution AH, so the env can step without a second GNN."""

    default_values = {"contribution": 0}

    def predict(self, state, reset_rnn, edge_index):
        return (th.ones_like(state["contribution"]),)


def run_seeded(fn):
    """Return (result, next 5 draws) with the RNG seeded before `fn`."""
    th.manual_seed(SEED)
    out = fn()
    return out, th.randn(5)


# --------------------------------------------------------------------------- #
# 1. the legacy path is untouched
# --------------------------------------------------------------------------- #
def test_default_model_matches_pre_change_path_bitwise():
    model = make_model()
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

    assert th.equal(new_pred, ref_pred)
    assert th.equal(new_proba, ref_proba)
    # same post-call RNG state -> the legacy path consumes exactly as before
    assert th.equal(new_rng, ref_rng)


# --------------------------------------------------------------------------- #
# 2. the copula path keeps marginals and correlates within the group only
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def copula_draws():
    """`N_REPEATS` sampled rounds from the copula model, plus the marginals of
    the independent path on the same weights and data."""
    n_batch = 4
    copula_model, legacy_model = paired_models(copula_rho=RHO)
    data = make_data(n_batch=n_batch)
    edge_index = copula_model.create_fully_connected(N_AGENTS, n_batch=n_batch)

    _, proba = legacy_model.predict_independent(
        data, sample=False, edge_index=edge_index
    )
    p_switch = proba[..., 1].detach().reshape(-1).numpy()
    # keeps the binomial SE and the correlations well defined
    assert np.all((p_switch > 0.05) & (p_switch < 0.95)), p_switch

    th.manual_seed(SEED)
    draws = np.empty((N_REPEATS, n_batch * N_AGENTS), dtype=np.int64)
    for i in range(N_REPEATS):
        y_pred, y_proba = copula_model.predict_independent(
            data, sample=True, edge_index=edge_index
        )
        # the copula must not touch the marginals it inverts
        assert th.equal(y_proba, proba)
        assert y_pred.shape == (n_batch, N_AGENTS, 1)
        draws[i] = y_pred.reshape(-1).numpy()
    cell = np.repeat(np.arange(n_batch), N_AGENTS) * 2 + np.tile(GROUPS, n_batch)
    return draws, p_switch, cell


def test_copula_preserves_per_agent_marginals(copula_draws):
    draws, p_switch, _ = copula_draws
    freq = draws.mean(axis=0)
    se = np.sqrt(p_switch * (1.0 - p_switch) / N_REPEATS)
    assert np.all(np.abs(freq - p_switch) < 5.0 * se), np.max(
        np.abs(freq - p_switch) / se
    )


def test_copula_correlates_within_cell_only(copula_draws):
    draws, _, cell = copula_draws
    c = np.corrcoef(draws.T)
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
    model = make_model(copula_rho=RHO, copula_phi=PHI, copula_switch_every=4)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        loaded = GraphNetwork.load(path, device="cpu")
        assert loaded.copula_rho == RHO
        assert loaded.copula_phi == PHI
        assert loaded.copula_switch_every == 4

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
# 4. AR(1) latent lifecycle
# --------------------------------------------------------------------------- #
def test_ar_latent_advances_only_on_decision_rounds():
    n_batch, switch_every = 2, 4
    model = make_model(copula_rho=RHO, copula_phi=PHI, copula_switch_every=switch_every)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=n_batch)

    th.manual_seed(SEED)
    seen = []
    for r in range(8):
        data = make_data(n_batch=n_batch, round_number=r)
        model.predict_independent(
            data, sample=True, reset_rnn=(r == 0), edge_index=edge_index
        )
        z = model._copula_z
        seen.append(None if z is None else z.clone())

    # rounds 0-2 leave the latent unset; round 3 (a decision round) sets it
    assert all(z is None for z in seen[:3])
    assert seen[3] is not None
    assert seen[3].shape == (2 * n_batch,)
    assert seen[3].dtype == th.float64
    # rounds 4-6 still draw, but must not overwrite the decision-round latent
    for r in (4, 5, 6):
        assert th.equal(seen[r], seen[3]), r
    # round 7 is the next decision round
    assert not th.equal(seen[7], seen[3])

    # a fresh episode (reset_rnn) drops the latent, as the GRU state is dropped
    model.predict_independent(
        make_data(n_batch=n_batch, round_number=0),
        sample=True,
        reset_rnn=True,
        edge_index=edge_index,
    )
    assert model._copula_z is None


def test_env_drives_the_copula_predictor():
    """The only simulation call site: `_run_switch_predictor` every round, one
    round per call, cells from the env's own pre-switch `agent_group`."""
    batch_size = 3
    switch = make_model(copula_rho=RHO, copula_phi=PHI, copula_switch_every=4)
    env = ArtificialHumanEnv(
        artifical_humans=_OnesContribution(),
        artifical_humans_valid=None,
        artifical_humans_switch=switch,
        switch_every=4,
        batch_size=batch_size,
        n_agents=N_AGENTS,
        n_contributions=21,
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
    th.manual_seed(SEED)
    env.reset()
    first_groups = env.state["agent_group"].clone()
    seen = []
    for _ in range(11):
        env.punish(th.zeros_like(env.state["punishment"]))
        env.step()
        z = switch._copula_z
        seen.append(None if z is None else z.clone())

    assert all(z is None for z in seen[:3])
    assert seen[3].shape == (2 * batch_size,)
    assert all(th.equal(seen[r], seen[3]) for r in (4, 5, 6))
    assert not th.equal(seen[7], seen[3])
    assert all(th.equal(seen[r], seen[7]) for r in (8, 9, 10))
    # the drawn decisions reached the membership
    assert not th.equal(env.state["agent_group"], first_groups)


def test_copula_rho_only_path_is_stateless():
    model = make_model(copula_rho=RHO)
    edge_index = model.create_fully_connected(N_AGENTS, n_batch=2)
    for r in range(5):
        model.predict_independent(
            make_data(n_batch=2, round_number=r),
            sample=True,
            reset_rnn=(r == 0),
            edge_index=edge_index,
        )
        assert model._copula_z is None


# --------------------------------------------------------------------------- #
# 5. guards and the unsampled path
# --------------------------------------------------------------------------- #
def test_phi_requires_switch_every():
    with pytest.raises(AssertionError, match="copula_switch_every"):
        make_model(copula_rho=RHO, copula_phi=PHI)


def test_rho_only_allowed_for_switch_and_contribution_heads():
    """The head gate opened to the contribution head
    (notes/autoresearch_log/contribution-herding-copula-v2.md, step 3);
    every other head must still reject rho > 0, so the validity model and
    any future head keep the legacy draw."""
    th.manual_seed(0)
    contribution = GraphNetwork(
        y_levels=21,
        y_name="contribution",
        hidden_size=4,
        add_rnn=True,
        add_edge_model=True,
        add_global_model=False,
        x_encoding=[
            {"name": "prev_contribution", "n_levels": 21, "encoding": "numeric"}
        ],
        edge_encoding=[],
        default_values={"contribution": 0},
        copula_rho=RHO,
    )
    assert contribution.copula_rho == RHO

    with pytest.raises(AssertionError, match="does_switch"):
        th.manual_seed(0)
        GraphNetwork(
            y_levels=2,
            y_name="contribution_valid",
            hidden_size=4,
            x_encoding=[],
            default_values={},
            copula_rho=RHO,
        )


@pytest.mark.parametrize("rho", [-0.01, 1.0])
def test_rejects_out_of_range_rho(rho):
    with pytest.raises(AssertionError, match="copula_rho"):
        make_model(copula_rho=rho)


def test_unsampled_path_ignores_copula():
    copula_model, legacy_model = paired_models(copula_rho=RHO)
    data = make_data(n_batch=2)
    edge_index = copula_model.create_fully_connected(N_AGENTS, n_batch=2)
    with_copula = copula_model.predict_independent(
        data, sample=False, edge_index=edge_index
    )
    without = legacy_model.predict_independent(
        data, sample=False, edge_index=edge_index
    )
    assert th.equal(with_copula[0], without[0])
    assert th.equal(with_copula[1], without[1])
    assert copula_model._copula_z is None
