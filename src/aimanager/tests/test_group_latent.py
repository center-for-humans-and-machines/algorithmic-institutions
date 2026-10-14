"""Tests for the shared per-(group, episode) contribution latent.

Covers the five properties the mechanism has to have: the disabled path is a
bit-identical no-op, the one-forward quadrature loss equals a brute-force
20-full-forward integral, a zero loading reduces the marginal likelihood to the
plain one, the simulation draw is episode-persistent and slot-attached, and the
artifact round-trips (including legacy artifacts that predate the feature).
"""

import os

import pytest
import torch as th

from aimanager.generic.graph import GraphNetwork, gauss_hermite_normal


GROUP_LATENT = {
    "dim": 1,
    "n_quadrature": 20,
    "pathway": "logit_skip",
    "loading_init": 0.1,
    "n_groups": 2,
}

REFERENCE_ARTIFACT = os.path.join(
    "artifacts",
    "artificial_humans",
    "group_switching_contribution_50ep",
    "model",
    "architecture_node+edge+rnn__dataset_50ep__epochs_575.pt",
)


def _make_model(group_latent=None, seed=0, **kwargs):
    """The small GNN the edge-encoder tests use, plus optional group_latent."""
    th.manual_seed(seed)
    model = GraphNetwork(
        y_levels=21,
        y_name="contribution",
        hidden_size=4,
        add_rnn=True,
        add_edge_model=True,
        add_global_model=False,
        x_encoding=[
            {"name": "prev_contribution", "n_levels": 21, "encoding": "numeric"},
            {"name": "agent_group", "n_levels": 2, "encoding": "onehot"},
        ],
        edge_encoding=[],
        default_values={"contribution": 10},
        group_latent=group_latent,
        **kwargs,
    )
    return model.to("cpu")


def _make_data(n_batch=2, n_player=4, n_rounds=6, seed=1):
    th.manual_seed(seed)
    shape = (n_batch, n_player, n_rounds)
    return {
        "contribution": th.randint(0, 21, shape),
        "prev_contribution": th.randint(0, 21, shape),
        "agent_group": th.randint(0, 2, shape),
        "contribution_valid": th.rand(shape) > 0.2,
    }


def _encode(model, data):
    return model.encode(data, mask="contribution_valid", device="cpu")


def _strip_latent_keys(path, out_path):
    """Rewrite an artifact without the two group-latent keys, i.e. exactly what
    every artifact trained before this change looks like on disk."""
    saved = th.load(path, map_location="cpu")
    saved.pop("group_latent", None)
    saved.pop("group_latent_loading", None)
    th.save(saved, out_path)
    return out_path


@pytest.fixture
def latent_model():
    return _make_model(group_latent=dict(GROUP_LATENT))


# --- (a) disabled parity ----------------------------------------------------


def test_disabled_path_is_a_bit_identical_no_op(tmp_path):
    """A legacy artifact must load latent-free and give bit-identical logits."""
    plain = _make_model()
    plain.eval()
    assert plain.group_latent is None
    assert plain.group_latent_loading is None

    sd = plain.state_dict()
    assert not any("group_latent" in k for k in sd)
    assert not any("group_latent" in n for n, _ in plain.named_parameters())

    path = os.path.join(tmp_path, "plain.pt")
    plain.save(path)
    legacy_path = _strip_latent_keys(path, os.path.join(tmp_path, "legacy.pt"))

    legacy = GraphNetwork.load(legacy_path, device="cpu")
    legacy.eval()
    assert legacy.group_latent is None
    assert legacy.group_latent_loading is None
    # (the load path registers submodules in a different order than fresh
    # construction, so compare as sets -- that is true on main as well)
    legacy_sd = legacy.state_dict()
    assert set(legacy_sd.keys()) == set(sd.keys())
    assert all(th.equal(legacy_sd[k], sd[k]) for k in sd)

    data = _make_data()
    encoded = _encode(plain, data)
    with th.no_grad():
        assert th.equal(plain(encoded), legacy(encoded))

    # a "z" in the dict is ignored outright when the model has no latent
    with th.no_grad():
        rogue = {**encoded, "z": th.ones(encoded["mask"].shape)}
        assert th.equal(plain(rogue), plain(encoded))

    # and the sampling path consumes no extra RNG draws
    th.manual_seed(7)
    first = plain.predict_independent(data)[0]
    th.manual_seed(7)
    assert th.equal(first, plain.predict_independent(data)[0])


@pytest.mark.skipif(
    not os.path.exists(REFERENCE_ARTIFACT)
    or os.path.getsize(REFERENCE_ARTIFACT) < 1024,
    reason="reference artifact absent or an un-pulled git-lfs pointer",
)
def test_real_reference_artifact_loads_latent_free():
    model = GraphNetwork.load(REFERENCE_ARTIFACT, device="cpu")
    assert model.group_latent is None
    assert model.group_latent_loading is None
    assert not any("group_latent" in k for k in model.state_dict())


# --- (b) quadrature loss vs brute force ------------------------------------


def test_quadrature_loss_equals_brute_force_integral(latent_model):
    """The one-forward logit-shift loss must equal 20 independent forwards."""
    latent_model.eval()
    data = _make_data()
    encoded = _encode(latent_model, data)
    cells = latent_model.group_latent_cells(data["agent_group"], device="cpu")
    nodes, log_w = gauss_hermite_normal(latent_model.n_quadrature)
    assert nodes.shape == (20,)
    assert th.allclose(log_w.exp().sum(), th.tensor(1.0), atol=1e-6)

    mask = encoded["mask"]
    y_true = encoded["y_enc"].flatten(0, 1)
    n_cells = int(cells.max().item()) + 1
    assert n_cells == 2 * 2  # (episode, group)

    with th.no_grad():
        fast = latent_model.group_latent_nll(encoded, cells, nodes, log_w)

        per_cell = th.zeros((len(nodes), n_cells))
        for k, z_k in enumerate(nodes):
            # a genuinely separate full forward pass per node
            logits = latent_model({**encoded, "z": th.full(mask.shape, float(z_k))})
            logp = (th.log_softmax(logits, dim=-1) * y_true).sum(-1) * mask
            for c in range(n_cells):
                per_cell[k, c] = logp[cells == c].sum()
        slow = -th.logsumexp(log_w.view(-1, 1) + per_cell, dim=0).sum() / mask.sum()

    assert th.allclose(fast, slow, atol=1e-4)


def test_loading_receives_gradient(latent_model):
    """The loss must actually train v (and only v under freeze_base)."""
    data = _make_data()
    encoded = _encode(latent_model, data)
    cells = latent_model.group_latent_cells(data["agent_group"], device="cpu")
    nodes, log_w = gauss_hermite_normal(latent_model.n_quadrature)

    for name, param in latent_model.named_parameters():
        param.requires_grad = name == "group_latent_loading"
    latent_model.group_latent_nll(encoded, cells, nodes, log_w).backward()

    grad = latent_model.group_latent_loading.grad
    assert grad is not None and grad.abs().sum() > 0
    for name, param in latent_model.named_parameters():
        if name != "group_latent_loading":
            assert param.grad is None


# --- (c) zero loading collapses to the plain likelihood --------------------


def test_zero_loading_matches_plain_likelihood(latent_model):
    latent_model.eval()
    with th.no_grad():
        latent_model.group_latent_loading.zero_()

    data = _make_data()
    encoded = _encode(latent_model, data)
    cells = latent_model.group_latent_cells(data["agent_group"], device="cpu")
    nodes, log_w = gauss_hermite_normal(latent_model.n_quadrature)

    with th.no_grad():
        marginal = latent_model.group_latent_nll(encoded, cells, nodes, log_w)

        logits = latent_model(encoded).flatten(end_dim=-2)
        y_true = encoded["y_enc"].flatten(end_dim=-2)
        mask = encoded["mask"].flatten()
        ce = th.nn.CrossEntropyLoss(reduction="none")(logits, y_true)
        plain = (ce * mask).sum() / mask.sum()

    assert th.allclose(marginal, plain, atol=1e-5)


# --- (d) simulation-time z -------------------------------------------------


def test_sim_z_persists_across_rounds_and_varies(latent_model):
    """z is drawn once per (group, episode) and held for the whole episode."""
    data = _make_data(n_batch=2, n_player=4, n_rounds=1)
    th.manual_seed(0)

    latent_model.predict_independent(data, reset_rnn=True)
    z_first = latent_model._group_z_cache.clone()
    assert z_first.shape == (2, 2)  # (episode, group slot)

    for _ in range(3):
        latent_model.predict_independent(data, reset_rnn=False)
        assert th.equal(latent_model._group_z_cache, z_first)

    # a new episode (reset_rnn) redraws
    latent_model.predict_independent(data, reset_rnn=True)
    assert not th.equal(latent_model._group_z_cache, z_first)

    # and the draws differ between the two group slots and the two episodes
    assert z_first[0, 0] != z_first[0, 1]
    assert not th.equal(z_first[0], z_first[1])


def test_z_is_attached_to_the_group_slot_not_the_member(latent_model):
    """An agent that switches groups comes under the new group's z."""
    # one episode, two agents, three rounds; agent 0 moves 0 -> 1 at round 2
    agent_group = th.tensor([[[0, 0, 1], [1, 1, 1]]])
    z = th.tensor([[-1.5, 2.5]])

    got = latent_model.group_latent_gather(agent_group, z, device="cpu")
    assert th.equal(got, th.tensor([[-1.5, -1.5, 2.5], [2.5, 2.5, 2.5]]))

    cells = latent_model.group_latent_cells(agent_group, device="cpu")
    assert th.equal(cells, th.tensor([[0, 0, 1], [1, 1, 1]]))

    # two episodes get disjoint cell ids
    two = th.cat([agent_group, agent_group], dim=0)
    assert th.equal(
        latent_model.group_latent_cells(two, device="cpu"),
        th.tensor([[0, 0, 1], [1, 1, 1], [2, 2, 3], [3, 3, 3]]),
    )


def test_z_reaches_the_logits(latent_model):
    latent_model.eval()
    data = _make_data()
    encoded = _encode(latent_model, data)
    shape = encoded["mask"].shape
    with th.no_grad():
        out_zero = latent_model({**encoded, "z": th.zeros(shape)})
        out_one = latent_model({**encoded, "z": th.ones(shape)})
        out_absent = latent_model(encoded)
    assert th.equal(out_absent, out_zero)
    assert not th.allclose(out_zero, out_one)


# --- (e) save / load -------------------------------------------------------


def test_save_load_round_trips_group_latent(latent_model, tmp_path):
    latent_model.eval()
    path = os.path.join(tmp_path, "latent.pt")
    latent_model.save(path)

    loaded = GraphNetwork.load(path, device="cpu")
    loaded.eval()
    assert loaded.group_latent == GROUP_LATENT
    assert loaded.n_latent_groups == 2
    assert loaded.n_quadrature == 20
    assert th.equal(loaded.group_latent_loading, latent_model.group_latent_loading)
    assert "group_latent_loading" in loaded.state_dict()

    data = _make_data()
    encoded = _encode(latent_model, data)
    z = th.randn(encoded["mask"].shape)
    with th.no_grad():
        assert th.equal(latent_model({**encoded, "z": z}), loaded({**encoded, "z": z}))

    # stripping the keys yields a legacy artifact: loads latent-free
    legacy_path = _strip_latent_keys(path, os.path.join(tmp_path, "legacy.pt"))
    legacy = GraphNetwork.load(legacy_path, device="cpu")
    assert legacy.group_latent is None
    assert legacy.group_latent_loading is None
    assert "group_latent_loading" not in legacy.state_dict()


def test_loading_init_is_a_non_degenerate_ramp():
    """A constant v is invisible to the softmax and v == 0 is a stationary
    point of the marginal likelihood, so the init must be a ramp."""
    model = _make_model(group_latent={**GROUP_LATENT, "loading_init": 0.25})
    v = model.group_latent_loading.detach()
    assert v.shape == (21,)
    assert th.allclose(v[0], th.tensor(-0.25))
    assert th.allclose(v[-1], th.tensor(0.25))
    assert th.allclose(v.mean(), th.tensor(0.0), atol=1e-6)
