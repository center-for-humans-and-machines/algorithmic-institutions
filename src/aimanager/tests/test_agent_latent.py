import os

import pytest
import torch as th

from aimanager.generic.graph import AgentLatentEncoder, GraphNetwork


AGENT_LATENT = {
    "dim": 4,
    "hidden_size": 6,
    "beta": 1.0,
    "free_bits": 0.0,
    "anneal_epochs": 1,
}


def _make_model(agent_latent=None, **kwargs):
    """Same small GNN the edge-encoder tests use, plus optional agent_latent."""
    model = GraphNetwork(
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
        default_values={},
        agent_latent=agent_latent,
        **kwargs,
    )
    return model.to("cpu")


def _make_data(n_batch=2, n_player=4, n_rounds=6):
    th.manual_seed(0)
    shape = (n_batch, n_player, n_rounds)
    return {
        "contribution": th.randint(0, 21, shape),
        "prev_contribution": th.randint(0, 21, shape),
        "agent_group": th.randint(0, 2, shape),
    }


def _node_in_features(model):
    """Width of the op1 node MLP input (x + edge + u features)."""
    return model.op1.node_model.node_mlp[0].in_features


@pytest.fixture
def latent_model():
    return _make_model(agent_latent=dict(AGENT_LATENT))


def test_disabled_latent_is_structural_no_op():
    """No agent_latent and agent_latent=None must be the very same model."""
    absent = _make_model()
    explicit_none = _make_model(agent_latent=None)

    for model in (absent, explicit_none):
        assert model.z_dim == 0
        assert model.z_encoder is None
        assert model.agent_latent is None

    sd_absent = absent.state_dict()
    sd_none = explicit_none.state_dict()
    assert list(sd_absent.keys()) == list(sd_none.keys())
    assert all(sd_absent[k].shape == sd_none[k].shape for k in sd_absent)
    assert not any("z_encoder" in k for k in sd_absent)

    assert _node_in_features(absent) == _node_in_features(explicit_none)
    # the latent model is the same network widened by exactly z_dim
    latent = _make_model(agent_latent=dict(AGENT_LATENT))
    assert latent.z_dim == 4
    assert _node_in_features(latent) == _node_in_features(absent) + 4

    data = _make_data()
    for model in (absent, explicit_none):
        model.eval()
        encoded = model.encode(data, y_encode=True, device="cpu")
        out = model(encoded)
        assert out.shape == (2 * 4, 6, 21)  # (N, n_rounds, y_levels)


def test_sample_posterior_shapes_and_zero_kl(latent_model):
    assert isinstance(latent_model.z_encoder, AgentLatentEncoder)
    # x features + one-hot y + mask channel
    assert latent_model.z_encoder.rnn.input_size == 1 + 21 + 1

    data = _make_data()
    encoded = latent_model.encode(data, y_encode=True, device="cpu")
    th.manual_seed(0)
    z, kl_per_dim = latent_model.sample_posterior(encoded)
    assert z.shape == (2 * 4, 1, 4)
    assert kl_per_dim.shape == (4,)
    assert th.isfinite(kl_per_dim).all()

    # a head that emits exactly mu=0, logvar=0 is the standard normal -> KL 0
    with th.no_grad():
        latent_model.z_encoder.head.weight.zero_()
        latent_model.z_encoder.head.bias.zero_()
    _, kl_zero = latent_model.sample_posterior(encoded)
    assert th.allclose(kl_zero, th.zeros(4), atol=1e-6)


def test_z_reaches_the_readout(latent_model):
    """Different z must produce different logits, i.e. z is really consumed."""
    latent_model.eval()
    data = _make_data()
    encoded = latent_model.encode(data, y_encode=True, device="cpu")
    n_nodes = 2 * 4

    encoded["z"] = th.zeros((n_nodes, 1, 4))
    out_zeros = latent_model(encoded)
    encoded["z"] = th.ones((n_nodes, 1, 4))
    out_ones = latent_model(encoded)

    assert out_zeros.shape == out_ones.shape == (n_nodes, 6, 21)
    assert not th.allclose(out_zeros, out_ones)


def test_prior_z_persists_within_an_episode(latent_model):
    """z is an episode-level trait: drawn on reset, reused for later rounds."""
    data = _make_data()
    th.manual_seed(0)

    latent_model.predict_independent(data, reset_rnn=False)
    z_first = latent_model._z_cache.clone()
    assert z_first.shape == (2 * 4, 1, 4)

    latent_model.predict_independent(data, reset_rnn=False)
    assert th.equal(latent_model._z_cache, z_first)

    latent_model.predict_independent(data, reset_rnn=True)
    assert not th.equal(latent_model._z_cache, z_first)


def test_save_load_round_trips_agent_latent(latent_model, tmp_path):
    path = os.path.join(tmp_path, "latent.pt")
    latent_model.eval()
    latent_model.save(path)
    loaded = GraphNetwork.load(path, device="cpu")
    loaded.eval()

    assert loaded.agent_latent == AGENT_LATENT
    assert loaded.z_dim == 4
    assert isinstance(loaded.z_encoder, AgentLatentEncoder)
    original_sd = latent_model.z_encoder.state_dict()
    loaded_sd = loaded.z_encoder.state_dict()
    assert list(original_sd.keys()) == list(loaded_sd.keys())
    for k, v in original_sd.items():
        assert th.allclose(loaded_sd[k], v)

    data = _make_data()
    encoded = latent_model.encode(data, y_encode=True, device="cpu")
    th.manual_seed(0)
    encoded["z"] = th.randn((2 * 4, 1, 4))
    assert th.allclose(latent_model(encoded), loaded(encoded))

    # legacy checkpoint: neither key present -> the __init__ defaults restore
    # a latent-free model.
    saved = th.load(path, map_location="cpu")
    del saved["agent_latent"]
    del saved["z_encoder"]
    stripped_path = os.path.join(tmp_path, "latent_stripped.pt")
    th.save(saved, stripped_path)
    stripped = GraphNetwork.load(stripped_path, device="cpu")
    assert stripped.z_dim == 0
    assert stripped.z_encoder is None
    assert stripped.agent_latent is None


def test_legacy_checkpoint_without_latent_keys_still_runs(tmp_path):
    """A genuine pre-latent artifact (un-widened op1) loads and predicts."""
    plain = _make_model()
    plain.eval()
    path = os.path.join(tmp_path, "plain.pt")
    plain.save(path)

    saved = th.load(path, map_location="cpu")
    del saved["agent_latent"]
    del saved["z_encoder"]
    legacy_path = os.path.join(tmp_path, "plain_legacy.pt")
    th.save(saved, legacy_path)

    legacy = GraphNetwork.load(legacy_path, device="cpu")
    legacy.eval()
    assert legacy.z_dim == 0
    assert legacy.z_encoder is None

    data = _make_data()
    encoded = legacy.encode(data, y_encode=True, device="cpu")
    assert legacy(encoded).shape == (2 * 4, 6, 21)
