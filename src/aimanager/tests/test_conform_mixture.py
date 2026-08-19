import math
import os
import tempfile

import pytest
import torch as th

from aimanager.generic.graph import GraphNetwork

HIDDEN_SIZE = 4
N_BATCH = 2
N_PLAYER = 4
N_ROUNDS = 3
N_NODES = N_BATCH * N_PLAYER

# x_encoding of the reference contribution config
X_ENCODING = [
    {"name": "prev_contribution", "n_levels": 21, "encoding": "numeric"},
    {"name": "prev_punishment", "n_levels": 31, "encoding": "numeric"},
    {"name": "agent_group", "n_levels": 2, "encoding": "onehot"},
]


def _make_model(seed=0, **kwargs):
    th.manual_seed(seed)
    model = GraphNetwork(
        y_levels=21,
        y_name="contribution",
        hidden_size=HIDDEN_SIZE,
        add_rnn=True,
        add_edge_model=True,
        add_global_model=False,
        x_encoding=X_ENCODING,
        edge_encoding=[],
        default_values={},
        **kwargs,
    )
    return model.to("cpu")


def _make_data(m=None):
    th.manual_seed(7)
    shape = (N_BATCH, N_PLAYER, N_ROUNDS)
    if m is None:
        m = th.rand(shape) * 20.0
    return {
        "contribution": th.randint(0, 21, shape),
        "prev_contribution": th.randint(0, 21, shape),
        "prev_punishment": th.randint(0, 31, shape),
        "agent_group": th.randint(0, 2, shape),
        "own_grp_prev_mean_contr": m,
    }


def test_off_flag_is_bit_identical():
    """Default build and an explicit conform_mixture=False build agree exactly,
    and the flag's own modules are created after all base modules so the base
    parameters draw the same RNG values as an on-flag build."""
    base = _make_model()
    off = _make_model(conform_mixture=False)
    on = _make_model(conform_mixture=True)

    assert base.conform_mixture is False
    assert base.conform_gate is None and base.conform_log_sigma is None
    assert list(base.state_dict().keys()) == list(off.state_dict().keys())

    data = _make_data()
    out_base = base(base.encode(data, device="cpu"))
    out_off = off(off.encode(data, device="cpu"))
    assert th.equal(out_base, out_off)
    assert out_base.shape == (N_NODES, N_ROUNDS, 21)

    # base parameters unchanged by the extra modules (RNG order preserved)
    on_state = on.state_dict()
    for k, v in base.state_dict().items():
        assert th.equal(v, on_state[k]), k
    assert set(on_state) - set(base.state_dict()) == {
        "conform_log_sigma",
        "conform_gate.weight",
        "conform_gate.bias",
    }
    assert float(on.conform_log_sigma) == pytest.approx(math.log(2.0))
    assert on.conform_gate.bias.item() == pytest.approx(-2.0)


def test_on_flag_returns_normalized_log_probs():
    model = _make_model(conform_mixture=True)
    data = _make_data()
    encoded = model.encode(data, device="cpu")
    assert encoded["m"].shape == (N_NODES, N_ROUNDS)
    out = model(encoded)
    assert out.shape == (N_NODES, N_ROUNDS, 21)
    total = out.exp().sum(-1)
    assert th.allclose(total, th.ones_like(total), atol=1e-5)
    renorm = th.log_softmax(out, -1)
    assert th.allclose(renorm, out, atol=1e-6)


def test_missing_feature_raises():
    model = _make_model(conform_mixture=True)
    data = _make_data()
    del data["own_grp_prev_mean_contr"]
    with pytest.raises(KeyError):
        model.encode(data, device="cpu")


def test_gradients_reach_gate_and_log_sigma():
    model = _make_model(conform_mixture=True)
    data = _make_data()
    encoded = model.encode(data, device="cpu")
    out = model(encoded)
    y = data["contribution"].flatten(0, 1).unsqueeze(-1)
    loss = -out.gather(-1, y).mean()
    loss.backward()
    for p in (
        model.conform_gate.weight,
        model.conform_gate.bias,
        model.conform_log_sigma,
    ):
        assert p.grad is not None
        assert th.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0


def test_save_load_round_trip_and_legacy_checkpoint():
    model = _make_model(conform_mixture=True)
    # perturb so defaults cannot mask a failed round trip
    with th.no_grad():
        model.conform_log_sigma.fill_(math.log(3.5))
        model.conform_gate.weight.add_(0.25)
    data = _make_data()
    out = model(model.encode(data, device="cpu"))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)

        loaded = GraphNetwork.load(path, device="cpu")
        assert loaded.conform_mixture is True
        assert loaded.conform_feature == "own_grp_prev_mean_contr"
        assert float(loaded.conform_log_sigma) == pytest.approx(math.log(3.5))
        assert th.equal(loaded.conform_gate.weight, model.conform_gate.weight)
        assert th.equal(loaded.conform_gate.bias, model.conform_gate.bias)
        assert th.equal(loaded(loaded.encode(data, device="cpu")), out)

        # legacy artifact: saved before the mixture existed -> flag off, and the
        # forward must be the plain categorical readout.
        saved = th.load(path, map_location="cpu")
        for k in (
            "conform_mixture",
            "conform_feature",
            "conform_gate",
            "conform_log_sigma",
        ):
            del saved[k]
        legacy_path = os.path.join(d, "legacy.pt")
        th.save(saved, legacy_path)
        legacy = GraphNetwork.load(legacy_path, device="cpu")

    assert legacy.conform_mixture is False
    assert legacy.conform_gate is None and legacy.conform_log_sigma is None
    model.conform_mixture = False  # same base modules, categorical path only
    expected = model(model.encode(data, device="cpu"))
    assert th.equal(legacy(legacy.encode(data, device="cpu")), expected)


def test_shift_sensitivity_towards_group_mean():
    """With a narrow conform width and the gate saturated on, the mode of the
    mixture is the level closest to the own-group previous mean."""
    model = _make_model(conform_mixture=True)
    with th.no_grad():
        model.conform_log_sigma.fill_(math.log(0.05))
        model.conform_gate.bias.fill_(20.0)
        model.conform_gate.weight.zero_()

    m_values = [0.0, 3.2, 7.9, 12.8, 19.9]
    for m_val in m_values:
        m = th.full((N_BATCH, N_PLAYER, N_ROUNDS), m_val)
        data = _make_data(m=m)
        out = model(model.encode(data, device="cpu"))
        expected = int(round(m_val))
        assert (out.argmax(-1) == expected).all(), m_val
