"""Tests for the numeric (continuous-target) readout of ``GraphNetwork``.

The default ``y_encoding='onehot'`` path must stay byte-compatible: a
``y_levels``-wide readout with a softmax proba. With ``y_encoding='numeric'``
the readout collapses to a single scalar that is clamped to [0, 1] and rounded
onto the 0..y_levels-1 grid, while the (N, R, y_levels) proba shape is kept as
a degenerate one-hot so downstream consumers keep working.

Runs on Raven only (``graph.py`` imports ``torch_scatter``).
"""

import os
import tempfile

import pytest
import torch as th

from aimanager.generic.graph import GraphNetwork

N_BATCH = 2
N_PLAYER = 4
N_ROUNDS = 3
Y_LEVELS = 21

# Mirrors the reference contribution config
# (configs/training/artificial_humans/contribution/group_switching_contribution_50ep.yml)
X_ENCODING = [
    {"name": "prev_contribution", "n_levels": 21, "encoding": "numeric"},
    {"name": "prev_punishment", "n_levels": 31, "encoding": "numeric"},
    {"name": "agent_group", "n_levels": 2, "encoding": "onehot"},
]


def _make_model(y_encoding=None):
    kwargs = {} if y_encoding is None else {"y_encoding": y_encoding}
    model = GraphNetwork(
        y_levels=Y_LEVELS,
        y_name="contribution",
        hidden_size=4,
        add_rnn=True,
        add_edge_model=True,
        add_global_model=False,
        x_encoding=X_ENCODING,
        default_values={"contribution": 0},
        **kwargs,
    )
    return model.to("cpu")


def _make_data(n_batch=N_BATCH, n_player=N_PLAYER, n_rounds=N_ROUNDS):
    th.manual_seed(0)
    shape = (n_batch, n_player, n_rounds)
    return {
        "contribution": th.randint(0, 21, shape),
        "prev_contribution": th.randint(0, 21, shape),
        "prev_punishment": th.randint(0, 31, shape),
        "agent_group": th.randint(0, 2, shape),
    }


def _readout(model):
    """The op2 node readout: a plain Linear (NodeModel with activation=None)."""
    return model.op2.node_model.node_mlp


def _force_constant_readout(model, value):
    """Make the readout emit ``value`` everywhere, to probe the [0, 1] clamp."""
    lin = _readout(model)
    with th.no_grad():
        lin.weight.zero_()
        lin.bias.fill_(value)


# ── (a) default is onehot ────────────────────────────────────────────


def test_default_is_onehot_with_full_width_readout():
    model = _make_model()
    assert model.y_encoding == "onehot"
    assert model.y_encoder.size == Y_LEVELS
    assert _readout(model).out_features == Y_LEVELS

    data = _make_data()
    y_pred, y_pred_proba = model.predict(data, sample=False)
    assert y_pred.shape == (N_BATCH, N_PLAYER, N_ROUNDS)
    assert y_pred_proba.shape == (N_BATCH, N_PLAYER, N_ROUNDS, Y_LEVELS)
    assert th.allclose(
        y_pred_proba.sum(-1), th.ones_like(y_pred_proba.sum(-1)), atol=1e-5
    )
    # a real softmax, not a degenerate one-hot
    assert y_pred_proba.max() < 1.0


def test_rejects_unknown_y_encoding():
    with pytest.raises(AssertionError):
        _make_model("ordinal")


# ── (b) numeric readout width ────────────────────────────────────────


def test_numeric_readout_is_scalar():
    model = _make_model("numeric")
    assert model.y_encoding == "numeric"
    assert model.y_encoder.size == 1
    assert _readout(model).out_features == 1

    data = _make_data()
    encoded = model.encode(data, y_encode=True, device="cpu")
    assert model(encoded).shape == (N_BATCH * N_PLAYER, N_ROUNDS, 1)
    # train.py flattens both to (n_samples, width) before the MSE
    n_samples = N_BATCH * N_PLAYER * N_ROUNDS
    assert encoded["y_enc"].flatten(end_dim=-2).shape == (n_samples, 1)


# ── (c) numeric prediction contract ──────────────────────────────────


def test_numeric_predictions_match_onehot_contract():
    numeric = _make_model("numeric")
    onehot = _make_model()
    data = _make_data()

    y_pred, y_pred_proba = numeric.predict(data, sample=False)
    y_pred_oh, y_pred_proba_oh = onehot.predict(data, sample=False)

    assert y_pred.shape == y_pred_oh.shape == (N_BATCH, N_PLAYER, N_ROUNDS)
    assert y_pred_proba.shape == y_pred_proba_oh.shape
    assert y_pred.dtype == th.int64 == y_pred_oh.dtype
    assert int(y_pred.min()) >= 0
    assert int(y_pred.max()) <= Y_LEVELS - 1

    # degenerate one-hot rows that agree with y_pred
    assert th.equal(y_pred_proba.sum(-1), th.ones_like(y_pred_proba.sum(-1)))
    assert th.equal(y_pred_proba.max(-1).values, th.ones(y_pred.shape))
    assert th.equal(y_pred_proba.argmax(-1), y_pred)


@pytest.mark.parametrize("value,expected", [(1000.0, Y_LEVELS - 1), (-1000.0, 0)])
def test_numeric_clamps_extreme_readout(value, expected):
    model = _make_model("numeric")
    _force_constant_readout(model, value)
    data = _make_data()

    y_pred, y_pred_proba = model.predict(data, sample=False)
    assert y_pred.shape == (N_BATCH, N_PLAYER, N_ROUNDS)
    assert th.equal(y_pred, th.full_like(y_pred, expected))
    assert th.equal(y_pred_proba.argmax(-1), y_pred)


# ── (d) sample is a no-op ────────────────────────────────────────────


def test_numeric_sampling_is_a_no_op():
    model = _make_model("numeric")
    data = _make_data()

    y_pred_s, y_pred_proba_s = model.predict(data, sample=True)
    y_pred_d, y_pred_proba_d = model.predict(data, sample=False)

    assert th.equal(y_pred_s, y_pred_d)
    assert th.equal(y_pred_proba_s, y_pred_proba_d)


# ── (e)/(f) save / load ──────────────────────────────────────────────


def test_save_load_round_trips_numeric_y_encoding():
    model = _make_model("numeric")
    data = _make_data()
    expected, _ = model.predict(data, sample=False)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        loaded = GraphNetwork.load(path, device="cpu")

    assert loaded.y_encoding == "numeric"
    assert loaded.y_encoder.size == 1
    assert _readout(loaded).out_features == 1
    y_pred, _ = loaded.predict(data, sample=False)
    assert th.equal(y_pred, expected)


def test_checkpoint_without_y_encoding_loads_as_onehot():
    """Old artifacts predate the key, so ``load`` must default to onehot."""
    model = _make_model("numeric")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        saved = th.load(path, map_location="cpu")
        assert saved["y_encoding"] == "numeric"

        del saved["y_encoding"]
        legacy_path = os.path.join(d, "legacy.pt")
        th.save(saved, legacy_path)
        legacy = GraphNetwork.load(legacy_path, device="cpu")

    assert legacy.y_encoding == "onehot"
    assert legacy.y_encoder.size == Y_LEVELS
