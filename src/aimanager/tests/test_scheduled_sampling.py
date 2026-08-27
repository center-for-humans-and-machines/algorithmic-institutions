"""Scheduled-sampling curriculum for the contribution model.

The enabled path unrolls a batch round by round and feeds the model's own
sampled contribution back into `prev_contribution`; at p = 0 it must reproduce
the legacy single fused pass. Imports PyG (via GraphNetwork), so these run on
the cluster.
"""

import pytest
import torch as th

from aimanager.artificial_humans.train import (
    parse_scheduled_sampling,
    scheduled_sampling_p,
    unroll_scheduled_sampling,
)
from aimanager.generic.data import shift
from aimanager.generic.graph import GraphNetwork

C_DEFAULT = 10  # stands in for the median contribution filler
MASK_NAME = "contribution_valid"

# the reference contribution model's encoding (see
# configs/training/artificial_humans/contribution/group_switching_contribution_50ep.yml)
X_ENCODING = [
    {"name": "prev_contribution", "n_levels": 21, "encoding": "numeric"},
    {"name": "prev_punishment", "n_levels": 31, "encoding": "numeric"},
    {"name": "agent_group", "n_levels": 2, "encoding": "onehot"},
]


def _make_model(seed=0):
    th.manual_seed(seed)
    model = GraphNetwork(
        y_levels=21,
        y_name="contribution",
        hidden_size=6,
        add_rnn=True,
        add_edge_model=True,
        add_global_model=False,
        x_encoding=X_ENCODING,
        default_values={"contribution": C_DEFAULT},
    )
    model.eval()  # no dropout / batchnorm; keeps the comparison explicit
    return model.to("cpu")


def _make_data(n_batch=2, n_player=4, n_rounds=6, seed=1):
    """Tensors shaped like `create_torch_data`'s: (n_batch, n_player, n_rounds).

    Two no-input cells: their round t+1 `prev_contribution` holds the filler
    default and `prev_contribution_valid` is False there.
    """
    th.manual_seed(seed)
    shape = (n_batch, n_player, n_rounds)
    contribution = th.randint(0, 21, shape)
    punishment = th.randint(0, 31, shape)
    contribution_valid = th.ones(shape, dtype=th.bool)
    contribution_valid[0, 0, 1] = False
    contribution_valid[-1, -1, 3] = False
    contribution = th.where(
        contribution_valid, contribution, th.full_like(contribution, C_DEFAULT)
    )
    return {
        "contribution": contribution,
        "punishment": punishment,
        MASK_NAME: contribution_valid,
        "agent_group": th.randint(0, 2, shape),
        "prev_contribution": shift(contribution, C_DEFAULT),
        "prev_punishment": shift(punishment, 0),
        "prev_contribution_valid": shift(contribution_valid, False),
    }


def _unroll(model, data, p):
    n_batch, n_player, _ = data["contribution"].shape
    edge_index = model.create_fully_connected(n_player, n_batch=n_batch)
    return unroll_scheduled_sampling(
        model,
        data,
        mask_name=MASK_NAME,
        edge_index=edge_index,
        device="cpu",
        p=p,
    )


def test_p_zero_reproduces_the_single_pass():
    """p = 0: same inputs bit for bit, same logits up to stepwise-GRU noise."""
    model = _make_model()
    data = _make_data()
    n_batch, n_player, _ = data["contribution"].shape
    edge_index = model.create_fully_connected(n_player, n_batch=n_batch)
    legacy_enc = model.encode(data, mask=MASK_NAME, edge_index=edge_index, device="cpu")
    with th.no_grad():
        legacy_logit = model(legacy_enc)

    out = _unroll(model, data, p=0.0)

    assert th.equal(out["x"], legacy_enc["x"])
    assert th.equal(out["prev_contribution"], data["prev_contribution"])
    assert out["y_logit"].shape == legacy_logit.shape
    assert th.allclose(out["y_logit"], legacy_logit, atol=1e-5)
    assert out["n_substituted"] == 0


def test_p_one_substitutes_every_valid_cell():
    model = _make_model()
    data = _make_data()
    original = data["prev_contribution"].clone()
    n_rounds = data["contribution"].shape[2]

    out = _unroll(model, data, p=1.0)

    prev_valid = data["prev_contribution_valid"]
    substituted = out["prev_contribution"]
    samples = out["samples"]
    # round 0 has no preceding sample, so it is never touched
    assert th.equal(substituted[:, :, 0], original[:, :, 0])
    for t in range(1, n_rounds):
        valid = prev_valid[:, :, t]
        assert th.equal(substituted[:, :, t][valid], samples[:, :, t - 1][valid])
        # invalid cells keep the no-input filler
        assert th.equal(substituted[:, :, t][~valid], original[:, :, t][~valid])
    n_valid = int(prev_valid[:, :, 1:].sum())
    assert out["n_eligible"] == n_valid
    assert out["n_substituted"] == n_valid
    # the source batch is never mutated
    assert th.equal(data["prev_contribution"], original)


def test_realized_rate_matches_p():
    model = _make_model()
    th.manual_seed(7)
    n_substituted = 0
    n_eligible = 0
    for rep in range(10):
        data = _make_data(n_batch=6, n_player=8, seed=100 + rep)
        out = _unroll(model, data, p=0.5)
        n_substituted += out["n_substituted"]
        n_eligible += out["n_eligible"]
    assert n_eligible > 2000
    assert n_substituted / n_eligible == pytest.approx(0.5, abs=0.05)


def test_sample_carries_no_gradient():
    model = _make_model()
    data = _make_data()

    out = _unroll(model, data, p=0.5)

    for key in ("samples", "prev_contribution"):
        assert out[key].dtype == th.int64
        assert not out[key].requires_grad
        assert out[key].grad_fn is None

    # the loss expression of train.py, one backward
    y_logit = out["y_logit"].flatten(end_dim=-2)
    y_true = out["batch_data"]["y_enc"].flatten(end_dim=-2)
    mask = out["batch_data"]["mask"].flatten()
    loss_fn = th.nn.CrossEntropyLoss(reduction="none")
    loss = (loss_fn(y_logit, y_true) * mask).sum() / mask.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(th.isfinite(g).all() for g in grads)


def test_schedule_values():
    kwargs = {"p_max": 0.5, "ramp_start_epoch": 86, "ramp_end_epoch": 345}
    assert scheduled_sampling_p(0, **kwargs) == 0.0
    assert scheduled_sampling_p(85, **kwargs) == 0.0
    assert scheduled_sampling_p(86, **kwargs) == 0.0
    assert scheduled_sampling_p(215, **kwargs) == pytest.approx(
        0.5 * (215 - 86) / (345 - 86)
    )
    assert scheduled_sampling_p(345, **kwargs) == 0.5
    assert scheduled_sampling_p(574, **kwargs) == 0.5


def _model_args(x_encoding):
    return {"y_name": "contribution", "x_encoding": x_encoding}


def test_parse_defaults_and_absent_key():
    assert parse_scheduled_sampling({}, _model_args(X_ENCODING)) is None
    parsed = parse_scheduled_sampling(
        {"scheduled_sampling": {"p_max": 0.25}}, _model_args(X_ENCODING)
    )
    assert parsed == {
        "p_max": 0.25,
        "ramp_start_epoch": 86,
        "ramp_end_epoch": 345,
    }


def test_guard_rejects_contribution_derived_feature():
    x_encoding = X_ENCODING + [
        {"name": "own_grp_prev_mean_contr", "etype": "float", "norm": 20.0}
    ]
    with pytest.raises(ValueError, match="own_grp_prev_mean_contr"):
        parse_scheduled_sampling(
            {"scheduled_sampling": {"p_max": 0.5}}, _model_args(x_encoding)
        )


def test_parse_rejects_missing_p_max_and_missing_prev_contribution():
    with pytest.raises(ValueError, match="p_max"):
        parse_scheduled_sampling(
            {"scheduled_sampling": {"ramp_start_epoch": 10}},
            _model_args(X_ENCODING),
        )
    with pytest.raises(ValueError, match="prev_contribution"):
        parse_scheduled_sampling(
            {"scheduled_sampling": {"p_max": 0.5}},
            _model_args([{"name": "agent_group", "n_levels": 2, "encoding": "onehot"}]),
        )
