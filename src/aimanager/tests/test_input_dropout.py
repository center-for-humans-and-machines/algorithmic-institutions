"""Tests for training-time input dropout (dropout_input_features).

Runs on Raven via scripts/remote_test.sh: importing the training module pulls
in the AH model registry, which imports PyG.
"""

import torch as th

from aimanager.artificial_humans.train import dropout_input_features

DEFAULTS = {"contribution": 10.0, "own_grp_prev_mean_contr": 10.0}


def make_data():
    return {
        "prev_contribution": th.full((6, 4, 24), 3, dtype=th.int64),
        "own_grp_prev_mean_contr": th.full((6, 4, 24), 7.5, dtype=th.float),
        "contribution": th.full((6, 4, 24), 5, dtype=th.int64),
    }


def test_rate_one_masks_everything_with_base_default():
    data = make_data()
    out = dropout_input_features(data, {"prev_contribution": 1.0}, DEFAULTS)
    # prev_ features inherit the base feature's default, as shift() does
    assert (out["prev_contribution"] == 10).all()
    assert out["prev_contribution"].dtype == th.int64


def test_rate_zero_is_identity():
    data = make_data()
    out = dropout_input_features(data, {"prev_contribution": 0.0}, DEFAULTS)
    assert (out["prev_contribution"] == 3).all()


def test_masked_fraction_matches_rate():
    th.random.manual_seed(0)
    data = {"prev_contribution": th.full((100, 8, 24), 3, dtype=th.int64)}
    out = dropout_input_features(data, {"prev_contribution": 0.3}, DEFAULTS)
    frac = (out["prev_contribution"] == 10).float().mean().item()
    assert abs(frac - 0.3) < 0.02


def test_direct_feature_uses_own_default():
    data = make_data()
    out = dropout_input_features(data, {"own_grp_prev_mean_contr": 1.0}, DEFAULTS)
    assert (out["own_grp_prev_mean_contr"] == 10.0).all()
    assert out["own_grp_prev_mean_contr"].dtype == th.float


def test_other_features_and_input_untouched():
    data = make_data()
    out = dropout_input_features(data, {"prev_contribution": 1.0}, DEFAULTS)
    # untouched keys pass through
    assert (out["contribution"] == 5).all()
    assert (out["own_grp_prev_mean_contr"] == 7.5).all()
    # the input dict and its tensors are not mutated
    assert (data["prev_contribution"] == 3).all()
