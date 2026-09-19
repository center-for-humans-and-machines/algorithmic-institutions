"""The artificial punisher at round t must see round t's contribution.

The human manager punishes the contribution just made (corr(p_t, c_t) =
-0.28 vs corr(p_t, c_{t-1}) = -0.19); both punisher families used to read
`prev_contribution` only. A two-round history with c_0 != c_1 pins which
round reaches the model input when round 1 is punished, on both paths:

  * linear adapter (`LinearAHAdapter.get_punishments`, rounds-driven) and
    the feature-legality rule -- CPU torch, no PyG, run locally too;
  * GNN (`api_manager.create_data` -> the config's input `Encoder`) --
    imports torch_scatter, so Raven (`scripts/remote_test.sh`).
"""

import numpy as np
import pytest

C0 = [5, 10, 20, 0, 7, 14, 3, 18]
C1 = [20, 0, 5, 15, 12, 1, 9, 6]
P0 = [2, 0, 0, 5, 1, 0, 4, 0]
AGENT_GROUP = [0, 0, 0, 0, 1, 1, 1, 1]
DEFAULTS = {
    "punishment": 0,
    "contribution": 9,
    "contribution_valid": False,
    "punishment_valid": False,
    "recorded": False,
    "common_good": 12.333333333333334,
    "agent_group": 0,
}
FEATS = ["contribution", "prev_contribution", "prev_punishment", "round_number"]


def _round(t, contributions, punishments):
    return {
        "contribution": contributions,
        "contribution_valid": [True] * 8,
        "punishment": punishments,
        "punishment_valid": [p is not None for p in punishments],
        "agent_group": AGENT_GROUP,
        "group": ["a"] * 4 + ["b"] * 4,
        "round": t,
    }


def _rounds():
    """Round 0 is played and punished, round 1 is played and awaits its
    punishment -- the state `get_punishments` is called in."""
    return [_round(0, C0, P0), _round(1, C1, [None] * 8)]


class _RecordingEstimator:
    """Multinomial stand-in that keeps every design matrix it is asked to
    score, so the test can read what reached the model."""

    classes_ = np.arange(31)

    def __init__(self):
        self.seen = []

    def predict_proba(self, X):
        self.seen.append(np.asarray(X, dtype=float).copy())
        return np.full((len(X), 31), 1 / 31)


class _IdentityScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)


def _bundle(features):
    return {
        "model": "multinomial",
        "estimator": _RecordingEstimator(),
        "scaler": _IdentityScaler(),
        "features": features,
        "target": "punishment",
        "target_type": "categorical",
        "n_levels": 31,
        "default_values": DEFAULTS,
        "switch_every": 4,
    }


def _adapter(features):
    from aimanager.simulation.linear_ah import LinearAHAdapter

    return LinearAHAdapter(_bundle(features), sample=False)


def test_linear_punisher_reads_current_contribution():
    ah = _adapter(FEATS)
    out = ah.get_punishments(_rounds())
    assert tuple(out.shape) == (8,)
    X = ah.estimator.seen[-1]
    col = {f: i for i, f in enumerate(FEATS)}
    np.testing.assert_array_equal(X[:, col["contribution"]], C1)
    np.testing.assert_array_equal(X[:, col["prev_contribution"]], C0)
    np.testing.assert_array_equal(X[:, col["prev_punishment"]], P0)
    assert (X[:, col["round_number"]] == 1).all()
    # round 0: the contribution is realised, the lag is the default
    ah.get_punishments(_rounds()[:1])
    X0 = ah.estimator.seen[-1]
    np.testing.assert_array_equal(X0[:, col["contribution"]], C0)
    assert (X0[:, col["prev_contribution"]] == DEFAULTS["contribution"]).all()


def test_linear_punisher_group_mean_is_current_round():
    ah = _adapter(["contribution_mean_group"])
    ah.get_punishments(_rounds())
    X = ah.estimator.seen[-1]
    c1 = np.array(C1, dtype=float)
    expected = [
        c1[[j for j in range(8) if AGENT_GROUP[j] == AGENT_GROUP[i] and j != i]].mean()
        for i in range(8)
    ]
    np.testing.assert_allclose(X[:, 0], expected)


def test_linear_punisher_rejects_same_round_punishment_features():
    for feat in ("punishment", "payoff", "common_good", "punishment_mean_group"):
        with pytest.raises(AssertionError, match="current-valued"):
            _adapter(["contribution", feat])


def test_feature_legality_per_target():
    # importing the adapter puts scripts/baselines on the path
    import aimanager.simulation.linear_ah  # noqa: F401
    from handcrafted_grid import validate_feature_legality

    def cfg(target, *feats):
        return {"data": {"target": target}, "blocks": {"b": {"sets": [list(feats)]}}}

    validate_feature_legality(cfg("punishment", "contribution", "prev_punishment"))
    validate_feature_legality(cfg("punishment", "contribution_mean_group"))
    validate_feature_legality(cfg("does_switch", "punishment", "common_good"))
    with pytest.raises(ValueError, match="punishment target"):
        validate_feature_legality(cfg("punishment", "punishment_mean_group"))
    with pytest.raises(ValueError, match="punishment target"):
        validate_feature_legality(cfg("punishment", "payoff"))
    with pytest.raises(ValueError, match="contribution target"):
        validate_feature_legality(cfg("contribution", "contribution"))


def test_gnn_punisher_data_reads_current_contribution():
    from aimanager.generic.encoder import Encoder
    from aimanager.manager.api_manager import create_data

    data = create_data(_rounds(), ["a", "b"], DEFAULTS)
    # batch 0 is manager "a": its four agents at the last index (round 1)
    assert data["contribution"][0, :4, -1].tolist() == C1[:4]
    assert data["prev_contribution"][0, :4, -1].tolist() == C0[:4]
    assert data["prev_punishment"][0, :4, -1].tolist() == P0[:4]
    # round 1's punishment is what the model is asked for: a placeholder
    assert data["punishment"][0, :4, -1].tolist() == [DEFAULTS["punishment"]] * 4
    # the same tensors through the input encoder the punisher config wires
    # (numeric: level / (n_levels - 1)); the model reads the last index
    enc = Encoder(
        [
            {"name": "contribution", "n_levels": 21, "encoding": "numeric"},
            {"name": "prev_contribution", "n_levels": 21, "encoding": "numeric"},
        ],
        refrence="punishment",
    )
    x = enc(**data)
    assert tuple(x.shape) == (2, 8, 2, 2)
    np.testing.assert_allclose(x[0, :4, -1, 0].numpy() * 20, C1[:4], atol=1e-5)
    np.testing.assert_allclose(x[0, :4, -1, 1].numpy() * 20, C0[:4], atol=1e-5)


def test_linear_punisher_reads_ceiling_indicator():
    feats = ["contribution", "contribution_max", "contribution_zero"]
    ah = _adapter(feats)
    ah.get_punishments(_rounds())
    X = ah.estimator.seen[-1]
    np.testing.assert_array_equal(X[:, 0], C1)
    np.testing.assert_array_equal(X[:, 1], [float(c == 20) for c in C1])
    np.testing.assert_array_equal(X[:, 2], [float(c == 0) for c in C1])
    # legal for the punishment target, illegal for the contribution target
    import aimanager.simulation.linear_ah  # noqa: F401
    from handcrafted_grid import validate_feature_legality

    validate_feature_legality(
        {"data": {"target": "punishment"}, "blocks": {"b": {"sets": [feats]}}}
    )
    with pytest.raises(ValueError, match="contribution target"):
        validate_feature_legality(
            {
                "data": {"target": "contribution"},
                "blocks": {"b": {"sets": [["contribution_max"]]}},
            }
        )


def test_gnn_punisher_data_has_ceiling_indicator():
    import torch as th

    from aimanager.generic.encoder import Encoder
    from aimanager.manager.api_manager import create_data

    data = create_data(_rounds(), ["a", "b"], DEFAULTS)
    assert data["contribution_max"].dtype == th.bool
    assert data["contribution_max"][0, :4, -1].tolist() == [c == 20 for c in C1[:4]]
    assert data["contribution_max"][0, :4, 0].tolist() == [c == 20 for c in C0[:4]]
    # other-group cells hold the default contribution -> never the maximum
    assert not data["contribution_max"][0, 4:, :].any()
    enc = Encoder(
        [{"etype": "bool", "name": "contribution_max"}], refrence="punishment"
    )
    x = enc(**data)
    assert tuple(x.shape) == (2, 8, 2, 1)
    np.testing.assert_array_equal(
        x[0, :4, -1, 0].numpy(), [float(c == 20) for c in C1[:4]]
    )


def test_linear_punisher_reads_contribution_onehot():
    from handcrafted_grid import CONTRIBUTION_ONEHOT, validate_feature_legality

    assert len(CONTRIBUTION_ONEHOT) == 21
    feats = list(CONTRIBUTION_ONEHOT)
    ah = _adapter(feats)
    ah.get_punishments(_rounds())
    X = ah.estimator.seen[-1]
    np.testing.assert_array_equal(X, np.eye(21)[C1])
    # the level-20 column IS contribution_max, the level-0 column is _zero
    np.testing.assert_array_equal(X[:, 20], [float(c == 20) for c in C1])
    np.testing.assert_array_equal(X[:, 0], [float(c == 0) for c in C1])
    # legal for the punishment target, illegal for the contribution target
    validate_feature_legality(
        {"data": {"target": "punishment"}, "blocks": {"b": {"sets": [feats]}}}
    )
    with pytest.raises(ValueError, match="contribution target"):
        validate_feature_legality(
            {
                "data": {"target": "contribution"},
                "blocks": {"b": {"sets": [["contribution_is_07"]]}},
            }
        )


def test_gnn_punisher_onehot_encodes_current_contribution():
    import torch as th

    from aimanager.generic.encoder import Encoder
    from aimanager.manager.api_manager import create_data

    data = create_data(_rounds(), ["a", "b"], DEFAULTS)
    enc = Encoder(
        [{"name": "contribution", "n_levels": 21, "encoding": "onehot"}],
        refrence="punishment",
    )
    x = enc(**data)
    assert tuple(x.shape) == (2, 8, 2, 21)
    np.testing.assert_array_equal(x[0, :4, -1].numpy(), np.eye(21)[C1[:4]])
    np.testing.assert_array_equal(x[0, :4, 0].numpy(), np.eye(21)[C0[:4]])
    # the level-20 column reproduces the ceiling bool the parent added
    np.testing.assert_array_equal(
        x[..., 20].to(th.bool).numpy(), data["contribution_max"].numpy()
    )
