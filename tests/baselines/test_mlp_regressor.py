"""Unit tests for the MLP point-regressor baseline and its registry wiring.

Two sides are covered:
  * scripts/baselines/mlp_regressor.py -- the estimator itself: seed
    determinism, predict contract, that the nonlinear mean actually buys
    something (beats both the intercept-only floor and a linear least-squares
    fit on a purely nonlinear target), and that coef_/intercept_ stay
    unavailable.
  * scripts/baselines/baseline_models.py -- the 'mlp' registry entries:
    resolve_model / metric_name / build_settings validation + Cartesian
    expansion, and that scoring routes 'mlp' through the MSE path (same floor
    as ridge). Plus a joblib round-trip, since the CV driver persists fitted
    estimators into bundles.

Everything here is CPU torch + numpy only (no PyG), so it runs locally.

Run:  .venv/bin/python -m pytest tests/baselines/test_mlp_regressor.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # tests/baselines -> repo root
sys.path.insert(0, str(ROOT / "scripts/baselines"))

import baseline_models as bm  # noqa: E402
from mlp_regressor import MLPRegressor  # noqa: E402

SEED = 20250818
# small + short: the point is the mechanism, not convergence quality
N, EPOCHS, HIDDEN = 300, 600, 32


def nonlinear_data(n=N, seed=SEED):
    """Target with (almost) no linear signal: a linear fit can only recover the
    mean, so any improvement over the floor is curvature."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = 3.0 * X[:, 0] ** 2 + np.cos(3.0 * X[:, 1]) + 0.05 * rng.standard_normal(n)
    return X, y


@pytest.fixture(scope="module")
def data():
    return nonlinear_data()


@pytest.fixture(scope="module")
def fitted(data):
    X, y = data
    return MLPRegressor(hidden=HIDDEN, epochs=EPOCHS, seed=0).fit(X, y)


def mse(pred, y):
    return float(np.mean((np.asarray(pred) - np.asarray(y)) ** 2))


# --------------------------------------------------------------------------- #
# estimator
# --------------------------------------------------------------------------- #
def test_seed_determinism_and_predict_contract(data):
    X, y = data
    a = MLPRegressor(hidden=8, epochs=100, seed=3).fit(X, y).predict(X)
    b = MLPRegressor(hidden=8, epochs=100, seed=3).fit(X, y).predict(X)
    c = MLPRegressor(hidden=8, epochs=100, seed=4).fit(X, y).predict(X)

    assert np.array_equal(a, b), "same seed must reproduce identical predictions"
    assert not np.allclose(a, c, atol=1e-6), "different seed must change predictions"

    assert isinstance(a, np.ndarray)
    assert a.shape == (len(y),)
    assert np.issubdtype(a.dtype, np.floating)
    assert np.isfinite(a).all()


def test_nonlinear_mean_beats_floor_and_linear_fit(data, fitted):
    X, y = data
    floor = mse(np.full(len(y), y.mean()), y)

    Xd = np.column_stack([X, np.ones(len(y))])
    coef = np.linalg.lstsq(Xd, y, rcond=None)[0]
    linear = mse(Xd @ coef, y)

    fit = mse(fitted.predict(X), y)

    assert fit < 0.5 * floor, f"mlp mse {fit:.4f} vs floor {floor:.4f}"
    assert fit < 0.5 * linear, f"mlp mse {fit:.4f} vs linear {linear:.4f}"


def test_no_linear_coefficients(fitted):
    with pytest.raises(NotImplementedError):
        fitted.coef_
    with pytest.raises(NotImplementedError):
        fitted.intercept_


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_resolve_model_and_metric():
    cfg = {"data": {"target_type": "continuous", "model": "mlp"}}
    assert bm.resolve_model(cfg) == "mlp"
    assert bm.metric_name("mlp") == "mse"
    assert set(bm.setting_keys("mlp")) == {"hidden", "weight_decay", "lr", "epochs"}


def test_build_settings_expands_grid():
    cfg = {"setting": {"hidden": [8, 16], "lr": [0.01, 0.05], "epochs": 200}}
    cells = bm.build_settings(cfg, "mlp")

    assert len(cells) == 4
    assert {(c["hidden"], c["lr"]) for c in cells} == {
        (8, 0.01),
        (8, 0.05),
        (16, 0.01),
        (16, 0.05),
    }
    assert all(c["epochs"] == 200 and c["weight_decay"] == 0.0 for c in cells)
    assert all(isinstance(c["hidden"], int) for c in cells)


def test_build_settings_rejects_foreign_keys():
    with pytest.raises(ValueError) as e:
        bm.build_settings({"setting": {"hidden": 8, "alpha": 1.0}}, "mlp")
    msg = str(e.value)
    assert "alpha" in msg and "mlp" in msg and "hidden" in msg

    with pytest.raises(ValueError, match="hidden"):
        bm.build_settings({"setting": {"hidden": 8}}, "gaussian")


def test_build_model_passes_setting_and_seed():
    setting = {"hidden": 4, "weight_decay": 0.1, "lr": 0.02, "epochs": 7}
    m = bm.build_model("mlp", setting, seed=11)

    assert isinstance(m, MLPRegressor)
    assert (m.hidden, m.weight_decay, m.lr, m.epochs, m.seed) == (4, 0.1, 0.02, 7, 11)


def test_scoring_uses_the_mse_path(data, fitted):
    X, y = data
    loss, ce = bm.predict_scores("mlp", fitted, X, y, n_levels=21)
    assert ce is None
    assert loss == pytest.approx(mse(fitted.predict(X), y))

    ytr, yte = y[:200], y[200:]
    mlp_floor = bm.floor_score("mlp", ytr, yte, n_levels=21)
    assert mlp_floor == bm.floor_score("ridge", ytr, yte, n_levels=21)
    assert mlp_floor[0] == pytest.approx(mse(np.full(len(yte), ytr.mean()), yte))


def test_joblib_roundtrip(data, fitted, tmp_path):
    import joblib

    X, _ = data
    path = tmp_path / "mlp.joblib"
    joblib.dump(fitted, path)
    loaded = joblib.load(path)

    assert np.array_equal(loaded.predict(X), fitted.predict(X))
