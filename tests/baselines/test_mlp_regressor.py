"""Unit tests for the MLP point-regressor baseline, its registry wiring, and
the simulation adapter running the saved mlp bundle unchanged.

Three sides are covered:
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
  * src/aimanager/simulation/linear_ah.py -- the claim that LinearAHAdapter
    needs NO mlp-specific code: load_ah_model dispatches on the .joblib
    extension, the current-valued leak guard runs for 'mlp' like any other
    prev-anchored target, and level sampling falls through to the homoscedastic
    `sample and sigma > 0` branch (the gaussian branch cannot be reached --
    MLPRegressor has no predict_std). Driven with the REAL artifact
    artifacts/baselines/contribution_mlp_best.joblib over the frozen fixture
    episode, teacher-forced through the same env-state replay the feature-parity
    test uses (episode_states from test_baseline_features).

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

BUNDLE_PATH = ROOT / "artifacts/baselines/contribution_mlp_best.joblib"
N_CONTRIBUTIONS = 21


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


# --------------------------------------------------------------------------- #
# simulation adapter, driven with the real saved bundle
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bundle_path():
    if not BUNDLE_PATH.exists():
        pytest.skip(f"artifact not available: {BUNDLE_PATH}")
    return str(BUNDLE_PATH)


@pytest.fixture(scope="module")
def bundle(bundle_path):
    import joblib

    return joblib.load(bundle_path)


@pytest.fixture(scope="module")
def episode():
    """Teacher-forced env states for the frozen fixture episode -- the same
    replay the feature-parity test drives the adapter with."""
    from test_baseline_features import episode_states

    states, n_agents, _ = episode_states("contribution")
    return states, n_agents


@pytest.fixture(scope="module")
def adapter(bundle_path, episode):
    import torch as th

    from aimanager.simulation.linear_ah import load_ah_model

    _, n_agents = episode
    return load_ah_model(
        bundle_path,
        device=th.device("cpu"),
        n_agents=n_agents,
        n_contributions=N_CONTRIBUTIONS,
    )


def replay(adapter, states, seed):
    """Whole-episode replay under a fixed torch seed -> [T, A] int64 levels."""
    import torch as th

    th.manual_seed(seed)
    return np.stack(
        [
            adapter.predict(s, reset_rnn=(t == 0))[0].reshape(-1).numpy()
            for t, s in enumerate(states)
        ]
    )


def test_load_ah_model_returns_an_mlp_adapter(bundle_path, episode):
    from handcrafted_grid import CURRENT_VALUED

    from aimanager.simulation.linear_ah import LinearAHAdapter, load_ah_model
    from test_baseline_features import CONTRIB_SAFE

    _, n_agents = episode
    # constructing at all means the prev-anchoring leak guard passed
    ad = load_ah_model(bundle_path, n_agents=n_agents, n_contributions=N_CONTRIBUTIONS)

    assert isinstance(ad, LinearAHAdapter)
    assert ad.model_type == "mlp"
    assert ad.target == "contribution" and not ad.is_switch
    assert ad.autoregressive is False
    assert ad.sample and ad.sigma > 0  # homoscedastic scalar sigma stored
    assert not set(ad.features) & CURRENT_VALUED
    # every feature the bundle uses is one the parity harness already checks
    assert set(ad.features) <= set(CONTRIB_SAFE)


def test_leak_guard_applies_to_the_mlp_bundle(bundle, episode):
    """The guard is not model-type gated: a tampered mlp bundle still trips."""
    from aimanager.simulation.linear_ah import LinearAHAdapter

    _, n_agents = episode
    tampered = {**bundle, "features": list(bundle["features"]) + ["contribution"]}
    with pytest.raises(AssertionError, match="current-valued"):
        LinearAHAdapter(tampered, n_agents=n_agents, n_contributions=N_CONTRIBUTIONS)


def test_sampled_levels_are_seeded_and_in_range(adapter, episode):
    states, n_agents = episode
    a = replay(adapter, states, 7)
    b = replay(adapter, states, 7)
    c = replay(adapter, states, 8)

    assert a.shape == (len(states), n_agents)
    assert a.dtype == np.int64
    assert a.min() >= 0 and a.max() <= N_CONTRIBUTIONS - 1
    assert np.array_equal(a, b), "same seed must reproduce identical levels"
    assert not np.array_equal(a, c), "noise must be applied -> seed must matter"


def test_noise_comes_from_the_homoscedastic_branch(adapter, episode, monkeypatch):
    """Pin the branch: the heteroscedastic path is unreachable (no sigma(x)
    head), sigma == 0 collapses to the deterministic mean, and scaling sigma
    scales the deviation from that mean under a fixed seed."""
    states, _ = episode
    assert not hasattr(adapter.estimator, "predict_std")

    stored = adapter.sigma
    monkeypatch.setattr(adapter, "sigma", 0.0)
    det = replay(adapter, states, 7)
    assert np.array_equal(det, replay(adapter, states, 99))

    monkeypatch.setattr(adapter, "sigma", stored)
    narrow = replay(adapter, states, 7)
    monkeypatch.setattr(adapter, "sigma", 4.0 * stored)
    wide = replay(adapter, states, 7)

    assert not np.array_equal(narrow, det)
    assert np.std(wide - det) > np.std(narrow - det)


def test_sampled_levels_are_dispersed(adapter, episode):
    lvl = replay(adapter, episode[0], 11)

    assert len(np.unique(lvl)) > 5, "levels collapsed to a near-constant"
    assert (lvl.std(axis=0) > 0).all(), "some agent is constant over the episode"
    assert (lvl.std(axis=1) > 0).mean() > 0.9, "rounds without cross-agent spread"
