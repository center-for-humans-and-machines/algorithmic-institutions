"""Unit tests for the nonlinear contribution baseline `gaussian_mlp`:
the `GaussianMLPRegressor` estimator (nonlinear mean, state-dependent sigma,
determinism, marginal at epochs=0, disabled linear-only accessors), its
registration in `baseline_models`, the simulation adapter's sampling branch,
and backward compatibility of the incumbent linear gaussian artifact.
Invariants and rationale: notes/autoresearch_log/contribution-gaussian-mlp.md.

Local test (CPU torch, no PyG):
    .venv/bin/python -m pytest tests/baselines/test_gaussian_mlp.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch as th  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # tests/baselines -> repo root
# this checkout's src must win over any installed/editable aimanager, so the
# adapter under test is the one in THIS worktree
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/baselines"))

import baseline_models  # noqa: E402
from aimanager.simulation.linear_ah import LinearAHAdapter  # noqa: E402
from gaussian_regressor import (  # noqa: E402
    GaussianMLPRegressor,
    GaussianRegressor,
)

N_CONTRIBUTIONS = 21  # contribution levels 0..20
FEATS = ["prev_contribution", "prev_punishment", "round_number", "is_first"]
DEFAULTS = {
    "punishment": 0.0,
    "contribution": 9.0,
    "common_good": 12.333333333333334,
    "agent_group": 0.0,
    "contribution_valid": 0.0,
    "punishment_valid": 0.0,
    "recorded": 0.0,
}
INCUMBENT = ROOT / "artifacts/baselines/contribution_gaussian_best.joblib"


# --------------------------------------------------------------------------- #
# synthetic data: nonlinear mean, heteroscedastic spread
# --------------------------------------------------------------------------- #
def toy_xy(n, seed):
    """y = 10 + 5 sin(2 x0) + eps, sd(eps) = 0.5 + 2 x1^2.

    Both the mean and the spread are nonlinear -- and SYMMETRIC in x1, so the
    linear model's affine log-sigma head cannot track the spread at all (its
    best fit is a near-flat sigma), while an MLP can. This is exactly the
    state-dependent-sigma claim the gaussian_mlp change rests on."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, n)
    x1 = rng.uniform(-1.5, 1.5, n)
    sd = 0.5 + 2.0 * x1**2
    y = 10.0 + 5.0 * np.sin(2.0 * x0) + rng.normal(0.0, 1.0, n) * sd
    return np.column_stack([x0, x1]), y


@pytest.fixture(scope="module")
def fitted():
    """One MLP and one linear model on the same train split, plus a held-out
    split. The linear model gets more epochs than the MLP, so a win cannot be
    an artefact of an unfair optimisation budget."""
    Xtr, ytr = toy_xy(600, 1)
    Xte, yte = toy_xy(400, 2)
    mlp = GaussianMLPRegressor(hidden=32, epochs=800, lr=0.05, seed=0).fit(Xtr, ytr)
    lin = GaussianRegressor(epochs=2000, lr=0.05, seed=0).fit(Xtr, ytr)
    return dict(mlp=mlp, lin=lin, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)


# --------------------------------------------------------------------------- #
# (1) the estimator
# --------------------------------------------------------------------------- #
def test_mlp_beats_linear_on_heldout_nll(fitted):
    nll_mlp = fitted["mlp"].nll(fitted["Xte"], fitted["yte"])
    nll_lin = fitted["lin"].nll(fitted["Xte"], fitted["yte"])
    assert np.isfinite(nll_mlp)
    assert nll_mlp < nll_lin - 0.1, f"mlp {nll_mlp:.4f} vs linear {nll_lin:.4f}"


def test_sigma_positive_and_state_dependent(fitted):
    """sigma(x) must be strictly positive (log-sigma parameterisation) and
    genuinely vary with the state -- more so than the linear head's, which is
    near-flat on this data."""
    s_mlp = fitted["mlp"].predict_std(fitted["Xte"])
    s_lin = fitted["lin"].predict_std(fitted["Xte"])
    assert np.all(s_mlp > 0.0)
    assert s_mlp.std() > 0.25, f"sigma(x) nearly flat: sd {s_mlp.std():.4f}"
    assert s_mlp.std() > s_lin.std(), f"mlp {s_mlp.std():.4f} <= lin {s_lin.std():.4f}"


def test_same_seed_bit_identical_different_seed_differs():
    Xtr, ytr = toy_xy(300, 3)
    Xte, _ = toy_xy(100, 4)
    a = GaussianMLPRegressor(hidden=16, epochs=100, seed=7).fit(Xtr, ytr)
    b = GaussianMLPRegressor(hidden=16, epochs=100, seed=7).fit(Xtr, ytr)
    c = GaussianMLPRegressor(hidden=16, epochs=100, seed=8).fit(Xtr, ytr)
    assert np.array_equal(a.predict(Xte), b.predict(Xte))
    assert np.array_equal(a.predict_std(Xte), b.predict_std(Xte))
    assert not np.array_equal(a.predict(Xte), c.predict(Xte))


def test_epochs_zero_emits_the_marginal():
    """Zero-init output weights + warm-started output biases mean the untrained
    net emits N(mean(y), std(y)) -- the same starting point as the linear model
    and as the intercept-only floor."""
    Xtr, ytr = toy_xy(300, 5)
    m = GaussianMLPRegressor(hidden=8, epochs=0, seed=0).fit(Xtr, ytr)
    mu, sigma = m.predict(Xtr), m.predict_std(Xtr)
    assert np.allclose(mu, ytr.mean(), atol=1e-5)
    # warm start is float32 torch std (ddof=1) round-tripped through log/exp
    want = float(th.as_tensor(ytr, dtype=th.float32).std())
    assert sigma == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize("attr", ["coef_", "intercept_"])
def test_linear_only_accessors_raise(fitted, attr):
    with pytest.raises(NotImplementedError, match="affine mean map"):
        getattr(fitted["mlp"], attr)
    getattr(fitted["lin"], attr)  # still available on the linear model


# --------------------------------------------------------------------------- #
# (2) registry: resolve / settings / build / scoring
# --------------------------------------------------------------------------- #
def test_resolve_model_gaussian_mlp():
    cfg = {"data": {"target_type": "continuous", "model": "gaussian_mlp"}}
    assert baseline_models.resolve_model(cfg) == "gaussian_mlp"
    assert baseline_models.metric_name("gaussian_mlp") == "nll"


@pytest.mark.parametrize(
    "data",
    [
        {"target_type": "continuous", "model": "gaussian_mpl"},
        {"target_type": "categorical", "model": "gaussian_mlp"},
    ],
)
def test_resolve_model_rejects_bogus(data):
    with pytest.raises(ValueError):
        baseline_models.resolve_model({"data": data})


def test_build_settings_expands_mixed_grid():
    cfg = {
        "setting": {
            "hidden": [16, 32],
            "weight_decay": 0.0,
            "lr": [0.01, 0.05],
            "epochs": 500,
        }
    }
    settings = baseline_models.build_settings(cfg, "gaussian_mlp")
    assert len(settings) == 4
    assert {(s["hidden"], s["lr"]) for s in settings} == {
        (16, 0.01),
        (16, 0.05),
        (32, 0.01),
        (32, 0.05),
    }
    for s in settings:
        assert sorted(s) == ["epochs", "hidden", "lr", "weight_decay"]
        assert isinstance(s["hidden"], int) and isinstance(s["epochs"], int)
        assert isinstance(s["lr"], float) and isinstance(s["weight_decay"], float)


def test_build_settings_rejects_unknown_key():
    with pytest.raises(ValueError, match="alpha"):
        baseline_models.build_settings({"setting": {"alpha": 1.0}}, "gaussian_mlp")


def test_build_model_applies_the_setting():
    setting = {"hidden": 16, "weight_decay": 1e-4, "lr": 0.01, "epochs": 250}
    m = baseline_models.build_model("gaussian_mlp", setting, seed=11)
    assert isinstance(m, GaussianMLPRegressor)
    assert (m.hidden, m.epochs, m.lr, m.weight_decay, m.seed) == (
        16,
        250,
        0.01,
        1e-4,
        11,
    )


def test_predict_scores_returns_nll_and_ce(fitted):
    nll, ce = baseline_models.predict_scores(
        "gaussian_mlp",
        fitted["mlp"],
        fitted["Xte"],
        fitted["yte"],
        n_levels=0,
        show_ce=True,
    )
    assert isinstance(nll, float) and isinstance(ce, float)
    assert np.isfinite(nll) and np.isfinite(ce)
    assert nll == pytest.approx(fitted["mlp"].nll(fitted["Xte"], fitted["yte"]))


def test_floor_score_shared_with_linear_gaussian():
    """No features -> both models reduce to the same marginal, so the floor must
    be identical (the CV driver compares runs across both models against it)."""
    _, ytr = toy_xy(300, 9)
    _, yte = toy_xy(200, 10)
    lin = baseline_models.floor_score("gaussian", ytr, yte, 0, show_ce=True)
    mlp = baseline_models.floor_score("gaussian_mlp", ytr, yte, 0, show_ce=True)
    assert lin == mlp
    assert all(np.isfinite(v) for v in mlp)


# --------------------------------------------------------------------------- #
# (3) simulation adapter: the heteroscedastic sampling branch
# --------------------------------------------------------------------------- #
def toy_features(n, seed):
    """Raw feature rows on the real game's scale -- contributions 0-20,
    punishments 0-7, 24 rounds -- so the fitted model is exercised where the
    simulation would exercise it."""
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.integers(0, 21, n).astype(float),
            rng.integers(0, 8, n).astype(float),
            rng.integers(1, 24, n).astype(float),
            np.zeros(n),
        ]
    )


def toy_bundle(**over):
    """A small but real gaussian_mlp contribution bundle: mean reverting to a
    nonlinear function of prev_contribution, spread widening with punishment."""
    from sklearn.preprocessing import StandardScaler

    X = toy_features(500, 21)
    rng = np.random.default_rng(22)
    sd = 0.5 + 0.4 * X[:, 1]
    y = np.clip(
        0.6 * X[:, 0] + 4.0 * np.sin(X[:, 0] / 3.0) + rng.normal(0.0, 1.0, len(X)) * sd,
        0,
        20,
    )
    scaler = StandardScaler().fit(X)
    est = GaussianMLPRegressor(hidden=16, epochs=400, lr=0.05, seed=0).fit(
        scaler.transform(X), y
    )
    bundle = dict(
        model="gaussian_mlp",
        estimator=est,
        scaler=scaler,
        features=list(FEATS),
        target="contribution",
        n_levels=0,
        default_values=dict(DEFAULTS),
        switch_every=4,
    )
    bundle.update(over)
    return bundle


@pytest.fixture(scope="module")
def contribution_bundle():
    return toy_bundle()


def _adapter(bundle, **kw):
    return LinearAHAdapter(bundle, n_agents=8, n_contributions=N_CONTRIBUTIONS, **kw)


def test_sample_levels_matches_reference_draw(contribution_bundle):
    """Bit-identical to clip(rint(mu + randn * sigma(x)), 0, 20) under one seed:
    pins both the formula and the RNG consumption (one draw of size n)."""
    ad = _adapter(contribution_bundle)
    X = toy_features(8, 33)
    Xs = ad.scaler.transform(X)
    est = contribution_bundle["estimator"]

    th.manual_seed(42)
    got = ad._sample_levels(Xs, N_CONTRIBUTIONS)
    got_next = th.randn(1).item()

    th.manual_seed(42)
    yhat = est.predict(Xs) + th.randn(len(Xs)).numpy() * est.predict_std(Xs)
    want = np.clip(np.rint(yhat), 0, N_CONTRIBUTIONS - 1).astype(np.int64)
    want_next = th.randn(1).item()

    assert np.array_equal(got, want), f"{got.tolist()} != {want.tolist()}"
    assert got.dtype == np.int64
    assert got_next == want_next, "RNG consumption differs from the reference"


def test_sample_false_is_the_rounded_mean(contribution_bundle):
    ad = _adapter(contribution_bundle, sample=False)
    Xs = ad.scaler.transform(toy_features(8, 34))
    want = np.clip(
        np.rint(contribution_bundle["estimator"].predict(Xs)), 0, N_CONTRIBUTIONS - 1
    ).astype(np.int64)
    th.manual_seed(42)
    got = ad._sample_levels(Xs, N_CONTRIBUTIONS)
    got_next = th.randn(1).item()
    th.manual_seed(42)
    want_next = th.randn(1).item()
    assert np.array_equal(got, want)
    assert got_next == want_next, "deterministic path must not consume the RNG"


def test_sampling_actually_varies(contribution_bundle):
    """The branch must be drawing, not collapsing onto the rounded mean."""
    ad = _adapter(contribution_bundle)
    Xs = ad.scaler.transform(toy_features(8, 35))
    th.manual_seed(1)
    draws = np.stack([ad._sample_levels(Xs, N_CONTRIBUTIONS) for _ in range(50)])
    assert draws.std(0).max() > 0.0
    assert draws.min() >= 0 and draws.max() <= N_CONTRIBUTIONS - 1


def test_copula_rho_rejected_on_gaussian_mlp():
    """`copula_rho` (PR #160's punisher field) is a multinomial-punishment
    feature; a gaussian_mlp bundle carrying it is a misconfiguration, exactly
    as for `gaussian`. This is about `copula_rho` specifically -- the group
    copula's own fields, `copula_rho_p` / `copula_rho_t`, are covered below."""
    for model in ("gaussian", "gaussian_mlp"):
        with pytest.raises(AssertionError, match="multinomial punishment"):
            _adapter(toy_bundle(model=model, copula_rho=0.35))


# --------------------------------------------------------------------------- #
# (3b) simulation adapter: the group-copula gate (`copula_rho_p` /
# `copula_rho_t`) -- step 3 only opens the gate, it draws nothing; the
# sampler that honours these fields is step 4.
# --------------------------------------------------------------------------- #
def test_copula_rho_p_t_accepted_on_gaussian_mlp_contribution():
    ad = _adapter(toy_bundle(copula_rho_p=0.05, copula_rho_t=0.02))
    assert ad.copula_rho_p == 0.05
    assert ad.copula_rho_t == 0.02


def test_copula_rho_p_t_accepted_on_gaussian_contribution():
    ad = _adapter(toy_bundle(model="gaussian", copula_rho_p=0.03, copula_rho_t=0.01))
    assert ad.copula_rho_p == 0.03
    assert ad.copula_rho_t == 0.01


def test_copula_rho_p_t_rejected_on_multinomial_punishment():
    bundle = toy_bundle(model="multinomial", target="punishment", copula_rho_p=0.05)
    with pytest.raises(AssertionError, match="Gaussian contribution sampler"):
        _adapter(bundle)


def test_copula_rho_p_t_rejected_on_ridge():
    with pytest.raises(AssertionError, match="Gaussian contribution sampler"):
        _adapter(toy_bundle(model="ridge", copula_rho_p=0.05))


def test_copula_rho_p_t_rejected_when_sum_at_least_one():
    with pytest.raises(AssertionError, match="must be < 1"):
        _adapter(toy_bundle(copula_rho_p=0.6, copula_rho_t=0.4))


@pytest.mark.parametrize(
    "over",
    [{"copula_rho_p": -0.1}, {"copula_rho_t": -0.1}],
)
def test_copula_rho_p_t_rejected_when_negative(over):
    with pytest.raises(AssertionError, match="must be >= 0"):
        _adapter(toy_bundle(**over))


@pytest.mark.parametrize(
    "over",
    [
        {},
        {"copula_rho_p": None, "copula_rho_t": None},
        {"copula_rho_p": 0.0, "copula_rho_t": 0.0},
    ],
)
def test_copula_rho_p_t_accepted_as_zero_when_absent_or_none(over):
    ad = _adapter(toy_bundle(**over))
    assert ad.copula_rho_p == 0.0
    assert ad.copula_rho_t == 0.0


def test_copula_rho_p_t_no_behaviour_change_yet(contribution_bundle):
    """Step 3 invariant, EXPECTED TO BE UPDATED IN STEP 4: setting the new
    fields to a non-zero pair must leave sampling completely unchanged --
    step 3 only opens the configuration gate, it adds no sampler branch. Once
    step 4 lands the sampler and wires it in at `predict()`, a non-zero pair
    will draw from the group-copula path and this test's premise (bit-identity
    with the fields absent) will no longer hold and must be replaced."""
    ad_legacy = _adapter(contribution_bundle)
    ad_copula = _adapter(toy_bundle(copula_rho_p=0.2, copula_rho_t=0.1))
    Xs = ad_legacy.scaler.transform(toy_features(8, 40))

    th.manual_seed(11)
    want = [ad_legacy._sample_levels(Xs, N_CONTRIBUTIONS) for _ in range(3)]
    want_next = th.randn(1).item()

    th.manual_seed(11)
    got = [ad_copula._sample_levels(Xs, N_CONTRIBUTIONS) for _ in range(3)]
    got_next = th.randn(1).item()

    for i, (w, g) in enumerate(zip(want, got)):
        assert np.array_equal(w, g), f"call {i}: {w.tolist()} != {g.tolist()}"
    assert got_next == want_next, "RNG consumption differs with the fields set"


# --------------------------------------------------------------------------- #
# (4) backward compatibility of the incumbent linear gaussian artifact
# --------------------------------------------------------------------------- #
def test_incumbent_gaussian_bundle_still_loads():
    """Guards the incumbent joblib against the class restructuring (_make_net /
    _out_layer hooks, new subclass): it must unpickle and still predict."""
    import joblib

    if not INCUMBENT.exists():  # pragma: no cover
        pytest.skip(f"{INCUMBENT} missing")
    if INCUMBENT.read_bytes()[:23] == b"version https://git-lfs":  # pragma: no cover
        pytest.skip(f"{INCUMBENT} is an unfetched git-lfs pointer (git lfs pull)")

    bundle = joblib.load(INCUMBENT)
    assert bundle["model"] == "gaussian"
    est = bundle["estimator"]
    assert isinstance(est, GaussianRegressor)
    assert not isinstance(est, GaussianMLPRegressor)

    rng = np.random.default_rng(0)
    Xs = rng.normal(size=(16, len(bundle["features"])))
    mu, sigma = est.predict(Xs), est.predict_std(Xs)
    assert mu.shape == (16,) and sigma.shape == (16,)
    assert np.all(np.isfinite(mu)) and np.all(sigma > 0.0)
    assert np.isfinite(est.coef_).all()

    ad = LinearAHAdapter(bundle, n_agents=8, n_contributions=N_CONTRIBUTIONS)
    assert ad.model_type == "gaussian" and ad.target == "contribution"
