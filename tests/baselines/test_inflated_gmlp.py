"""Unit tests for the status-quo- and corner-inflated contribution emission:
the `InflatedGaussianMLPRegressor` estimator (proper distribution over the 21
levels, atoms that sit where the raw prev_contribution sits, colliding atoms
that add, the atoms=() degeneracy, recovery of a planted repeat mass,
determinism, joblib round-trip), the binned-Gaussian convention it shares
with `binned_logloss`, and its registry / CV plumbing in `baseline_models`
(the griddable `atoms` knob, the CE metric and floor, and the task-local
`prev_index` that must not be confused with the feature pool's column index).
Invariants and rationale: notes/autoresearch_log/contribution-inflated-gmlp.md.

Local test (CPU torch, no PyG):
    .venv/bin/python -m pytest tests/baselines/test_inflated_gmlp.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch as th  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # tests/baselines -> repo root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/baselines"))

from gaussian_regressor import (  # noqa: E402
    InflatedGaussianMLPRegressor,
    binned_log_probs,
    binned_logloss,
    binned_probs,
)

K = 21  # contribution levels 0..20


# --------------------------------------------------------------------------- #
# synthetic panel: 40% of rounds repeat last round's contribution exactly
# --------------------------------------------------------------------------- #
def toy_panel(n, seed, p_repeat=0.4):
    """Raw features [prev_contribution, prev_punishment] and a target that is
    an exact repeat of prev with probability `p_repeat`, and otherwise a noisy
    partial reversion to the middle -- the human pattern the atoms encode."""
    rng = np.random.default_rng(seed)
    prev = rng.integers(0, K, n).astype(float)
    pun = rng.integers(0, 8, n).astype(float)
    body = np.clip(np.rint(0.6 * prev + 4.0 + rng.normal(0.0, 3.0, n)), 0, K - 1)
    y = np.where(rng.random(n) < p_repeat, prev, body)
    return np.column_stack([prev, pun]), y


def fit_toy(n=3000, seed=0, atoms=("prev", "0", "20"), epochs=300):
    X, y = toy_panel(n, seed)
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    m = InflatedGaussianMLPRegressor(
        hidden=8, epochs=epochs, lr=0.05, seed=3, atoms=atoms, prev_index=0
    ).fit(Z, y, prev=X[:, 0])
    return m, Z, X, y


@pytest.fixture(scope="module")
def fitted():
    m, Z, X, y = fit_toy()
    return dict(m=m, Z=Z, X=X, y=y)


# --------------------------------------------------------------------------- #
# (1) the shared binned-Gaussian convention
# --------------------------------------------------------------------------- #
def test_torch_body_matches_the_numpy_convention():
    """`binned_log_probs` is the differentiable twin of `binned_probs`, the
    convention `binned_logloss` scores; if they drift, the fitted body no
    longer means what the reported cross-entropy means."""
    rng = np.random.default_rng(4)
    mu = rng.uniform(-4.0, 24.0, 500)
    sigma = rng.uniform(0.3, 9.0, 500)
    want = binned_probs(mu, sigma, K)
    got = np.exp(
        binned_log_probs(
            th.as_tensor(mu, dtype=th.float64),
            th.as_tensor(np.log(sigma), dtype=th.float64),
            K,
        ).numpy()
    )
    assert np.allclose(got, want, rtol=1e-6, atol=1e-12)


def test_binned_logloss_still_scores_that_matrix():
    rng = np.random.default_rng(5)
    mu, sigma = rng.uniform(0, 20, 200), rng.uniform(1.0, 5.0, 200)
    y = rng.integers(0, K, 200)
    P = binned_probs(mu, sigma, K)
    want = float(-np.mean(np.log(P[np.arange(len(y)), y])))
    assert binned_logloss(mu, y, sigma, K) == want


# --------------------------------------------------------------------------- #
# (2) the mixture is a proper distribution
# --------------------------------------------------------------------------- #
def test_rows_sum_to_one(fitted):
    P = fitted["m"].predict_proba(fitted["Z"])
    assert P.dtype == np.float64 and P.shape == (len(fitted["Z"]), K)
    assert np.all(P > 0.0)
    assert np.allclose(P.sum(1), 1.0, rtol=0.0, atol=1e-12)


def test_classes_are_the_levels(fitted):
    assert np.array_equal(fitted["m"].classes_, np.arange(K))


def _body_probs(m, Z):
    """The mixture's Gaussian body alone, [N, K] (straight off the net, so no
    float32 round-trip stands between this and what the mixture used)."""
    with th.no_grad():
        mu, log_sigma, _ = m.net(th.as_tensor(np.asarray(Z, float), dtype=th.float32))
    return np.exp(binned_log_probs(mu, log_sigma, K).numpy())


def test_colliding_atoms_add_their_mass(fitted):
    """prev == 0 puts the status-quo atom on the same level as the '0' atom.
    The masses must ADD: level 0 then carries pi_body * body_0 + pi_rep + pi_0,
    not whichever atom was written last."""
    m = fitted["m"]
    zs = fitted["Z"][:200].copy()
    zs[:, 0] = (0.0 - m.prev_mean_) / m.prev_scale_  # every row's prev at 0
    P = m.predict_proba(zs)
    w = m.mixture_probs(zs)
    body = _body_probs(m, zs)
    want = w[:, 0] * body[:, 0] + w[:, 1] + w[:, 2]  # body + prev atom + 0 atom
    assert np.allclose(P[:, 0], want, rtol=1e-12, atol=1e-14)
    assert np.allclose(P.sum(1), 1.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# (3) the atoms sit where the raw prev column sits
# --------------------------------------------------------------------------- #
def test_prev_affine_map_recovered_exactly(fitted):
    """The atom's level comes from the RAW prev value, recovered by inverting
    the standardiser's affine map for that column. A silent mismatch here would
    place the mass at the wrong level and only show up in the simulation."""
    m, Z, X = fitted["m"], fitted["Z"], fitted["X"]
    raw = Z[:, 0] * m.prev_scale_ + m.prev_mean_
    assert np.allclose(raw, X[:, 0], atol=1e-9)
    assert np.array_equal(m._prev_levels(Z), X[:, 0].astype(int))


@pytest.mark.parametrize("level", [0, 7, 13, 20])
def test_prev_atom_moves_with_the_prev_column(fitted, level):
    """Hold everything else fixed, move the prev column: the excess mass over
    the body must follow to the matching level. At the corners the fixed atom
    sits there too, so the excess is the sum of the two."""
    m = fitted["m"]
    z = fitted["Z"][:1].copy()
    z[:, 0] = (level - m.prev_mean_) / m.prev_scale_
    w = m.mixture_probs(z)[0]
    excess = m.predict_proba(z)[0] - w[0] * _body_probs(m, z)[0]
    want = w[1] + (w[2] if level == 0 else 0.0) + (w[3] if level == K - 1 else 0.0)
    assert int(np.argmax(excess)) == level
    assert excess[level] == pytest.approx(want, abs=1e-12)


# --------------------------------------------------------------------------- #
# (4) degeneracy, recovery, determinism, round-trip
# --------------------------------------------------------------------------- #
def test_no_atoms_is_exactly_a_binned_gaussian():
    """atoms=() must reduce to the binned Gaussian body fitted by the same
    21-way cross-entropy -- the comparison that makes the atoms' gain readable."""
    m, Z, _, y = fit_toy(atoms=())
    P = m.predict_proba(Z)
    want = binned_probs(m.predict(Z), m.predict_std(Z), K)
    assert np.allclose(P, want, rtol=1e-6, atol=1e-12)
    assert m.mixture_probs(Z).shape == (len(Z), 1)
    assert m.nll(Z, y) == pytest.approx(
        binned_logloss(m.predict(Z), y, m.predict_std(Z), K), rel=1e-6
    )


def test_planted_repeat_mass_is_recovered(fitted):
    """40% of the panel repeats prev exactly; the fitted status-quo weight must
    find at least 0.30 of it (the body itself explains some repeats)."""
    m, Z, y = fitted["m"], fitted["Z"], fitted["y"]
    pi_rep = float(m.mixture_probs(Z)[:, 1].mean())
    assert pi_rep >= 0.30, f"pi_rep {pi_rep:.4f}"
    flat, _, _, _ = fit_toy(atoms=())
    assert m.nll(Z, y) < flat.nll(Z, y) - 0.05


def test_same_seed_bit_identical(fitted):
    again, _, _, _ = fit_toy()
    assert np.array_equal(
        again.predict_proba(fitted["Z"]), fitted["m"].predict_proba(fitted["Z"])
    )


def test_joblib_round_trip(fitted, tmp_path):
    path = tmp_path / "inflated.joblib"
    joblib.dump(fitted["m"], path)
    back = joblib.load(path)
    assert back.atoms == fitted["m"].atoms
    assert np.array_equal(
        back.predict_proba(fitted["Z"]), fitted["m"].predict_proba(fitted["Z"])
    )


# --------------------------------------------------------------------------- #
# (5) refusals
# --------------------------------------------------------------------------- #
def test_fit_without_prev_raises():
    X, y = toy_panel(200, 11)
    Z = StandardScaler().fit_transform(X)
    m = InflatedGaussianMLPRegressor(hidden=4, epochs=1, seed=0)
    with pytest.raises(ValueError, match="raw prev_contribution"):
        m.fit(Z, y)


def test_fit_with_the_wrong_prev_index_raises():
    X, y = toy_panel(200, 12)
    Z = StandardScaler().fit_transform(X)
    m = InflatedGaussianMLPRegressor(hidden=4, epochs=1, seed=0, prev_index=1)
    with pytest.raises(ValueError, match="affine image"):
        m.fit(Z, y, prev=X[:, 0])


@pytest.mark.parametrize("atoms", [("prev", "prev"), ("prev", "10")])
def test_bad_atom_sets_rejected(atoms):
    with pytest.raises(ValueError):
        InflatedGaussianMLPRegressor(atoms=atoms)


# --------------------------------------------------------------------------- #
# (6) registry + CV plumbing (baseline_models / run_baseline_cv)
# --------------------------------------------------------------------------- #
from baseline_models import (  # noqa: E402
    CE_MODELS,
    NEEDS_PREV,
    PREV_FEATURE,
    build_model,
    build_settings,
    floor_score,
    metric_name,
    parse_atoms,
    predict_scores,
    prev_position,
    resolve_model,
    setting_keys,
)

MODEL = "gaussian_mlp_inflated"


def _cfg(atoms):
    return {
        "data": {"target_type": "continuous", "model": MODEL},
        "setting": {
            "hidden": 8,
            "weight_decay": 0.0003,
            "lr": 0.01,
            "epochs": 1000,
            "atoms": atoms,
        },
    }


def test_registered_for_continuous_targets():
    assert resolve_model(_cfg("prev,0,20")) == MODEL
    assert metric_name(MODEL) == "ce"
    assert MODEL in NEEDS_PREV and MODEL in CE_MODELS
    assert "atoms" in setting_keys(MODEL)
    with pytest.raises(ValueError):
        resolve_model({"data": {"target_type": "categorical", "model": MODEL}})


def test_settings_expand_to_the_three_atom_sets():
    """The declared grid: one griddable knob, three atom sets, everything else
    pinned to the incumbent's setting."""
    settings = build_settings(_cfg(["prev", "prev,20", "prev,0,20"]), MODEL)
    assert [s["atoms"] for s in settings] == ["prev", "prev,20", "prev,0,20"]
    assert {s["hidden"] for s in settings} == {8}
    assert {s["epochs"] for s in settings} == {1000}
    assert build_settings({"data": {}}, MODEL)[0]["atoms"] == "prev,0,20"


@pytest.mark.parametrize("bad", ["prev,10", "10", "prev,prev", "body"])
def test_unknown_atom_string_rejected(bad):
    with pytest.raises(ValueError):
        build_settings(_cfg(bad), MODEL)


def test_parse_atoms_round_trips():
    assert parse_atoms("prev, 0 ,20") == ("prev", "0", "20")
    assert parse_atoms("") == () and parse_atoms(None) == ()


def test_prev_position_is_the_task_position_not_the_pool_index():
    """The whole hazard of the plumbing: col_of[PREV_FEATURE] is the column in
    the feature POOL, while the estimator indexes the task's own matrix."""
    feats = ["prev_punishment", PREV_FEATURE, "rounds_since_switch"]
    col_of = {"prev_punishment": 7, PREV_FEATURE: 0, "rounds_since_switch": 19}
    cols = [col_of[f] for f in feats]
    assert prev_position(MODEL, feats, PREV_FEATURE) == 1
    assert prev_position(MODEL, cols, col_of[PREV_FEATURE]) == 1
    assert prev_position("gaussian_mlp", feats, PREV_FEATURE) is None


def test_feature_set_without_prev_is_a_config_error():
    with pytest.raises(ValueError, match=PREV_FEATURE):
        prev_position(MODEL, ["prev_punishment"], PREV_FEATURE)
    with pytest.raises(ValueError, match=PREV_FEATURE):
        prev_position(MODEL, [7, 19], None)


def test_build_model_needs_the_prev_index():
    setting = build_settings(_cfg("prev,0,20"), MODEL)[0]
    m = build_model(MODEL, setting, seed=3, prev_index=2)
    assert isinstance(m, InflatedGaussianMLPRegressor)
    assert m.atoms == ("prev", "0", "20") and m.prev_index == 2
    assert (m.hidden, m.epochs, m.lr, m.weight_decay) == (8, 1000, 0.01, 0.0003)
    with pytest.raises(ValueError, match="prev_index"):
        build_model(MODEL, setting, seed=3)


def test_a_wrong_prev_index_raises_instead_of_fitting():
    """`prev_index` pointing at another feature must hard-error at fit, not
    silently place the status-quo atom at the wrong level."""
    X, y = toy_panel(300, 21)
    Z = StandardScaler().fit_transform(X)
    setting = build_settings(_cfg("prev,0,20"), MODEL)[0]
    setting["epochs"] = 1
    m = build_model(MODEL, setting, seed=3, prev_index=1)  # prev_punishment
    with pytest.raises(ValueError, match="affine image"):
        m.fit(Z, y, prev=X[:, 0])


def test_predict_scores_reports_the_cross_entropy_twice(fitted):
    """Primary metric and the show_ce column are the same 21-way CE."""
    m, Z, y = fitted["m"], fitted["Z"], fitted["y"]
    primary, ce = predict_scores(MODEL, m, Z, y, n_levels=0, show_ce=True)
    assert primary == ce == m.nll(Z, y)
    assert predict_scores(MODEL, m, Z, y, n_levels=0)[1] is None


def test_floor_is_the_marginal_histogram(fitted):
    """The floor is the multinomial branch's: the smoothed marginal histogram
    of the levels, scored by the same cross-entropy."""
    ytr, yte = fitted["y"][:2000], fitted["y"][2000:]
    primary, ce = floor_score(MODEL, ytr, yte, n_levels=0, show_ce=True)
    c = np.bincount(np.rint(ytr).astype(int), minlength=K) + 1.0
    want = float(-np.mean(np.log((c / c.sum())[np.rint(yte).astype(int)])))
    assert primary == ce == pytest.approx(want, rel=1e-12)
    assert fitted["m"].nll(fitted["Z"], fitted["y"]) < primary
