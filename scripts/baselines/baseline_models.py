"""Model dispatch for the hand-crafted linear baselines (issue #119).

One source of truth, shared by the CV driver (run_baseline_cv) and the best-model
inspector (inspect_best_model), for:
  * choosing the estimator from `data.target_type` + `data.model`
    - categorical -> multinomial logistic  (data.model is implicitly 'multinomial')
    - continuous  -> 'ridge' (fast MSE point model, good for shrinking a huge
      feature grid), 'gaussian' (heteroscedastic N(mu, sigma) by MLE -- samples
      in the sim, gives a proper cross-entropy) or 'gaussian_mlp' (the same
      heads behind a 2-layer net, so mu and sigma become state-dependent)
  * validating + expanding the `setting:` sweep. EVERY key is griddable (scalar or
    list -> Cartesian product). A key that does not belong to the chosen model is
    an error (fail fast on a misconfigured run).
  * the primary CV metric name per model, and scoring a fitted model / the floor.

Each model's allowed `setting` keys, with (default, caster):
  * multinomial  : C
  * ridge        : alpha
  * gaussian     : weight_decay, lr, epochs
  * gaussian_mlp : hidden, weight_decay, lr, epochs
"""

import itertools

import numpy as np

# model -> {setting_key: (default, caster)}
_SPEC = {
    "multinomial": {"C": (1.0, float)},
    "ridge": {"alpha": (1.0, float)},
    "gaussian": {
        "weight_decay": (0.0, float),
        "lr": (0.05, float),
        "epochs": (3000, int),
    },
    "gaussian_mlp": {
        "hidden": (32, int),
        "weight_decay": (0.0, float),
        "lr": (0.05, float),
        "epochs": (500, int),
    },
}
_METRIC = {
    "multinomial": "log_loss",
    "ridge": "mse",
    "gaussian": "nll",
    "gaussian_mlp": "nll",
}
# Models sharing the heteroscedastic-Gaussian scoring path (NLL + binned CE).
GAUSSIAN_MODELS = ("gaussian", "gaussian_mlp")
MAX_ITER = 1000  # multinomial logistic solver cap


def as_list(x):
    """Scalar -> [scalar]; list/tuple -> list. Makes any setting knob griddable."""
    return list(x) if isinstance(x, (list, tuple)) else [x]


def resolve_model(cfg):
    """Pick the model from cfg['data'] (target_type + optional model), validated."""
    tt = cfg["data"]["target_type"]
    model = cfg["data"].get("model")
    if tt == "categorical":
        if model not in (None, "multinomial"):
            raise ValueError(
                f"categorical target uses the multinomial model; got model={model!r}"
            )
        return "multinomial"
    if tt == "continuous":
        model = model or "gaussian"
        if model not in ("ridge",) + GAUSSIAN_MODELS:
            raise ValueError(
                "continuous target model must be 'ridge', 'gaussian' or "
                f"'gaussian_mlp'; got {model!r}"
            )
        return model
    raise ValueError(f"unknown target_type {tt!r}")


def build_settings(cfg, model):
    """Validate cfg['setting'] against the model and return the Cartesian product
    of its griddable knobs as a list of dicts (one per grid cell)."""
    spec = _SPEC[model]
    given = cfg.get("setting", {}) or {}
    extra = set(given) - set(spec)
    if extra:
        raise ValueError(
            f"setting key(s) {sorted(extra)} are not valid for model '{model}'; "
            f"allowed: {sorted(spec)}"
        )
    keys = list(spec)
    axes = [
        [cast(v) for v in as_list(given.get(k, default))]
        for k, (default, cast) in spec.items()
    ]
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes)]


def setting_keys(model):
    return list(_SPEC[model])


def metric_name(model):
    return _METRIC[model]


def build_model(model, setting, seed):
    """Construct (unfitted) estimator for one setting. `seed` used by gaussian*."""
    if model == "multinomial":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=setting["C"], max_iter=MAX_ITER)
    if model == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=setting["alpha"])
    if model == "gaussian_mlp":
        from gaussian_regressor import GaussianMLPRegressor

        return GaussianMLPRegressor(
            hidden=setting["hidden"],
            weight_decay=setting["weight_decay"],
            lr=setting["lr"],
            epochs=setting["epochs"],
            seed=seed,
        )
    from gaussian_regressor import GaussianRegressor

    return GaussianRegressor(
        weight_decay=setting["weight_decay"],
        lr=setting["lr"],
        epochs=setting["epochs"],
        seed=seed,
    )


def predict_scores(model, m, Xte, yte, n_levels, show_ce=False, ce_levels=21):
    """(primary_loss, ce) for a fitted model on (Xte, yte). ce is the binned
    21-way cross-entropy for the gaussian models when show_ce, else None. The
    fall-through is the shared GAUSSIAN_MODELS path (same surface either way)."""
    if model == "multinomial":
        from sklearn.metrics import log_loss

        p = np.full((len(yte), n_levels), 1e-12)
        p[:, m.classes_] = m.predict_proba(Xte)
        ll = log_loss(yte, p / p.sum(1, keepdims=True), labels=list(range(n_levels)))
        return float(ll), None
    if model == "ridge":
        return float(np.mean((m.predict(Xte) - yte) ** 2)), None
    from gaussian_regressor import binned_logloss

    ce = (
        binned_logloss(m.predict(Xte), yte, m.predict_std(Xte), ce_levels)
        if show_ce
        else None
    )
    return m.nll(Xte, yte), ce


def floor_score(model, ytr, yte, n_levels, show_ce=False, ce_levels=21):
    """(primary_loss, ce) of the intercept-only floor (no features). The
    fall-through floor is shared by GAUSSIAN_MODELS: with no features both
    reduce to the same marginal N(mean(ytr), std(ytr))."""
    if model == "multinomial":
        from sklearn.metrics import log_loss

        c = np.bincount(ytr, minlength=n_levels) + 1.0
        ll = log_loss(
            yte, np.tile(c / c.sum(), (len(yte), 1)), labels=list(range(n_levels))
        )
        return float(ll), None
    if model == "ridge":
        return float(np.mean((ytr.mean() - yte) ** 2)), None
    mu, sigma = float(ytr.mean()), max(float(ytr.std()), 1e-3)
    var = sigma**2
    nll = float(np.mean(0.5 * (np.log(2 * np.pi * var) + (yte - mu) ** 2 / var)))
    ce = None
    if show_ce:
        from gaussian_regressor import binned_logloss

        ce = binned_logloss(
            np.full(len(yte), mu), yte, np.full(len(yte), sigma), ce_levels
        )
    return nll, ce
