"""Model dispatch for the hand-crafted linear baselines (issue #119).

One source of truth, shared by the CV driver (run_baseline_cv) and the best-model
inspector (inspect_best_model), for:
  * choosing the estimator from `data.target_type` + `data.model`
    - categorical -> multinomial logistic  (data.model is implicitly 'multinomial')
    - continuous  -> 'ridge' (fast MSE point model, good for shrinking a huge
      feature grid) or 'gaussian' (heteroscedastic N(mu, sigma) by MLE -- samples
      in the sim, gives a proper cross-entropy)
  * validating + expanding the `setting:` sweep. EVERY key is griddable (scalar or
    list -> Cartesian product). A key that does not belong to the chosen model is
    an error (fail fast on a misconfigured run).
  * the primary CV metric name per model, and scoring a fitted model / the floor.

Each model's allowed `setting` keys, with (default, caster):
  * multinomial : C
  * xgb         : n_estimators, max_depth, learning_rate, min_child_weight,
                  subsample, reg_lambda
  * ridge       : alpha
  * gaussian    : weight_decay, lr, epochs
"""

import itertools

import numpy as np

# model -> {setting_key: (default, caster)}
_SPEC = {
    "multinomial": {"C": (1.0, float)},
    "xgb": {
        "n_estimators": (300, int),
        "max_depth": (4, int),
        "learning_rate": (0.05, float),
        "min_child_weight": (1.0, float),
        "subsample": (1.0, float),
        "reg_lambda": (1.0, float),
    },
    "ridge": {"alpha": (1.0, float)},
    "gaussian": {
        "weight_decay": (0.0, float),
        "lr": (0.05, float),
        "epochs": (3000, int),
    },
}
_METRIC = {
    "multinomial": "log_loss",
    "xgb": "log_loss",
    "ridge": "mse",
    "gaussian": "nll",
}
MAX_ITER = 1000  # multinomial logistic solver cap


def as_list(x):
    """Scalar -> [scalar]; list/tuple -> list. Makes any setting knob griddable."""
    return list(x) if isinstance(x, (list, tuple)) else [x]


def resolve_model(cfg):
    """Pick the model from cfg['data'] (target_type + optional model), validated."""
    tt = cfg["data"]["target_type"]
    model = cfg["data"].get("model")
    if tt == "categorical":
        if model not in (None, "multinomial", "xgb"):
            raise ValueError(
                f"categorical target model must be 'multinomial' or 'xgb'; "
                f"got model={model!r}"
            )
        return model or "multinomial"
    if tt == "continuous":
        model = model or "gaussian"
        if model not in ("ridge", "gaussian"):
            raise ValueError(
                f"continuous target model must be 'ridge' or 'gaussian'; got {model!r}"
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


class XGBLevelClassifier:
    """XGBClassifier over possibly non-contiguous integer levels.

    xgboost requires y in {0..k-1}; a CV train fold can miss some of the 21
    contribution levels, so fit() re-encodes the labels and `classes_` holds
    the ORIGINAL levels -- the shared categorical scoring / sim-adapter path
    maps predict_proba onto the full label grid via `classes_`, exactly as
    for the multinomial. Picklable by reference (joblib bundles resolve
    `baseline_models` through the scripts/baselines sys.path entry)."""

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        from xgboost import XGBClassifier

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        enc = {c: i for i, c in enumerate(self.classes_)}
        self._m = XGBClassifier(**self.params)
        self._m.fit(X, np.array([enc[v] for v in y]))
        return self

    def predict_proba(self, X):
        return self._m.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def feature_importances_(self):
        return self._m.feature_importances_


def build_model(model, setting, seed):
    """Construct (unfitted) estimator for one setting. `seed` is used by the
    gaussian init and the xgb subsampling RNG."""
    if model == "multinomial":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=setting["C"], max_iter=MAX_ITER)
    if model == "xgb":
        return XGBLevelClassifier(
            n_estimators=int(setting["n_estimators"]),
            max_depth=int(setting["max_depth"]),
            learning_rate=setting["learning_rate"],
            min_child_weight=setting["min_child_weight"],
            subsample=setting["subsample"],
            reg_lambda=setting["reg_lambda"],
            objective="multi:softprob",
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
            verbosity=0,
        )
    if model == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=setting["alpha"])
    from gaussian_regressor import GaussianRegressor

    return GaussianRegressor(
        weight_decay=setting["weight_decay"],
        lr=setting["lr"],
        epochs=setting["epochs"],
        seed=seed,
    )


def predict_scores(model, m, Xte, yte, n_levels, show_ce=False, ce_levels=21):
    """(primary_loss, ce) for a fitted model on (Xte, yte). ce is the binned
    21-way cross-entropy for the gaussian model when show_ce, else None."""
    if model in ("multinomial", "xgb"):
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
    """(primary_loss, ce) of the intercept-only floor (no features)."""
    if model in ("multinomial", "xgb"):
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
