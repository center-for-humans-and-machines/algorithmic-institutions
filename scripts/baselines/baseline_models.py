"""Model dispatch for the hand-crafted linear baselines (issue #119).

One source of truth, shared by the CV driver (run_baseline_cv) and the best-model
inspector (inspect_best_model), for:
  * choosing the estimator from `data.target_type` + `data.model`
    - categorical -> multinomial logistic  (data.model is implicitly 'multinomial')
    - continuous  -> 'ridge' (fast MSE point model, good for shrinking a huge
      feature grid), 'gaussian' (heteroscedastic N(mu, sigma) by MLE -- samples
      in the sim, gives a proper cross-entropy), 'gaussian_mlp' (the same heads
      behind a 2-layer net, so mu and sigma become state-dependent) or
      'gaussian_mlp_inflated' (that trunk with probability atoms at the status
      quo and the two corners; a proper categorical, scored by the 21-way CE)
  * validating + expanding the `setting:` sweep. EVERY key is griddable (scalar or
    list -> Cartesian product). A key that does not belong to the chosen model is
    an error (fail fast on a misconfigured run).
  * the primary CV metric name per model, and scoring a fitted model / the floor.

Each model's allowed `setting` keys, with (default, caster):
  * multinomial           : C
  * ridge                 : alpha
  * gaussian              : weight_decay, lr, epochs
  * gaussian_mlp          : hidden, weight_decay, lr, epochs
  * gaussian_mlp_inflated : hidden, weight_decay, lr, epochs, atoms

`gaussian_mlp_inflated` is the one model that needs more than (Z, y) to fit:
its status-quo atom sits at the RAW prev_contribution level, so `build_model`
takes the position of that feature INSIDE the task's feature list as
`prev_index` and `fit` takes the raw column itself. See `prev_position` for
why that position is not `prep['col_of']['prev_contribution']`.
"""

import itertools

import numpy as np

PREV_FEATURE = "prev_contribution"
# Models whose `fit` needs the raw PREV_FEATURE column (and a `prev_index`).
NEEDS_PREV = ("gaussian_mlp_inflated",)


def parse_atoms(value):
    """Griddable `atoms` string -> validated tuple of atom names.

    'prev,0,20' -> ('prev', '0', '20'); '' / None -> (). An unknown or
    repeated name is an error (the estimator owns the vocabulary)."""
    from gaussian_regressor import InflatedGaussianMLPRegressor

    text = "" if value is None else str(value).strip()
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    return InflatedGaussianMLPRegressor._check_atoms(names)


def _atoms_setting(value):
    """Caster for the `atoms` knob: validate, return the normalised string (so
    it round-trips through the CV CSV and the saved bundle as one token)."""
    return ",".join(parse_atoms(value))


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
    "gaussian_mlp_inflated": {
        "hidden": (32, int),
        "weight_decay": (0.0, float),
        "lr": (0.05, float),
        "epochs": (500, int),
        # which probability atoms the emission carries; griddable like the rest
        "atoms": ("prev,0,20", _atoms_setting),
    },
}
_METRIC = {
    "multinomial": "log_loss",
    "ridge": "mse",
    "gaussian": "nll",
    "gaussian_mlp": "nll",
    # the inflated emission IS a categorical: its primary metric is the CE
    "gaussian_mlp_inflated": "ce",
}
# Models sharing the heteroscedastic-Gaussian scoring path (NLL + binned CE).
GAUSSIAN_MODELS = ("gaussian", "gaussian_mlp")
# Models that report a binned 21-way cross-entropy under cv.show_ce.
CE_MODELS = GAUSSIAN_MODELS + NEEDS_PREV
MAX_ITER = 1000  # multinomial logistic solver cap


def prev_position(model, ordering, key):
    """Position of PREV_FEATURE inside ONE TASK'S ordering -- the estimator's
    `prev_index` -- or None for models that do not need it.

    Two different indices are in play and swapping them is silent: the POOL
    column index (`prep['col_of'][PREV_FEATURE]`, the column in the full
    feature pool) and this POSITION inside the task's own `cols` / `features`,
    which changes with every feature set. The estimator indexes the
    standardised matrix it is handed, so it needs the position; the pool index
    would point the status-quo atom at some other feature. Call as
    `prev_position(model, cols, col_of[PREV_FEATURE])` or
    `prev_position(model, features, PREV_FEATURE)`.

    A feature set without PREV_FEATURE is a config error for these models."""
    if model not in NEEDS_PREV:
        return None
    seq = list(ordering)
    if key is None or key not in seq:
        raise ValueError(
            f"model {model!r} requires the feature {PREV_FEATURE!r}: its "
            "status-quo atom has no level without it, and this feature set "
            "does not contain it -- config error"
        )
    return seq.index(key)


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
        if model not in ("ridge",) + GAUSSIAN_MODELS + NEEDS_PREV:
            raise ValueError(
                "continuous target model must be 'ridge', 'gaussian', "
                f"'gaussian_mlp' or 'gaussian_mlp_inflated'; got {model!r}"
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


def build_model(model, setting, seed, prev_index=None):
    """Construct (unfitted) estimator for one setting. `seed` used by gaussian*.
    `prev_index` is required by NEEDS_PREV models and ignored by the rest; it
    is a task-local POSITION, see `prev_position`."""
    if model == "multinomial":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=setting["C"], max_iter=MAX_ITER)
    if model == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=setting["alpha"])
    if model == "gaussian_mlp_inflated":
        from gaussian_regressor import InflatedGaussianMLPRegressor

        if prev_index is None:
            raise ValueError(
                f"build_model({model!r}) needs prev_index (the position of "
                f"{PREV_FEATURE!r} inside this task's features)"
            )
        return InflatedGaussianMLPRegressor(
            hidden=setting["hidden"],
            weight_decay=setting["weight_decay"],
            lr=setting["lr"],
            epochs=setting["epochs"],
            seed=seed,
            atoms=parse_atoms(setting["atoms"]),
            prev_index=prev_index,
        )
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
    if model in NEEDS_PREV:
        # the mixture's own 21-way cross-entropy is BOTH the primary metric and
        # what show_ce reports -- one number, not two.
        ce = m.nll(Xte, yte)
        return ce, (ce if show_ce else None)
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
    if model in NEEDS_PREV:
        # same floor as the multinomial: the smoothed marginal histogram of the
        # levels, scored by the same 21-way cross-entropy the model reports.
        tr = np.clip(np.rint(np.asarray(ytr, float)), 0, ce_levels - 1).astype(int)
        te = np.clip(np.rint(np.asarray(yte, float)), 0, ce_levels - 1).astype(int)
        c = np.bincount(tr, minlength=ce_levels) + 1.0
        ce = float(-np.mean(np.log((c / c.sum())[te])))
        return ce, (ce if show_ce else None)
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
