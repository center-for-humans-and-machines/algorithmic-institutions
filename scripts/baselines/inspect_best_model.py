"""Refit the best (or n-th) model from a baseline CV-output CSV on the FULL train
split and report its standardized coefficients, and optionally save it + evaluate
on the locked test split (--save-best).

The estimator + its swept hyper-parameters come from the config (data.model) and
the CSV row (see baseline_models). Features are standardized as in CV. Per-feature
value reported:
  * ridge / gaussian (mu-head) -> signed standardized coefficient
  * multinomial (one coef per class) -> magnitude via --cat-metric:
    absmag (mean |coef|, default) / l2 (coef norm) / perm (delta in-sample log-loss).

Usage:
    .venv/bin/python scripts/baselines/inspect_best_model.py out.csv \
        --config configs/training/baselines/contribution/ridge.yml
    .venv/bin/python scripts/baselines/inspect_best_model.py out.csv --save-best
"""

import argparse
import copy
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/baselines"))
from handcrafted_grid import load_config, prepare_data  # noqa: E402
from gaussian_regressor import binned_logloss  # noqa: E402
from baseline_models import (  # noqa: E402
    build_model,
    floor_score,
    predict_scores,
    resolve_model,
    setting_keys,
)

DEFAULT_CFG = ROOT / "configs/training/baselines/contribution/ridge.yml"
ARTIFACTS = ROOT / "artifacts/baselines"


def _feats_of(row):
    return [] if pd.isna(row["features"]) else str(row["features"]).split(";")


def _row_setting(row, model, cfg):
    """(label, setting) for a CV-output row: read the model's swept hyper-parameter
    columns off the row. label is a short display string."""
    setting = {
        k: (int(row[k]) if k == "epochs" else float(row[k]))
        for k in setting_keys(model)
    }
    label = " ".join(f"{k}={v}" for k, v in setting.items())
    return label, setting


def _y(prep, model):
    return prep["y_cat"] if model == "multinomial" else prep["y_cont"]


def _fit(prep, model, feats, setting, seed):
    """Fit `model` on ALL of prep's rows with `setting`; return (m, scaler, cols)."""
    cols = [prep["col_of"][f] for f in feats]
    sc = StandardScaler().fit(prep["X"][:, cols])
    m = build_model(model, setting, seed).fit(
        sc.transform(prep["X"][:, cols]), _y(prep, model)
    )
    return m, sc, cols


def _primary(model, m, sc, cols, prep, n_levels):
    X = sc.transform(prep["X"][:, cols])
    return predict_scores(model, m, X, _y(prep, model), n_levels)[0]


def save_best(args, df, cfg, model, n_levels, metric_col, prep_tr):
    """Refit the CV-best model on all train with the row's swept setting, evaluate
    on the locked test split, and save the fitted model + metadata under artifacts/.

    Both continuous models are sampleable in the sim (#121): gaussian from its
    trained heteroscedastic head, contribution ~ N(mu(x), sigma(x)); ridge is a
    point model, so it stores a homoscedastic `sigma` = sqrt(train MSE) and samples
    contribution ~ N(mu(x), sigma). Features are standardized (scaler saved)."""
    import joblib

    row = df[df["rank"] == args.rank].iloc[0]
    feats = _feats_of(row)
    seed = cfg["cv"]["seed"]
    label, setting = _row_setting(row, model, cfg)

    cfg_te = copy.deepcopy(cfg)
    cfg_te["data"]["data_file"] = cfg["data"]["data_file"].replace("_train", "_test")
    prep_te = prepare_data(cfg_te, ROOT)

    m, sc, cols = _fit(prep_tr, model, feats, setting, seed)
    train_m = _primary(model, m, sc, cols, prep_tr, n_levels)
    test_m = _primary(model, m, sc, cols, prep_te, n_levels)
    floor = floor_score(model, _y(prep_tr, model), _y(prep_te, model), n_levels)[0]

    # default_values + switch_every let the sim adapter rebuild features exactly as
    # training did; default_values["contribution"] is what the env fills invalid
    # agents with (see #121).
    default_values = {
        k: (float(v) if hasattr(v, "__float__") else v)
        for k, v in prep_tr["default_values"].items()
    }
    bundle = {
        "model": model,
        "estimator": m,
        "scaler": sc,
        "features": feats,
        **setting,
        "target": cfg["data"]["target"],
        "target_type": cfg["data"]["target_type"],
        "n_levels": n_levels,
        "config": str(args.config),
        "metric": metric_col,
        "cv_metric": float(row[metric_col]),
        "train_metric": train_m,
        "test_metric": test_m,
        "test_floor": floor,
        "default_values": default_values,
        "switch_every": prep_tr["switch_every"],
    }

    # Both continuous models are distributional -> report a binned 21-way test CE
    # (comparable to the multinomial / GNN). ridge: N(mu(x), sigma) homoscedastic;
    # gaussian: N(mu(x), sigma(x)) from its trained head.
    extra = []
    if model in ("ridge", "gaussian"):
        Xte, yte = sc.transform(prep_te["X"][:, cols]), prep_te["y_cont"]
        k = int(cfg["data"].get("categorical_levels", 21))
        if model == "ridge":  # point model -> homoscedastic sigma makes it sampleable
            mu_tr = m.predict(sc.transform(prep_tr["X"][:, cols]))
            sigma = float(np.sqrt(np.mean((mu_tr - prep_tr["y_cont"]) ** 2)))
            bundle["sigma"] = sigma  # sim samples N(mu(x), sigma) on standardized feats
            sig_te = sigma
            extra.append(f"  sigma (homoscedastic, sqrt train MSE) = {sigma:.4f}")
        else:  # gaussian: trained sigma(x) head (heteroscedastic, no scalar sigma)
            sig_te = m.predict_std(Xte)
            bundle["sigma_mean"] = float(sig_te.mean())  # info; sim uses sigma(x)
            extra.append(
                f"  mean sigma(x) = {bundle['sigma_mean']:.2f}  "
                f"(heteroscedastic head)"
            )
        test_ll = binned_logloss(m.predict(Xte), yte, sig_te, k)
        bundle["test_logloss_binned"] = test_ll
        extra.append(
            f"  TEST binned log-loss = {test_ll:.4f}   "
            f"({k}-way; comparable to the GNN's categorical log-loss)"
        )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{cfg['data']['target']}_{model}_best.joblib"
    joblib.dump(bundle, ARTIFACTS / name)

    print(f"\nbest model [{model}]: {row['config']}  ({label})  n={len(feats)}")
    print(f"  cv    {metric_col} = {row[metric_col]:.4f}")
    print(f"  train {metric_col} = {train_m:.4f}  ({len(prep_tr['fold_row'])} rows)")
    print(
        f"  TEST  {metric_col} = {test_m:.4f}  ({len(prep_te['fold_row'])} rows)  "
        f"[floor {floor:.4f}]"
    )
    for line in extra:
        print(line)
    print(f"  saved -> {(ARTIFACTS / name).relative_to(ROOT)}")


def _fit_betas(
    prep, model, n_levels, feats, setting, seed, cat_metric="absmag", n_repeats=10
):
    """{feature: value}, intercept, in-sample metric. Continuous -> signed
    standardized mu-coefficient. Multinomial -> per-feature magnitude (cat_metric:
    absmag / l2 / perm delta-log-loss)."""
    y = _y(prep, model)
    cols = [prep["col_of"][f] for f in feats]
    if not feats:
        return {}, None, None
    sc = StandardScaler().fit(prep["X"][:, cols])
    Xs = sc.transform(prep["X"][:, cols])
    m = build_model(model, setting, seed).fit(Xs, y)

    if model == "multinomial":

        def _ll(Z):
            p = np.full((len(y), n_levels), 1e-12)
            p[:, m.classes_] = m.predict_proba(Z)
            return log_loss(
                y, p / p.sum(1, keepdims=True), labels=list(range(n_levels))
            )

        base = _ll(Xs)
        if cat_metric == "l2":
            vals = {
                f: float(np.linalg.norm(m.coef_[:, j])) for j, f in enumerate(feats)
            }
        elif cat_metric == "absmag":
            vals = {f: float(np.abs(m.coef_[:, j]).mean()) for j, f in enumerate(feats)}
        else:  # perm
            rng = np.random.default_rng(0)
            vals = {}
            for j, f in enumerate(feats):
                d = 0.0
                for _ in range(n_repeats):
                    Zp = Xs.copy()
                    Zp[:, j] = Xs[rng.permutation(len(Xs)), j]
                    d += _ll(Zp) - base
                vals[f] = d / n_repeats
        return vals, None, base

    insample = float(np.mean((m.predict(Xs) - y) ** 2))
    betas = {f: float(b) for f, b in zip(feats, m.coef_)}
    return betas, float(m.intercept_), insample


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="baseline CV-output CSV (best model = rank 1)")
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument(
        "--rank", type=int, default=1, help="which row to refit (single view)"
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=1,
        help=">1: side-by-side coefficient table of the top N distinct fits",
    )
    ap.add_argument(
        "--cat-metric",
        choices=["perm", "l2", "absmag"],
        default="absmag",
        help="multinomial importance: absmag / l2 / perm (delta log-loss)",
    )
    ap.add_argument(
        "--save-best",
        action="store_true",
        help="refit the best (rank) model on all train, save under "
        "artifacts/, and evaluate on the locked test split",
    )
    ap.add_argument(
        "--name",
        default=None,
        help="bundle filename for --save-best "
        "(default <target>_<model>_best.joblib)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    metric_col = next(m for m in ("log_loss", "nll", "mse") if m in df.columns)
    cfg = load_config(Path(args.config))
    model = resolve_model(cfg)
    n_levels = cfg["data"].get("categorical_levels", 0)
    seed = cfg["cv"]["seed"]
    prep = prepare_data(cfg, ROOT)
    if args.save_best:
        save_best(args, df, cfg, model, n_levels, metric_col, prep)
        return

    cm = args.cat_metric
    value_name = (
        {"perm": "perm_dloss", "l2": "coef_l2", "absmag": "coef_absmag"}[cm]
        if model == "multinomial"
        else "std_beta"
    )
    print(
        f"train rows={len(prep['fold_row'])}, "
        f"folds={sorted(set(prep['fold_row'].tolist()))}, value={value_name}"
    )

    if args.top_n > 1:
        top = df.sort_values("rank").drop_duplicates("config").head(args.top_n)
        fits = []
        print(f"\ntop {len(top)} distinct fits:")
        for k, (_, row) in enumerate(top.iterrows(), start=1):
            feats = _feats_of(row)
            label, setting = _row_setting(row, model, cfg)
            betas, _, _ = _fit_betas(
                prep, model, n_levels, feats, setting, seed, cat_metric=cm
            )
            fits.append((f"#{k}", betas))
            print(
                f"  #{k}: {label}  n={row['n_features']}  "
                f"cv_{metric_col}={row[metric_col]:.4f}  [{row['config']}]"
            )
        all_feats = {f for _, b in fits for f in b}
        order = sorted(all_feats, key=lambda f: -max(abs(b.get(f, 0)) for _, b in fits))
        table = pd.DataFrame(index=order)
        for lbl, betas in fits:
            table[lbl] = [betas.get(f, np.nan) for f in order]
        print()
        print(table.to_string(float_format=lambda x: f"{x:.3f}", na_rep=""))
        print(
            f"\n(rows = features, cols = fits #1..#{len(fits)}, values = {value_name}; "
            f"blank = feature not in that fit)"
        )
        return

    row = df[df["rank"] == args.rank].iloc[0]
    feats = _feats_of(row)
    label, setting = _row_setting(row, model, cfg)
    print(
        f"\nrank {args.rank}: config='{row['config']}'  {label}  "
        f"n_features={len(feats)}  cv_{metric_col}={row[metric_col]:.4f}"
    )
    if not feats:
        print("floor model -- no features, nothing to report.")
        return
    betas, intercept, insample = _fit_betas(
        prep, model, n_levels, feats, setting, seed, cat_metric=cm
    )
    extra = f"   intercept={intercept:.3f}" if intercept is not None else ""
    print(
        f"in-sample {'mse' if model != 'multinomial' else 'log_loss'}="
        f"{insample:.4f}{extra}"
    )
    print(f"\n  {'feature':<32}{value_name:>12}")
    for f, v in sorted(betas.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {f:<32}{v:>12.3f}")


if __name__ == "__main__":
    main()
