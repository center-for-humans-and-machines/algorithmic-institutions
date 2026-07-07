"""Refit the best (or n-th) model from a baseline CV-output CSV on the FULL train
split (all 4 folds, no held-out set) and report its standardized coefficients.

The test fold stays locked -- this only uses the train file the config points at.
Features are standardized (as in CV). Per-feature value reported:
  * continuous (Ridge)  -> signed standardized coefficient (direction + magnitude)
  * categorical (multinomial, one coef per class) -> magnitude via --cat-metric:
    absmag (mean |coef| across classes, default) / l2 (coef norm) / perm (mean
    increase in in-sample log-loss when the feature is shuffled).

Usage:
    .venv/bin/python scripts/baselines/inspect_best_model.py smoke_cv_output.csv
    .venv/bin/python scripts/baselines/inspect_best_model.py out.csv --rank 3 \
        --config configs/training/baselines/contribution/handcrafted_grid_cont.yml
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/baselines"))
from handcrafted_grid import load_config, prepare_data  # noqa: E402

DEFAULT_CFG = ROOT / "configs/training/baselines/contribution/handcrafted_grid_cont.yml"


def _fit_betas(prep, cat, n_levels, feats, reg, cat_metric="perm", n_repeats=10):
    """Refit on ALL train rows; return {feature: value}, intercept, in-sample metric.

    Continuous -> signed standardized beta. Categorical value depends on
    `cat_metric`:
      * perm   -> mean increase in in-sample log-loss when the feature is shuffled
      * l2     -> L2 norm of the feature's coefficients across the 21 classes
      * absmag -> mean absolute coefficient across the 21 classes
    All three are magnitude, not direction."""
    y = prep["y_cat"] if cat else prep["y_cont"]
    cols = [prep["col_of"][f] for f in feats]
    if not feats:
        return {}, None, None
    sc = StandardScaler().fit(prep["X"][:, cols])
    Xs = sc.transform(prep["X"][:, cols])
    if cat:
        m = LogisticRegression(C=reg, max_iter=2000).fit(Xs, y)

        def _ll(Z):
            p = np.full((len(y), n_levels), 1e-12)
            p[:, m.classes_] = m.predict_proba(Z)
            return log_loss(y, p / p.sum(1, keepdims=True),
                            labels=list(range(n_levels)))

        base = _ll(Xs)
        if cat_metric == "l2":
            vals = {f: float(np.linalg.norm(m.coef_[:, j]))
                    for j, f in enumerate(feats)}
        elif cat_metric == "absmag":
            vals = {f: float(np.abs(m.coef_[:, j]).mean())
                    for j, f in enumerate(feats)}
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
    m = Ridge(alpha=reg).fit(Xs, y)
    insample = float(np.mean((m.predict(Xs) - y) ** 2))
    betas = {f: float(b) for f, b in zip(feats, m.coef_)}
    return betas, float(m.intercept_), insample


def _feats_of(row):
    return [] if pd.isna(row["features"]) else str(row["features"]).split(";")


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="baseline CV-output CSV (best model = rank 1)")
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--rank", type=int, default=1, help="which row to refit (single view)")
    ap.add_argument("--top-n", type=int, default=1,
                    help=">1: side-by-side beta table of the top N distinct fits")
    ap.add_argument("--cat-metric", choices=["perm", "l2", "absmag"], default="absmag",
                    help="categorical importance: absmag (mean |coef|) / "
                         "l2 (coef norm) / perm (delta log-loss on shuffle)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    metric_col = "log_loss" if "log_loss" in df.columns else "mse"
    cfg = load_config(Path(args.config))
    cat = cfg["data"]["target_type"] == "categorical"
    n_levels = cfg["data"].get("categorical_levels", 0)
    prep = prepare_data(cfg, ROOT)
    cm = args.cat_metric
    value_name = {"perm": "perm_dloss", "l2": "coef_l2",
                  "absmag": "coef_absmag"}[cm] if cat else "std_beta"
    print(f"train rows={len(prep['fold_row'])}, folds={sorted(set(prep['fold_row'].tolist()))}, "
          f"value={value_name}")

    if args.top_n > 1:
        # top-N DISTINCT configs (best row per feature-set), side by side
        top = (df.sort_values("rank").drop_duplicates("config").head(args.top_n))
        fits = []  # (col_label, betas)
        print(f"\ntop {len(top)} distinct fits:")
        for k, (_, row) in enumerate(top.iterrows(), start=1):
            feats = _feats_of(row)
            betas, _, _ = _fit_betas(prep, cat, n_levels, feats, float(row["reg"]),
                                     cat_metric=cm)
            fits.append((f"#{k}", betas))
            print(f"  #{k}: reg={row['reg']}  n={row['n_features']}  "
                  f"cv_{metric_col}={row[metric_col]:.4f}  [{row['config']}]")
        all_feats = {f for _, b in fits for f in b}
        order = sorted(all_feats, key=lambda f: -max(abs(b.get(f, 0)) for _, b in fits))
        table = pd.DataFrame(index=order)
        for label, betas in fits:
            table[label] = [betas.get(f, np.nan) for f in order]
        print()
        print(table.to_string(float_format=lambda x: f"{x:.3f}", na_rep=""))
        print(f"\n(rows = features, cols = fits #1..#{len(fits)}, values = {value_name}; "
              f"blank = feature not in that fit)")
        return

    # single-model detailed view
    row = df[df["rank"] == args.rank].iloc[0]
    feats = _feats_of(row)
    reg = float(row["reg"])
    print(f"\nrank {args.rank}: config='{row['config']}'  reg={reg}  "
          f"n_features={len(feats)}  cv_{metric_col}={row[metric_col]:.4f}")
    if not feats:
        print("floor model -- no features, nothing to report.")
        return
    betas, intercept, insample = _fit_betas(prep, cat, n_levels, feats, reg,
                                            cat_metric=cm)
    extra = f"   intercept={intercept:.3f}" if intercept is not None else ""
    print(f"in-sample {metric_col}={insample:.4f}{extra}")
    print(f"\n  {'feature':<32}{value_name:>10}")
    for f, v in sorted(betas.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {f:<32}{v:>10.3f}")
    if cat:
        print("  (perm importance: mean increase in in-sample log-loss when the "
              "feature is shuffled; magnitude only)")
    else:
        print("  (standardized signed coefficients)")


if __name__ == "__main__":
    main()
