"""Hand-crafted linear baseline: CV sweep over block/feature-set x hyper-parameter
combinations (issue #119).

Each block lists a `components` pool and candidate feature `sets`; every block is
independently OFF or set to one of its `sets`, and the Cartesian product over
blocks gives the feature sets. Orthogonally, `setting:` lists the model's
hyper-parameters, each griddable (scalar or list). Every (feature-set x setting)
is scored by k-fold CV on the TRAIN split (the test fold lives in a separate
locked file and is never opened here); results are written best-to-worst to
`cv.output`.

The estimator is chosen by data.target_type + data.model (see baseline_models):
categorical -> multinomial logistic; continuous -> ridge (fast MSE) or gaussian
(heteroscedastic N(mu, sigma) by MLE). With cv.show_ce, the gaussian run also
reports a binned 21-way cross-entropy alongside its NLL.

Usage:
    .venv/bin/python scripts/baselines/run_baseline_cv.py [config.yml]
"""

import itertools
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
# One BLAS thread per process -- we parallelize across processes, so multi-threaded
# BLAS would oversubscribe the cores. Set before numpy imports (also in spawned
# children, which re-import this module).
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
import torch as th
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/baselines"))
from handcrafted_grid import load_config, prepare_data  # noqa: E402
from baseline_models import (  # noqa: E402
    build_model,
    build_settings,
    floor_score,
    metric_name,
    predict_scores,
    resolve_model,
    setting_keys,
)

# --------------------------------------------------------------------------- #
# worker: read-only data shared once per process via the initializer
# --------------------------------------------------------------------------- #
_W = {}


def _init(
    X, y, fold_row, model, n_levels, settings, dev_folds, seed, show_ce, ce_levels
):
    th.set_num_threads(1)  # we parallelize across processes; keep torch single-threaded
    _W.update(
        X=X,
        y=y,
        fr=fold_row,
        model=model,
        nl=n_levels,
        settings=settings,
        dev=dev_folds,
        seed=seed,
        show_ce=show_ce,
        ce_levels=ce_levels,
    )


def _score(Xtr, ytr, Xte, yte):
    """(primary, ce) per setting for one train/val split. Standardisation is fit
    once on the fold's train; the floor (no features) is setting-independent."""
    model, nl, settings = _W["model"], _W["nl"], _W["settings"]
    seed, show_ce, k = _W["seed"], _W["show_ce"], _W["ce_levels"]
    if Xtr.shape[1] == 0:
        return [floor_score(model, ytr, yte, nl, show_ce, k) for _ in settings]
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    out = []
    for s in settings:
        m = build_model(model, s, seed).fit(Ztr, ytr)
        out.append(predict_scores(model, m, Zte, yte, nl, show_ce, k))
    return out


def _worker(cols):
    """Return, per setting, (setting, mean, se, ce_mean, ce_se) across folds.
    ce_* are None unless cv.show_ce is set (gaussian only)."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    X, y, fr, settings, dev = (_W["X"], _W["y"], _W["fr"], _W["settings"], _W["dev"])
    cols = list(cols)
    n = np.sqrt(len(dev))
    per_fold = [
        _score(X[fr != vf][:, cols], y[fr != vf], X[fr == vf][:, cols], y[fr == vf])
        for vf in dev
    ]
    prim = np.array([[t[0] for t in fold] for fold in per_fold])  # [folds, settings]
    if per_fold[0][0][1] is not None:
        ce = np.array([[t[1] for t in fold] for fold in per_fold])
        return [
            (
                s,
                prim[:, j].mean(),
                prim[:, j].std() / n,
                ce[:, j].mean(),
                ce[:, j].std() / n,
            )
            for j, s in enumerate(settings)
        ]
    return [
        (s, prim[:, j].mean(), prim[:, j].std() / n, None, None)
        for j, s in enumerate(settings)
    ]


# --------------------------------------------------------------------------- #
# enumeration
# --------------------------------------------------------------------------- #
def enumerate_feature_sets(cfg):
    """Every combination: each block OFF or one of its `sets`. Yields
    (label, feature_names) with features concatenated across chosen blocks."""
    names = list(cfg["blocks"])
    per_block = []
    for b in names:
        opts = [("off", [])]
        for i, s in enumerate(cfg["blocks"][b]["sets"]):
            opts.append((f"s{i}", list(s)))
        per_block.append(opts)
    for combo in itertools.product(*per_block):
        feats, parts = [], []
        for b, (lab, fl) in zip(names, combo):
            if lab != "off":
                parts.append(f"{b}:{lab}")
                feats += fl
        seen, uniq = set(), []
        for f in feats:
            if f not in seen:
                seen.add(f)
                uniq.append(f)
        yield ("+".join(parts) if parts else "floor", uniq)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    cfg_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else (
            ROOT / "configs/training/baselines/contribution/ridge.yml"
        )
    )
    cfg = load_config(cfg_path)
    model = resolve_model(cfg)
    cat = model in ("multinomial", "xgb")
    n_levels = cfg["data"].get("categorical_levels", 0)
    seed = cfg["cv"]["seed"]  # gaussian init seed reuses cv.seed
    settings = build_settings(cfg, model)  # validated + Cartesian expanded
    metric = metric_name(model)
    show_ce = bool(cfg["cv"].get("show_ce", False)) and model == "gaussian"
    ce_levels = int(cfg["data"].get("categorical_levels", 21))

    prep = prepare_data(cfg, ROOT)
    X = np.ascontiguousarray(prep["X"])
    y = prep["y_cat"] if cat else prep["y_cont"]
    fr = prep["fold_row"]
    dev_folds = sorted(set(fr.tolist()))
    print(
        f"target={cfg['data']['target']} model={model}, rows={len(y)}, "
        f"folds={dev_folds}, settings={len(settings)}"
    )
    print(
        f"masked rows (mask={cfg['data']['mask']}) = {len(fr)}  "
        f"-- these are the ONLY data points used"
    )
    for vf in dev_folds:
        n_val = int((fr == vf).sum())
        print(f"  fold {vf}: validate on {n_val:5d}  |  train on {len(fr) - n_val:5d}")

    combos = list(enumerate_feature_sets(cfg))
    print(
        f"feature-sets={len(combos)}  ->  {len(combos) * len(settings)} "
        f"(set x setting) rows"
    )

    col_of = prep["col_of"]
    tasks = [tuple(col_of[f] for f in feats) for _, feats in combos]

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(x, **_):
            return x

    n_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"parallel fits across {n_workers} workers")
    results = [None] * len(tasks)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init,
        initargs=(
            X,
            y,
            fr,
            model,
            n_levels,
            settings,
            dev_folds,
            seed,
            show_ce,
            ce_levels,
        ),
    ) as ex:
        futs = {ex.submit(_worker, t): i for i, t in enumerate(tasks)}
        for fut in tqdm(
            as_completed(futs), total=len(futs), desc="feature-sets", unit="set"
        ):
            results[futs[fut]] = fut.result()  # per-task -> responsive bar

    keys = setting_keys(model)
    rows = []
    for (label, feats), res in zip(combos, results):
        for setting, mean, se, ce_mean, ce_se in res:
            row = {
                **setting,
                "mean_loss": mean,
                "se_loss": se,
                "n_features": len(feats),
                "config": label,
                "features": ";".join(feats),
            }
            if ce_mean is not None:
                row["ce"], row["ce_se"] = ce_mean, ce_se
            rows.append(row)

    df = pd.DataFrame(rows).sort_values("mean_loss").reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    df = df.rename(columns={"mean_loss": metric})
    # column order: rank, metric, se, [ce, ce_se], swept settings, n, config, features
    order = (
        ["rank", metric, "se_loss"]
        + (["ce", "ce_se"] if "ce" in df.columns else [])
        + keys
        + ["n_features", "config", "features"]
    )
    df = df[order]
    out_path = ROOT / cfg["cv"]["output"]
    df.to_csv(out_path, index=False)

    print(f"\nwrote {len(df)} rows -> {out_path.relative_to(ROOT)}")
    print(f"\ntop 10 by {metric}:")
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
