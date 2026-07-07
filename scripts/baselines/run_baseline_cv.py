"""Hand-crafted linear baseline: CV sweep over block/feature-set combinations
(issue #119).

New config idea (configs/training/baselines/*/handcrafted_grid.yml): each block
lists a `components` pool and a set of candidate feature `sets`. Every block is
independently OFF or set to one of its `sets`; the Cartesian product over blocks
gives the feature sets to evaluate. Each (feature-set x regularization) is scored
by k-fold CV on the TRAIN split (the test fold lives in a separate locked file
and is never opened here). Results are written best-to-worst to `cv.output`.

No new data engineering: features come from handcrafted_grid.build_feature_pool
(the existing, validated pool). This script only enumerates, fits, and ranks.

Parallelized across CPU cores (12k+ feature-sets x reg x folds).

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
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/baselines"))
from handcrafted_grid import load_config, prepare_data  # noqa: E402

MAX_ITER = 1000

# --------------------------------------------------------------------------- #
# worker: read-only data shared once per process via the initializer
# --------------------------------------------------------------------------- #
_W = {}


def _init(X, y, fold_row, cat, n_levels, regs, dev_folds):
    _W.update(X=X, y=y, fr=fold_row, cat=cat, nl=n_levels,
              regs=regs, dev=dev_folds)


def _score(Xtr, ytr, Xte, yte):
    """CV score for one train/val split (log_loss if categorical else mse)."""
    scores = []
    cat, nl, regs = _W["cat"], _W["nl"], _W["regs"]
    for reg in regs:
        if Xtr.shape[1] == 0:  # intercept-only floor
            if cat:
                c = np.bincount(ytr, minlength=nl) + 1.0
                proba = np.tile(c / c.sum(), (len(yte), 1))
                scores.append(log_loss(yte, proba, labels=list(range(nl))))
            else:
                scores.append(float(np.mean((ytr.mean() - yte) ** 2)))
            continue
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        if cat:
            m = LogisticRegression(C=reg, max_iter=MAX_ITER).fit(Ztr, ytr)
            p = np.full((len(yte), nl), 1e-12)
            p[:, m.classes_] = m.predict_proba(Zte)
            scores.append(log_loss(yte, p / p.sum(1, keepdims=True),
                                   labels=list(range(nl))))
        else:
            m = Ridge(alpha=reg).fit(Ztr, ytr)
            scores.append(float(np.mean((m.predict(Zte) - yte) ** 2)))
    return scores  # one score per reg


def _worker(cols):
    """Return, per reg, the mean & se of the CV score across folds."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    X, y, fr, regs, dev = _W["X"], _W["y"], _W["fr"], _W["regs"], _W["dev"]
    cols = list(cols)
    per_fold = []  # [n_folds][n_regs]
    for vf in dev:
        tr, te = fr != vf, fr == vf
        per_fold.append(_score(X[tr][:, cols], y[tr], X[te][:, cols], y[te]))
    arr = np.array(per_fold)  # [n_folds, n_regs]
    return list(zip(regs, arr.mean(0), arr.std(0) / np.sqrt(len(dev))))


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
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "configs/training/baselines/contribution/handcrafted_grid_cont.yml")
    cfg = load_config(cfg_path)
    cat = cfg["data"]["target_type"] == "categorical"
    n_levels = cfg["data"].get("categorical_levels", 0)
    regs = cfg["regularization"]["C" if cat else "alpha"]

    prep = prepare_data(cfg, ROOT)
    X = np.ascontiguousarray(prep["X"])
    y = prep["y_cat"] if cat else prep["y_cont"]
    fr = prep["fold_row"]
    dev_folds = sorted(set(fr.tolist()))
    print(f"target={cfg['data']['target']} ({'categorical' if cat else 'continuous'}), "
          f"rows={len(y)}, folds={dev_folds}, reg={regs}")
    # prove the training pool: rows kept by the mask, and the per-fold CV split
    # (each fit trains on all-but-one fold, validates on the held-out fold)
    print(f"masked rows (mask={cfg['data']['mask']}) = {len(fr)}  "
          f"-- these are the ONLY data points used")
    for vf in dev_folds:
        n_val = int((fr == vf).sum())
        print(f"  fold {vf}: validate on {n_val:5d}  |  train on {len(fr) - n_val:5d}")

    combos = list(enumerate_feature_sets(cfg))
    print(f"feature-sets={len(combos)}  ->  {len(combos) * len(regs)} (set x reg) rows")

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
        max_workers=n_workers, initializer=_init,
        initargs=(X, y, fr, cat, n_levels, regs, dev_folds),
    ) as ex:
        futs = {ex.submit(_worker, t): i for i, t in enumerate(tasks)}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc="feature-sets", unit="set"):
            results[futs[fut]] = fut.result()  # per-task -> responsive bar

    rows = []
    for (label, feats), res in zip(combos, results):
        for reg, mean, se in res:
            rows.append({
                "mean_loss": mean, "se_loss": se, "reg": reg,
                "n_features": len(feats), "config": label,
                "features": ";".join(feats),
            })

    metric = "log_loss" if cat else "mse"
    df = pd.DataFrame(rows).sort_values("mean_loss").reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    df = df.rename(columns={"mean_loss": metric})
    out_path = ROOT / cfg["cv"]["output"]
    df.to_csv(out_path, index=False)

    print(f"\nwrote {len(df)} rows -> {out_path.relative_to(ROOT)}")
    print(f"\ntop 10 by {metric}:")
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
