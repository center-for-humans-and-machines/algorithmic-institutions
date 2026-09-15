"""F1 pre-flight: the contribution trunk's size profile, teacher-forced.

Reads the candidate (gaussian_mlp_v2 + `inv_group_size`) and the incumbent
(gaussian_mlp_v2, no own-group-size input) on the REAL human rows -- no
simulation -- and prints, per own-group size `n` and per split:

  * the teacher-forced residual `y - mu` with its standard error and the cell
    count. This is the defect the experiment exists to close: the incumbent
    under-predicts lone players by +2.599 at 7.8 SE on the train split and is
    unbiased at n >= 3 (log Declaration, "Where the incumbent actually fails").
  * mean `mu` by `n`, against the observed `y` mean of the same cells and
    against the human marginal means of the Declaration's size table.
  * mean `sigma(x)` by `n` -- the emission width the sim samples with.

Rows are restricted to `round_number >= 4` (the Declaration's convention; it
reproduces its cell counts 190/432/573/657/950/1304/1372/745 on train).

F1, the pre-registered stop rule of this experiment: the candidate's TRAIN
residual at n = 1 must lie within +-0.7 of zero AND its 4-fold CV NLL (read
from the bundle's own `cv_metric`) must not exceed 2.7125. A failure means the
committed pipeline differs from the measurement the plan was written on -- a
bug to fix, not a result to accept.

Runs locally (CPU torch, no PyG):
    .venv/bin/python scripts/baselines/size_profile_preflight.py \
        [--config configs/training/baselines/contribution/\
gaussian_mlp_v2_inv_size.yml] \
        [--candidate PATH] [--incumbent PATH]
"""

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

from handcrafted_grid import load_config, prepare_data  # noqa: E402

DEFAULT_CFG = (
    ROOT / "configs/training/baselines/contribution/gaussian_mlp_v2_inv_size.yml"
)
CANDIDATE = (
    ROOT / "artifacts/baselines/contribution_gaussian_mlp_v2_inv_size_best.joblib"
)
INCUMBENT = ROOT / "artifacts/baselines/contribution_gaussian_mlp_v2_best.joblib"

MIN_ROUND = 4  # the Declaration's "rounds >= 4"
SIZES = list(range(1, 9))
# human marginal mean contribution by own group size, rounds >= 4, single-copy
# data (Declaration, "The behavioral finding this experiment acts on").
HUMAN_MEAN = {1: 10.54, 2: 9.66, 3: 8.82, 4: 10.55, 5: 10.06, 6: 9.82, 7: 9.75, 8: 8.22}
HUMAN_SE = {1: 0.54, 2: 0.29, 3: 0.26, 4: 0.26, 5: 0.19, 6: 0.16, 7: 0.15, 8: 0.15}
# F1
F1_RESID_TOL = 0.7
F1_CV_NLL_MAX = 2.7125


def _rel(path):
    p = Path(path)
    return str(p.relative_to(ROOT)) if p.is_absolute() and ROOT in p.parents else path


def table(header, rows, indent="  "):
    cells = [[str(c) for c in header]] + [[str(c) for c in r] for r in rows]
    w = [max(len(r[i]) for r in cells) for i in range(len(header))]
    for j, r in enumerate(cells):
        line = r[0].ljust(w[0]) + "".join(f"  {r[i]:>{w[i]}}" for i in range(1, len(r)))
        print(indent + line)
        if j == 0:
            print(indent + "-" * len(line))
    print()


def score(bundle, prep):
    """Teacher-forced mu / sigma of one bundle on one prepared split, each on
    ITS OWN feature list + scaler."""
    cols = [prep["col_of"][f] for f in bundle["features"]]
    X = bundle["scaler"].transform(prep["X"][:, cols])
    m = bundle["estimator"]
    mu = np.asarray(m.predict(X), float).reshape(-1)
    sigma = np.asarray(m.predict_std(X), float).reshape(-1)
    return dict(mu=mu, sigma=sigma, resid=prep["y"] - mu)


def load_split(cfg, split):
    c = copy.deepcopy(cfg)
    if split == "test":
        c["data"]["data_file"] = cfg["data"]["data_file"].replace("_train", "_test")
    prep = prepare_data(c, ROOT)
    keep = prep["X"][:, prep["col_of"]["round_number"]] >= MIN_ROUND
    prep["n_all"] = len(prep["y_cont"])
    prep["X"] = prep["X"][keep]
    prep["y"] = prep["y_cont"][keep]
    prep["size"] = np.rint(prep["X"][:, prep["col_of"]["group_size"]]).astype(int)
    prep["file"] = c["data"]["data_file"]
    return prep


def cells(prep):
    return [(n, prep["size"] == n) for n in SIZES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--candidate", default=str(CANDIDATE))
    ap.add_argument("--incumbent", default=str(INCUMBENT))
    args = ap.parse_args()

    import joblib

    cfg = load_config(Path(args.config))
    models = [
        ("candidate", joblib.load(args.candidate)),
        ("incumbent", joblib.load(args.incumbent)),
    ]
    splits = {}
    for split in ("train", "test"):
        prep = load_split(cfg, split)
        splits[split] = (prep, {n: score(b, prep) for n, b in models})

    print("size-profile pre-flight -- teacher-forced on human rows, no simulation")
    print(f"  config    {_rel(args.config)}")
    for (name, b), path in zip(models, (args.candidate, args.incumbent)):
        size_feat = "inv_group_size" in b["features"]
        print(
            f"  {name} {b['model']}  {len(b['features'])} features"
            f"  size feature: {'inv_group_size' if size_feat else 'NONE'}"
            f"  cv_nll {b.get('cv_metric', float('nan')):.6f}"
        )
        print(f"    {_rel(path)}")
    for split, (prep, _) in splits.items():
        print(
            f"  {split:<5} {prep['file']}  ({len(prep['y'])} of {prep['n_all']} rows "
            f"with round_number >= {MIN_ROUND})"
        )
    print()

    print("[1] teacher-forced residual  y - mu  by own group size")
    rows = []
    for split, (prep, sc) in splits.items():
        for n, m in cells(prep):
            k = int(m.sum())
            row = [split, n, k]
            for name, _ in models:
                r = sc[name]["resid"][m]
                row += (
                    ["-", "-"]
                    if k == 0
                    else [f"{r.mean():+.4f}", f"{r.std(ddof=1) / np.sqrt(k):.4f}"]
                )
            rows.append(row)
    table(
        ["split", "n", "rows", "cand resid", "SE", "inc resid", "SE"],
        rows,
    )

    print("[2] mean mu by own group size, vs observed y and the human marginal")
    rows = []
    for split, (prep, sc) in splits.items():
        for n, m in cells(prep):
            k = int(m.sum())
            rows.append(
                [
                    split,
                    n,
                    k,
                    "-" if k == 0 else f"{sc['candidate']['mu'][m].mean():.4f}",
                    "-" if k == 0 else f"{sc['incumbent']['mu'][m].mean():.4f}",
                    "-" if k == 0 else f"{prep['y'][m].mean():.4f}",
                    f"{HUMAN_MEAN[n]:.2f}",
                    f"{HUMAN_SE[n]:.2f}",
                ]
            )
    table(
        ["split", "n", "rows", "cand mu", "inc mu", "y mean", "human", "hum SE"],
        rows,
    )
    print(
        "  human = the Declaration's marginal human mean over the whole\n"
        "  single-copy data (rounds >= 4), not a per-split figure.\n"
    )

    print("[3] mean sigma(x) by own group size")
    rows = []
    for split, (prep, sc) in splits.items():
        for n, m in cells(prep):
            k = int(m.sum())
            rows.append(
                [
                    split,
                    n,
                    k,
                    "-" if k == 0 else f"{sc['candidate']['sigma'][m].mean():.4f}",
                    "-" if k == 0 else f"{sc['incumbent']['sigma'][m].mean():.4f}",
                ]
            )
    table(["split", "n", "rows", "cand sigma", "inc sigma"], rows)

    prep_tr, sc_tr = splits["train"]
    m1 = prep_tr["size"] == 1
    r1 = float(sc_tr["candidate"]["resid"][m1].mean())
    se1 = float(sc_tr["candidate"]["resid"][m1].std(ddof=1) / np.sqrt(int(m1.sum())))
    cv = float(models[0][1].get("cv_metric", float("nan")))
    ok_r = abs(r1) <= F1_RESID_TOL
    ok_cv = cv <= F1_CV_NLL_MAX
    print("[4] F1 -- pre-registered stop rule")
    print(
        f"  train residual at n = 1 : {r1:+.4f} (SE {se1:.4f}, "
        f"{int(m1.sum())} rows)  |{r1:+.4f}| <= {F1_RESID_TOL} -> "
        f"{'PASS' if ok_r else 'FAIL'}"
    )
    print(
        f"  4-fold CV NLL           : {cv:.6f} <= {F1_CV_NLL_MAX} -> "
        f"{'PASS' if ok_cv else 'FAIL'}"
    )
    print(f"F1: {'PASS' if (ok_r and ok_cv) else 'FAIL -- STOP and debug'}")


if __name__ == "__main__":
    main()
