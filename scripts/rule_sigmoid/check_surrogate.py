"""Score the fitted surrogate on a seed it was never fitted on.

The surrogate is the deliverable, so it gets validated as one. The whole
design is re-run on a held-out seed and the GP -- already fitted, loaded from
disk, not refitted -- is asked to predict those measurements. The number that
matters is the residual RMSE against the **noise floor**: with thousands of
cheap evaluations a GP will happily fit noise, and the only way to see that
is to ask it about data it has not seen.

`R2_vs_noise` is 1 - RMSE^2 / noise_var: at 0 the surrogate is no better
than the measurement noise on a single point, at 1 it predicts the held-out
seed perfectly. A surrogate that has fitted noise scores badly here while
scoring perfectly in sample.

Usage:
    PYTHONPATH=src python scripts/rule_sigmoid/check_surrogate.py \
        --fit-dir plots/data_analysis/rule_sigmoid \
        --run runs/sweep_check --design runs/design_1024.csv --seeds 44
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from aggregate import OBJECTIVES, load_sweep, per_point
from fit_surrogate import natural, to_unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--seeds", default="44")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    design = pd.read_csv(args.design)
    table = design.merge(
        per_point(load_sweep(args.run), seeds=seeds), on="name", how="inner"
    )
    sob = table[table["kind"] == "sobol"].reset_index(drop=True)
    u = to_unit(natural(sob))

    out, rows = {}, []
    for obj in OBJECTIVES:
        tag = obj.replace("focal_", "")
        gp = joblib.load(os.path.join(args.fit_dir, f"gp_{tag}.joblib"))
        pred = gp.predict(u)
        y = sob[obj].to_numpy(dtype=float)
        se = sob[f"se_{obj}"].to_numpy(dtype=float)
        resid = y - pred
        rmse = float(np.sqrt(np.mean(resid**2)))
        noise = float(np.sqrt(np.mean(se**2)))
        out[tag] = {
            "n": int(len(y)),
            "holdout_rmse": rmse,
            "holdout_noise_sd": noise,
            "R2_vs_noise": 1 - rmse**2 / noise**2,
            "R2_vs_variance": 1 - rmse**2 / float(np.var(y)),
            "spearman_pred_vs_measured": float(
                pd.Series(pred).corr(pd.Series(y), method="spearman")
            ),
            "bias": float(np.mean(resid)),
        }
        rows.append(
            pd.DataFrame(
                {
                    "name": sob["name"],
                    "objective": obj,
                    "predicted": pred,
                    "measured": y,
                }
            )
        )

    pd.concat(rows, ignore_index=True).to_csv(
        os.path.join(args.fit_dir, "surrogate_holdout.csv"), index=False
    )
    with open(os.path.join(args.fit_dir, "surrogate_holdout.json"), "w") as f:
        json.dump({"seeds": seeds, **out}, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
