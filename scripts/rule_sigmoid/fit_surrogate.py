"""Fit a noisy Gaussian process to the design, and read the landscape off it.

The surrogate matters more than the argmax. A point estimate hides a flat
ridge, and a flat ridge is the finding whenever it is there: it says the
objective does not care about a parameter the family was built around. So
this script reports, for each objective separately:

  * the fitted optimum, and the measured value at the nearest design points;
  * the ARD length scales, which say directly how far you can move in each
    standardised direction before the objective changes;
  * the Hessian's eigenvalues and eigenvectors at the optimum, which say the
    same thing locally and in mixed directions;
  * one-dimensional profiles through the optimum with posterior bands;
  * the *set* of parameter vectors the surrogate cannot distinguish from the
    optimum, described by its extent in every parameter.

The GP is fitted **with noise**, two ways at once: `alpha` carries each
design point's own measured squared standard error (the simulator's
heteroscedastic noise, which is known here because every point is a mean over
hundreds of episodes), and a `WhiteKernel` on top absorbs whatever that does
not explain. The seed spread in this project routinely exceeds the effects
being chased, and a noiseless fit walks straight into a sharp false optimum.

Fitted on the fit seeds only; the held-out seeds are never seen here and are
what `validate.py` scores the chosen parameters on.

Usage:
    PYTHONPATH=src python scripts/rule_sigmoid/fit_surrogate.py \
        --run runs/sweep_clone --design runs/design_1024.csv \
        --fit-seeds 42,43 --out plots/data_analysis/rule_sigmoid
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from aggregate import OBJECTIVES, load_sweep, per_point

#: The design box, in the coordinates the GP is fitted in. Every parameter is
#: mapped to [0, 1]; `tau` enters as log10 because that is how it was sampled.
AXES = ("p_max", "c0", "log10_tau", "gamma_ep", "gamma_sw")
LO = np.array([0.0, 0.0, -2.0, 0.0, 0.0])
HI = np.array([30.0, 20.0, 1.0, 3.0, 3.0])


def natural(df):
    x = df[["p_max", "c0"]].to_numpy(dtype=float)
    return np.column_stack(
        [
            x,
            np.log10(df["tau"].to_numpy(dtype=float)),
            df[["gamma_ep", "gamma_sw"]].to_numpy(dtype=float),
        ]
    )


def to_unit(x_nat):
    return (np.clip(x_nat, LO, HI) - LO) / (HI - LO)


def to_natural(u):
    return LO + np.asarray(u) * (HI - LO)


def fit_gp(u, y, se, seed=0):
    kernel = ConstantKernel(1.0, (1e-3, 1e4)) * Matern(
        length_scale=np.ones(u.shape[1]),
        length_scale_bounds=(0.03, 30.0),
        nu=2.5,
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-4, 1e4))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=np.maximum(se, 1e-6) ** 2,
        normalize_y=True,
        n_restarts_optimizer=8,
        random_state=seed,
    )
    gp.fit(u, y)
    return gp


def argmax_mean(gp, seed=0, n_starts=64):
    """Maximise the posterior mean over the unit box, multi-start L-BFGS-B."""
    rng = np.random.default_rng(seed)
    starts = qmc.Sobol(d=len(AXES), scramble=True, seed=seed).random(1024)
    best_idx = np.argsort(-gp.predict(starts))[:n_starts]
    starts = np.vstack([starts[best_idx], rng.random((8, len(AXES)))])
    best = None
    for s in starts:
        r = minimize(
            lambda z: -float(gp.predict(z.reshape(1, -1))[0]),
            s,
            bounds=[(0.0, 1.0)] * len(AXES),
            method="L-BFGS-B",
        )
        if best is None or r.fun < best.fun:
            best = r
    return np.clip(best.x, 0.0, 1.0), -float(best.fun)


def hessian(gp, u0, h=0.02):
    """Numerical Hessian of the posterior mean, in unit coordinates."""
    d = len(u0)
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            pts, signs = [], []
            for si in (1, -1):
                for sj in (1, -1):
                    e = np.zeros(d)
                    e[i] += si * h
                    e[j] += sj * h
                    pts.append(np.clip(u0 + e, 0, 1))
                    signs.append(si * sj)
            f = gp.predict(np.array(pts))
            H[i, j] = H[j, i] = float(np.dot(signs, f)) / (4 * h * h)
    return H


def profiles(gp, u0, n=81):
    """One-dimensional cuts through the optimum, with posterior bands."""
    rows = []
    for i, axis in enumerate(AXES):
        grid = np.linspace(0, 1, n)
        u = np.tile(u0, (n, 1))
        u[:, i] = grid
        mu, sd = gp.predict(u, return_std=True)
        nat = to_natural(u)[:, i]
        rows.append(
            pd.DataFrame(
                {"axis": axis, "value": nat, "mean": mu, "sd": sd, "unit": grid}
            )
        )
    return pd.concat(rows, ignore_index=True)


def flat_region(gp, best_value, delta, seed=0, n=1 << 16):
    """What the surrogate cannot tell apart from the optimum.

    A dense Sobol sweep of the box, kept where the posterior mean is within
    `delta` of the best. Reported as the extent of that set in every
    parameter: a parameter whose extent fills the box is one the objective
    does not care about.
    """
    u = qmc.Sobol(d=len(AXES), scramble=True, seed=seed).random(n)
    mu = gp.predict(u)
    keep = u[mu >= best_value - delta]
    nat = to_natural(keep)
    rows = []
    for i, axis in enumerate(AXES):
        v = nat[:, i]
        rows.append(
            {
                "axis": axis,
                "box_lo": LO[i],
                "box_hi": HI[i],
                "min": v.min(),
                "q05": np.quantile(v, 0.05),
                "median": np.median(v),
                "q95": np.quantile(v, 0.95),
                "max": v.max(),
                # 1.0 means the indistinguishable set spans the whole box in
                # this direction, i.e. the objective is flat along it
                "span_frac": (np.quantile(v, 0.95) - np.quantile(v, 0.05))
                / (HI[i] - LO[i]),
            }
        )
    return pd.DataFrame(rows), int(len(keep)), int(n)


def run(args):
    os.makedirs(args.out, exist_ok=True)
    design = pd.read_csv(args.design)
    episodes = load_sweep(args.run)
    fit_seeds = [int(s) for s in args.fit_seeds.split(",")]

    summary = per_point(episodes, seeds=fit_seeds)
    table = design.merge(summary, on="name", how="inner")
    table.to_csv(os.path.join(args.out, "design_summary_fit.csv"), index=False)

    sob = table[table["kind"] == "sobol"].reset_index(drop=True)
    u = to_unit(natural(sob))
    report = {"n_design": int(len(sob)), "fit_seeds": fit_seeds}
    models, optima = {}, []

    for obj in OBJECTIVES:
        y = sob[obj].to_numpy(dtype=float)
        se = sob[f"se_{obj}"].to_numpy(dtype=float)
        gp = fit_gp(u, y, se, seed=args.seed)
        models[obj] = gp
        k = gp.kernel_
        ls = np.atleast_1d(k.k1.k2.length_scale)
        white = float(k.k2.noise_level)
        # leave-one-out by refit is too slow at n ~ 1000; the GP's own
        # log-marginal-likelihood and the held-out seed check in validate.py
        # carry that job instead.
        pred = gp.predict(u)
        resid = y - pred
        u_star, f_star = argmax_mean(gp, seed=args.seed)
        H = hessian(gp, u_star)
        evals, evecs = np.linalg.eigh(H)
        delta = float(np.sqrt(white))
        flat, n_keep, n_tot = flat_region(gp, f_star, delta, seed=args.seed)

        tag = obj.replace("focal_", "")
        joblib.dump(gp, os.path.join(args.out, f"gp_{tag}.joblib"))
        profiles(gp, u_star).to_csv(
            os.path.join(args.out, f"profiles_{tag}.csv"), index=False
        )
        flat.to_csv(os.path.join(args.out, f"flat_region_{tag}.csv"), index=False)
        pd.DataFrame(
            {
                "axis": AXES,
                "length_scale_unit": ls,
                "length_scale_natural": ls * (HI - LO),
            }
        ).to_csv(os.path.join(args.out, f"length_scales_{tag}.csv"), index=False)
        pd.DataFrame(evecs.T, columns=list(AXES)).assign(eigenvalue=evals).to_csv(
            os.path.join(args.out, f"hessian_{tag}.csv"), index=False
        )

        nat_star = to_natural(u_star)
        optima.append(
            {
                "objective": obj,
                "p_max": nat_star[0],
                "c0": nat_star[1],
                "tau": 10 ** nat_star[2],
                "gamma_ep": nat_star[3],
                "gamma_sw": nat_star[4],
                "gp_mean": f_star,
                "gp_sd": float(
                    gp.predict(u_star.reshape(1, -1), return_std=True)[1][0]
                ),
                "best_design_point": sob.loc[int(np.argmax(y)), "name"],
                "best_design_value": float(y.max()),
            }
        )
        report[tag] = {
            "kernel": str(k),
            "white_noise_sd": delta,
            "median_measured_se": float(np.median(se)),
            "fit_rmse": float(np.sqrt(np.mean(resid**2))),
            "y_sd": float(y.std()),
            "hessian_eigenvalues": evals.tolist(),
            "flat_region_share": n_keep / n_tot,
            "flat_region_delta": delta,
        }

    pd.DataFrame(optima).to_csv(os.path.join(args.out, "optima.csv"), index=False)
    with open(os.path.join(args.out, "surrogate_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(pd.DataFrame(optima).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--fit-seeds", default="42,43")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
