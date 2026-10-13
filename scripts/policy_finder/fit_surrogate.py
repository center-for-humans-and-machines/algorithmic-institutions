"""Fit a Gaussian process to a policy-finder sweep: landscapes.json (#238).

A sweep scores each design point once (500 episodes, one seed), so its best
measured point is the top of hundreds of noisy estimates and reads high by
luck. A Gaussian process fitted to all of them pools each point's neighbours:
its surface says what the rule's pool looks like across its ranges, where it
peaks, and which params matter at all. The fit follows #219's
(scripts/rule_sigmoid/fit_surrogate.py), made generic over a rule's
`sweep_config`:

  * each ranged param is one axis, mapped to [0, 1] of its range (in log for
    `log` ranges); fixed params are not axes
  * a Matern-5/2 kernel with one length scale per axis (ARD), fitted with
    noise two ways: `alpha` carries each point's own squared `pool_se`, and a
    `WhiteKernel` absorbs whatever that leaves unexplained
  * the peak is the maximum of the posterior mean over the box, `int` params
    kept on integers, the rule's constraints met; it can lie between design
    points and has never been simulated

The peak is a candidate, not a result: the sweep's final best comes from the
confirmation sim that reruns it and the top measured points on fresh seeds
(#238 step 2).

Writes <sim dir>/landscapes.json beside the sweep's sweep.json:

    {"rule": ..., "fit": {"kernel", "n_points", "median_se", "extra_noise_sd",
                          "fit_rmse"},
     "params": {"<axis>": {"range", "log", "type", "effect",
                           "length_scale"}},
     "peak": {"params": {...}, "mean", "sd", "at_edge": [...]},
     "points": [{"name", "mean", "sd"}, ...],
     "profiles": {"<axis>": {"x": [...], "mean": [...], "sd": [...]}}}

`at_edge` names the params whose peak sits at an end of its range and that
move the pool by at least EDGE_EFFECT: the pool still rises towards the
range's end, so the optimum may lie outside it, where the surface is least
sure. `effect` is how far the pool moves along the axis's profile (max minus
min of its mean, in pool points): a param that does not matter moves it by
less than the points' se. `length_scale` is the kernel's, in fractions of
the axis's range; it only compares axes (a smooth bowl fits long length
scales and a large amplitude), and one far above the others' marks a flat
axis. `points` is the surface at each design point, in sweep.json's order;
`profiles` cut the surface along each axis through the peak, the other
params held there. `mean` and `sd` are the surface's pool and its posterior
standard deviation.

Usage:
    python scripts/policy_finder/fit_surrogate.py <sweep sim dir> [--rule <rule.yml>]

The rule defaults to sweep.json's `rule`, resolved from the working
directory or the sim dir's parents. Local, CPU, seconds.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from aimanager.manager.rule import check_params, read_rule
from aimanager.simulation.sweep import SWEEP_FILE

LANDSCAPES_FILE = "landscapes.json"
#: Candidates the peak search scores before polishing the best of them
N_CANDIDATES = 1 << 14
N_POLISH = 16
#: Points per float profile; an int profile takes every integer
PROFILE_POINTS = 101
#: How close to a range's end (in [0, 1]) a peak counts as on its edge, and
#: how far (pool points) a param must move the pool for that to count: along
#: a flat param the peak's value is arbitrary
EDGE = 0.02
EDGE_EFFECT = 1.0


def axes(rule):
    """The ranged params of `sweep_config`, in its order."""
    out = []
    for name, spec in rule["sweep_config"].items():
        if isinstance(spec, list):
            out.append(
                {
                    "name": name,
                    "low": float(spec[0]),
                    "high": float(spec[1]),
                    "log": len(spec) == 3,
                    "int": rule["params"][name]["type"] == "int",
                }
            )
    return out


def to_unit(values, ax):
    """Natural param values [n, d] -> [0, 1] of each axis's range."""
    u = np.empty_like(values, dtype=float)
    for j, a in enumerate(ax):
        x, lo, hi = values[:, j], a["low"], a["high"]
        if a["log"]:
            x, lo, hi = np.log(x), np.log(lo), np.log(hi)
        u[:, j] = (x - lo) / (hi - lo)
    return u


def to_natural(u, ax):
    """[0, 1] -> natural values; `int` axes rounded onto their integers."""
    x = np.empty_like(u, dtype=float)
    for j, a in enumerate(ax):
        lo, hi = a["low"], a["high"]
        if a["log"]:
            x[:, j] = np.exp(np.log(lo) + u[:, j] * (np.log(hi) - np.log(lo)))
        else:
            x[:, j] = lo + u[:, j] * (hi - lo)
        # a range's ends exactly, not their float round trip
        x[:, j] = np.where(u[:, j] <= 0, lo, np.where(u[:, j] >= 1, hi, x[:, j]))
        if a["int"]:
            x[:, j] = np.clip(np.round(x[:, j]), lo, hi)
    return x


def snap(u, ax):
    """Move `int` coordinates onto their integers, in [0, 1]."""
    return to_unit(to_natural(u, ax), ax)


def fit_gp(u, y, se, seed=0):
    """Matern-5/2 ARD plus two noise terms, as in #219.

    `normalize_y` scales the target by its standard deviation but not
    `alpha` or the `WhiteKernel`, so `alpha` is given in those units. A length
    scale at its upper bound is a param that does not matter and a noise
    term at its lower bound one the points' se explains: findings, so
    sklearn's warnings about them are silenced.
    """
    y_std = float(np.std(y))
    kernel = ConstantKernel(1.0, (1e-3, 1e4)) * Matern(
        length_scale=np.full(u.shape[1], 0.5),
        length_scale_bounds=(0.01, 100.0),
        nu=2.5,
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e2))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=(np.maximum(se, 1e-6) / y_std) ** 2,
        normalize_y=True,
        n_restarts_optimizer=8,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(u, y)
    return gp, y_std


def find_peak(gp, ax, accept, seed=0):
    """The maximum of the posterior mean over the box, in [0, 1].

    Scores a Sobol cloud (ints snapped), polishes the best candidates along
    their float axes with L-BFGS-B, int axes held, and keeps the best that
    `accept` (the rule's constraints) lets through.
    """
    cloud = snap(
        qmc.Sobol(d=len(ax), scramble=True, seed=seed).random(N_CANDIDATES), ax
    )
    order = np.argsort(-gp.predict(cloud))
    floats = [j for j, a in enumerate(ax) if not a["int"]]
    best, best_mean = None, -np.inf
    for start in cloud[order[:N_POLISH]]:
        u = start.copy()
        if floats:

            def neg_mean(z, u=u):
                v = u.copy()
                v[floats] = z
                return -float(gp.predict(v[None])[0])

            r = minimize(neg_mean, u[floats], bounds=[(0, 1)] * len(floats))
            u[floats] = np.clip(r.x, 0, 1)
        for cand in (u, start):
            mean = float(gp.predict(cand[None])[0])
            if mean > best_mean and accept(cand):
                best, best_mean = cand, mean
                break
    if best is None:  # nothing polished passed: first passing cloud point
        best = next((c for c in cloud[order] if accept(c)), None)
    if best is None:
        raise ValueError("no point of the box meets the rule's constraints")
    return best


def profiles(gp, ax, peak):
    """Cuts of the surface along each axis through the peak."""
    out = {}
    for j, a in enumerate(ax):
        if a["int"]:
            ints = np.arange(int(a["low"]), int(a["high"]) + 1, dtype=float)
            if len(ints) > PROFILE_POINTS:
                ints = np.unique(
                    np.round(np.linspace(a["low"], a["high"], PROFILE_POINTS))
                )
            nat = np.tile(to_natural(peak[None], ax), (len(ints), 1))
            nat[:, j] = ints
            grid = to_unit(nat, ax)
        else:
            grid = np.tile(peak, (PROFILE_POINTS, 1))
            grid[:, j] = np.linspace(0, 1, PROFILE_POINTS)
        mean, sd = gp.predict(grid, return_std=True)
        out[a["name"]] = {
            "x": to_natural(grid, ax)[:, j].tolist(),
            "mean": mean.tolist(),
            "sd": sd.tolist(),
        }
    return out


def landscapes(sweep, rule, rule_path, seed=0):
    """Fit the sweep's points; return the landscapes.json content."""
    ax = axes(rule)
    if not ax:
        raise ValueError(f"{rule_path}: no ranged params to fit")
    names = [a["name"] for a in ax]
    pts = sweep["points"]
    values = np.array([[p["params"][n] for n in names] for p in pts], dtype=float)
    y = np.array([p["pool"] for p in pts], dtype=float)
    se = np.array([p["pool_se"] for p in pts], dtype=float)
    u = to_unit(values, ax)

    gp, y_std = fit_gp(u, y, se, seed=seed)
    fixed = {n: s for n, s in rule["sweep_config"].items() if not isinstance(s, list)}

    def params_of(cand):
        nat = to_natural(cand[None], ax)[0]
        out = dict(fixed)
        for a, v in zip(ax, nat):
            out[a["name"]] = int(v) if a["int"] else float(v)
        return {n: out[n] for n in rule["sweep_config"]}

    def accept(cand):
        try:
            check_params(rule_path, rule, params_of(cand), "peak")
        except ValueError:
            return False
        return True

    peak = find_peak(gp, ax, accept, seed=seed)
    peak_mean, peak_sd = gp.predict(peak[None], return_std=True)
    smooth, smooth_sd = gp.predict(u, return_std=True)
    kernel = gp.kernel_
    length = np.atleast_1d(kernel.k1.k2.length_scale)
    cuts = profiles(gp, ax, peak)
    effect = {n: max(c["mean"]) - min(c["mean"]) for n, c in cuts.items()}
    return {
        "rule": str(rule_path),
        "fit": {
            "kernel": str(kernel),
            "n_points": len(pts),
            "median_se": float(np.median(se)),
            # the scatter the points' own se leaves unexplained, in pool units
            "extra_noise_sd": float(np.sqrt(kernel.k2.noise_level) * y_std),
            "fit_rmse": float(np.sqrt(np.mean((y - smooth) ** 2))),
        },
        "params": {
            a["name"]: {
                "range": [a["low"], a["high"]],
                "log": a["log"],
                "type": "int" if a["int"] else "float",
                "effect": effect[a["name"]],
                "length_scale": float(ls),
            }
            for a, ls in zip(ax, length)
        },
        "peak": {
            "params": params_of(peak),
            "mean": float(peak_mean[0]),
            "sd": float(peak_sd[0]),
            "at_edge": [
                a["name"]
                for a, v in zip(ax, peak)
                if (v <= EDGE or v >= 1 - EDGE) and effect[a["name"]] >= EDGE_EFFECT
            ],
        },
        "points": [
            {"name": p["name"], "mean": float(m), "sd": float(s)}
            for p, m, s in zip(pts, smooth, smooth_sd)
        ],
        "profiles": cuts,
    }


def find_rule(rule, sim_dir):
    """sweep.json's rule path, from the working directory or a sim dir parent."""
    for base in (Path.cwd(), *sim_dir.resolve().parents):
        if (base / rule).exists():
            return base / rule
    sys.exit(f"cannot find the rule {rule}: pass --rule")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sim_dir", help="the sweep's sim dir, holding sweep.json")
    ap.add_argument("--rule", help="the rule YAML (default: sweep.json's rule)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sim_dir = Path(args.sim_dir)
    sweep = json.loads((sim_dir / SWEEP_FILE).read_text())
    rule_path = Path(args.rule) if args.rule else find_rule(sweep["rule"], sim_dir)
    try:
        rule, _ = read_rule(rule_path)
        result = landscapes(sweep, rule, rule_path, seed=args.seed)
    except ValueError as e:
        sys.exit(str(e))
    result["rule"] = sweep["rule"]
    out = sim_dir / LANDSCAPES_FILE
    out.write_text(json.dumps(result, indent=2) + "\n")

    peak, fit = result["peak"], result["fit"]
    best = max(sweep["points"], key=lambda p: p["pool"])
    at_best = next(p for p in result["points"] if p["name"] == best["name"])
    print(f"wrote {out}")
    print(
        f"  fit: {fit['n_points']} points, median se {fit['median_se']:.2f},"
        f" extra noise {fit['extra_noise_sd']:.2f}, rmse {fit['fit_rmse']:.2f}"
    )
    for name, p in result["params"].items():
        print(
            f"  {name}: moves the pool by {p['effect']:.1f},"
            f" length scale {p['length_scale']:.2f} of its range"
        )
    print(f"  peak {peak['params']}: {peak['mean']:.1f} +- {peak['sd']:.1f}", end="")
    print(f" (on the edge of {', '.join(peak['at_edge'])})" if peak["at_edge"] else "")
    print(
        f"  measured best {best['name']}: {best['pool']:.1f} +- {best['pool_se']:.1f},"
        f" surface {at_best['mean']:.1f} +- {at_best['sd']:.1f}"
    )


if __name__ == "__main__":
    main()
