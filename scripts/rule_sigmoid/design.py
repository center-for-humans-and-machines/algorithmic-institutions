"""The space-filling design over the sigmoid rule's five parameters.

Sobol rather than uniform random: a uniform sample leaves clumps and holes,
and both get worse as the dimension grows, which is exactly the regime a
surrogate is least able to repair. A scrambled Sobol sequence at a power of
two is balanced in every one- and two-dimensional projection, which is where
the surrogate will be read.

`tau` is sampled on a log scale because it spans the interesting range
multiplicatively: 0.01 is a hard threshold to within a rounding, 10 is almost
flat across the whole contribution axis. Sampling it linearly would spend
nine tenths of the design on rules that barely differ.

The named anchors are appended to the design and run in the same rollouts, so
the incumbent, the do-nothing control and the intermediate thresholds carry
in-harness numbers rather than numbers quoted from another arm's RNG stream.

Usage:
    python scripts/rule_sigmoid/design.py --n 1024 --out <path.csv>
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import qmc

#: `tau` enters as log10; every other parameter is sampled linearly.
BOUNDS = {
    "p_max": (0.0, 30.0),
    "c0": (0.0, 20.0),
    "log10_tau": (-2.0, 1.0),
    "gamma_ep": (0.0, 3.0),
    "gamma_sw": (0.0, 3.0),
}

#: Hard thresholds are `tau -> 0`; 1e-6 saturates the logistic in float32 on
#: every integer contribution, so these rows ARE the step rules exactly
#: (asserted in tests/test_sigmoid_rule.py).
TAU0 = 1e-6

#: `kind == "clone"` puts the behavioural clone itself in the focal seat --
#: the symmetric control that says whether an asymmetry is the rule or the
#: seat. Its parameter columns are never read.
ANCHORS = [
    # name,      kind,     p_max,  c0,   tau,  gamma_ep, gamma_sw
    ("never", "anchor", 0.0, 10.0, 1.0, 0.0, 0.0),
    ("thr9_p10", "anchor", 10.0, 9.5, TAU0, 0.0, 0.0),
    ("thr9_p5", "anchor", 5.0, 9.5, TAU0, 0.0, 0.0),
    ("thr14_p5", "anchor", 5.0, 14.5, TAU0, 0.0, 0.0),
    ("thr4_p10", "anchor", 10.0, 4.5, TAU0, 0.0, 0.0),
    ("ah_punisher", "clone", 0.0, 10.0, 1.0, 0.0, 0.0),
]


def sobol_design(n, seed=0):
    """`n` scrambled Sobol points, as a frame in the natural parameters."""
    sampler = qmc.Sobol(d=len(BOUNDS), scramble=True, seed=seed)
    u = sampler.random(n)
    lo = np.array([b[0] for b in BOUNDS.values()])
    hi = np.array([b[1] for b in BOUNDS.values()])
    x = lo + u * (hi - lo)
    df = pd.DataFrame(x, columns=list(BOUNDS))
    df["tau"] = 10.0 ** df.pop("log10_tau")
    df["name"] = [f"sobol{i:04d}" for i in range(n)]
    df["kind"] = "sobol"
    return df[["name", "kind", "p_max", "c0", "tau", "gamma_ep", "gamma_sw"]]


def anchor_design():
    return pd.DataFrame(
        ANCHORS,
        columns=["name", "kind", "p_max", "c0", "tau", "gamma_ep", "gamma_sw"],
    )


def make_design(n=1024, seed=0):
    return pd.concat([anchor_design(), sobol_design(n, seed)], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = make_design(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} design rows -> {args.out}")


if __name__ == "__main__":
    main()
