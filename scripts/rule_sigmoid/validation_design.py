"""Build the held-out validation design from the fitted surrogate.

Three kinds of row, and each is there to answer a stated question:

  * the two fitted optima, one per objective -- do they hold up on seeds the
    surrogate never saw, and do they differ in kind or only in degree?
  * the incumbents (`thr9_p10`, `thr9_p5`, `never`) and the clone control --
    the bar. A fitted rule that ties with `thr9_p10` has not beaten it.
  * ridge probes and boundary probes -- walks along the flattest Hessian
    direction at each optimum, which turn "the surrogate says this direction
    is flat" into a measurement; and a copy of each optimum with the horizon
    exponents pushed below the design box, which tests the premise the
    multipliers were built on (punishment is an investment, so `gamma >= 0`)
    rather than assuming it.

Usage:
    PYTHONPATH=src python scripts/rule_sigmoid/validation_design.py \
        --fit-dir plots/data_analysis/rule_sigmoid --out runs/validation.csv
"""

import argparse

import numpy as np
import pandas as pd

from design import ANCHORS, TAU0
from fit_surrogate import AXES, HI, LO, natural, to_natural, to_unit

#: How far along a flat direction to step, in units of the unit box.
RIDGE_STEPS = (-0.35, -0.175, 0.175, 0.35)

#: Exponents outside the design box, on both sides.
#:
#: Below zero tests the premise the multipliers were built on -- punishment
#: is an investment, so you discount it towards the end of the horizon rather
#: than the start. A rule with `gamma < 0` punishes HARDER as the episode
#: runs out and as the reshuffle approaches, which is the opposite claim.
#:
#: Above the box matters because the upper edge is arbitrary in a way the
#: others are not: `P_max`'s 30 is the action space's own ceiling and `c0`'s
#: [0, 20] is the contribution scale, but nothing fixes `gamma <= 3`. If the
#: fitted optimum sits against that edge, the box is the binding constraint
#: and these rows say how much is left outside it.
BOUNDARY_GAMMAS = (-1.0, -0.5, 4.0, 6.0)


def _row(name, kind, nat, phase=0.0):
    return {
        "name": name,
        "kind": kind,
        "p_max": nat[0],
        "c0": nat[1],
        "tau": float(10 ** nat[2]),
        "gamma_ep": nat[3],
        "gamma_sw": nat[4],
        "phase": float(phase),
    }


def ridge_rows(tag, u_star, hess_csv):
    """Walk the smallest-curvature eigenvector of the posterior mean."""
    h = pd.read_csv(hess_csv)
    order = np.argsort(np.abs(h["eigenvalue"].to_numpy()))
    rows = []
    for rank, idx in enumerate(order[:2]):
        v = h.loc[idx, list(AXES)].to_numpy(dtype=float)
        for k, step in enumerate(RIDGE_STEPS):
            u = np.clip(u_star + step * v, 0.0, 1.0)
            rows.append(_row(f"{tag}_ridge{rank}_{k}", "ridge", to_natural(u)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    optima = pd.read_csv(f"{args.fit_dir}/optima.csv")
    rows = []
    for _, o in optima.iterrows():
        tag = "opt_" + o["objective"].replace("focal_", "")
        nat = natural(pd.DataFrame([o]))[0]
        rows.append(_row(tag, "optimum", nat))
        for g in BOUNDARY_GAMMAS:
            for axis in ("gamma_ep", "gamma_sw"):
                nat_b = nat.copy()
                nat_b[AXES.index(axis)] = g
                rows.append(_row(f"{tag}_{axis}{g}", "boundary", nat_b))
        # `m_sw`'s trough lands exactly on the round the switch decision is
        # taken, so `gamma_sw > 0` could be spending where tenure is long or
        # simply hiding punishment from the switch predictor. A phase shift
        # keeps the duty cycle and the average discount and moves the trough
        # off the decision round, which separates the two.
        for ph in (1.0, 2.0, 3.0):
            rows.append(_row(f"{tag}_phase{int(ph)}", "phase", nat, phase=ph))
        rows += ridge_rows(
            tag,
            to_unit(nat.reshape(1, -1))[0],
            f"{args.fit_dir}/hessian_{o['objective'].replace('focal_', '')}.csv",
        )

    anchors = [
        {
            "name": n,
            "kind": k,
            "p_max": p,
            "c0": c,
            "tau": t,
            "gamma_ep": ge,
            "gamma_sw": gs,
            "phase": 0.0,
        }
        for n, k, p, c, t, ge, gs in ANCHORS
    ]
    df = pd.DataFrame(anchors + rows)
    df = df.drop_duplicates(subset="name").reset_index(drop=True)
    # tau must stay inside the family's domain even after a ridge step
    df["tau"] = df["tau"].clip(lower=TAU0)
    assert (df["p_max"] >= 0).all() and (df["p_max"] <= HI[0] + 1e-9).all()
    assert (df["c0"] >= LO[1] - 1e-9).all()
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\n{len(df)} validation rows -> {args.out}")


if __name__ == "__main__":
    main()
