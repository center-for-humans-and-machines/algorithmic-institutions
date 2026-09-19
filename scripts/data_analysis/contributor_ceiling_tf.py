"""The cheap pre-simulation screen: the contributor's TEACHER-FORCED response
at the contribution ceiling.

RCC is the contribution change of full contributors, punished minus
unpunished. PR #192 fixed the punisher's half of it (the punished share at
c = 20 is now 4.0% against the human 3.9%) and measured what is left: the
simulated player who gave the maximum and was punished cuts back 3.75 where a
real person cuts back 8.66. This script asks, in about a minute and without
spending a simulation, whether a candidate contributor installs that response
on the human manifold.

Everything is `rcb_teacher_forced.py`'s machinery, unmodified and imported:
the same `create_torch_data` lag construction, the same human default values,
the same teacher-forced forward pass (`E[c_{t+1}]` given `prev_contribution =
c_t` and `prev_punishment = p_t`, with the alignment assertion), the same
50 single-copy games. Only the population differs: this script keeps the
CEILING rows (`c_t = 20`, a valid punishment, a valid next contribution),
which `rcb_population` excludes by construction (RCB's rate is undefined
there), and reports the punished-minus-unpunished contrast.

Column `(C)`, the OBSERVED contrast over exactly these rows, is the
self-check: it must reproduce the canonical human RCC, -7.035003317850032.

Measurement only: trains nothing, writes no artifact.

Imports graph.py, so this runs on Raven only:
    .venv/bin/python scripts/data_analysis/contributor_ceiling_tf.py \
        --model baseline=<pt> --model candidate=<pt>
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "artificial_humans"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(ROOT / "scripts" / "data_analysis"))

import contribution_copula_rho as cc  # noqa: E402
import rcb_teacher_forced as tf  # noqa: E402

from aimanager.generic.graph import GraphNetwork  # noqa: E402

CEILING = 20
HUMAN_RCC = -7.035003317850032  # canonical, from the evaluation suite
OUT = ROOT / "plots/data_analysis/evaluation/contributor_ceiling_indicator"


def ceiling_population(df):
    """Full contributors with a valid punishment and a valid dc -- RCC's own
    population, built on the same frame RCB's is built on."""
    return df[df["p_valid"] & (df["c_t"] == CEILING)].copy()


def contrast(pop, col):
    p, u = pop[pop["p_t"] > 0], pop[pop["p_t"] == 0]
    return {
        "contrast": p[col].mean() - u[col].mean(),
        "dc_punished": p[col].mean(),
        "n_punished": len(p),
        "dc_unpunished": u[col].mean(),
        "n_unpunished": len(u),
        "punished_share": len(p) / len(pop) if len(pop) else np.nan,
        # how the response scales with the dose, over the punished rows only
        "slope_on_p": (
            tf.ols_slope(p["p_t"], p[col]) if p["p_t"].nunique() > 1 else np.nan
        ),
    }


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="a contributor .pt to screen; repeatable",
    )
    args = ap.parse_args()

    data, pair_id, key_to_idx, defaults = cc.load_full()
    tr = cc.select_split(key_to_idx, cc.TRAIN, cc.N_TRAIN_EP)
    te = cc.select_split(key_to_idx, cc.TEST, cc.N_TEST_EP)
    idx = np.array(sorted(set(tr.tolist()) | set(te.tolist())))
    print(f"data      {cc.rel(cc.FULL)} ({len(idx)} single-copy games)")
    print(f"  contribution default={tf.f(defaults['contribution'])}")

    rows, observed = [], None
    for spec in args.model:
        label, _, path = spec.partition("=")
        path = Path(path or label).resolve()
        model = GraphNetwork.load(str(path), device="cpu")
        model.eval()
        assert model.y_name == "contribution", f"not a contributor: {model.y_name}"
        print(f"\nmodel     {label}: {cc.rel(path)}")
        print(f"  x_encoding={[e['name'] for e in model.x_encoding]}")
        print(f"  copula_rho={model.copula_rho} copula_phi={model.copula_phi}")
        df, dense = tf.stimulus_frame(model, data, idx, label)
        tf.check_alignment(model, data, df, dense, label)
        pop = ceiling_population(df)
        rows.append({"model": label, **contrast(pop, "dc_model")})
        obs = {"model": "observed (C)", **contrast(pop, "dc_human")}
        if observed is None:
            observed = obs
        else:
            assert observed == obs, "the observed rows moved between models"

    tab = pd.DataFrame([{"model": "human canonical RCC", "contrast": HUMAN_RCC}])
    tab = pd.concat([tab, pd.DataFrame([observed] + rows)], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "teacher_forced_ceiling.csv", index=False)
    print("\n=============== TEACHER-FORCED CEILING CONTRAST ===============")
    print(tab.to_string(index=False, float_format=lambda v: f"{v: .4f}"))

    gap = abs(observed["contrast"] - HUMAN_RCC)
    print(f"\nSELF-CHECK observed vs canonical RCC: gap {tf.f(gap)}")
    ok = gap <= 5e-3
    print(f"SELF-CHECK: {'PASS' if ok else 'FAIL'}")
    print(f"wall {time.time() - t0:.1f}s")
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
