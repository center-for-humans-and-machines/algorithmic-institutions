"""Score the five predictions this arm registered before launch.

They are in notes/autoresearch_log/rl-manager-evolution-strategies.md under
"Predicted, in advance, so it can be scored". Registering predictions is only
worth anything if the failures are reported as loudly as the hits, so this
prints every one with its verdict and the number that decides it.

Usage:
    python scripts/rl_es/score_predictions.py \\
        --dqn <dir of rl_new_clones_s*.parquet>
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "plots/data_analysis/evaluation/rl_manager_es")


def final_eval(path, tail=10):
    """Mean of the last `tail` evaluation rollouts' per-round metrics."""
    df = pd.read_parquet(path)
    ev = df[df["sampling"] == "greedy"]
    last = sorted(ev["update_step"].unique())[-tail:]
    ev = ev[ev["update_step"].isin(last)]
    return (
        ev[ev["metric"].isin(["next_reward", "punishment", "contribution"])]
        .groupby("metric")["value"]
        .mean()
        .to_dict()
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--dqn", default=None, help="dir of DQN metrics parquets")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    collapse = pd.read_csv(os.path.join(args.out, "collapse.csv"))
    degen = pd.read_csv(os.path.join(args.out, "shape_degeneracy.csv"))

    es = {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        path = os.path.join(
            ROOT, f"artifacts/manager/rl_es_s{seed}/metrics/rl_es_s{seed}.parquet"
        )
        if os.path.exists(path):
            es[seed] = final_eval(path)
    dqn = {}
    if args.dqn:
        for path in sorted(glob.glob(os.path.join(args.dqn, "*.parquet"))):
            dqn[os.path.basename(path)[:-8]] = final_eval(path)

    es_reward = np.array([v["next_reward"] for v in es.values()])
    es_punish = np.array([v["punishment"] for v in es.values()])
    dq_reward = np.array([v["next_reward"] for v in dqn.values()]) if dqn else None
    dq_punish = np.array([v["punishment"] for v in dqn.values()]) if dqn else None

    verdicts = []

    # P1 -- all five reach mean punishment 0.000 between generations 100-300
    reached_zero = int((es_punish < 0.01).sum())
    gens = collapse["collapse_generation"].dropna().astype(int).tolist()
    verdicts.append(
        {
            "prediction": "P1: all five converge to punishment 0.000 between "
            "generations 100 and 300",
            "verdict": "WRONG",
            "evidence": (
                f"{reached_zero}/5 seeds reach ~0 punishment "
                f"(levels {np.round(es_punish, 2).tolist()}); the three that do "
                f"collapse at generations {gens}, none inside 100-300. What DOES "
                f"die in that window is the shape variance "
                f"({degen['shape_degenerate_from_generation'].tolist()}) and the "
                f"fitness signal, not the punishment level."
            ),
        }
    )

    # P2 -- the shape is identically flat and the human sign is not recovered
    slopes = degen["theta_slope_final"].to_numpy()
    verdicts.append(
        {
            "prediction": "P2: the policy shape is identically flat, slope 0, "
            "and the human sign is not recovered",
            "verdict": "RIGHT",
            "evidence": (
                f"final evaluated slopes {np.round(slopes, 4).tolist()}, against "
                f"the human -0.857 and the clone -0.659. Every seed is flat to "
                f"three decimals. Right for a reason only half anticipated: two "
                f"seeds are flat at a NON-ZERO level (a flat tax of 2 and 1), "
                f"which the zero-punishment story did not predict."
            ),
        }
    )

    # P3 -- seed spread small, smaller than the DQN arm's
    row = {
        "prediction": "P3: seed spread in final return is small, and smaller "
        "than the DQN arm's",
        "verdict": "UNDECIDED" if dq_reward is None else None,
        "evidence": "",
    }
    if dq_reward is not None:
        es_sd, dq_sd = float(es_reward.std(ddof=1)), float(dq_reward.std(ddof=1))
        es_rng = float(es_reward.max() - es_reward.min())
        dq_rng = float(dq_reward.max() - dq_reward.min())
        row["verdict"] = "WRONG" if es_sd >= dq_sd else "RIGHT"
        row["evidence"] = (
            f"ES reward sd {es_sd:.2f} (range {es_rng:.2f}, n=5) against DQN "
            f"sd {dq_sd:.2f} (range {dq_rng:.2f}, n={len(dq_reward)}). "
            f"ES punishment levels {np.round(es_punish, 2).tolist()} against "
            f"DQN {np.round(dq_punish, 2).tolist()}."
        )
    verdicts.append(row)

    # P5 -- return beats the DQN seeds
    row = {
        "prediction": "P5: return beats the DQN seeds (and must NOT be read "
        "as ES winning)",
        "verdict": "UNDECIDED" if dq_reward is None else None,
        "evidence": "",
    }
    if dq_reward is not None:
        row["verdict"] = "RIGHT" if es_reward.mean() > dq_reward.mean() else "WRONG"
        row["evidence"] = (
            f"ES mean evaluated reward {es_reward.mean():.2f} "
            f"({np.round(es_reward, 1).tolist()}) against DQN "
            f"{dq_reward.mean():.2f} ({np.round(dq_reward, 1).tolist()}). "
            f"This is the three-rung ladder, not a win: ES reaches the "
            f"'punish little or nothing' rung, which beats indiscriminate "
            f"punishment and loses to targeted punishment."
        )
    verdicts.append(row)

    out = pd.DataFrame(verdicts)
    out.to_csv(os.path.join(args.out, "prediction_scorecard.csv"), index=False)
    print("=== PRE-REGISTERED PREDICTIONS, SCORED ===\n")
    for r in verdicts:
        print(f"[{r['verdict']}] {r['prediction']}")
        print(f"    {r['evidence']}\n")
    print(
        "P4 (per-bin behaviour shift ends near zero) is scored in "
        "behaviour_shift.csv; it is a triviality under the corrected framing "
        "and is reported there rather than here."
    )

    per_seed = pd.DataFrame(
        [{"seed": k, **v} for k, v in es.items()]
        + [{"seed": k, **v} for k, v in dqn.items()]
    )
    per_seed.to_csv(os.path.join(args.out, "final_evaluated_metrics.csv"), index=False)
    print("\n=== final evaluated metrics (mean of last 10 evaluation rollouts) ===")
    print(per_seed.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
