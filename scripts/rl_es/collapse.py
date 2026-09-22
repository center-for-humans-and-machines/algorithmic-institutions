"""The collapse detector: when, if ever, each seed's population stopped
being a population.

READ THIS BEFORE ANY OTHER NUMBER FROM THESE RUNS. Once the argmax over the
31 ordinal punishment levels saturates at 0, every population member
implements the identical zero policy, every fitness is that same policy
scored on different episodes, and the rank vector the update is built from is
noise. **Everything after a seed's collapse generation is a random walk and
must not be interpreted as learning.**

Collapse is declared at the first generation after which the population never
recovers, using two independent signals that must agree:

  * `members_punishing_nothing` reaches the population size (or within
    `--tolerance` of it) and stays there -- the behavioural signal;
  * `signal_to_noise` (between-member fitness spread over the within-member
    standard error) falls to ~1 and stays there -- the statistical signal,
    and the one that says the ranking carries nothing.

Reporting both, rather than one, is deliberate: a population could in
principle be degenerate in punishment while still differing in some other
behaviour, and a signal-to-noise of 1 with a live population would mean
something quite different from a signal-to-noise of 1 with a dead one.

Usage:
    python scripts/rl_es/collapse.py --seeds 42,43,44,45,46
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "plots/data_analysis/evaluation/rl_manager_es")


def collapse_generation(dead, population, tolerance):
    """First generation from which `dead` never again falls below the
    threshold. None if the population never collapses."""
    alive = np.flatnonzero(dead.to_numpy() < population - tolerance)
    if len(alive) == 0:
        return 0
    last_alive = int(alive[-1])
    if last_alive == len(dead) - 1:
        return None
    return last_alive + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--population", type=int, default=40)
    ap.add_argument(
        "--tolerance",
        type=int,
        default=2,
        help="members allowed to still punish and the population still count "
        "as collapsed; 0 demands every member be dead",
    )
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, trajectories = [], []
    for seed in [int(s) for s in args.seeds.split(",")]:
        path = os.path.join(
            ROOT,
            f"artifacts/manager/rl_es_s{seed}/metrics/"
            f"rl_es_s{seed}_generations.parquet",
        )
        if not os.path.exists(path):
            continue
        g = pd.read_parquet(path).sort_values("generation")
        g["seed"] = seed
        trajectories.append(g)
        gen = collapse_generation(
            g["members_punishing_nothing"], args.population, args.tolerance
        )
        # The uninterpretable tail, and the part that is real optimisation.
        pre = g if gen is None else g[g["generation"] < gen]
        post = g.iloc[0:0] if gen is None else g[g["generation"] >= gen]
        evals = g.dropna(subset=["eval_punishment"])
        rows.append(
            {
                "seed": seed,
                "collapse_generation": gen,
                "collapsed": gen is not None,
                "generations_before_collapse": len(pre),
                "generations_of_random_walk": len(post),
                "fraction_of_budget_wasted": len(post) / len(g),
                "dead_members_final": int(g["members_punishing_nothing"].iloc[-1]),
                "signal_to_noise_final_100": float(
                    g["signal_to_noise"].tail(100).mean()
                ),
                "signal_to_noise_pre_collapse": (
                    float(pre["signal_to_noise"].mean()) if len(pre) else np.nan
                ),
                "eval_punishment_final": float(evals["eval_punishment"].iloc[-1]),
                "eval_next_reward_final": float(evals["eval_next_reward"].iloc[-1]),
                "fitness_final_100": float(g["fitness_mean"].tail(100).mean()),
                "theta_norm_final": float(g["theta_norm"].iloc[-1]),
            }
        )

    if not rows:
        print("no generation diagnostics found")
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out, "collapse.csv"), index=False)

    traj = pd.concat(trajectories)
    keep = [
        "seed",
        "generation",
        "members_punishing_nothing",
        "signal_to_noise",
        "member_punishment_sd",
        "behaviour_punishment",
        "eval_punishment",
        "fitness_mean",
        "theta_norm",
    ]
    traj[keep].to_csv(os.path.join(args.out, "collapse_trajectory.csv"), index=False)

    print("=== COLLAPSE DETECTOR (read before any other number) ===")
    print(
        summary[
            [
                "seed",
                "collapse_generation",
                "generations_of_random_walk",
                "fraction_of_budget_wasted",
                "dead_members_final",
                "signal_to_noise_final_100",
                "eval_punishment_final",
                "eval_next_reward_final",
            ]
        ].to_string(index=False)
    )
    print(
        "\ncollapse_generation None = the population never collapsed; that "
        "seed's whole run is interpretable."
    )


if __name__ == "__main__":
    main()
