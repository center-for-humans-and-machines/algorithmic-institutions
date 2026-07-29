"""Evaluation-suite main script (#128).

    python -m aimanager evaluate <simulation config>

Reads the finished simulation's per_round.parquet from the config's
output_dir (the simulation must have run with save_per_round: true),
converts the human reference and every pairing into the canonical frame,
and writes two files to the output_dir:

- metrics.csv: one row per (pairing, metric) with the raw discrepancy
  d(human, sim), the signed std diagnostic where retained (CA/CC/CE),
  and the observation counts -- raw-unit quick-look values
- scores.csv: the #132 normalised scores against the human-vs-human
  noise ceiling (score, numerator, denominator, repeats_used,
  n_repeats, seed); scoring_repeats and scoring_seed in the sim config
  override the 500 / 42 defaults
"""

import os
import sys

import pandas as pd

from aimanager.evaluation_suite.convert import (
    HUMAN_DATA_FILE,
    load_human,
    load_sim,
)
from aimanager.evaluation_suite.metrics import GROUPS
from aimanager.evaluation_suite.scoring import score_all

DIAGNOSTIC_ROWS = ["CA", "CC", "CE"]


def evaluate(human, sims):
    """d(human, sim) for every metric row and pairing, as a long frame."""
    rows = []
    for run, sim in sims.items():
        for group in GROUPS.values():
            for name, kind in group.KINDS.items():
                extract = getattr(group, name.lower())
                rows.append(
                    {
                        "run": run,
                        "metric": name,
                        "kind": kind,
                        "d": group.d(name, human, sim),
                        "std_diff": (
                            group.std_diff(name, sim, human)
                            if name in DIAGNOSTIC_ROWS
                            else None
                        ),
                        "n_human": len(extract(human)),
                        "n_sim": len(extract(sim)),
                    }
                )
    return pd.DataFrame(rows)


def run_cli(config, config_path):
    if "output_dir" in config:
        output_dir = config["output_dir"]
    else:
        output_dir = os.path.splitext(config_path)[0]

    parquet_path = os.path.join(output_dir, "per_round.parquet")
    if not os.path.exists(parquet_path):
        print(
            f"Error: {parquet_path} not found. Run the simulation with "
            "save_per_round: true first (python -m aimanager simulate "
            f"{config_path}).",
            file=sys.stderr,
        )
        sys.exit(1)

    switch_every = config.get("switch_every", 4)
    human = load_human(HUMAN_DATA_FILE, switch_every=switch_every)
    sims = load_sim(parquet_path, switch_every=switch_every)

    result = evaluate(human, sims)
    out_path = os.path.join(output_dir, "metrics.csv")
    result.to_csv(out_path, index=False)
    print(f"{len(result)} rows ({len(sims)} pairings) -> {out_path}")

    scores = score_all(
        human,
        sims,
        n_repeats=config.get("scoring_repeats", 500),
        seed=config.get("scoring_seed", 42),
    )
    scores_path = os.path.join(output_dir, "scores.csv")
    scores.to_csv(scores_path, index=False)
    print(f"{len(scores)} scores -> {scores_path}")
