"""Seed spread of the interventional targeting gradient.

``contributor_punishment_intervention.py`` measures one model.  Whether the
dependence it finds is a property of the architecture or of one training draw
is a different question, and the only way to answer it is to run the same probe
on retrains of the same model.  This collects the ``summary.json`` files of
those runs into one table and reports the spread, so the gradient can be judged
against the noise the model's own training carries rather than against a seed
sd borrowed from a different statistic.

    python scripts/data_analysis/contributor_intervention_seed_spread.py \
        <shipped summary.json> <seed summary.json> ... [--out DIR]

Pure pandas: runs locally.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BAND_LABELS = ["0-4", "5-9", "10-14", "15-19"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--names", default=None, help="comma-separated arm names")
    ap.add_argument("--out", default=None, help="directory for seed_spread.csv")
    args = ap.parse_args()

    names = (
        args.names.split(",")
        if args.names
        else [Path(p).parent.name for p in args.summaries]
    )
    assert len(names) == len(args.summaries), "names and summaries disagree"

    rows = []
    for name, path in zip(names, args.summaries):
        s = json.loads(Path(path).read_text())
        row = {"arm": name}
        row.update({b: s["model_forced_slopes"][b] for b in BAND_LABELS})
        row["gradient"] = s["model_targeting_gradient"]
        row["mean_slope"] = s["model_mean_slope"]
        row["ceiling_contrast"] = s["ceiling"]["model_rcc_contrast"]
        rows.append(row)
    tab = pd.DataFrame(rows)

    cols = BAND_LABELS + ["gradient", "mean_slope", "ceiling_contrast"]
    stats = pd.DataFrame(
        {
            "arm": ["mean", "sd", "min", "max"],
            **{
                c: [
                    tab[c].mean(),
                    tab[c].std(ddof=1),
                    tab[c].min(),
                    tab[c].max(),
                ]
                for c in cols
            },
        }
    )
    out = pd.concat([tab, stats], ignore_index=True)
    print(out.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    human_gradient = 0.30123532671395425
    sd = float(tab["gradient"].std(ddof=1))
    mean = float(tab["gradient"].mean())
    print(
        f"\n  gradient over {len(tab)} arms: mean {mean!r}  sd {sd!r}\n"
        f"  mean / sd = {mean / sd!r}   (how many seed sd the gradient is)\n"
        f"  share of the human gradient: "
        f"{mean / human_gradient!r} "
        f"[{tab['gradient'].min() / human_gradient!r}, "
        f"{tab['gradient'].max() / human_gradient!r}]"
    )
    sign_ok = bool((tab[BAND_LABELS[0]] > 0).all())
    print(f"  0-4 slope positive in every arm: {sign_ok}")
    print(f"  15-19 slope range: {tab['15-19'].min()!r} to {tab['15-19'].max()!r}")
    print(
        f"  no arm reaches the human 15-19 slope of -0.1615: "
        f"{bool((tab['15-19'] > -0.1615).all())}"
    )

    if args.out:
        d = Path(args.out)
        d.mkdir(parents=True, exist_ok=True)
        out.to_csv(d / "seed_spread.csv", index=False)
        print(f"\nwrote {d / 'seed_spread.csv'}")
    assert np.isfinite(tab["gradient"]).all(), "a seed produced no gradient"


if __name__ == "__main__":
    main()
