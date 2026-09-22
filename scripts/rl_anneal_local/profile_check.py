"""The three-statistic targeting check, applied to all ten finished runs.

Raised by a sibling arm after it withdrew one of its own headline seeds. A
difference of endpoint bin means confuses force with aim: a manager that
punishes hard at contribution 0 and flat everywhere else has a large endpoint
contrast and no profile at all. A rank correlation alone has the opposite
failure, scoring a profile that is flat to within noise as perfectly
targeted, because six nearly equal numbers still have an ordering.

So three statistics together:

  rank            Spearman of the six bin means against the bin index.
                  Human managers punish free-riders hardest, so the human
                  profile scores -1.000 and an inverted manager scores near
                  +1. Reported beside Kendall tau-b and the monotonicity
                  flags, because Spearman is dragged toward zero by ties and
                  two of these profiles saturate at exactly 0 in several bins.
  relative_range  (max - min) of the bin means over their mean: is the
                  profile big enough to be worth a direction.
  gate            the range over its standard error, resampled over episodes:
                  is it distinguishable from flat at all. Produced by
                  `guard.py shape --profile-out`, which needs a GPU rollout.

Rank and relative range are functions of the six bin means alone, so this
script computes them from the committed `policy_shape.csv` and needs no
cluster. The gate is merged in from `profile_*.csv` when those exist; without
them the gate column is NaN and the verdict is taken on rank and size only.
That is stated in the output rather than hidden, and it is sound in one
direction: a noise gate can only ever turn a verdict into "no targeting", so
a run failing on rank fails whatever the gate says.

    python scripts/rl_anneal_local/profile_check.py
"""

import glob
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "plots/data_analysis/rl_anneal_local")
SEEDS = (42, 43, 44, 45, 46)
ARMS = {"arm": "rl_anneal_local_s{}", "control": "rl_new_clones_s{}"}

# The sibling's passing band for a real profile is |rank| 0.943 to 1.000 and
# the human and clone references sit at 1.000, but control seed 43 is monotone
# decreasing across all six bins and still scores only -0.845 on three tied
# bins. 0.8 admits it; tau-b and the monotonicity flags are carried so a
# reader can see which verdicts turn on the threshold. None of them do.
RANK_MIN = 0.8
RELATIVE_MIN = 1.0
GATE_MIN = 1.65


def stats(means):
    idx = pd.Series(range(len(means)), index=means.index, dtype=float)
    diffs = means.diff().dropna()
    rng = float(means.max() - means.min())
    return {
        "rank": float(means.corr(idx, method="spearman")),
        "kendall_tau_b": float(means.corr(idx, method="kendall")),
        "n_tied_bins": int(len(means) - means.nunique()),
        "monotone_decreasing": bool((diffs <= 0).all()),
        "monotone_increasing": bool((diffs >= 0).all()),
        "range": rng,
        "relative_range": rng / float(means.mean()),
    }


def verdict(r):
    strong = abs(r["rank"]) >= RANK_MIN
    big = r["relative_range"] >= RELATIVE_MIN
    # A missing gate cannot rescue anything, so it is treated as passing and
    # flagged in the output.
    real = pd.isna(r["gate"]) or r["gate"] >= GATE_MIN
    if not (strong and big and real):
        return "no targeting"
    return "targets free-riders" if r["rank"] < 0 else "targets contributors"


def load_gates():
    """range / SE per manager, from whatever profile_*.csv exist."""
    gates = {}
    for f in glob.glob(os.path.join(OUT, "profile_*.csv")):
        if os.path.basename(f) == "profile_check.csv":
            continue
        d = pd.read_csv(f).set_index("manager")
        for name, row in d.iterrows():
            gates.setdefault(name, row["gate"])
    return gates


def main():
    shape = pd.read_csv(os.path.join(OUT, "policy_shape.csv")).set_index(
        "contribution_bin"
    )
    gates = load_gates()

    rows = []
    for label, col, key in (
        [
            ("human", "human", "human managers"),
            ("clone", "clone", "clone (same rollout)"),
        ]
        + [(f"arm {s}", f"arm_s{s}", ARMS["arm"].format(s)) for s in SEEDS]
        + [(f"control {s}", f"control_s{s}", ARMS["control"].format(s)) for s in SEEDS]
    ):
        rows.append(
            {
                "profile": label,
                "arm": label.split()[0],
                **stats(shape[col]),
                "gate": gates.get(key, float("nan")),
            }
        )

    d = pd.DataFrame(rows)
    d["verdict"] = [verdict(r) for _, r in d.iterrows()]
    d.to_csv(os.path.join(OUT, "profile_check.csv"), index=False)

    pd.set_option("display.width", 220)
    print("=== the three-statistic targeting check ===")
    print(
        f"thresholds: |rank| >= {RANK_MIN}, relative_range >= {RELATIVE_MIN}, "
        f"gate >= {GATE_MIN}"
    )
    if d["gate"].isna().any():
        missing = int(d["gate"].isna().sum())
        print(
            f"NOTE: {missing} of {len(d)} gates not yet measured (needs the GPU "
            "rollout). A gate can only tighten a verdict, never rescue one."
        )
    print()
    cols = [
        "profile",
        "rank",
        "kendall_tau_b",
        "n_tied_bins",
        "monotone_decreasing",
        "monotone_increasing",
        "range",
        "relative_range",
        "gate",
        "verdict",
    ]
    print(d[cols].round(3).to_string(index=False))
    print()
    print("=== verdict counts ===")
    runs = d[d["arm"].isin(["arm", "control"])]
    print(runs.groupby(["arm", "verdict"]).size().to_string())

    t = pd.read_csv(os.path.join(OUT, "targeting.csv"))
    t["profile"] = [f"{a} {s}" for a, s in zip(t["arm"], t["seed"])]
    m = runs.merge(t[["profile", "leaver_minus_stayer"]], on="profile")
    print()
    print("=== against the leaver/stayer diagnostic ===")
    print(
        m[["profile", "verdict", "leaver_minus_stayer"]].round(3).to_string(index=False)
    )


if __name__ == "__main__":
    main()
