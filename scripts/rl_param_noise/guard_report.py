"""Read the matched guard pilots and report what this arm is meant to move.

Two 200-step training runs, identical but for the exploration mechanism
(`rl_epsgreedy_guard` and `rl_pnoise_guard`, both seed 42), read from the
metrics parquets the training loop writes. Two outputs:

  guard_gap.md     WHAT WAS SAMPLED. Mean punishment in the behaviour
                   rollouts against mean punishment in the evaluation rollout
                   at the same update steps, and their ratio. This is a
                   DESCRIPTION, not a defect: DQN is off-policy and a
                   behaviour policy that differs from the evaluated one is
                   what the algorithm is for. On the finished runs the ratio
                   was 1.7 to 6.6; reporting it here says what the buffer
                   contains, not that the buffer is wrong.

  guard_shape.csv  THE OUTCOME. Mean punishment per contribution bin, on the
                   evaluation suite's own RPA bins, for each run and each
                   sampling, with the row count per bin. Beside them: the
                   artificial punisher (the clone of a human manager),
                   measured on the opponent's group in the very same
                   rollouts, and the human managers from
                   experiments/2group_8agent_50ep.csv.

  guard_noise.csv  This arm's own diagnostics: the adapted noise scale and
                   the divergence it held, per update step, under all three
                   measures.

Usage (local):
    python scripts/rl_param_noise/guard_report.py \\
        artifacts/manager/rl_epsgreedy_guard/metrics/rl_epsgreedy_guard.parquet \\
        artifacts/manager/rl_pnoise_guard/metrics/rl_pnoise_guard.parquet \\
        --out plots/data_analysis/evaluation/rl_manager_param_noise
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from aimanager.evaluation_suite.convert import (  # noqa: E402
    HUMAN_DATA_FILE,
    load_human,
)
from aimanager.evaluation_suite.metrics import (  # noqa: E402
    RPA_LABELS,
    ResponseMetrics,
)

EVAL_TAG = "greedy"


def md(df, fmt="{:.3f}"):
    """A markdown table without pulling in `tabulate`."""

    def cell(v):
        if isinstance(v, float):
            return "" if v != v else fmt.format(v)
        return str(v)

    names = list(df.index.names)
    head = [" / ".join(str(n or "") for n in names)] + [str(c) for c in df.columns]
    rows = [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for idx, row in df.iterrows():
        key = " / ".join(str(k) for k in idx) if isinstance(idx, tuple) else str(idx)
        rows.append("| " + " | ".join([key] + [cell(v) for v in row]) + " |")
    return "\n".join(rows)


def wide(paths):
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df.pivot_table(
        index=["job_id", "sampling", "update_step", "round_number"],
        columns="metric",
        values="value",
    ).reset_index()


def weighted_profile(df, prefix):
    """Mean punishment per bin, weighting each recorded rollout-round mean by
    the number of agent-rounds behind it -- identical to pooling the raw rows."""
    out = {}
    for label in RPA_LABELS:
        n = df.get(f"{prefix}_n_{label}")
        v = df.get(f"{prefix}_mean_{label}")
        if n is None or v is None:
            out[label], out[f"n[{label}]"] = float("nan"), 0
            continue
        ok = n.notna() & v.notna() & (n > 0)
        total = float(n[ok].sum())
        out[label] = float((v[ok] * n[ok]).sum() / total) if total else float("nan")
        out[f"n[{label}]"] = int(total)
    return out


def drag_table(df, job, tag, eps, n_actions=31):
    """Does the behaviour buffer's SHAPE differ from the evaluated policy's,
    and in the particular way uniform action noise would make it differ?

    Epsilon-greedy replaces a fraction `eps` of actions with a uniform draw,
    whose mean is (n_actions - 1) / 2 = 15, *independently of the contribution
    the action was aimed at*. So within every contribution bin the behaviour
    mean should sit at

        evaluated + eps * (15 - evaluated)

    A prediction with no free parameters. Regressing the observed per-bin
    shift on that prediction through the origin gives a slope near 1 if the
    distortion is exactly the uniform drag.

    Weight noise has no uniform action mean to drag anything toward, so the
    slope should collapse even where the per-bin shifts themselves are large:
    a perturbed network still maps contribution to punishment coherently, it
    just maps it differently. `shift_spread` separates those two cases -- the
    standard deviation ACROSS episodes of each bin's behaviour mean is small
    when every episode is flattened the same way and large when every episode
    carries its own contingency.
    """
    uniform_mean = (n_actions - 1) / 2.0
    ev = df[(df["job_id"] == job) & (df["sampling"] == EVAL_TAG)]
    bh = df[(df["job_id"] == job) & (df["sampling"] == tag)]
    if ev.empty or bh.empty:
        return None
    e = weighted_profile(ev, "rpa")
    b = weighted_profile(bh, "rpa")
    rows = []
    for label in RPA_LABELS:
        spread = bh.groupby("update_step")[f"rpa_mean_{label}"].mean().std()
        rows.append(
            {
                "job_id": job,
                "sampling": tag,
                "bin": label,
                "evaluated": e[label],
                "behaviour": b[label],
                "observed_shift": b[label] - e[label],
                "predicted_uniform_drag": eps * (uniform_mean - e[label]),
                "shift_spread": float(spread),
                "n": b[f"n[{label}]"],
            }
        )
    t = pd.DataFrame(rows)
    x, y = t["predicted_uniform_drag"], t["observed_shift"]
    ok = x.notna() & y.notna()
    t["drag_slope"] = float((x[ok] * y[ok]).sum() / (x[ok] ** 2).sum())
    t["mean_abs_shift"] = float(y[ok].abs().mean())
    ce, cb = e["{0}"] - e["{20}"], b["{0}"] - b["{20}"]
    t["contrast_evaluated"] = ce
    t["contrast_behaviour"] = cb
    t["contrast_flattening"] = (cb - ce) / ce if ce else float("nan")
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = wide(args.parquets)
    behaviour_tags = sorted(set(df["sampling"]) - {EVAL_TAG})

    # -- the gap ------------------------------------------------------
    lines = [
        "# Behaviour versus evaluated, 200-step guard pilots",
        "",
        "Mean punishment served by the RL manager to its own group. The",
        "behaviour rows are the rollouts that fill the replay buffer; the",
        "evaluated rows are the fully deterministic rollout at the same",
        "update steps, every exploration mechanism off. A ratio away from 1",
        "describes what the buffer holds; it is not a defect, because DQN is",
        "off-policy and is meant to evaluate the greedy policy whatever",
        "collected the data.",
        "",
        "| run | sampling | behaviour | evaluated | ratio | update steps |",
        "|---|---|---|---|---|---|",
    ]
    gap_rows = []
    for job in sorted(df["job_id"].unique()):
        sub = df[df["job_id"] == job]
        ev = sub[sub["sampling"] == EVAL_TAG]
        for tag in behaviour_tags:
            bh = sub[sub["sampling"] == tag]
            if bh.empty or ev.empty:
                continue
            steps = sorted(set(bh["update_step"]) & set(ev["update_step"]))
            b = float(bh[bh["update_step"].isin(steps)]["punishment"].mean())
            e = float(ev[ev["update_step"].isin(steps)]["punishment"].mean())
            ratio = b / e if e else float("nan")
            lines.append(
                f"| {job} | {tag} | {b:.4f} | {e:.4f} | {ratio:.3f} | {len(steps)} |"
            )
            gap_rows.append(
                {
                    "job_id": job,
                    "sampling": tag,
                    "behaviour": b,
                    "evaluated": e,
                    "ratio": ratio,
                    "n_update_steps": len(steps),
                }
            )
    pd.DataFrame(gap_rows).to_csv(os.path.join(args.out, "guard_gap.csv"), index=False)

    # -- the shape ----------------------------------------------------
    rows = {}
    for job in sorted(df["job_id"].unique()):
        for tag in sorted(df["sampling"].unique()):
            sub = df[(df["job_id"] == job) & (df["sampling"] == tag)]
            if sub.empty:
                continue
            rows[f"{job} [{tag}]"] = weighted_profile(sub, "rpa")
    # The artificial punisher, on the opponent's group in the same rollouts.
    ev_all = df[df["sampling"] == EVAL_TAG]
    if not ev_all.empty:
        rows["artificial punisher (clone)"] = weighted_profile(ev_all, "rpa_opp")

    human = load_human(os.path.join(ROOT, HUMAN_DATA_FILE))
    h = ResponseMetrics().rpa(human)
    hm = h.groupby(level=0).mean().reindex(RPA_LABELS)
    hn = h.groupby(level=0).size().reindex(RPA_LABELS)
    rows["human managers"] = {
        **{k: float(hm[k]) for k in RPA_LABELS},
        **{f"n[{k}]": int(hn[k]) for k in RPA_LABELS},
    }

    shape = pd.DataFrame(rows).T
    shape.index.name = "manager"
    shape = shape[list(RPA_LABELS) + [f"n[{k}]" for k in RPA_LABELS]]
    shape["contrast_0_minus_20"] = shape["{0}"] - shape["{20}"]
    shape.to_csv(os.path.join(args.out, "guard_shape.csv"))

    lines += [
        "",
        "## Policy shape",
        "",
        "Mean punishment per contribution bin, evaluation-suite RPA bins.",
        "Human managers fall from 4.76 at contribution 0 to 0.27 at 20; two",
        "of the three finished learned seeds rose instead. `contrast` is",
        "{0} minus {20}: positive is the human sign.",
        "",
        md(shape),
    ]

    # -- the uniform-drag discriminator -------------------------------
    drags = [
        t
        for job in sorted(df["job_id"].unique())
        for tag in behaviour_tags
        if (t := drag_table(df, job, tag, args.eps)) is not None
    ]
    if drags:
        drag = pd.concat(drags, ignore_index=True)
        drag.to_csv(os.path.join(args.out, "guard_drag.csv"), index=False)
        summary = (
            drag.groupby(["job_id", "sampling"])[
                [
                    "drag_slope",
                    "mean_abs_shift",
                    "contrast_evaluated",
                    "contrast_behaviour",
                    "contrast_flattening",
                ]
            ]
            .first()
            .join(
                drag.groupby(["job_id", "sampling"])["shift_spread"]
                .mean()
                .rename("mean_shift_spread")
            )
        )
        lines += [
            "",
            "## Does the buffer's shape differ the way uniform noise would",
            "",
            "A description of what was SAMPLED, not of what was learned.",
            "",
            "Within each contribution bin, epsilon-greedy should drag the",
            "behaviour mean toward the uniform mean of 15 by exactly",
            "`eps * (15 - evaluated)` -- no free parameters. `drag_slope`",
            "regresses the observed per-bin shift on that prediction through",
            "the origin. Weight noise has no uniform action mean to drag",
            "toward, so its slope should collapse even where the shifts",
            "themselves are large; `mean_shift_spread` is the standard",
            "deviation across episodes of each bin's behaviour mean, small",
            "when every episode is flattened the same way and large when each",
            "episode carries its own contingency.",
            "",
            md(summary, "{:.4f}"),
        ]

    # -- this arm's diagnostics ---------------------------------------
    noise_cols = [c for c in df.columns if c.startswith("param_noise_")]
    if noise_cols:
        noise = (
            df[df["sampling"] == "param-noise"]
            .groupby(["job_id", "update_step"])[noise_cols]
            .mean()
            .reset_index()
        )
        noise.to_csv(os.path.join(args.out, "guard_noise.csv"), index=False)
        last = noise.tail(1).iloc[0]
        lines += [
            "",
            "## Noise scale",
            "",
            f"Final scale {last['param_noise_scale']:.5f}, holding divergence "
            f"{last['param_noise_divergence']:.3f} punishment levels "
            f"(l2 {last['param_noise_divergence_l2']:.5f}, "
            f"w1 {last['param_noise_divergence_w1']:.3f}).",
        ]

    path = os.path.join(args.out, "guard_gap.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
