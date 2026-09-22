#!/usr/bin/env python3
"""The behaviour policy and the evaluated policy of the RL manager, measured.

The manager explores with a fixed epsilon-greedy over 31 ordinal punishment
levels (0..30). A uniform draw has expectation 15, so at eps = 0.1 the
behaviour policy injects 1.5 punishment points per member per round on top of
whatever the greedy policy asks for -- the same magnitude as the whole learned
signal. `rl_manager.run_batch` logs both rollouts into the same metrics
parquet under `sampling`, so the two managers can be read off the committed
runs directly:

  sampling == "eps-greedy"  the training rollout; the ONLY one written to the
                            replay buffer (`run_batch(..., replay_mem, ...)`)
  sampling == "greedy"      the evaluation rollout, `replay_mem=None`

Three checks, in increasing strength:

1. **The two policies**, over the final update steps: mean punishment and mean
   contribution under each sampling mode.

2. **The mixture arithmetic.** `environment.punish` zeroes a punishment aimed
   at a player who gave no input, so the realised behaviour mean is
       E[valid * ((1-eps) * a_greedy + eps * U)]  =  (1-eps) g + eps * v * 15
   and the predicted gap is `1.5 * v - eps * g`, with `v` the share of cells
   that gave input. `v` is taken from the launch-guard evidence of the same
   world rather than assumed.

3. **Round 0 versus the episode.** `env.reset()` draws round-0 contributions
   and validity *before* the manager acts, so at round 0 the two rollouts face
   the same state distribution and the mixture identity must hold exactly.
   From round 1 on they are on different trajectories and it need not. Fitting
   `gap_0` on `g_0` across the runs recovers eps and v without assuming
   either; the same fit over the episode average does not.

Usage:
    python scripts/data_analysis/rl_manager_exploration_gap.py
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
OUT = os.path.join(ROOT, "plots/data_analysis/evaluation/rl_manager_exploration_gap")

# Identical in all six configs; `_check_configs` re-reads whichever are on the
# branch. The per-capita arm's configs live on `auto/rl-manager-percapita-reward`
# and differ from the pool arm's in exactly `reward_mode`, `job_id` and
# `output_dir` (that branch's log, section 3), so these hold for all six.
EPS = 0.1
N_PUNISHMENTS = 31

# The last 1000 update steps. Eval points are logged every 20 steps up to 3980,
# so this window is the 51 points 2980..3980 inclusive.
DEFAULT_SINCE = 2980

# Launch-guard evidence for the timeout rate: same world, same validity model,
# same batch shape, committed by the parent branch.
GUARD_FILE = os.path.join(
    ROOT, "plots/data_analysis/evaluation/rl_manager_two_worlds/guards_after_fix.json"
)

RUNS = [
    ("pool", 42, "rl_new_clones_s42"),
    ("pool", 43, "rl_new_clones_s43"),
    ("pool", 44, "rl_new_clones_s44"),
    ("per capita", 42, "rl_new_clones_percapita_s42"),
    ("per capita", 43, "rl_new_clones_percapita_s43"),
    ("per capita", 44, "rl_new_clones_percapita_s44"),
]


def _check_configs():
    """Fail loudly if a config on this branch disagrees with EPS / levels."""
    cfg_dir = os.path.join(ROOT, "configs/training/rl_manager")
    seen = 0
    for _, _, job in RUNS:
        path = os.path.join(cfg_dir, f"{job}.yml")
        if not os.path.exists(path):
            continue
        text = open(path).read()
        assert f"eps: {EPS}" in text, f"{path}: eps is not {EPS}"
        assert f"n_punishments: {N_PUNISHMENTS}" in text, f"{path}: levels differ"
        seen += 1
    assert seen, "no RL manager config found to check against"
    return seen


def load(job: str) -> pd.DataFrame:
    path = os.path.join(ROOT, f"artifacts/manager/{job}/metrics/{job}.parquet")
    return pd.read_parquet(path)


def by_round(df, metric, since):
    """Per-round mean of `metric` under each sampling mode, over the window."""
    w = df[(df["update_step"] >= since) & (df["metric"] == metric)]
    t = w.pivot_table(
        index="round_number", columns="sampling", values="value", aggfunc="mean"
    )
    t["gap"] = t["eps-greedy"] - t["greedy"]
    return t


def episode(df, metric, since):
    """Episode mean of `metric` under each sampling mode, over the window."""
    w = df[(df["update_step"] >= since) & (df["metric"] == metric)]
    s = w.groupby("sampling")["value"].mean()
    return float(s["greedy"]), float(s["eps-greedy"])


def two_policies(since):
    rows = []
    for arm, seed, job in RUNS:
        df = load(job)
        pg, pb = episode(df, "punishment", since)
        cg, cb = episode(df, "contribution", since)
        p0 = by_round(df, "punishment", since).loc[0]
        c0 = by_round(df, "contribution", since).loc[0]
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "job": job,
                "greedy": pg,
                "behaviour": pb,
                "ratio": pb / pg,
                "gap": pb - pg,
                "greedy_round0": float(p0["greedy"]),
                "behaviour_round0": float(p0["eps-greedy"]),
                "gap_round0": float(p0["gap"]),
                "contribution_greedy": cg,
                "contribution_behaviour": cb,
                "contribution_gap": cb - cg,
                "contribution_gap_round0": float(c0["gap"]),
            }
        )
    return pd.DataFrame(rows)


def guard_validity_rate():
    g = json.load(open(GUARD_FILE))["guard2"]
    cells = g["agent_cells_observed"]
    timed_out = g["timeout_cells_observed"]
    return 1.0 - timed_out / cells, cells, timed_out


def mixture_check(tp, v):
    """Predicted gap with and without the free-punishment zeroing."""
    mean_draw = (N_PUNISHMENTS - 1) / 2
    out = tp[["arm", "seed"]].copy()
    for tag, g, gap in (
        ("round0", "greedy_round0", "gap_round0"),
        ("episode", "greedy", "gap"),
    ):
        out[f"{tag}_gap"] = tp[gap]
        out[f"{tag}_pred_naive"] = EPS * mean_draw - EPS * tp[g]
        out[f"{tag}_pred_zeroed"] = EPS * mean_draw * v - EPS * tp[g]
        out[f"{tag}_res_naive"] = tp[gap] - out[f"{tag}_pred_naive"]
        out[f"{tag}_res_zeroed"] = tp[gap] - out[f"{tag}_pred_zeroed"]
    return out


def mixture_fit(tp, v):
    """Fit gap = a*g + b. Theory says a = -eps and b = eps*15*v."""
    mean_draw = (N_PUNISHMENTS - 1) / 2
    rows = []
    for tag, g, gap in (
        ("round 0", "greedy_round0", "gap_round0"),
        ("episode", "greedy", "gap"),
    ):
        x, y = tp[g].to_numpy(), tp[gap].to_numpy()
        a, b = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)[0]
        pred = a * x + b
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append(
            {
                "window": tag,
                "slope": a,
                "slope_theory": -EPS,
                "intercept": b,
                "intercept_theory": EPS * mean_draw * v,
                "implied_validity_rate": b / (EPS * mean_draw),
                "r2": r2,
            }
        )
    return pd.DataFrame(rows)


def gap_by_round(since):
    frames = []
    for arm, seed, job in RUNS:
        df = load(job)
        for metric in ("punishment", "contribution"):
            t = by_round(df, metric, since)
            frames.append(
                pd.DataFrame(
                    {
                        "arm": arm,
                        "seed": seed,
                        "metric": metric,
                        "round_number": t.index,
                        "greedy": t["greedy"].to_numpy(),
                        "behaviour": t["eps-greedy"].to_numpy(),
                        "gap": t["gap"].to_numpy(),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def window_sensitivity(a, b):
    """The same table at two window edges, so the convention is auditable."""
    ta, tb = two_policies(a), two_policies(b)
    out = ta[["arm", "seed"]].copy()
    for col in ("greedy", "behaviour", "ratio"):
        out[f"{col}_{a}"] = ta[col]
        out[f"{col}_{b}"] = tb[col]
        out[f"{col}_delta"] = tb[col] - ta[col]
    return out


# Colour-blind safe; one per run, arms distinguished by the legend text.
COLOURS = ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#56b4e9"]


def figure(gbr, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    keys = [(a, s) for a, s, _ in RUNS]

    ax = axes[0]
    pun = gbr[gbr["metric"] == "punishment"]
    for colour, (arm, seed) in zip(COLOURS, keys):
        sub = pun[(pun["arm"] == arm) & (pun["seed"] == seed)]
        ax.plot(sub["round_number"], sub["behaviour"], color=colour, lw=1.4)
        ax.plot(sub["round_number"], sub["greedy"], color=colour, lw=1.4, ls="--")
    ax.plot([], [], color="0.3", lw=1.4, label="behaviour (eps-greedy)")
    ax.plot([], [], color="0.3", lw=1.4, ls="--", label="evaluated (greedy)")
    ax.set_ylim(bottom=0)
    ax.set_title("Mean punishment: two policies, six runs", fontsize=10)
    ax.set_ylabel("punishment points per member")
    ax.legend(fontsize=8, loc="lower right", frameon=False)

    ax = axes[1]
    con = gbr[gbr["metric"] == "contribution"]
    for colour, (arm, seed) in zip(COLOURS, keys):
        sub = con[(con["arm"] == arm) & (con["seed"] == seed)]
        ax.plot(
            sub["round_number"],
            sub["gap"],
            color=colour,
            lw=1.4,
            label=f"{arm} s{seed}",
        )
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_title("Contribution: behaviour minus evaluated", fontsize=10)
    ax.set_ylabel("contribution units per member")
    ax.legend(fontsize=8, loc="upper left", frameon=False, ncol=2)

    for ax in axes:
        ax.set_xlabel("round_number")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", type=int, default=DEFAULT_SINCE)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    n_cfg = _check_configs()
    v, cells, timed_out = guard_validity_rate()
    os.makedirs(args.out, exist_ok=True)

    tp = two_policies(args.since)
    mc = mixture_check(tp, v)
    mf = mixture_fit(tp, v)
    gbr = gap_by_round(args.since)
    ws = window_sensitivity(args.since, args.since + 20)

    tp.to_csv(os.path.join(args.out, "two_policies.csv"), index=False)
    mc.to_csv(os.path.join(args.out, "mixture_check.csv"), index=False)
    mf.to_csv(os.path.join(args.out, "mixture_fit.csv"), index=False)
    gbr.to_csv(os.path.join(args.out, "gap_by_round.csv"), index=False)
    ws.to_csv(os.path.join(args.out, "window_sensitivity.csv"), index=False)
    figure(gbr, os.path.join(args.out, "gap_by_round.jpg"))

    print(f"eps {EPS}, {N_PUNISHMENTS} levels, checked against {n_cfg} config(s)")
    print(f"window: update_step >= {args.since}")
    print(f"validity rate v = {v:.4f} ({cells - timed_out}/{cells} gave input)")
    print(f"injected per member-round: {EPS * (N_PUNISHMENTS - 1) / 2 * v:.4f}\n")
    cols = ["arm", "seed", "greedy", "behaviour", "ratio", "gap"]
    print(tp[cols].round(4).to_string(index=False), "\n")
    print("contribution")
    ccols = ["arm", "seed", "contribution_greedy", "contribution_behaviour"]
    ccols += ["contribution_gap", "contribution_gap_round0"]
    print(tp[ccols].round(4).to_string(index=False), "\n")
    print("mixture check (residual = measured gap - prediction)")
    mcols = ["arm", "seed", "round0_res_naive", "round0_res_zeroed"]
    mcols += ["episode_res_naive", "episode_res_zeroed"]
    print(mc[mcols].round(4).to_string(index=False), "\n")
    print("mixture fit")
    print(mf.round(4).to_string(index=False), "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
