"""Per-episode fingerprint for high-performing episodes.

Usage:
    python experiment_analysis/top_episode_breakdown.py

For each dataset (legacy, 50ep), identifies the episodes that beat the
dummy-at-`p=0` baseline (focus group only) and writes a per-episode
fingerprint to `reports/<dataset>_top_episodes.md`. 50ep uses both avg
and sum criteria (switching makes group recruitment a separate axis);
legacy uses only avg (1g4a, no switching, sum = 4 × avg trivially).
"""

import pandas as pd


# `has_groups`: include mean_size / group_size_change columns (only
# useful in 2g8a switching runs). Legacy is 1g4a with fixed group size.
DATASETS = [
    {
        "name": "Legacy",
        "path": "experiments/pilot_random1_player_round_slim.csv",
        "report": "reports/legacy_top_episodes.md",
        "baseline_avg": 24.0,
        "baseline_sum": None,
        "has_groups": False,
    },
    {
        "name": "GS (50 ep)",
        "path": "experiments/2group_8agent_50ep.csv",
        "report": "reports/50ep_top_episodes.md",
        "baseline_avg": 25.18,
        "baseline_sum": 122.87,
        "has_groups": True,
    },
]


def compute_payoff(df):
    df = df.copy()
    valid = (df["player_no_input"] == 0).astype(int)
    n_valid = (
        valid.groupby(
            [df["episode_id"], df["round_number"], df["group_id"]]
        )
        .transform("sum")
        .clip(lower=1)
    )
    df["payoff_calc"] = (
        20
        - df["contribution"]
        - df["punishment"]
        + df["common_good"] / n_valid
    )
    return df


def derive_prev(df):
    df = df.sort_values(["episode_id", "player_id", "round_number"]).copy()
    grp = df.groupby(["episode_id", "player_id"])
    df["prev_contribution"] = grp["contribution"].shift(1)
    df["prev_player_no_input"] = grp["player_no_input"].shift(1)
    return df


def fingerprint(g0):
    by_round = g0.groupby("round_number")
    size_series = by_round.size()
    first_r, last_r = size_series.index.min(), size_series.index.max()
    mean_size = len(g0) / g0["round_number"].nunique()

    contrib_by_round = by_round["contribution"].mean()
    c_min_r = contrib_by_round.idxmin()
    c_max_r = contrib_by_round.idxmax()

    real_fr = g0[
        (g0["prev_contribution"] == 0)
        & (g0["prev_player_no_input"] == 0)
    ]
    if len(real_fr):
        p_at_fr = real_fr["punishment"].mean()
        n_fr = len(real_fr)
    else:
        p_at_fr = float("nan")
        n_fr = 0

    return {
        "avg": g0["payoff_calc"].mean(),
        "sum": by_round["payoff_calc"].sum().mean(),
        "mean_size": mean_size,
        "size_first": size_series.loc[first_r],
        "size_last": size_series.loc[last_r],
        "c_mean": contrib_by_round.mean(),
        "c_min": contrib_by_round.loc[c_min_r],
        "c_min_r": int(c_min_r),
        "c_max": contrib_by_round.loc[c_max_r],
        "c_max_r": int(c_max_r),
        "mean_p": g0["punishment"].mean(),
        "p_heavy_rate": (g0["punishment"] >= 8).mean(),
        "p_at_fr": p_at_fr,
        "n_fr": n_fr,
    }


def render(ds, episodes, fps):
    name = ds["name"]
    b_avg = ds["baseline_avg"]
    b_sum = ds["baseline_sum"]
    has_groups = ds["has_groups"]

    lines = []
    lines.append(f"# {name} top-performing episode fingerprints")
    lines.append("")
    if b_sum is not None:
        lines.append(
            f"Episodes that beat the dummy-at-`p=0` baseline on either "
            f"avg ({b_avg:.2f}) or sum ({b_sum:.2f}) payoff, focus "
            f"group only (group_id == 0)."
        )
    else:
        lines.append(
            f"Episodes that beat the dummy-at-`p=0` baseline on "
            f"avg ({b_avg:.2f}). Single group, no switching, so sum "
            f"is trivially 4 × avg."
        )
    lines.append("")
    lines.append(
        f"- {len(episodes)} episodes above baseline (sorted by "
        f"{'sum' if has_groups else 'avg'} descending)"
    )
    lines.append("")
    lines.append("## Per-episode fingerprint")
    lines.append("")

    if has_groups:
        lines.append(
            "| ep | avg | sum | mean_size | size 0→last | "
            "c mean | c min @ r | c max @ r | mean p | p≥8 rate | "
            "p at real free-riders |"
        )
        lines.append(
            "|---:|----:|----:|---------:|----------:|"
            "------:|---------:|---------:|"
            "------:|--------:|---------------------:|"
        )
    else:
        lines.append(
            "| ep | avg | c mean | c min @ r | c max @ r | "
            "mean p | p≥8 rate | p at real free-riders |"
        )
        lines.append(
            "|---:|----:|------:|---------:|---------:|"
            "------:|--------:|---------------------:|"
        )

    fps.sort(key=lambda r: -(r[1]["sum"] if has_groups else r[1]["avg"]))

    for ep, f in fps:
        avg_v = f"{f['avg']:.2f}"
        sum_v = f"{f['sum']:.1f}"
        if f["avg"] > b_avg:
            avg_v = f"**{avg_v}**"
        if b_sum is not None and f["sum"] > b_sum:
            sum_v = f"**{sum_v}**"
        p_at_fr_str = (
            f"{f['p_at_fr']:.1f} (n={f['n_fr']})"
            if f["n_fr"] > 0
            else "—"
        )
        c_min_str = f"{f['c_min']:.1f} @ r{f['c_min_r']}"
        c_max_str = f"{f['c_max']:.1f} @ r{f['c_max_r']}"
        if has_groups:
            lines.append(
                f"| {ep} | {avg_v} | {sum_v} | {f['mean_size']:.1f} | "
                f"{f['size_first']:.0f}→{f['size_last']:.0f} | "
                f"{f['c_mean']:.1f} | {c_min_str} | {c_max_str} | "
                f"{f['mean_p']:.2f} | {f['p_heavy_rate'] * 100:.1f}% | "
                f"{p_at_fr_str} |"
            )
        else:
            lines.append(
                f"| {ep} | {avg_v} | {f['c_mean']:.1f} | "
                f"{c_min_str} | {c_max_str} | "
                f"{f['mean_p']:.2f} | {f['p_heavy_rate'] * 100:.1f}% | "
                f"{p_at_fr_str} |"
            )

    return "\n".join(lines) + "\n"


def main():
    for ds in DATASETS:
        df = compute_payoff(pd.read_csv(ds["path"]))
        df = derive_prev(df)
        g0_all = df[df["group_id"] == 0]
        avg = g0_all.groupby("episode_id")["payoff_calc"].mean()

        above = set(avg[avg > ds["baseline_avg"]].index)
        if ds["baseline_sum"] is not None:
            sum_mean = (
                g0_all.groupby(["episode_id", "round_number"])[
                    "payoff_calc"
                ]
                .sum()
                .groupby("episode_id")
                .mean()
            )
            above |= set(sum_mean[sum_mean > ds["baseline_sum"]].index)

        episodes = sorted(above)
        fps = []
        for ep in episodes:
            g0 = g0_all[g0_all["episode_id"] == ep]
            fps.append((ep, fingerprint(g0)))

        text = render(ds, episodes, fps)
        with open(ds["report"], "w") as fh:
            fh.write(text)
        print(f"Wrote {ds['report']} ({len(episodes)} episodes)")


if __name__ == "__main__":
    main()
