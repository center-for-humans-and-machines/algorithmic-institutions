"""All 21 evaluation scores along the two trees' success spines.

For every spine PR the confirmed run's evaluation/scores.csv is fetched
from the PR's head branch (LFS: fetch -> smudge) and identified by
matching its 21-row mean against the mean recorded in the PR's results
log — no hand-pinned paths or numbers. Fetched scores are cached in
data/spine_scores.json; delete it to force a refetch.

Usage:
    python scripts/data_analysis/autoresearch_summary/score_progressions.py
"""

import csv
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA = Path(__file__).parent / "data"
CACHE = DATA / "spine_scores.json"
OUT = Path("plots/data_analysis/autoresearch_summary/score_progressions.png")

ROOT_SCORES_CSV = Path(
    "plots/simulation/23_2g8a_self_gnn_contr_gnn_switch/evaluation/scores.csv"
)
ROOT_RUN = "ah group_switching managed by lin_multinomial_self"

# Grouped by the stack slot each row measures.
METRICS = [
    "CA", "CB", "CC", "CD", "CE", "CF", "CG", "RCA", "RCB", "RCC", "RCD",
    "SA", "SB", "SC", "RSA",
    "PA", "PB", "PC", "PD", "RPA", "RPB",
]
FAMILY = {
    **{m: "contribution" for m in METRICS[:11]},
    **{m: "switch" for m in METRICS[11:15]},
    **{m: "punisher" for m in METRICS[15:]},
}
FAMILY_COLOR = {
    "contribution": "#2a78d6",
    "switch": "#eb6834",
    "punisher": "#1baf7a",
}
TREE_COLOR = {"gnn": "#6b6a66", "gmlp": "#b08968"}
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
MEAN_TOL = 5e-3


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, check=True,
                          **kw).stdout


def spines(experiments):
    """The two success spines, root-first, discovered from the data."""
    by_pr = {e["pr"]: e for e in experiments}
    tips = {}
    for e in experiments:
        if e["verdict"] != "SUCCESS":
            continue
        chain, node = [], e
        while node:
            if node["verdict"] == "SUCCESS":
                chain.append(node["pr"])
            node = by_pr.get(node["parent_pr"])
        root = chain[-1]
        if len(chain) > len(tips.get(root, [])):
            tips[root] = chain[::-1]
    deep = sorted(tips.values(), key=len, reverse=True)[:2]
    # gnn tree is the one rooted in the punisher-slot trunk
    a, b = deep
    return {"gnn": a, "gmlp": b} if by_pr[a[0]]["slot"] and "punish" in str(
        by_pr[a[0]]["slot"]) else {"gnn": b, "gmlp": a}


def parse_scores(text):
    """scores.csv text -> {run: {metric: score}}."""
    runs = defaultdict(dict)
    for row in csv.DictReader(io.StringIO(text)):
        runs[row["run"]][row["metric"]] = float(row["score"])
    return {r: m for r, m in runs.items() if len(m) == len(METRICS)}


def fetch_pr_scores(e):
    """The confirmed run's 21 scores for one spine PR, from its branch."""
    branch, pr = e["head_branch"], e["pr"]
    want = e["metrics"]["mean"]
    sh(["git", "fetch", "origin",
        f"+refs/heads/{branch}:refs/remotes/origin/{branch}"])
    files = json.loads(sh(
        ["gh", "api", f"repos/{{owner}}/{{repo}}/pulls/{pr}/files",
         "--paginate", "--jq", "[.[].filename]"]
    ))
    for path in files:
        if not path.endswith("evaluation/scores.csv"):
            continue
        try:
            blob = sh(["git", "rev-parse", f"origin/{branch}:{path}"]).strip()
            sh(["git", "lfs", "fetch", "origin", f"origin/{branch}",
                "-I", path])
            pointer = sh(["git", "cat-file", "-p", blob])
            text = sh(["git", "lfs", "smudge"], input=pointer)
        except subprocess.CalledProcessError:
            continue
        for run, scores in parse_scores(text).items():
            if abs(sum(scores.values()) / len(scores) - want) < MEAN_TOL:
                print(f"#{pr}: {path} [{run}]")
                return scores
    raise RuntimeError(f"#{pr}: no run matches recorded mean {want}")


def load_spine_scores(experiments, spine_prs):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    root = parse_scores(ROOT_SCORES_CSV.read_text())[ROOT_RUN]
    by_pr = {e["pr"]: e for e in experiments}
    for pr in spine_prs:
        if str(pr) not in cache:
            cache[str(pr)] = fetch_pr_scores(by_pr[pr])
    CACHE.write_text(json.dumps(cache, indent=2) + "\n")
    return root, cache


def main():
    experiments = json.loads((DATA / "experiments.json").read_text())
    trees = spines(experiments)
    all_prs = [pr for chain in trees.values() for pr in chain]
    root, cache = load_spine_scores(experiments, all_prs)

    fig, axes = plt.subplots(3, 7, figsize=(16, 8), facecolor=SURFACE,
                             sharex=True, sharey=True)
    for ax, metric in zip(axes.flat, METRICS):
        ax.set_facecolor(SURFACE)
        for band in (1.0, 2.0, 5.0):
            ax.axhline(band, color="#eceae6", lw=0.8, zorder=1)
        for tree, chain in trees.items():
            ys = [root[metric]] + [cache[str(pr)][metric] for pr in chain]
            ax.plot(range(len(ys)), ys, color=TREE_COLOR[tree], lw=2,
                    marker="o", markersize=4.5, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.set_yscale("log")
        ax.set_yticks([0.5, 1, 2, 5, 10])
        ax.set_yticklabels(["0.5", "1", "2", "5", "10"])
        ax.set_title(metric, fontsize=10, fontweight="bold",
                     color=FAMILY_COLOR[FAMILY[metric]])
        ax.tick_params(colors=INK_2, labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_color("#d6d5d0")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("spine step", fontsize=8, color=INK_2)
        ax.set_xticks(range(max(len(c) for c in trees.values()) + 1))

    gnn_tip, gmlp_tip = trees["gnn"][-1], trees["gmlp"][-1]
    fig.legend(handles=[
        Line2D([], [], color=TREE_COLOR["gnn"], lw=2, marker="o",
               label=f"gnn-stack spine (main -> #{gnn_tip})"),
        Line2D([], [], color=TREE_COLOR["gmlp"], lw=2, marker="o",
               label=f"gaussian-MLP spine (main -> #{gmlp_tip})"),
        *(Line2D([], [], ls="", marker="s", color=c,
                 label=f"{f} row (panel title)")
          for f, c in FAMILY_COLOR.items()),
    ], loc="lower right", frameon=False, fontsize=9, ncol=5)
    fig.suptitle(
        "All 21 evaluation scores along the two success spines "
        "(step 0 = main; guides at the 1 / 2 / 5 band edges, log scale)",
        fontsize=12, color=INK,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
