"""Spine scores: fetch + cache, and the shared metric/palette constants.

For every PR on the two success spines, the confirmed run's
evaluation/scores.csv is fetched from the PR's head branch (LFS: fetch
-> smudge) and identified by matching its 21-row mean against the mean
recorded in the PR's results log — no hand-pinned paths or numbers.
Fetched scores are cached in data/spine_scores.json (committed, so the
report pipeline runs offline); delete the cache to force a refetch.

The metric list, family grouping, palettes and parsing helpers here are
imported by every other pipeline script.

Usage:
    python scripts/data_analysis/autoresearch_summary/score_progressions.py
"""

import csv
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).parent / "data"
CACHE = DATA / "spine_scores.json"

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
    files = sh(
        ["gh", "api", f"repos/{{owner}}/{{repo}}/pulls/{pr}/files",
         "--paginate", "--jq", ".[].filename"]
    ).splitlines()
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
    _, cache = load_spine_scores(experiments, all_prs)
    print(f"spine cache complete: {len(cache)} PRs "
          f"({', '.join(f'{t} {len(c)}' for t, c in trees.items())})")


if __name__ == "__main__":
    main()
