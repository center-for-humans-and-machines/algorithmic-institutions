"""Fetch the before/after evaluation figures for the report.

"Before" is the pre-campaign reference stack's evaluation visuals (the
local root run the tree is anchored on); "after" is each frontier tip's
confirmed run, whose sim dir is found on the tip PR's branch by matching
a scores.csv run mean against the tip mean recorded in the frozen corpus
-- no hand-pinned paths. The figures (plain git blobs, not LFS) are
copied under plots/data_analysis/autoresearch_summary/stack_visuals/ and
indexed in data/stack_visuals.json for build_report.py.

Usage:
    python scripts/data_analysis/autoresearch_summary/stack_visuals.py
"""

import json
import subprocess
from pathlib import Path

from score_progressions import DATA, parse_scores, sh, spines

OUTDIR = Path("plots/data_analysis/autoresearch_summary/stack_visuals")
MANIFEST = DATA / "stack_visuals.json"
ROOT_VISUALS = Path(
    "plots/simulation/23_2g8a_self_gnn_contr_gnn_switch/evaluation/visuals"
)
MEAN_TOL = 1e-6


def shb(cmd, **kw):
    """sh, but binary stdout (for image blobs)."""
    return subprocess.run(cmd, capture_output=True, check=True, **kw).stdout


def tip_sim_dir(branch, want):
    """The branch's sim dir whose scores.csv holds a run at the tip mean."""
    paths = sh(["git", "ls-tree", "-r", f"origin/{branch}",
                "--name-only"]).splitlines()
    for path in paths:
        if not path.endswith("evaluation/scores.csv"):
            continue
        try:
            pointer = sh(["git", "cat-file", "-p", f"origin/{branch}:{path}"])
            text = sh(["git", "lfs", "smudge"], input=pointer)
        except subprocess.CalledProcessError:
            continue
        for run, scores in parse_scores(text).items():
            if abs(sum(scores.values()) / len(scores) - want) < MEAN_TOL:
                print(f"{branch}: {path} [{run}]")
                return path.rsplit("/evaluation/", 1)[0]
    raise RuntimeError(f"{branch}: no run matches mean {want}")


def copy_branch_visuals(branch, sim_dir, dest):
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    for path in sh(["git", "ls-tree", "-r", f"origin/{branch}",
                    "--name-only"]).splitlines():
        if path.startswith(f"{sim_dir}/evaluation/visuals/") and \
                path.endswith(".jpg"):
            name = path.rsplit("/", 1)[1]
            (dest / name).write_bytes(
                shb(["git", "cat-file", "-p", f"origin/{branch}:{path}"]))
            names.append(name)
    return sorted(names)


def main():
    experiments = json.loads((DATA / "experiments.json").read_text())
    by_pr = {e["pr"]: e for e in experiments}
    trees = spines(experiments)

    stacks = {"main": {"label": "before: the pre-campaign reference stack",
                       "files": []}}
    # before: the local root run's figures
    dest = OUTDIR / "main"
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(ROOT_VISUALS.glob("*.jpg")):
        (dest / f.name).write_bytes(f.read_bytes())
        stacks["main"]["files"].append(f.name)

    # after: each tree's tip, from its PR branch
    for tree, chain in trees.items():
        tip = by_pr[chain[-1]]
        want = tip["metrics"]["mean"]
        sim_dir = tip_sim_dir(tip["head_branch"], want)
        stacks[tree] = {
            "label": f"after: {tree} tip #{tip['pr']} (mean {want:.3f})",
            "pr": tip["pr"],
            "sim_dir": sim_dir,
            "files": copy_branch_visuals(tip["head_branch"], sim_dir,
                                         OUTDIR / tree),
        }

    MANIFEST.write_text(json.dumps(stacks, indent=2) + "\n")
    total = sum(len(s["files"]) for s in stacks.values())
    print(f"wrote {total} figures under {OUTDIR}/ and {MANIFEST}")


if __name__ == "__main__":
    main()
