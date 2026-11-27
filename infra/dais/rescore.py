"""Rerun committed policy-finder final runs through the cluster backend and
compare each point with its recorded result.

    python infra/dais/rescore.py <out dir> <branch>:<stem> [<branch>:<stem> ...]

For each rule version: the commit that added its final run's config
(configs/simulation/policy_finder/<stem>_sweep_final.yml) on <branch> is
snapshotted and the config rerun unchanged (same points, episodes and seed;
the code and rule of that commit). The recorded result is the final run's
sweep.json on <branch>. All configs go into the pool at once. Writes
<out dir>/<stem>/ (the new output) and <out dir>/rescore.json, and prints one
line per point: recorded pool +- se, new pool +- se, difference, z.
Run it in infra/dais/pipeline.slurm.sh (`python` mode) for the DAIS pool.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAIN / "scripts/policy_finder"))
import cluster_proxy  # noqa: E402

CONFIGS = "configs/simulation/policy_finder"
OUT = "plots/simulation/policy_finder"


def git(*args):
    return subprocess.run(["git", "-C", str(MAIN), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def main():
    out = Path(sys.argv[1])
    targets = [a.split(":", 1) for a in sys.argv[2:]]
    if not targets or any(len(t) != 2 for t in targets):
        sys.exit(__doc__)
    jobs = {}
    for branch, stem in targets:
        config = f"{CONFIGS}/{stem}_sweep_final.yml"
        commit = git("log", "-1", "--format=%H", branch, "--", config)
        if not commit:
            sys.exit(f"no {config} on {branch}")
        recorded = json.loads(git("show", f"{branch}:{OUT}/{stem}_sweep/final/sweep.json"))
        sha = cluster_proxy.snapshot(commit)
        job = cluster_proxy.submit(config, sha, run_group="rescore")
        print(f"{stem}: commit {commit[:12]}, job {job}", flush=True)
        jobs[stem] = (sha, job, recorded)
    cluster_proxy.wait([j for _, j, _ in jobs.values()])
    rows, failed = [], []
    for stem, (sha, job, recorded) in jobs.items():
        try:
            cluster_proxy.fetch(sha, f"{OUT}/{stem}_sweep/final", out / stem)
            new = json.loads((out / stem / "sweep.json").read_text())
        except (cluster_proxy.ClusterError, OSError) as e:
            failed.append(f"{stem}: {e}")
            continue
        fresh = {p["name"]: p for p in new["points"]}
        for p in recorded["points"]:
            q = fresh[p["name"]]
            diff = q["pool"] - p["pool"]
            se = math.hypot(p["pool_se"], q["pool_se"])
            rows.append({
                "stem": stem, "point": p["name"], "gp": p["name"] == recorded.get("best_name"),
                "raven": p["pool"], "raven_se": p["pool_se"], "dais": q["pool"],
                "dais_se": q["pool_se"], "diff": diff, "z": diff / se, "job": job,
            })
            print(f"{p['name']:32s} raven {p['pool']:8.2f} +- {p['pool_se']:.2f}"
                  f"  dais {q['pool']:8.2f} +- {q['pool_se']:.2f}"
                  f"  diff {diff:+6.2f} +- {se:.2f}  z {diff / se:+.2f}"
                  + ("  (GP peak)" if rows[-1]["gp"] else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "rescore.json").write_text(json.dumps(rows, indent=2) + "\n")
    if failed:
        sys.exit("FAILED: " + "; ".join(failed))


if __name__ == "__main__":
    main()
