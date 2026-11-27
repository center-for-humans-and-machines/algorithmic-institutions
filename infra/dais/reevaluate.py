"""Re-evaluate every version of a policy-finder campaign on fresh seeds, and
its best-so-far curve and improvement, at a precision a comparison can carry.

    python infra/dais/reevaluate.py <campaign state.json> <out dir>
        [--refs policy-finder/] [--target-se 0.3] [--seed 100000] [--slices K]
        [--keep-snapshots]

<campaign state.json> is campaign.py's state ($PF_RUNS/campaign/<name>.json:
the members of each generation, {"generations": [{"index", "members":
[{"name", "version"}]}]}), or the same written by hand for a campaign that
ran elsewhere (docs/reports/policy_finder_reference_curves/). Each version's
files come from branch <refs><name>: its final run's config and result
(configs/simulation/policy_finder/<stem>_sweep_final.yml,
plots/simulation/policy_finder/<stem>_sweep/final/sweep.json, whose `best_name`
is the GP peak the campaign scored) and the rule and code of the commit that
added that config. A version without them (it failed, or stopped before its
final run) is listed as "no final" and counts as no improvement, as in
campaign.py.

Each version's GP peak is played alone (with the setting's other managers,
`ah` in exp2) in K slices of the final run's size, on seeds `--seed` + the
batches of the slices before it, the same seeds for every version and none
the campaign used. K is the smallest number for which the largest recorded
GP-peak standard error of the campaign, divided by sqrt(K), is at most
--target-se. A slice is a whole number of batches, single-threaded, in the
cluster backend's pool (PF_CLUSTER_BACKEND=local: infra/dais/pipeline.slurm.sh
`python` mode). Fails if any slice fails.

Writes to <out dir>: versions.tsv (generation, lineage, version, stem, status,
recorded score and se, re-evaluated score and se, episodes, commit),
curve.tsv (per generation, the best so far by recorded score, its
re-evaluated score and its paired gain over generation 1's best) and
summary.json (the improvement: that gain at the last generation, its se, and
70% of it).
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

MAIN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAIN / "scripts/policy_finder"))
import cluster_proxy  # noqa: E402
import local_cluster  # noqa: E402

CONFIGS = "configs/simulation/policy_finder"
OUT = "plots/simulation/policy_finder"
EVAL_CONFIGS = "configs/simulation/reevaluate"
EVAL_OUT = "plots/simulation/reevaluate"


def stem_of(name, version):
    return name if version == 1 else f"{name}-v{version}"


def git(*args, check=True):
    result = subprocess.run(["git", "-C", str(MAIN), *args], capture_output=True,
                            text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip() if result.returncode == 0 else None


def version_files(ref, stem):
    """(final config commit, final config dict, recorded final sweep.json), or
    None when the version has no final run on `ref`."""
    config = f"{CONFIGS}/{stem}_sweep_final.yml"
    result = git("show", f"{ref}:{OUT}/{stem}_sweep/final/sweep.json", check=False)
    commit = git("log", "-1", "--format=%H", ref, "--", config, check=False)
    if not result or not commit:
        return None
    final = json.loads(result)
    if "best_name" not in final:
        return None  # not marked by --score-final: the version did not finish
    return commit, yaml.safe_load(git("show", f"{commit}:{config}")), final


def gp_config(final_config, gp, seed, out):
    """The final run's config with only the GP peak among the version's points
    (its rival managers and pairings stay), at `seed`, writing to `out`."""
    stem = gp.rsplit("_", 1)[0]
    config = dict(final_config)
    dropped = {m for m in config["managers"] if m.startswith(f"{stem}_") and m != gp}
    config["managers"] = {k: v for k, v in config["managers"].items() if k not in dropped}
    if "pairings" in config:
        config["pairings"] = [p for p in config["pairings"]
                              if p["group_0"] not in dropped and p["group_1"] not in dropped]
    config.update(seed=seed, output_dir=out, figure_name=Path(out).name,
                  save_per_round=False)
    return config


def batches(config):
    """Seed batches one slice of `config` takes (generate_sim_config's seeding)."""
    if config.get("seed_cycle"):
        return config["seed_cycle"]
    pairings = len(config.get("pairings") or []) or 1
    return math.ceil(config["n_episodes"] * pairings / config.get("episode_batch_size", 1))


def combine(slices):
    """Equal-sized slices: the mean of their means, its se."""
    k = len(slices)
    return (sum(p for p, _ in slices) / k,
            math.sqrt(sum(se * se for _, se in slices)) / k)


def paired(a, b):
    """Mean and se of b - a over slices played on the same seeds (common
    random numbers: slice j of every version plays seed j's games), from the
    spread of the per-slice differences."""
    d = [y - x for (x, _), (y, _) in zip(a, b)]
    k = len(d)
    mean = sum(d) / k
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (k - 1)) if k > 1 else math.nan
    return mean, sd / math.sqrt(k)


def curve(rows, slices=None):
    """Per generation: the best so far by recorded score (the campaign's
    selection), with its re-evaluated score and, with `slices` ({stem: per
    slice (pool, se)}), its paired gain over generation 1's best."""
    out, best, first = [], None, None
    for g in sorted({r["generation"] for r in rows}):
        for r in rows:
            if r["generation"] == g and r["recorded"] is not None and (
                    best is None or r["recorded"] > best["recorded"]):
                best = r
        if first is None and best is not None:
            first = best
        point = {"generation": g, "best_so_far": best["stem"] if best else None}
        for key in ("recorded", "recorded_se", "reevaluated", "reevaluated_se"):
            point[key] = best[key] if best else None
        point["gain_vs_first"] = point["gain_se"] = None
        if slices and best is not None:
            point["gain_vs_first"], point["gain_se"] = paired(
                slices[first["stem"]], slices[best["stem"]])
        out.append(point)
    return out


def improvement(points):
    """The final best so far minus generation 1's best, re-evaluated: the
    paired gain of the curve's last point, and 70% of it."""
    first, last = points[0], points[-1]
    gain, se = last["gain_vs_first"], last["gain_se"]
    return {"first_generation_best": first["best_so_far"], "final_best": last["best_so_far"],
            "improvement": gain, "improvement_se": se, "bar_70": 0.7 * gain,
            "bar_70_se": 0.7 * se, "se_method": "paired over common-seed slices"}


def write_tsv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                             for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("state")
    ap.add_argument("out")
    ap.add_argument("--refs", default="policy-finder/")
    ap.add_argument("--target-se", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=100_000)
    ap.add_argument("--slices", type=int, help="instead of --target-se")
    ap.add_argument("--keep-snapshots", action="store_true")
    args = ap.parse_args()
    if cluster_proxy.BACKEND != "local":
        sys.exit("set PF_CLUSTER_BACKEND=local (infra/dais/pipeline.slurm.sh)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state = json.loads(Path(args.state).read_text())

    rows, versions = [], {}
    for generation in state["generations"]:
        for m in generation["members"]:
            stem = stem_of(m["name"], m["version"])
            files = version_files(args.refs + m["name"], stem)
            row = {"generation": generation["index"], "lineage": m["name"],
                   "version": m["version"], "stem": stem,
                   "status": "scored" if files else "no final",
                   "recorded": None, "recorded_se": None, "reevaluated": None,
                   "reevaluated_se": None, "episodes": 0, "commit": None}
            if files:
                commit, config, final = files
                gp = {p["name"]: p for p in final["points"]}[final["best_name"]]
                row.update(recorded=gp["pool"], recorded_se=gp["pool_se"], commit=commit)
                versions[stem] = (commit, config, final["best_name"])
            rows.append(row)
    missing = [r["stem"] for r in rows if r["status"] == "no final"]
    print(f"{len(rows)} versions, {len(versions)} with a final run; no final: "
          f"{', '.join(missing) or 'none'}", flush=True)
    if not versions:
        sys.exit("no version has a final run")

    largest = max(r["recorded_se"] for r in rows if r["recorded_se"] is not None)
    k = args.slices or math.ceil((largest / args.target_se) ** 2)
    print(f"largest recorded se {largest:.2f}: {k} slices per version", flush=True)

    snapshots, jobs = {}, {}
    for stem, (commit, config, gp) in versions.items():
        # the slices' configs go into the snapshot: the local backend's only
        existed = (local_cluster.tree(commit).parent / "complete").exists()
        sha = cluster_proxy.snapshot(commit)
        if not existed:
            snapshots[stem] = sha
        tree = local_cluster.tree(sha)
        seed = args.seed
        for j in range(k):
            name = f"{stem}_r{j:02d}"
            cfg = gp_config(config, gp, seed, f"{EVAL_OUT}/{name}")
            seed += batches(cfg)
            path = tree / EVAL_CONFIGS / f"{name}.yml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            jobs[name] = (stem, sha, cluster_proxy.submit(
                f"{EVAL_CONFIGS}/{name}.yml", sha, run_group="reevaluate"))
    print(f"{len(jobs)} slices submitted", flush=True)
    cluster_proxy.wait([job for _, _, job in jobs.values()])

    results, failed = {}, []
    for name, (stem, sha, job) in jobs.items():
        try:
            cluster_proxy.fetch(sha, f"{EVAL_OUT}/{name}", out / "slices" / name)
            sweep = json.loads((out / "slices" / name / "sweep.json").read_text())
            gp = {p["name"]: p for p in sweep["points"]}[versions[stem][2]]
            results.setdefault(stem, []).append((gp["pool"], gp["pool_se"]))
        except (cluster_proxy.ClusterError, OSError, KeyError) as e:
            failed.append(f"{name} (job {job}): {e!r}")
    if failed:
        sys.exit("FAILED slices:\n" + "\n".join(failed))
    for r in rows:
        if r["stem"] in results:
            r["reevaluated"], r["reevaluated_se"] = combine(results[r["stem"]])
            r["episodes"] = k * versions[r["stem"]][1]["n_episodes"]

    points = curve(rows, results)
    summary = {"state": str(args.state), "refs": args.refs, "seed": args.seed,
               "slices": k, "target_se": args.target_se, "no_final": missing,
               **improvement(points)}
    write_tsv(out / "versions.tsv", rows)
    write_tsv(out / "curve.tsv", points)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for p in points:
        print(f"gen {p['generation']}: {p['best_so_far']} recorded {p['recorded']:.2f}"
              f" re-evaluated {p['reevaluated']:.2f} +- {p['reevaluated_se']:.2f}")
    print(f"improvement {summary['improvement']:.2f} +- {summary['improvement_se']:.2f},"
          f" 70%: {summary['bar_70']:.2f}")
    if not args.keep_snapshots:  # only those this run made
        for sha in snapshots.values():
            local_cluster.cleanup(sha)


if __name__ == "__main__":
    main()
