"""Re-evaluate every version of a policy-finder campaign on fresh seeds, and
its best-so-far curve and improvement, at a precision a comparison can carry.

    python infra/dais/reevaluate.py <campaign state.json> <out dir>
        [--refs policy-finder/] [--target-se 0.3] [--seed 100000] [--slices K]
        [--keep-snapshots] [--references <file>] [--references-only]

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

A campaign that iterated more lineages after generation 1 than
--compare-top-k (4, the reference campaigns' top_k) gets two results: the
comparable one, over the k lineages best after generation 1 by recorded score
(what a top_k k campaign iterates throughout; the headline of summary.json),
and the one over all lineages (summary.json "subsets"; curve_comparable.tsv
beside curve.tsv). lineages.tsv holds each lineage's best so far per
generation.

--single-idea: a single-idea campaign (every lineage starts from the seed,
`punishment = 0`, and runs to the end): the result over all lineages (the
first generation is the seed in every lineage, so a "best after generation
1" subset is noise), labelled as the gain of the best rule over the seed,
never punishing, which is not comparable with the free-mode campaigns'
improvement and its 70% bar.

--resume: reuse the slices an interrupted run of the same arguments already
finished in <out dir> (a version slice whose sweep.json was fetched, a
reference slice whose score was saved); the rest are run. Seeds are those
of a run from scratch.

--references <file> (infra/dais/references/<setting>.yml) names fixed
managers to play beside the versions: each in K slices of the same final
run's setting and size (the first version with a final is the template: its
artificial humans, its other managers and its score), with the code of
--code or of this checkout's HEAD. A batched reference plays the versions'
seeds; one that runs only unbatched (`rl`, `human`) plays its own, disjoint
ones (`common_seeds` false), as a batch's seeds differ from an episode's. A
reference is scored from its slices' per_round.parquet as the sweeps score a
point (pool against `ah` in exp2, the average payoff in exp1) and listed in
references.tsv and the summary. --references-only plays the references
alone, with the campaign's versions as the template only.

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
import os
import re
import subprocess
import sys
import tempfile
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


def git(*args, check=True, env=None):
    result = subprocess.run(["git", "-C", str(MAIN), *args], capture_output=True,
                            text=True, env={**os.environ, **(env or {})})
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip() if result.returncode == 0 else None


def version_files(ref, stem):
    """(final config commit, final config dict, recorded final sweep.json), or
    None when the version has no final run on `ref`. A missing branch is an
    error, not a version without a final."""
    if git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False) is None:
        raise SystemExit(f"no branch {ref} here: fetch it first")
    config = f"{CONFIGS}/{stem}_sweep_final.yml"
    result = git("show", f"{ref}:{OUT}/{stem}_sweep/final/sweep.json", check=False)
    commit = git("log", "-1", "--format=%H", ref, "--", config, check=False)
    if not result or not commit:
        return None
    final = json.loads(result)
    if "best_name" not in final:
        return None  # not marked by --score-final: the version did not finish
    return commit, yaml.safe_load(git("show", f"{commit}:{config}")), final


def with_code(commit, code_ref):
    """A commit (on no branch) with `commit`'s files but src/aimanager of
    `code_ref`: the version's rule and configs played by other simulator
    code."""
    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_INDEX_FILE": os.path.join(tmp, "index")}
        git("read-tree", commit, env=env)
        git("rm", "-r", "-q", "-f", "--cached", "src/aimanager", env=env)
        git("read-tree", "--prefix=src/aimanager/", f"{code_ref}:src/aimanager", env=env)
        tree = git("write-tree", env=env)
    return git("commit-tree", tree, "-p", commit, "-m",
               f"reevaluate: {commit[:12]} with src/aimanager of {code_ref}",
               env={"GIT_AUTHOR_NAME": "reevaluate", "GIT_AUTHOR_EMAIL": "-",
                    "GIT_COMMITTER_NAME": "reevaluate", "GIT_COMMITTER_EMAIL": "-"})


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
        if g == min(r["generation"] for r in rows):
            first = best  # None when no generation-1 version has a final
        point = {"generation": g, "best_so_far": best["stem"] if best else None}
        for key in ("recorded", "recorded_se", "reevaluated", "reevaluated_se"):
            point[key] = best[key] if best else None
        point["gain_vs_first"] = point["gain_se"] = None
        if slices and best is not None and first is not None:
            point["gain_vs_first"], point["gain_se"] = paired(
                slices[first["stem"]], slices[best["stem"]])
        out.append(point)
    return out


def improvement(points):
    """The final best so far minus generation 1's best, re-evaluated: the
    paired gain of the curve's last point, and 70% of it."""
    first, last = points[0], points[-1]
    gain, se = last["gain_vs_first"], last["gain_se"]
    if gain is None:
        return {"first_generation_best": None, "final_best": last["best_so_far"],
                "improvement": None, "improvement_se": None, "bar_70": None,
                "bar_70_se": None,
                "note": "no improvement defined: no generation-1 version has a final run"}
    return {"first_generation_best": first["best_so_far"], "final_best": last["best_so_far"],
            "improvement": gain, "improvement_se": se, "bar_70": 0.7 * gain,
            "bar_70_se": 0.7 * se, "se_method": "paired over common-seed slices"}


def with_ci(summary, k):
    """95% intervals (t with k - 1 degrees of freedom) of the improvement and
    its bar."""
    if summary["improvement"] is None:
        return summary
    t = t95(k)
    for key in ("improvement", "bar_70"):
        m, e = summary[key], summary[f"{key}_se"]
        summary[f"{key}_ci95"] = [m - t * e, m + t * e]
    summary["t95"] = t
    return summary


def t95(k):
    """The two-sided 95% t value for a mean over k slices."""
    from scipy.stats import t

    return float(t.ppf(0.975, k - 1))


def gain(slices, first, last, use=None):
    """The paired gain of `last` over `first` on the slices `use` (all)."""
    pick = (lambda v: v) if use is None else (lambda v: [v[j] for j in use])
    return paired(pick(slices[first]), pick(slices[last]))


def robust(rows, slices):
    """Companions of the improvement that do not hinge on which rules the
    campaign's noisy recorded scores selected.

    - `runner_up`: from generation 1's second best by recorded score instead
      (how far the campaign's own choice moves the number)
    - `in_sample`: from generation 1's best to the best overall, both chosen
      by re-evaluated score on the same slices (optimistic: selection and
      estimate share their noise)
    - `cross_fit`: the same choice made on one half of the slices (even or
      odd j) and measured on the other half, both ways. When both halves
      choose the same two rules, their estimates are averaged (se from both);
      when they disagree, the two estimates are reported separately and the
      interval spans both halves' intervals (`agree`: false)

    {"undefined": reason} when no generation-1 version has a final run.
    """
    first_gen = min(x["generation"] for x in rows)
    scored = [r for r in rows if r["stem"] in slices]
    gen1 = [r for r in scored if r["generation"] == first_gen]
    if not gen1:
        return {"undefined": "no generation-1 version has a final run"}
    k = len(next(iter(slices.values())))
    out = {}
    by_rec = sorted(gen1, key=lambda r: -r["recorded"])
    final = max(scored, key=lambda r: r["recorded"])["stem"]

    def entry(first, last, use=None, dof=k):
        m, e = gain(slices, first, last, use)
        t = t95(dof)
        return {"first": first, "last": last, "improvement": m, "se": e,
                "ci95": [m - t * e, m + t * e]}

    if len(by_rec) > 1:
        out["runner_up"] = entry(by_rec[1]["stem"], final)

    def best(group, use):
        def mean(stem):
            return sum(slices[stem][j][0] for j in use) / len(use)
        return max((r["stem"] for r in group), key=mean)

    everything = list(range(k))
    out["in_sample"] = entry(best(gen1, everything), best(scored, everything))
    if k >= 4:
        halves = {"even": everything[0::2], "odd": everything[1::2]}
        parts = []
        for name, measure in (("even", halves["odd"]), ("odd", halves["even"])):
            pick = halves[name]
            part = entry(best(gen1, pick), best(scored, pick), measure, len(measure))
            parts.append({"selected_on": name, **part})
        agree = len({(p["first"], p["last"]) for p in parts}) == 1
        mean = sum(p["improvement"] for p in parts) / 2
        if agree:
            se = math.hypot(*(p["se"] for p in parts)) / 2
            t = t95(len(halves["odd"]))
            ci = [mean - t * se, mean + t * se]
        else:  # different rules: no common estimate, the halves' range
            se = None
            ci = [min(p["ci95"][0] for p in parts), max(p["ci95"][1] for p in parts)]
        out["cross_fit"] = {"first": parts[0]["first"] if agree else "(differs by half)",
                            "last": parts[0]["last"] if agree else "(differs by half)",
                            "improvement": mean, "se": se, "ci95": ci, "agree": agree,
                            "halves": parts}
    return out


#: Managers that punish only unbatched (CLAUDE.md, Manager Comparison)
UNBATCHED = ("rl", "human")
#: The sweep metrics a reference can be scored by (sweep.py), with the anchor
#: a reference plays against (None: alone with the clones)
REFERENCE_METRICS = {"pool_vs_anchor": "ah", "group_payoff": None}


def reference_config(template, name, spec, seed, out):
    """A final run's config `template` with its design points replaced by the
    reference `name` (`spec` a sim config's manager entry, or None for the
    template's own manager of that name), not a sweep, keeping per_round."""
    metric = template.get("sweep_metric", "pool_vs_anchor")
    if metric not in REFERENCE_METRICS:
        raise SystemExit(f"references: setting with sweep_metric {metric} not supported")
    anchor = REFERENCE_METRICS[metric]
    own = {k: v for k, v in template["managers"].items() if v.get("type") != "rule_based"}
    if spec is None and name not in own:
        raise SystemExit(f"references: {name} is null but the setting has no such manager")
    managers = {**own, name: spec if spec is not None else own[name]}
    config = {k: v for k, v in template.items() if not k.startswith("sweep")}
    config["managers"] = {k: v for k, v in managers.items() if k in (name, anchor)}
    if anchor is not None:
        config["pairings"] = [{"name": f"{name}_vs_{anchor}", "group_0": name,
                               "group_1": anchor}]
    if config["managers"][name].get("type") in UNBATCHED:
        config["episode_batch_size"] = 1
    config.update(seed=seed, output_dir=out, figure_name=Path(out).name,
                  save_per_round=True)
    return config


def reference_score(per_round, metric, name):
    """(score, se) of reference `name` in one slice, as sweep.py scores a point
    by `metric` (the template's sweep_metric)."""
    sys.path.insert(0, str(MAIN / "src"))
    import pandas as pd
    from aimanager.simulation import pool_scores

    df = pool_scores.add_pool(pd.read_parquet(per_round))
    anchor = REFERENCE_METRICS[metric]
    scores = (pool_scores.group_payoff_scores(df) if anchor is None
              else pool_scores.against_anchor(df, anchor))
    pool = scores[name]["pool"]
    return float(pool.mean()), float(pool.std(ddof=1) / math.sqrt(len(pool)))


def top_lineages(rows, k):
    """The k lineages with the highest recorded generation-1 score (ties: the
    lower lineage number): those a top_k = k campaign iterates throughout,
    since a lineage best only rises and the others' stay fixed (campaign.py
    select). Ranked by the recorded GP-peak pool, as campaign.py ranks exp1
    and exp2 (a setting with a baseline ranks by its paired gain there)."""
    first_gen = min(r["generation"] for r in rows)
    gen1 = {r["lineage"]: r["recorded"] for r in rows if r["generation"] == first_gen}

    def number(name):
        tail = name.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    ranked = sorted(gen1, key=lambda n: (-(gen1[n] if gen1[n] is not None else -math.inf),
                                         number(n)))
    return ranked[:k]


def analysis(rows, slices, k):
    """Curve, improvement and its robust companions over `rows`."""
    points = curve(rows, slices)
    return points, {**with_ci(improvement(points), k), "robust": robust(rows, slices)}


def lineage_curves(rows):
    """Per lineage and generation: its best version so far by recorded score,
    with that version's re-evaluated score."""
    out = []
    for name in sorted({r["lineage"] for r in rows},
                       key=lambda n: int(n.rsplit("-", 1)[-1]) if n.rsplit("-", 1)[-1].isdigit()
                       else 0):
        best = None
        for g in sorted({r["generation"] for r in rows}):
            for r in rows:
                if r["lineage"] == name and r["generation"] == g and r["recorded"] is not None \
                        and (best is None or r["recorded"] > best["recorded"]):
                    best = r
            mine = [r for r in rows if r["lineage"] == name and r["generation"] == g]
            out.append({"lineage": name, "generation": g,
                        "version_this_generation": mine[0]["stem"] if mine else None,
                        "best_so_far": best["stem"] if best else None,
                        "recorded": best["recorded"] if best else None,
                        "reevaluated": best["reevaluated"] if best else None,
                        "reevaluated_se": best["reevaluated_se"] if best else None})
    return out


def write_tsv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("NA" if v is None else f"{v:.4f}" if isinstance(v, float)
                                 else v) for k, v in r.items()})


#: The headline of a single-idea campaign's summary (--single-idea)
SINGLE_IDEA_HEADLINE = (
    "all lineages; single-idea mode: the improvement is the gain of the best rule"
    " over the seed (never punishing), not comparable with the free-mode campaigns'"
    " improvement and its 70% bar")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("state")
    ap.add_argument("out")
    ap.add_argument("--refs", default="policy-finder/")
    ap.add_argument("--target-se", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=100_000)
    ap.add_argument("--slices", type=int, help="instead of --target-se")
    ap.add_argument("--keep-snapshots", action="store_true")
    ap.add_argument("--code", help="play every version with src/aimanager of this ref")
    ap.add_argument("--references", help="a YAML of fixed managers to play beside "
                    "the versions (infra/dais/references/)")
    ap.add_argument("--references-only", action="store_true",
                    help="play only the references, on the campaign's setting")
    ap.add_argument("--single-idea", action="store_true",
                    help="a single-idea campaign: all lineages, the gain over the seed")
    ap.add_argument("--resume", action="store_true",
                    help="reuse the slices an interrupted run finished in <out>")
    ap.add_argument("--compare-top-k", type=int, default=4,
                    help="a campaign that iterated more lineages also gets the result "
                    "over the k best after generation 1 (the reference's top_k)")
    args = ap.parse_args()
    if cluster_proxy.BACKEND != "local":
        sys.exit("set PF_CLUSTER_BACKEND=local (infra/dais/pipeline.slurm.sh)")
    out = Path(args.out)
    if (not args.resume and (out / "slices").exists()
            and any((out / "slices").iterdir())):
        sys.exit(f"{out} holds slices of an earlier run: choose a new out dir, or"
                 " --resume it")
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

    refs = yaml.safe_load(Path(args.references).read_text()) if args.references else {}
    if args.references_only and not refs:
        sys.exit("--references-only needs --references")
    snapshots, jobs = {}, {}
    try:
        results, ref_results = run_slices(args, versions, k, out, snapshots, jobs, refs)
    finally:
        if not args.keep_snapshots:  # only those this run made, also after a failure
            for sha in snapshots.values():
                local_cluster.cleanup(sha)
    ref_rows = [{"reference": name, **ref_results[name]} for name in refs]
    if ref_rows:
        write_tsv(out / "references.tsv", ref_rows)
    if args.references_only:
        print("\n".join(reference_lines(ref_rows)))
        return
    for r in rows:
        if r["stem"] in results:
            r["reevaluated"], r["reevaluated_se"] = combine(results[r["stem"]])
            r["episodes"] = k * versions[r["stem"]][1]["n_episodes"]

    # the tables first: they hold whatever the summary below can make of them
    with open(out / "slices.tsv", "w") as f:  # the raw per-slice scores
        f.write("stem\tslice\tpool\tpool_se\n")
        for stem, values in results.items():
            for j, (pool, se) in enumerate(values):
                f.write(f"{stem}\t{j}\t{pool:.6f}\t{se:.6f}\n")
    write_tsv(out / "versions.tsv", rows)
    write_tsv(out / "lineages.tsv", lineage_curves(rows))
    # a campaign that iterated more lineages than the reference (top_k 4):
    # two results, the reference-comparable one over the lineages a top_k 4
    # campaign would have iterated, and the one over all
    first_gen = min(r["generation"] for r in rows)
    iterated = {r["lineage"] for r in rows if r["generation"] > first_gen}
    subsets = {"all": sorted({r["lineage"] for r in rows})}
    if len(iterated) > args.compare_top_k and not args.single_idea:
        subsets["comparable"] = top_lineages(rows, args.compare_top_k)
    results_by = {}
    for name, lineages in subsets.items():
        part = [r for r in rows if r["lineage"] in lineages]
        points, result = analysis(part, results, k)
        write_tsv(out / ("curve.tsv" if name == "all" else f"curve_{name}.tsv"), points)
        results_by[name] = {"lineages": lineages, **result, "points": points}
    head = results_by.get("comparable", results_by["all"])
    summary = {"state": str(args.state), "refs": args.refs, "seed": args.seed,
               "slices": k, "target_se": args.target_se, "code": args.code,
               "no_final": missing, "references": ref_rows,
               "headline": (SINGLE_IDEA_HEADLINE if args.single_idea else
                            "comparable: the %d lineages best after generation 1 by "
                            "recorded score, as a top_k %d campaign iterates"
                            % (args.compare_top_k, args.compare_top_k)
                            if "comparable" in results_by else "all lineages"),
               "mode": "single_idea" if args.single_idea else "free",
               **{k_: v for k_, v in head.items() if k_ not in ("points", "lineages")},
               "subsets": {n: {k_: v for k_, v in r.items() if k_ != "points"}
                           for n, r in results_by.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, r in results_by.items():
        print(f"== {name}: {len(r['lineages'])} lineages")
        print(report(r["points"], r))
    if ref_rows:
        print("== references")
        print("\n".join(reference_lines(ref_rows)))


def reference_lines(ref_rows):
    return [f"{r['reference']}: {_num(r['score'], r['se'])} over {r['episodes']} episodes"
            + ("" if r["common_seeds"] else " (own seeds: runs unbatched)")
            for r in ref_rows]


def _num(x, se=None):
    if x is None:
        return "NA"
    return f"{x:.2f}" + ("" if se is None else f" +- {se:.2f}")


def report(points, summary):
    """The run's summary lines."""
    lines = [f"gen {p['generation']}: {p['best_so_far']} recorded {_num(p['recorded'])}"
             f" re-evaluated {_num(p['reevaluated'], p['reevaluated_se'])}" for p in points]
    if summary["improvement"] is None:
        lines.append(summary["note"])
    else:
        lines.append(f"improvement {_num(summary['improvement'], summary['improvement_se'])},"
                     f" 70%: {_num(summary['bar_70'])}")
    for name, r in summary["robust"].items():
        if name == "undefined":
            lines.append(f"robust improvements: {r}")
            continue
        lines.append(f"{name}: {r['first']} -> {r['last']}: {_num(r['improvement'], r['se'])}"
                     f" (95%: {_num(r['ci95'][0])} to {_num(r['ci95'][1])})")
        if name == "cross_fit" and not r["agree"]:
            lines.append("  the halves chose different rules: " + "; ".join(
                f"chosen on {h['selected_on']}: {h['first']} -> {h['last']}"
                f" {_num(h['improvement'], h['se'])}" for h in r["halves"]))
    return "\n".join(lines)


def run_slices(args, versions, k, out, snapshots, jobs, refs=None):
    """Every version's and reference's K slices into the pool; ({stem: [(pool,
    se)] by slice}, {reference: its row}). Fails if a slice's sim failed or its
    output is missing. With --resume, slices an earlier run finished in `out`
    are read, not run."""
    done = {}
    for stem, (commit, config, gp) in ({} if args.references_only
                                       else versions).items():
        if args.code:
            commit = with_code(commit, args.code)
        seed, todo = args.seed, []
        for j in range(k):
            name = f"{stem}_r{j:02d}"
            cfg = gp_config(config, gp, seed, f"{EVAL_OUT}/{name}")
            seed += batches(cfg)
            if args.resume and (out / "slices" / name / "sweep.json").exists():
                done[name] = stem  # finished by an earlier run: its seeds are these
            else:
                todo.append((name, cfg))
        if not todo:
            continue
        # the slices' configs go into the snapshot: the local backend's only
        existed = (local_cluster.tree(commit).parent / "complete").exists()
        sha = cluster_proxy.snapshot(commit)
        if not existed:
            snapshots[stem] = sha
        tree = local_cluster.tree(sha)
        for name, cfg in todo:
            path = tree / EVAL_CONFIGS / f"{name}.yml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            jobs[name] = (stem, sha, cluster_proxy.submit(
                f"{EVAL_CONFIGS}/{name}.yml", sha, run_group="reevaluate"))
    ref_jobs, ref_done = submit_references(args, refs or {}, versions, k, snapshots, out)
    print(f"{len(jobs) + len(ref_jobs)} slices submitted"
          + (f", {len(done) + len(ref_done)} finished before" if args.resume else ""),
          flush=True)
    cluster_proxy.wait([job for _, _, job in jobs.values()]
                       + [job for *_, job in ref_jobs.values()])

    results, failed = {}, []
    ref_slices = {}
    for name, (ref, sha, cfg, job) in ref_jobs.items():
        code = local_cluster.status(job).get("exit_code")
        try:
            if code != 0:
                raise RuntimeError(f"exit code {code}")
            cluster_proxy.fetch(sha, f"{EVAL_OUT}/{name}", out / "slices" / name)
            per_round = out / "slices" / name / "per_round.parquet"
            metric = next(iter(versions.values()))[1].get("sweep_metric", "pool_vs_anchor")
            score = reference_score(per_round, metric, ref)
            ref_slices.setdefault(ref, []).append(score)
            # kept for --resume; the per-round data is large
            (out / "slices" / name / "score.json").write_text(json.dumps(list(score)))
            per_round.unlink()
        except (cluster_proxy.ClusterError, OSError, KeyError, ValueError,
                RuntimeError) as e:
            failed.append(f"{name} (job {job}): {e!r}")
    for name, (ref, cfg) in ref_done.items():
        score = json.loads((out / "slices" / name / "score.json").read_text())
        ref_slices.setdefault(ref, []).append(tuple(score))
    for name, stem in done.items():
        sweep = json.loads((out / "slices" / name / "sweep.json").read_text())
        gp = {p["name"]: p for p in sweep["points"]}[versions[stem][2]]
        results.setdefault(stem, []).append((gp["pool"], gp["pool_se"]))
    for name, (stem, sha, job) in jobs.items():
        code = local_cluster.status(job).get("exit_code")
        if code != 0:
            failed.append(f"{name} (job {job}): exit code {code}")
            continue
        try:
            cluster_proxy.fetch(sha, f"{EVAL_OUT}/{name}", out / "slices" / name)
            sweep = json.loads((out / "slices" / name / "sweep.json").read_text())
            gp = {p["name"]: p for p in sweep["points"]}[versions[stem][2]]
            results.setdefault(stem, []).append((gp["pool"], gp["pool_se"]))
        except (cluster_proxy.ClusterError, OSError, KeyError, ValueError) as e:
            failed.append(f"{name} (job {job}): {e!r}")
    if failed:
        sys.exit("FAILED slices:\n" + "\n".join(failed))
    ref_rows = {}
    for ref, values in ref_slices.items():
        score, se = combine(values)
        cfg = next([c for r, _, c, _ in ref_jobs.values() if r == ref]
                   + [c for r, c in ref_done.values() if r == ref])
        ref_rows[ref] = {"score": score, "se": se, "slices": len(values),
                         "episodes": len(values) * cfg["n_episodes"],
                         "common_seeds": cfg.get("episode_batch_size", 1)
                         == next(iter(versions.values()))[1].get("episode_batch_size", 1)}
    return results, ref_rows


def submit_references(args, refs, versions, k, snapshots, out=None):
    """Each reference's K slices, on the first version's final run setting;
    ({slice name: (reference, sha, config, job)}, {slice name: (reference,
    config)} of those an earlier run finished (--resume))."""
    if not refs:
        return {}, {}
    template = next(iter(versions.values()))[1]
    commit = git("rev-parse", "HEAD")
    if args.code:
        commit = with_code(commit, args.code)
    existed = (local_cluster.tree(commit).parent / "complete").exists()
    sha = cluster_proxy.snapshot(commit)
    if not existed:
        snapshots["references"] = sha
    tree = local_cluster.tree(sha)
    size = template.get("episode_batch_size", 1)
    # the out dir's name in the slices' names: re-evaluations that share a
    # snapshot (the same code) must not share a reference's slice dirs
    run = re.sub(r"[^A-Za-z0-9.-]", "-", Path(args.out).resolve().name)
    ref_jobs, ref_done = {}, {}
    for ref, spec in refs.items():
        seed = args.seed
        for j in range(k):
            name = f"ref-{run}-{ref}_r{j:02d}"
            cfg = reference_config(template, ref, spec, seed, f"{EVAL_OUT}/{name}")
            # the versions' seeds when batched alike; else disjoint ones of its own
            seed += max(batches(cfg), batches({**cfg, "episode_batch_size": size}))
            if (getattr(args, "resume", False) and out is not None
                    and (out / "slices" / name / "score.json").exists()):
                ref_done[name] = (ref, cfg)
                continue
            path = tree / EVAL_CONFIGS / f"{name}.yml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            ref_jobs[name] = (ref, sha, cfg, cluster_proxy.submit(
                f"{EVAL_CONFIGS}/{name}.yml", sha, run_group="reevaluate"))
    return ref_jobs, ref_done


if __name__ == "__main__":
    main()
