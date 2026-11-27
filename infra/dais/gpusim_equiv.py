"""Equivalence of sim results across platforms (sim image, device): the
design is in docs/reports/dais_gpu_sim_equivalence.md.

    python infra/dais/gpusim_equiv.py configs
        writes the reference sim configs, one per (reference, job), into
        infra/dais/gpusim_configs/ (committed: Raven runs them from a
        snapshot of this branch). Needs only PyYAML.
    python infra/dais/gpusim_equiv.py episodes CONFIG OUT.csv [--per-round P]
        one row per (point, episode) of a finished sim: the sweep score's
        per-episode pool and members (aimanager.simulation.pool_scores, as
        sweep.json averages them), from its per_round.parquet
    python infra/dais/gpusim_equiv.py sweep-parts RULE OUT_DIR [--n-parts 2]
        a real sweep of RULE (256 points x 500 episodes) as the pipeline
        splits it, its part configs and outputs in OUT_DIR
    python infra/dais/gpusim_equiv.py analyze DIR [--out REPORT.md]
        DIR/<platform>/<config stem>.csv for every platform and job; prints
        per-point means, platform differences, the null comparison (two seed
        sets on one platform) and the decision, as markdown tables
"""

import argparse
import itertools
import math
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "gpusim_configs"
#: Sim parameters not set here come from the stack's reference sim
TEMPLATE = (
    "configs/simulation/manager_testing/25_LEVIN_run1_ah_zero_pairings_batched.yml"
)
#: (reference, rule, {point: params}, episodes per point and job, batch)
REFERENCES = {
    # the 6-point final run of pf-20261104-1 (its Raven result: 2000 episodes,
    # seed 1000, on that instance's branch)
    "final1": (
        "infra/dais/gpusim_configs/pf-20261104-1.yml",
        {
            "pf-20261104-1_gp": dict(
                slope=3.0, tol=0.0, cap=15, t_stop=12, pre_switch=1.0
            ),
            "pf-20261104-1_s183": dict(
                slope=2.163661020776924,
                tol=0.3547392413020134,
                cap=15,
                t_stop=11,
                pre_switch=0.5826417710632086,
            ),
            "pf-20261104-1_s063": dict(
                slope=1.9397214686212731,
                tol=0.24269776046276093,
                cap=4,
                t_stop=5,
                pre_switch=0.9085054006427526,
            ),
            "pf-20261104-1_s043": dict(
                slope=0.6617625103864022,
                tol=0.652161743491888,
                cap=10,
                t_stop=11,
                pre_switch=0.7788509540259838,
            ),
            "pf-20261104-1_s255": dict(
                slope=1.7707028529540205,
                tol=1.4091276582330465,
                cap=13,
                t_stop=15,
                pre_switch=0.6705854739993811,
            ),
            "pf-20261104-1_s103": dict(
                slope=2.551283698044575,
                tol=1.7682885602116585,
                cap=6,
                t_stop=5,
                pre_switch=0.6221125880256295,
            ),
        },
        6000,
        2000,
    ),
    # a different rule: 8 points of pf-20261104-12-v7's sweep at the 0, 15,
    # ..., 90 and 100 % quantiles of its sweep pool (64.8 to 77.4)
    "v7slice": (
        "configs/managers/rule_based/pf-20261104-12-v7.yml",
        {
            "pf-20261104-12-v7_s167": dict(
                thr=10.836938478983939,
                peer=0.1370829939842224,
                ceil=13.237979426980019,
                slope=17.032649676879224,
                decay=0.9654433452483281,
                stop=3,
            ),
            "pf-20261104-12-v7_s219": dict(
                thr=9.581387992016971,
                peer=0.4116171136498451,
                ceil=14.119859956204891,
                slope=49.9276286850355,
                decay=0.7371207236656455,
                stop=3,
            ),
            "pf-20261104-12-v7_s087": dict(
                thr=10.497339916415513,
                peer=0.1046727865934372,
                ceil=13.277786570601165,
                slope=6.342775438771845,
                decay=0.8079123486122839,
                stop=1,
            ),
            "pf-20261104-12-v7_s002": dict(
                thr=7.104958914220333,
                peer=0.6997926436364651,
                ceil=9.370147556066513,
                slope=5.56739448863647,
                decay=0.7846665244750323,
                stop=1,
            ),
            "pf-20261104-12-v7_s101": dict(
                thr=6.360220777802169,
                peer=0.45575982332229614,
                ceil=9.470439782366157,
                slope=13.02524406999795,
                decay=0.3775091078690453,
                stop=3,
            ),
            "pf-20261104-12-v7_s120": dict(
                thr=10.937460301443934,
                peer=0.8038776852190495,
                ceil=14.488934378139675,
                slope=19.791269702961085,
                decay=0.460307427568258,
                stop=3,
            ),
            "pf-20261104-12-v7_s253": dict(
                thr=7.119151911698282,
                peer=0.10016933307051659,
                ceil=12.401599420234561,
                slope=33.30986107259669,
                decay=0.47053076831715235,
                stop=3,
            ),
            "pf-20261104-12-v7_s008": dict(
                thr=10.673051610589027,
                peer=0.6021612554788589,
                ceil=14.76794362720102,
                slope=8.11983387380725,
                decay=0.5341744628221813,
                stop=4,
            ),
        },
        4500,
        1500,
    ),
}
#: Jobs per reference and platform; jobs 0-1 are seed set S1, 2-3 set S2.
#: Every platform runs the same seeds.
N_JOBS = 4
SEED_BASE = {"final1": 71000, "v7slice": 72000}
SEED_STEP = 10


def write_configs(_):
    with open(TEMPLATE) as f:
        template = yaml.safe_load(f)
    CONFIG_DIR.mkdir(exist_ok=True)
    for ref, (rule, points, episodes, batch) in REFERENCES.items():
        for job in range(N_JOBS):
            config = dict(template)
            config["managers"] = {"ah": template["managers"]["ah"]}
            config["pairings"] = []
            for name, params in points.items():
                config["managers"][name] = {
                    "type": "rule_based",
                    "rule": rule,
                    "params": params,
                }
                config["pairings"].append(
                    {"name": f"{name}_vs_ah", "group_0": name, "group_1": "ah"}
                )
            stem = f"{ref}_j{job}"
            config.update(
                seed=SEED_BASE[ref] + SEED_STEP * job,
                # every point plays the same games: batch b of each point is
                # seeded seed + b (common random numbers for the contrasts)
                seed_cycle=episodes // batch,
                n_episodes=episodes,
                episode_batch_size=batch,
                output_dir=f"plots/simulation/gpusim/{stem}",
                figure_name=f"gpusim_{stem}",
                save_per_round=True,
                sweep=True,
            )
            out = CONFIG_DIR / f"{stem}.yml"
            with open(out, "w") as f:
                f.write(
                    "# Platform equivalence of sims (infra/dais/gpusim_equiv.py"
                    " configs; docs/reports/dais_gpu_sim_equivalence.md)\n"
                )
                yaml.safe_dump(config, f, sort_keys=False, width=1000)
            print(out)


def episodes(a):
    import pandas as pd

    from aimanager.simulation.pool_scores import add_pool, against_anchor

    with open(a.config) as f:
        config = yaml.safe_load(f)
    per_round = a.per_round or Path(config["output_dir"]) / "per_round.parquet"
    df = add_pool(pd.read_parquet(per_round))
    scores = against_anchor(df, "ah")
    frames = []
    for name in config["managers"]:
        if name == "ah":
            continue
        s = scores[name]
        frames.append(
            pd.DataFrame(
                {
                    "point": name,
                    "episode": s["episode"].to_numpy(),
                    "pool": s["pool"].to_numpy(),
                    "members": s["members"].to_numpy(),
                }
            )
        )
    out = pd.concat(frames)
    if out[["pool", "members"]].isna().any().any():
        sys.exit("non-finite per-episode scores")
    n = out.groupby("point").size()
    if (n != config["n_episodes"]).any():
        sys.exit(f"episodes per point {n.to_dict()}, expected {config['n_episodes']}")
    out.to_csv(a.out, index=False, float_format="%.6g")
    print(a.out, len(out), "rows")


def sweep_parts(a):
    """A real sweep's part configs, as the pipeline writes them
    (generate_sim_config.py's functions), into a folder outside the repo."""
    sys.path.insert(0, str(HERE.parents[1] / "scripts" / "policy_finder"))
    import generate_sim_config as g

    from aimanager.manager.rule import validate_rule

    design = validate_rule(a.rule, 1, None, g.SOBOL_POINTS)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for suffix, config in g.build_sim_configs(
        a.rule, design, g.N_EPISODES, g.SEED, a.n_parts
    ):
        config["output_dir"] = str(out / f"sweep{suffix}")
        path = out / f"sweep{suffix}.yml"
        with open(path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False, width=1000)
        print(path, len(config["pairings"]), "points x", config["n_episodes"])


# ---------------------------------------------------------------- analysis


def _load(root, platform, ref, jobs):
    """{job: frame} of one platform's per-episode csvs for `ref`."""
    import pandas as pd

    frames = {}
    for job in jobs:
        path = Path(root) / platform / f"{ref}_j{job}.csv"
        if path.exists():
            frames[job] = pd.read_csv(path).assign(job=job)
    return frames


def _units(frames, metric, points):
    """[units x points] matrix: one row per game (job, episode), which every
    point played with the same seed (common random numbers)."""
    import pandas as pd

    df = pd.concat(frames.values())
    wide = df.pivot_table(index=["job", "episode"], columns="point", values=metric)
    return wide[points].to_numpy()


def _compare(x, y):
    """Two [units x points] samples: per-point difference y - x with its
    covariance, the common shift, and the test of the point contrasts."""
    import numpy as np
    from scipy import stats

    d = y.mean(0) - x.mean(0)
    v = np.cov(x, rowvar=False) / len(x) + np.cov(y, rowvar=False) / len(y)
    k = len(d)
    w = np.full(k, 1 / k)
    shift, shift_se = float(w @ d), float(math.sqrt(w @ v @ w))
    # contrasts: each point against the mean of the points (k - 1 independent)
    c = (np.eye(k) - w)[:-1]
    cd, cv = c @ d, c @ v @ c.T
    chi2 = float(cd @ np.linalg.solve(cv, cd))
    full = float(d @ np.linalg.solve(v, d))
    return {
        "d": d,
        "se": np.sqrt(np.diag(v)),
        "shift": shift,
        "shift_se": shift_se,
        "shift_p": float(2 * stats.norm.sf(abs(shift) / shift_se)),
        "contrast": (np.eye(k) - w) @ d,
        "contrast_se": np.sqrt(np.diag((np.eye(k) - w) @ v @ (np.eye(k) - w).T)),
        "contrast_chi2": chi2,
        "contrast_p": float(stats.chi2.sf(chi2, k - 1)),
        "chi2": full,
        "p": float(stats.chi2.sf(full, k)),
        "k": k,
    }


def _fmt_p(p):
    return f"{p:.2g}" if p >= 0.001 else f"{p:.0e}"


def analyze(a):
    import numpy as np
    from scipy import stats

    root = Path(a.dir)
    platforms = sorted(p.name for p in root.iterdir() if p.is_dir())
    lines = []
    say = lines.append
    tests = []  # (label, p) of every platform test, for the decision
    pool_rows = []  # the platform comparisons of the pool
    for ref, (_, points, n_ep, _) in REFERENCES.items():
        names = list(points)
        short = [n.rsplit("_", 1)[-1] for n in names]
        data = {}
        for p in platforms:
            frames = _load(root, p, ref, range(N_JOBS))
            if frames:
                data[p] = frames
        if not data:
            continue
        say(f"\n### {ref}: {len(names)} points, {n_ep} episodes per point and job\n")
        for metric in ("pool", "members"):
            full = {p: _units(f, metric, names) for p, f in data.items()}
            say(f"\n#### {metric}: mean +- se per point (all jobs)\n")
            say(
                "| point | "
                + " | ".join(f"{p} (n={len(full[p])})" for p in full)
                + " |"
            )
            say("|---" * (len(full) + 1) + "|")
            for j, n in enumerate(short):
                cells = [
                    f"{u[:, j].mean():.2f} +- {u[:, j].std(ddof=1) / math.sqrt(len(u)):.2f}"
                    for u in full.values()
                ]
                say(f"| {n} | " + " | ".join(cells) + " |")
            sd = np.mean([u.std(0, ddof=1).mean() for u in full.values()])
            rho = np.mean(
                [
                    np.corrcoef(u, rowvar=False)[np.triu_indices(len(names), 1)].mean()
                    for u in full.values()
                ]
            )
            say(
                f"\nPer-episode sd {sd:.1f}; mean correlation between points"
                f" over common games {rho:.2f}.\n"
            )

            say(f"\n#### {metric}: comparisons (difference = second minus first)\n")
            say(
                "| comparison | n per side | shift +- se | p | contrasts chi2 (df)"
                " | p | largest contrast +- se | per-point d (se) |"
            )
            say("|---|---|---|---|---|---|---|---|")
            rows = []
            for p in data:  # null: seed set S2 against S1 on one platform
                s1 = {j: f for j, f in data[p].items() if j < N_JOBS // 2}
                s2 = {j: f for j, f in data[p].items() if j >= N_JOBS // 2}
                if s1 and s2:
                    rows.append(
                        (
                            f"null {p}: S1 vs S2",
                            _units(s1, metric, names),
                            _units(s2, metric, names),
                            False,
                        )
                    )
            for p, q in itertools.combinations(data, 2):
                rows.append((f"{p} vs {q}", full[p], full[q], True))
            for label, x, y, is_platform in rows:
                r = _compare(x, y)
                big = int(np.argmax(np.abs(r["contrast"])))
                say(
                    f"| {label} | {len(x)}/{len(y)} | {r['shift']:+.2f} +- {r['shift_se']:.2f}"
                    f" | {_fmt_p(r['shift_p'])} | {r['contrast_chi2']:.1f} ({r['k'] - 1})"
                    f" | {_fmt_p(r['contrast_p'])}"
                    f" | {short[big]} {r['contrast'][big]:+.2f} +- {r['contrast_se'][big]:.2f}"
                    f" | "
                    + ", ".join(
                        f"{n} {d:+.1f} ({s:.1f})"
                        for n, d, s in zip(short, r["d"], r["se"])
                    )
                    + " |"
                )
                if is_platform and metric == "pool":
                    tests += [
                        (f"{ref} {label} shift", r["shift_p"]),
                        (f"{ref} {label} contrasts", r["contrast_p"]),
                    ]
                    pool_rows.append({**r, "label": label, "ref": ref})
            if metric == "pool":
                say("\nRanking (point order by mean pool, best first):\n")
                for p, u in full.items():
                    order = np.argsort(-u.mean(0))
                    say(f"- {p}: " + " > ".join(short[i] for i in order))
                for p, q in itertools.combinations(full, 2):
                    rho_s = stats.spearmanr(
                        full[p].mean(0), full[q].mean(0)
                    ).correlation
                    say(f"- Spearman {p} vs {q}: {rho_s:.2f}")

    # decision: Holm over every platform test of the pool; equivalence bound
    say("\n### Decision\n")
    alpha = 0.05
    ordered = sorted(tests, key=lambda t: t[1])
    rejected = []
    for i, (label, p) in enumerate(ordered):
        if p > alpha / (len(ordered) - i):
            break
        rejected.append(label)
    say(
        f"Holm over {len(tests)} platform tests of the pool at alpha {alpha}:"
        f" {'rejected: ' + '; '.join(rejected) if rejected else 'none rejected'}.\n"
    )
    say(
        "| comparison | 90% bound on the shift | 90% bound on the largest point difference"
        " | 90% bound on the largest contrast |"
    )
    say("|---|---|---|---|")
    for r in pool_rows:
        z = stats.norm.ppf(0.95)
        say(
            f"| {r['ref']} {r['label']} | {abs(r['shift']) + z * r['shift_se']:.2f}"
            f" | {np.max(np.abs(r['d']) + z * r['se']):.2f}"
            f" | {np.max(np.abs(r['contrast']) + z * r['contrast_se']):.2f} |"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("configs")
    e = sub.add_parser("episodes")
    e.add_argument("config")
    e.add_argument("out")
    e.add_argument("--per-round")
    s = sub.add_parser("sweep-parts")
    s.add_argument("rule")
    s.add_argument("out_dir")
    s.add_argument("--n-parts", type=int, default=2)
    n = sub.add_parser("analyze")
    n.add_argument("dir")
    n.add_argument("--out")
    a = p.parse_args()
    commands = {
        "configs": write_configs,
        "episodes": episodes,
        "sweep-parts": sweep_parts,
        "analyze": analyze,
    }
    commands[a.cmd](a)


if __name__ == "__main__":
    main()
