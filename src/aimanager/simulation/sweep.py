"""Sweep sims (#227): `sweep: true` in a sim config.

A sweep config (scripts/policy_finder/generate_sim_config.py writes them)
plays every design point of one rule against `ah`: pairings
`<point>_vs_ah`, the point in group 0, each point a `rule_based` manager with
its params inline. The sim then skips its plots and writes `sweep.json`:
every point's params and its definition 4 score (pool_scores), and under
`best` the params of the point with the highest pool, so the file loads as a
manager's `params:` (manager/rule.py); and, with the sim's own plotting code,
the best point's matchup against `ah` (BEST_PLOTS). PyG-free, so it runs
locally.
"""

import json
import math
import os
import shutil

from aimanager.simulation.pool_scores import add_pool, against_anchor
from aimanager.simulation.sim_plots import plot_group_size, plot_pairing_side, prepare

#: The manager every design point plays (definition 4 of #226).
SWEEP_ANCHOR = "ah"
SWEEP_FILE = "sweep.json"
#: The plots a sweep draws of its best point against the anchor.
BEST_PLOTS = ("comparison_pairing_side.jpg", "group_size_evolution_global.jpg")


def check_sweep(config: dict) -> str:
    """Check a `sweep: true` config before the sim runs; return its rule path."""
    managers, pairings = config.get("managers", {}), config.get("pairings")
    if SWEEP_ANCHOR not in managers:
        raise ValueError(f"sweep: needs the anchor manager `{SWEEP_ANCHOR}`")
    if not pairings:
        raise ValueError("sweep: needs `pairings`, one `<point>_vs_ah` per point")
    rules = set()
    for p in pairings:
        point = p.get("group_0")
        if p.get("group_1") != SWEEP_ANCHOR or point == SWEEP_ANCHOR:
            raise ValueError(
                f"sweep: pairing {p.get('name')!r} must put a design point in"
                f" group_0 and `{SWEEP_ANCHOR}` in group_1"
            )
        manager = managers.get(point, {})
        if manager.get("type") != "rule_based" or not isinstance(
            manager.get("params"), dict
        ):
            raise ValueError(
                f"sweep: `{point}` must be a rule_based manager with inline params"
            )
        rules.add(manager["rule"])
    if len(rules) != 1:
        raise ValueError(f"sweep: all points must share one rule, got {sorted(rules)}")
    return rules.pop()


def sweep_result(df, config: dict) -> dict:
    """The contents of sweep.json, from the sim's per-round frame."""
    scores = against_anchor(add_pool(df), SWEEP_ANCHOR)
    managers = config["managers"]
    points = []
    for p in config["pairings"]:
        name = p["group_0"]
        s = scores[name]
        n = len(s)
        se = float(s["pool"].std(ddof=1) / math.sqrt(n)) if n > 1 else None
        points.append(
            {
                "name": name,
                "params": managers[name]["params"],
                "pool": float(s["pool"].mean()),
                "pool_se": se,
                "members": float(s["members"].mean()),
                "n_episodes": n,
            }
        )
    best = _best(points)
    return {
        "best": best["params"],
        "best_name": best["name"],
        "rule": check_sweep(config),
        "anchor": SWEEP_ANCHOR,
        "score": (
            "definition 4 of #226: the group's common pool 1.6*sum(c) - sum(p)"
            " per round against the anchor, empty group-rounds 0, mean over"
            " rounds then episodes; pool_se its standard error over episodes"
        ),
        "seed": config.get("seed"),
        "n_episodes": config.get("n_episodes"),
        "episode_batch_size": config.get("episode_batch_size", 1),
        "points": points,
    }


def write_sweep(df, config: dict, output_dir: str) -> str:
    """Write sweep.json into the sim's output dir; return its path."""
    path = os.path.join(output_dir, SWEEP_FILE)
    with open(path, "w") as f:
        json.dump(sweep_result(df, config), f, indent=2)
    return path


def plot_best(df, output_dir: str) -> None:
    """Draw BEST_PLOTS for the best point in output_dir's sweep.json, from the
    sim's per-round frame (the rest of a sweep's plots are skipped)."""
    with open(os.path.join(output_dir, SWEEP_FILE)) as f:
        sweep = json.load(f)
    best = sweep["best_name"]
    pairing = {
        "name": f"{best}_vs_{SWEEP_ANCHOR}",
        "group_0": best,
        "group_1": SWEEP_ANCHOR,
    }
    rows = df[df["run"].str.endswith(f"managed by {pairing['name']}")]
    rows = prepare(rows.reset_index(drop=True).copy())
    title = f"{best}: " + ", ".join(f"{k}={v:g}" for k, v in sweep["best"].items())
    plot_pairing_side(rows, [pairing], output_dir, title)
    plot_group_size(rows, output_dir, title)


def copy_best_plots(part_dirs: list, best_name: str, out_dir: str) -> list:
    """Copy BEST_PLOTS from the part whose own best is `best_name` (the
    overall best is always its part's best); return the copied paths."""
    copied = []
    for part in part_dirs:
        with open(os.path.join(part, SWEEP_FILE)) as f:
            if json.load(f).get("best_name") != best_name:
                continue
        for name in BEST_PLOTS:
            src = os.path.join(part, name)
            if os.path.exists(src):
                copied.append(shutil.copy(src, os.path.join(out_dir, name)))
    return copied


def _best(points: list) -> dict:
    return max(points, key=lambda point: point["pool"])


def merge_sweeps(results: list) -> dict:
    """One sweep.json from the sweep.json of a sweep's parts (--n-parts).

    The parts must share their rule, anchor, score and episodes; points are
    joined in design order (the index in `<rule>_s<index>`) and `best` is
    taken over all of them.
    """
    if not results:
        raise ValueError("merge: no sweep results")
    first = results[0]
    for key in ("rule", "anchor", "score", "n_episodes", "episode_batch_size"):
        values = {json.dumps(r.get(key)) for r in results}
        if len(values) > 1:
            raise ValueError(f"merge: the parts differ in `{key}`: {sorted(values)}")
    points = [p for r in results for p in r["points"]]
    points.sort(key=lambda p: int(p["name"].rsplit("_s", 1)[1]))
    names = [p["name"] for p in points]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"merge: points in more than one part: {duplicates}")
    best = _best(points)
    merged = {k: v for k, v in first.items() if k not in ("points", "seed")}
    merged.update(
        {
            "best": best["params"],
            "best_name": best["name"],
            "seeds": [r.get("seed") for r in results],
            "points": points,
        }
    )
    return merged
