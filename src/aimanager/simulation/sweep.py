"""Sweep sims (#227): `sweep: true` in a sim config.

A sweep config (scripts/policy_finder/generate_sim_config.py writes them)
plays every design point of one rule against `ah`: pairings
`<point>_vs_ah`, the point in group 0, each point a `rule_based` manager with
its params inline. The sim then skips its plots and writes `sweep.json`:
every point's params and its definition 4 score (pool_scores), and under
`best` the params of the point with the highest pool, so the file loads as a
manager's `params:` (manager/rule.py). Pandas only, so it runs locally.
"""

import json
import math
import os

from aimanager.simulation.pool_scores import add_pool, against_anchor

#: The manager every design point plays (definition 4 of #226).
SWEEP_ANCHOR = "ah"
SWEEP_FILE = "sweep.json"


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
    best = max(points, key=lambda point: point["pool"])
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
