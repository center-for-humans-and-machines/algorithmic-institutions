"""Write the head-to-head that tests common games (docs/plans/policy-finder-common-games.md):
the top 3 lineage bests of the campaign with common games (pf-20261113) and of the one
without (pf-20261112), the multi leader (pf-20261106-9-v5, w = 0) and `zero`, each against
`ah` in both seats, 2000 games per pairing, on fresh seeds (40000-45000). Every row plays
the same games (`seed_cycle`), so each comparison between rows is paired game by game.

    python scripts/data_analysis/exp2_diff_crn_h2h_configs.py <new root> <old root> 40000 ...
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp2_diff_h2h_configs import MULTI, REFERENCE  # noqa: E402

OUT = Path("configs/simulation/manager_testing/35_CRN_h2h")
HEADER = """# Head-to-head of common games against the old seeding (docs/plans/policy-finder-common-games.md),
# seed {seed}: each row against ah in both seats, 2000 games per pairing; every row plays the
# same games (seed_cycle). Written by scripts/data_analysis/exp2_diff_crn_h2h_configs.py.

"""


def bests(root: Path, k: int, label: str, key) -> dict:
    """{label_<lineage>_v<k>: (stem, params)} of the k lineages with the best final, by key."""
    best = {}
    for path in root.glob("pf-*.state.json"):
        state = json.loads(path.read_text())
        score = (state.get("data") or {}).get("score")
        if state.get("status") != "scored" or not score:
            continue
        stem = path.name.removesuffix(".state.json")
        lineage = stem.split("-v")[0]
        if lineage not in best or key(score) > best[lineage][0]:
            best[lineage] = (key(score), stem, score["params"])
    top = sorted(best.values(), key=lambda b: -b[0])[:k]
    return {f"{label}_" + stem.split("-", 2)[2].replace("-", "_"): (stem, params)
            for _, stem, params in top}


def config(seed: int, rows: dict, roots: dict):
    base = yaml.safe_load(REFERENCE.read_text())
    managers = {"ah": base["managers"]["ah"], "zero": base["managers"]["zero"],
                "multi_9_w0": yaml.safe_load(MULTI.read_text())["managers"]["pf_9_w0"]}
    for key, (stem, params) in rows.items():
        rule = Path(f"configs/managers/rule_based/{stem}.yml")
        if not rule.exists():  # the rule lives on its lineage's branch
            rule.write_text((roots[key] / stem.split("-v")[0] / rule).read_text())
        managers[key] = {"type": "rule_based", "rule": str(rule), "params": params,
                         "setting": "exp2_diff"}
    names = [*rows, "multi_9_w0", "zero"]
    pairings = [{"name": f"{a}_vs_{b}", "group_0": a, "group_1": b}
                for r in names for a, b in ((r, "ah"), ("ah", r))]
    out = dict(base)
    batch = out["episode_batch_size"]
    assert out["n_episodes"] % batch == 0
    out.update(seed=seed, managers=managers, pairings=pairings,
               seed_cycle=2 * out["n_episodes"] // batch,
               output_dir=f"plots/simulation/35_CRN_h2h/seed_{seed}",
               figure_name=f"CRN_h2h_{seed}")
    path = OUT / f"seed_{seed}.yml"
    path.write_text(HEADER.format(seed=seed) + yaml.safe_dump(out, sort_keys=False))
    print(path, len(pairings), "pairings, seed_cycle", out["seed_cycle"])


if __name__ == "__main__":
    new, old, seeds = Path(sys.argv[1]), Path(sys.argv[2]), [int(a) for a in sys.argv[3:]]
    rows = {**bests(new, 3, "new", lambda s: s.get("gain", s["pool"])),
            **bests(old, 3, "old", lambda s: s["pool"])}
    roots = {k: (new if k.startswith("new") else old) for k in rows}
    print(json.dumps(rows, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        config(s, rows, roots)
