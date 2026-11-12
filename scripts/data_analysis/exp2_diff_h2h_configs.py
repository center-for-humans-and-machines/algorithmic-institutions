"""Write the exp2_diff head-to-head configs (docs/plans/policy-finder-pool-difference.md,
step 7): the campaign's top rules at their GP peaks, `zero` and the earlier campaigns' best
rules, each against `ah` in both seats, 2000 games per pairing, on the 33_MULTI_h2h seeds
(20000-25000), which no sweep or final of this campaign used.

The rows are read from the campaign's state files (the lineage bests by final GP peak);
`zero`, `prev_best`, `cem_best` and the multi campaign's `pf_9_w0` (the largest lead over
ah in 33_MULTI_h2h) play again here so every row is measured on the same seeds and code.

    PF_WORKTREE_ROOT=<root> python scripts/data_analysis/exp2_diff_h2h_configs.py 3 20000 21000 ...
"""
import json
import os
import sys
from pathlib import Path

import yaml

REFERENCE = Path("configs/simulation/manager_testing/32_LEVIN_multi_reference.yml")
MULTI = Path("configs/simulation/manager_testing/33_MULTI_h2h/seed_20000_ablation.yml")
OUT = Path("configs/simulation/manager_testing/34_DIFF_h2h")
REFS = ["zero", "prev_best", "cem_best"]
HEADER = """# Head-to-head of the exp2_diff campaign (docs/plans/policy-finder-pool-difference.md,
# step 7), seed {seed}: each row against ah in both seats, 2000 games per pairing.
# Written by scripts/data_analysis/exp2_diff_h2h_configs.py.

"""


def lineage_bests(root: Path, k: int) -> dict:
    """{short name: (rule stem, params)} of the k lineages with the best final GP peak."""
    best = {}
    for path in root.glob("pf-*.state.json"):
        state = json.loads(path.read_text())
        score = (state.get("data") or {}).get("score")
        if state.get("status") != "scored" or not score:
            continue
        stem = path.name.removesuffix(".state.json")
        lineage = stem.split("-v")[0]
        if lineage not in best or score["pool"] > best[lineage][0]:
            best[lineage] = (score["pool"], stem, score["params"])
    top = sorted(best.values(), key=lambda b: -b[0])[:k]
    return {"pf_" + stem.split('-', 2)[2].replace('-', '_'): (stem, params) for _, stem, params in top}


def config(seed: int, rows: dict, root: Path):
    base = yaml.safe_load(REFERENCE.read_text())
    multi = yaml.safe_load(MULTI.read_text())["managers"]
    managers = {k: v for k, v in base["managers"].items() if k in {"ah", *REFS}}
    managers["pf_9_w0"] = multi["pf_9_w0"]
    for key, (stem, params) in rows.items():
        lineage = stem.split("-v")[0]
        managers[key] = {"type": "rule_based",
                         "rule": f"configs/managers/rule_based/{stem}.yml",
                         "params": params, "setting": "exp2_diff"}
        # the rule files live on the lineage's branch; copy them into this checkout
        src = root / lineage / "configs/managers/rule_based" / f"{stem}.yml"
        dst = Path(managers[key]["rule"])
        if not dst.exists():
            dst.write_text(src.read_text())
    names = [*rows, *REFS, "pf_9_w0"]
    pairings = [{"name": f"{a}_vs_{b}", "group_0": a, "group_1": b}
                for r in names for a, b in ((r, "ah"), ("ah", r))]
    out = dict(base)
    out.update(seed=seed, managers=managers, pairings=pairings,
               output_dir=f"plots/simulation/34_DIFF_h2h/seed_{seed}",
               figure_name=f"DIFF_h2h_{seed}")
    batches = len(pairings) * out["n_episodes"] // out["episode_batch_size"]
    assert batches < 1000, batches
    path = OUT / f"seed_{seed}.yml"
    path.write_text(HEADER.format(seed=seed) + yaml.safe_dump(out, sort_keys=False))
    print(path, len(pairings), "pairings", out["n_episodes"], "games each")


if __name__ == "__main__":
    k, seeds = int(sys.argv[1]), [int(a) for a in sys.argv[2:]]
    root = Path(os.environ["PF_WORKTREE_ROOT"])
    rows = lineage_bests(root, k)
    print(json.dumps(rows, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        config(s, rows, root)
