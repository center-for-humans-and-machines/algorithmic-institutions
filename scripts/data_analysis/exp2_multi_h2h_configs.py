"""Write the exp2_multi head-to-head configs (docs/plans/policy-finder-multi-opponent.md,
step 8): the campaign's top 3 rules at their GP peaks and five reference governors, each
against the four training opponents, two held-out ones (Cem's best, thr9_p10) and itself,
in both seats, on seeds no sweep, final or reference run used.

Per seed two configs: `rules` (the three new rules) and `refs` (the references). Batch i
of a config is seeded seed + i, and a config has under 200 batches, so seeds 1000 apart
never share a batch seed.

    python scripts/data_analysis/exp2_multi_h2h_configs.py 20000 21000 22000
"""
import sys
from pathlib import Path

import yaml

REFERENCE = Path("configs/simulation/manager_testing/32_LEVIN_multi_reference.yml")
OUT = Path("configs/simulation/manager_testing/33_MULTI_h2h")
OPPONENTS = ["levin", "ah", "prev_best", "zero", "cem_best", "thr9_p10"]
REFS = ["prev_best", "cem_best", "levin", "ah", "zero"]
RULES = {
    "pf_9": ("pf-20261106-9-v5", {"k0": 12.15253432568619, "k1": 3.941831314044691,
              "w": 0.3, "s0": 1.0809098622263695, "quiet": 2, "t_end": 16}),
    "pf_11": ("pf-20261106-11-v7", {"rate": 8.554446607770156, "delta": -1.3978606663724003,
               "decay": 0.4414150385064816, "mid": 1.3231990751160767,
               "mu": 0.07230281332795951, "theta": 6.0}),
    "pf_7": ("pf-20261106-7-v5", {"c0": 10, "w": 0.5386950376481652, "k": 5.018051125947482,
              "decay": 0.5623720337021627, "follow": 0.6136408739081771}),
}
HEADER = """# Head-to-head of the exp2_multi campaign (docs/plans/policy-finder-multi-opponent.md,
# step 8), {part}, seed {seed}: each row against levin, ah, prev_best, zero (the training
# opponents), cem_best and thr9_p10 (held out) and itself, in both seats, 2000 games per
# pairing. Written by scripts/data_analysis/exp2_multi_h2h_configs.py.
# Report: docs/reports/policy_finder_multi_opponent.md

"""


def pairings(rows):
    names = []
    for r in rows:
        for o in OPPONENTS + [r]:
            for g0, g1 in ((r, o), (o, r)):
                name = f"{g0}_vs_{g1}"
                if name not in names:
                    names.append(name)
    return [{"name": n, "group_0": n.split("_vs_")[0], "group_1": n.split("_vs_")[1]}
            for n in names]


def config(seed, part, rows):
    base = yaml.safe_load(REFERENCE.read_text())
    managers = {k: v for k, v in base["managers"].items()
                if k in set(OPPONENTS) | set(rows)}
    for key, (stem, params) in RULES.items():
        if key in rows:
            managers[key] = {"type": "rule_based",
                             "rule": f"configs/managers/rule_based/{stem}.yml",
                             "params": params, "setting": "exp2_multi"}
    out = dict(base)
    out.update(seed=seed, managers=managers, pairings=pairings(rows),
               output_dir=f"plots/simulation/33_MULTI_h2h/seed_{seed}_{part}",
               figure_name=f"MULTI_h2h_{seed}_{part}")
    batches = len(out["pairings"]) * out["n_episodes"] // out["episode_batch_size"]
    assert batches < 1000, batches
    path = OUT / f"seed_{seed}_{part}.yml"
    path.write_text(HEADER.format(part=part, seed=seed) + yaml.safe_dump(out, sort_keys=False))
    print(path, len(out["pairings"]), "pairings")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for s in map(int, sys.argv[1:]):
        config(s, "rules", list(RULES))
        config(s, "refs", REFS)
