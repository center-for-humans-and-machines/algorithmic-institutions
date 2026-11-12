"""Write the sweep sim config for a rule config (#227).

Validates the rule (`validate-rule`'s checks), draws its Sobol design and
writes a sim config that plays every design point against `ah`: one manager
`<rule>_s<index>` per point with its params inline, one pairing
`<rule>_s<index>_vs_ah` each, and `sweep: true`. The stack and the sim
parameters come from TEMPLATE (Levin's stack, the #226 run 1 reference sim);
the seed, the episodes per point and the output paths are set here.

`--n-parts N` splits the design into N contiguous parts, one sim config each
(`<rule>_sweep_p<k>of<N>`), to submit as N sims: a sim job has 16 GB of RAM
for its per-round frame. Point names stay global; part k's seed starts after
the batches of the parts before it, so no batch seed repeats across parts.
scripts/policy_finder/merge_sweeps.py then joins the parts' sweep.json.

`--final <sweep dir>/sweep.json` writes the sweep's final run instead: the
Gaussian process's peak (landscapes.json beside it,
scripts/policy_finder/fit_surrogate.py), named `<rule>_gp`, and the `--top`
(default 5) best measured points under their sweep names, on fresh seeds at
more episodes. It is a sweep sim too, so its own sweep.json (in `final/`
inside the sweep's dir) holds the rule's final result once
`--score-final <sweep dir>/final/sweep.json` has marked it: `best` is the GP
peak's result, the rule's score; `best_of_n` the best of all the final run's
points, the score of Cem's policy finder (the old validation), which reads
high because it picks the best of several noisy estimates.

Usage:
    python scripts/policy_finder/generate_sim_config.py \
        --config configs/managers/rule_based/<rule>.yml \
        [--sobol-points 256] [--n-episodes 500] [--seed 42] [--n-parts 1] \
        [--min-params 1] [--max-params 4|none] [--out <sim config>]
    python scripts/policy_finder/generate_sim_config.py \
        --final plots/simulation/policy_finder/<rule>_sweep/sweep.json \
        [--top 5] [--n-episodes 2000] [--seed 1000] [--out <sim config>]
    python scripts/policy_finder/generate_sim_config.py \
        --score-final plots/simulation/policy_finder/<rule>_sweep/final/sweep.json

Writes configs/simulation/policy_finder/<rule>_sweep[_p<k>of<N>].yml by
default; each sim writes to plots/simulation/policy_finder/ under the same
name. The final run's config is <rule>_sweep_final.yml, its output
plots/simulation/policy_finder/<rule>_sweep/final/. `--root` sets the
repo the paths are relative to (default: this script's), for a worktree
whose own copy of this script predates `--final`.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

from aimanager.manager.rule import SOBOL_POINTS, validate_rule

#: Paths in the sim config are relative to the repo root, where sims run.
ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(
    "configs/simulation/manager_testing/25_LEVIN_run1_ah_zero_pairings_batched.yml"
)
#: The manager every design point plays (definition 4 of #226).
ANCHOR = "ah"
#: The games the policy finder runs on (manager/rule.py SETTINGS): the sim a
#: sweep is built from, the manager its points play (None: each point plays
#: the clones alone), its score, and overrides of the template's sim keys.
#: `opponents` (instead of an anchor): a file of governors under opaque keys
#: (key -> name, manager); each point plays every one in both seats (group 0
#: and group 1), the score is the mean over the opponents of its pool there.
SETTINGS = {
    "exp2": {"template": TEMPLATE, "anchor": ANCHOR, "overrides": {}},
    "exp2_multi": {
        "template": TEMPLATE,
        "anchor": None,
        "opponents": Path("configs/simulation/opponents/exp2_multi.yml"),
        "overrides": {"sweep_metric": "pool_vs_opponents"},
    },
    # experiment 2 scored by the rule's group's lead in pool over `ah`'s, in
    # the same game, both seats (docs/plans/policy-finder-pool-difference.md)
    "exp2_diff": {
        "template": TEMPLATE,
        "anchor": None,
        "opponents": Path("configs/simulation/opponents/exp2_diff.yml"),
        "overrides": {"sweep_metric": "pool_diff_vs_opponents"},
    },
    "exp1": {
        "template": Path("configs/simulation/manager_testing/29_EXP1_reference.yml"),
        "anchor": None,
        # the template's human-like manager runs unbatched; rules batch
        "overrides": {"episode_batch_size": 1000, "sweep_metric": "group_payoff"},
    },
}
SETTING = "exp2"
N_EPISODES = 500
SEED = 42
#: The final run of the Gaussian process's peak: its episodes (se about half
#: a sweep point's) and its seed, past every seed a sweep's batches take
FINAL_EPISODES = 2000
FINAL_SEED = 1000
FINAL_DIR = "final"
#: The measured sweep points the final run plays beside the GP peak
FINAL_TOP = 5
CONFIG_DIR = Path("configs/simulation/policy_finder")
OUTPUT_DIR = Path("plots/simulation/policy_finder")


def opponents(setting):
    """The setting's governors under their opaque keys, {key: manager config}
    in key order; {} for a setting without them."""
    path = SETTINGS[setting or SETTING].get("opponents")
    if path is None:
        return {}
    with open(ROOT / path) as f:
        entries = yaml.safe_load(f)
    return {key: entries[key]["manager"] for key in sorted(entries)}


def pairings_per_point(setting):
    """Pairings each design point plays: one per opponent and seat, else one."""
    return 2 * len(opponents(setting)) or 1


class _Flow(dict):
    """A mapping written on one line: a point's params, a pairing."""


class _FlowList(list):
    """A list written on one line: agent_groups."""


yaml.SafeDumper.add_representer(
    _Flow,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map", data, flow_style=True
    ),
)
yaml.SafeDumper.add_representer(
    _FlowList,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    ),
)


def build_sim_config(
    rule_path,
    design,
    n_episodes=N_EPISODES,
    seed=SEED,
    start=0,
    suffix="",
    names=None,
    output=None,
    setting=None,
):
    """The sweep's sim config, as a dict: the setting's template with the
    design's managers (SETTINGS; default exp2, TEMPLATE against ANCHOR).

    `design` is the points of this sim, the first of them point `start` of
    the whole design; `suffix` tells a part's output and figure name apart.
    `names` replaces the points' names `<rule>_s<index>`, `output` the
    output dir `<rule>_sweep<suffix>` (under OUTPUT_DIR).
    """
    rule_path = Path(rule_path).resolve()
    name = rule_path.stem
    if "_vs_" in name:
        raise ValueError(
            f"rule name {name!r} contains `_vs_`, which pairing names split on"
        )
    try:
        rule_path = rule_path.relative_to(ROOT)
    except ValueError:
        raise ValueError(f"{rule_path} is outside the repo {ROOT}") from None
    spec = SETTINGS[setting or SETTING]
    anchor = spec["anchor"]
    with open(ROOT / spec["template"]) as f:
        template = yaml.safe_load(f)
    rivals = opponents(setting)

    managers = {anchor: template["managers"][anchor]} if anchor else {}
    managers.update(rivals)
    pairings = []
    if names is None:
        names = [f"{name}_s{i:03d}" for i in range(start, start + len(design))]
    for manager, point in zip(names, design):
        managers[manager] = {
            "type": "rule_based",
            "rule": str(rule_path),
            "params": _Flow(point),
        }
        if setting and setting != "exp2":
            managers[manager]["setting"] = setting
        if anchor:
            pairings.append(
                _Flow(name=f"{manager}_vs_{anchor}", group_0=manager, group_1=anchor)
            )
        for other in rivals:  # both seats against each opponent
            pairings.append(
                _Flow(name=f"{manager}_vs_{other}", group_0=manager, group_1=other)
            )
            pairings.append(
                _Flow(name=f"{other}_vs_{manager}", group_0=other, group_1=manager)
            )

    # TEMPLATE's keys in its order, the sweep's own values where it has them
    own = {
        "seed": seed,
        "managers": managers,
        "n_episodes": n_episodes,
        "output_dir": str(OUTPUT_DIR / (output or f"{name}_sweep{suffix}")),
        "figure_name": f"{name}_sweep{suffix}",
        **spec["overrides"],
    }
    if pairings:
        own["pairings"] = pairings
    if rivals:
        own["sweep_opponents"] = list(rivals)
    config = {k: own.get(k, v) for k, v in template.items()}
    if "agent_groups" in config:
        config["agent_groups"] = _FlowList(config["agent_groups"])
    config.update(own)  # in case the template lacks one
    config["sweep"] = True
    return config


def build_sim_configs(
    rule_path, design, n_episodes=N_EPISODES, seed=SEED, n_parts=1, setting=None
):
    """The sweep as `n_parts` sim configs: [(name suffix, config)].

    Contiguous parts of near-equal size. A sim seeds its batch i with
    `seed + i`, so part k's seed is `seed` plus the batches of the parts
    before it.
    """
    if not 1 <= n_parts <= len(design):
        raise ValueError(f"--n-parts must be in 1..{len(design)}, got {n_parts}")
    if n_parts == 1:
        return [
            ("", build_sim_config(rule_path, design, n_episodes, seed, setting=setting))
        ]
    edges = [round(k * len(design) / n_parts) for k in range(n_parts + 1)]
    configs, part_seed = [], seed
    for k in range(n_parts):
        part = design[edges[k] : edges[k + 1]]
        suffix = f"_p{k + 1}of{n_parts}"
        config = build_sim_config(
            rule_path, part, n_episodes, part_seed, edges[k], suffix, setting=setting
        )
        configs.append((suffix, config))
        batch = config.get("episode_batch_size", 1)
        part_seed += math.ceil(len(part) * pairings_per_point(setting) * n_episodes / batch)
    return configs


def write_final(args):
    """--final: the sweep's final run config, the Gaussian process's peak."""
    sweep_file = Path(args.final)
    sweep = json.loads(sweep_file.read_text())
    landscapes = sweep_file.with_name("landscapes.json")
    if not landscapes.exists():
        sys.exit(f"no {landscapes}: fit the sweep first (fit_surrogate.py)")
    peak = json.loads(landscapes.read_text())["peak"]["params"]
    n_episodes = args.n_episodes or FINAL_EPISODES
    stem = Path(sweep["rule"]).stem
    top = FINAL_TOP if args.top is None else args.top
    if top < 0:
        sys.exit("--top must be 0 or more")
    measured = sorted(sweep["points"], key=lambda p: p["pool"], reverse=True)[:top]
    try:
        config = build_sim_config(
            ROOT / sweep["rule"],
            [peak] + [p["params"] for p in measured],
            n_episodes,
            FINAL_SEED if args.seed is None else args.seed,
            names=[f"{stem}_gp"] + [p["name"] for p in measured],
            output=f"{stem}_sweep/{FINAL_DIR}",
            setting=args.setting,
        )
    except ValueError as e:
        sys.exit(f"Invalid: {e}")
    config["figure_name"] = f"{stem}_sweep_{FINAL_DIR}"
    out = Path(args.out or ROOT / CONFIG_DIR / f"{stem}_sweep_{FINAL_DIR}.yml")
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Final run of {sweep_file}, written by"
        f" scripts/policy_finder/generate_sim_config.py:\n"
        f"# the Gaussian process's peak{_against(args.setting)} on fresh seeds;"
        f" its sweep.json holds the\n# rule's final result. Stack and sim"
        f" parameters from {SETTINGS[args.setting]['template']}.\n\n"
    )
    with open(out, "w") as f:
        f.write(header)
        yaml.safe_dump(config, f, sort_keys=False, width=1000)
    extra = f" and the {len(measured)} best measured points" if measured else ""
    print(f"Wrote {out}: the peak{extra} x {n_episodes} episodes")


def _against(setting):
    anchor = SETTINGS[setting]["anchor"]
    if anchor:
        return f" against `{anchor}`"
    rivals = opponents(setting)
    if len(rivals) == 1:
        return f" against `{next(iter(rivals))}` in both seats ({SETTINGS[setting]['opponents']})"
    if rivals:
        return (
            f" against each of {len(rivals)} opponents in both seats"
            f" ({SETTINGS[setting]['opponents']})"
        )
    return " alone with the clones"


def score_final(path):
    """Mark a final run's sweep.json: `best` the GP peak, `best_of_n` the best.

    The sim writes `best` as its best point; for a final run that is the best
    of the GP peak and the top measured points, kept here as `best_of_n`
    (`best_of_n_name`, `n_points`). `best` and `best_name` become the GP
    peak's (`<rule>_gp`), the rule's final result. Idempotent.
    """
    path = Path(path)
    final = json.loads(path.read_text())
    stem = Path(final["rule"]).stem
    by_name = {p["name"]: p for p in final["points"]}
    gp = by_name.get(f"{stem}_gp")
    if gp is None:
        raise ValueError(f"{path}: no point {stem}_gp")
    top = max(final["points"], key=lambda p: p["pool"])
    final.update(
        {
            "best": gp["params"],
            "best_name": gp["name"],
            "best_of_n": top["params"],
            "best_of_n_name": top["name"],
            "n_points": len(final["points"]),
        }
    )
    path.write_text(json.dumps(final, indent=2) + "\n")
    return final


def main():
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--config", help="the rule config (YAML)")
    what.add_argument("--final", help="a sweep's sweep.json, to run its GP peak")
    what.add_argument(
        "--score-final", help="a final run's sweep.json, to mark its GP peak as best"
    )
    ap.add_argument(
        "--top", type=int, help=f"--final: measured points beside the peak ({FINAL_TOP})"
    )
    ap.add_argument("--sobol-points", type=int, default=SOBOL_POINTS)
    ap.add_argument(
        "--n-episodes", type=int, help="per point (default 500, --final 2000)"
    )
    ap.add_argument("--seed", type=int, help="default 42, --final 1000")
    ap.add_argument("--n-parts", type=int, default=1, help="sims to split into")
    ap.add_argument("--min-params", type=int, default=1)
    ap.add_argument(
        "--max-params",
        type=lambda v: None if v == "none" else int(v),
        default=4,
        help="a number, or none for no upper bound",
    )
    ap.add_argument("--root", help="the repo the paths are relative to")
    ap.add_argument(
        "--setting", default=SETTING, choices=sorted(SETTINGS),
        help="the game (manager/rule.py SETTINGS), default exp2",
    )
    ap.add_argument("--out", help="the sim config to write (one part only)")
    args = ap.parse_args()
    if args.root:
        ROOT = Path(args.root).resolve()
    if args.final:
        write_final(args)
        return
    if args.score_final:
        try:
            final = score_final(args.score_final)
        except (ValueError, KeyError) as e:
            sys.exit(f"Invalid: {e}")
        gp = next(p for p in final["points"] if p["name"] == final["best_name"])
        top = next(p for p in final["points"] if p["name"] == final["best_of_n_name"])
        print(
            f"final {final['best_name']}: {gp['pool']:.1f} +- {gp['pool_se'] or 0:.1f};"
            f" best of {final['n_points']} {top['name']}: {top['pool']:.1f}"
        )
        return
    args.n_episodes = args.n_episodes or N_EPISODES
    args.seed = SEED if args.seed is None else args.seed
    if args.out and args.n_parts != 1:
        sys.exit("--out writes one config; leave it out with --n-parts")

    try:
        design = validate_rule(
            args.config, args.min_params, args.max_params, args.sobol_points,
            setting=args.setting,
        )
        configs = build_sim_configs(
            args.config, design, args.n_episodes, args.seed, args.n_parts,
            setting=args.setting,
        )
    except ValueError as e:
        sys.exit(f"Invalid: {e}")

    stem = Path(args.config).stem
    for suffix, config in configs:
        out = Path(args.out or ROOT / CONFIG_DIR / f"{stem}_sweep{suffix}.yml")
        out.parent.mkdir(parents=True, exist_ok=True)
        rivals = config.get("sweep_opponents", ())
        n_points = sum(
            1 for k, m in config["managers"].items()
            if m["type"] == "rule_based" and k not in rivals
        )
        each = "each" + _against(args.setting)
        header = (
            f"# Sweep of {args.config} (#227), written by"
            f" scripts/policy_finder/generate_sim_config.py:\n"
            f"# {n_points} of {len(design)} distinct design points"
            f" ({args.sobol_points} Sobol points), {each};"
            f" stack and sim parameters from\n# {SETTINGS[args.setting]['template']}.\n\n"
        )
        with open(out, "w") as f:
            f.write(header)
            yaml.safe_dump(config, f, sort_keys=False, width=1000)
        pairs = pairings_per_point(args.setting)
        each = f" x {pairs // 2} opponent(s) x 2 seats" if pairs > 1 else ""
        print(
            f"Wrote {out}: {n_points} design points{each} x {args.n_episodes} episodes"
            f" = {n_points * pairs * args.n_episodes} episodes"
        )


if __name__ == "__main__":
    main()
