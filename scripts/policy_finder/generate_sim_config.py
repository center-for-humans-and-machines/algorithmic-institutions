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
scripts/policy_finder/fit_surrogate.py), named `<rule>_gp`, on fresh seeds
at more episodes. It is a sweep sim too, so its own sweep.json (in `final/`
inside the sweep's dir) holds the rule's final result.

Usage:
    python scripts/policy_finder/generate_sim_config.py \
        --config configs/managers/rule_based/<rule>.yml \
        [--sobol-points 256] [--n-episodes 500] [--seed 42] [--n-parts 1] \
        [--min-params 1] [--max-params 4|none] [--out <sim config>]
    python scripts/policy_finder/generate_sim_config.py \
        --final plots/simulation/policy_finder/<rule>_sweep/sweep.json \
        [--n-episodes 2000] [--seed 1000] [--out <sim config>]

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
N_EPISODES = 500
SEED = 42
#: The final run of the Gaussian process's peak: its episodes (se about half
#: a sweep point's) and its seed, past every seed a sweep's batches take
FINAL_EPISODES = 2000
FINAL_SEED = 1000
FINAL_DIR = "final"
CONFIG_DIR = Path("configs/simulation/policy_finder")
OUTPUT_DIR = Path("plots/simulation/policy_finder")


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
):
    """The sweep's sim config, as a dict: TEMPLATE with the design's managers.

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
    with open(ROOT / TEMPLATE) as f:
        template = yaml.safe_load(f)

    managers = {ANCHOR: template["managers"][ANCHOR]}
    pairings = []
    if names is None:
        names = [f"{name}_s{i:03d}" for i in range(start, start + len(design))]
    for manager, point in zip(names, design):
        managers[manager] = {
            "type": "rule_based",
            "rule": str(rule_path),
            "params": _Flow(point),
        }
        pairings.append(
            _Flow(name=f"{manager}_vs_{ANCHOR}", group_0=manager, group_1=ANCHOR)
        )

    # TEMPLATE's keys in its order, the sweep's own values where it has them
    own = {
        "seed": seed,
        "managers": managers,
        "pairings": pairings,
        "n_episodes": n_episodes,
        "output_dir": str(OUTPUT_DIR / (output or f"{name}_sweep{suffix}")),
        "figure_name": f"{name}_sweep{suffix}",
    }
    config = {k: own.get(k, v) for k, v in template.items()}
    config["agent_groups"] = _FlowList(config["agent_groups"])
    config.update(own)  # in case TEMPLATE lacks one
    config["sweep"] = True
    return config


def build_sim_configs(rule_path, design, n_episodes=N_EPISODES, seed=SEED, n_parts=1):
    """The sweep as `n_parts` sim configs: [(name suffix, config)].

    Contiguous parts of near-equal size. A sim seeds its batch i with
    `seed + i`, so part k's seed is `seed` plus the batches of the parts
    before it.
    """
    if not 1 <= n_parts <= len(design):
        raise ValueError(f"--n-parts must be in 1..{len(design)}, got {n_parts}")
    if n_parts == 1:
        return [("", build_sim_config(rule_path, design, n_episodes, seed))]
    edges = [round(k * len(design) / n_parts) for k in range(n_parts + 1)]
    configs, part_seed = [], seed
    for k in range(n_parts):
        part = design[edges[k] : edges[k + 1]]
        suffix = f"_p{k + 1}of{n_parts}"
        config = build_sim_config(
            rule_path, part, n_episodes, part_seed, edges[k], suffix
        )
        configs.append((suffix, config))
        batch = config.get("episode_batch_size", 1)
        part_seed += math.ceil(len(part) * n_episodes / batch)
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
    try:
        config = build_sim_config(
            ROOT / sweep["rule"],
            [peak],
            n_episodes,
            FINAL_SEED if args.seed is None else args.seed,
            names=[f"{stem}_gp"],
            output=f"{stem}_sweep/{FINAL_DIR}",
        )
    except ValueError as e:
        sys.exit(f"Invalid: {e}")
    config["figure_name"] = f"{stem}_sweep_{FINAL_DIR}"
    out = Path(args.out or ROOT / CONFIG_DIR / f"{stem}_sweep_{FINAL_DIR}.yml")
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Final run of {sweep_file}, written by"
        f" scripts/policy_finder/generate_sim_config.py:\n"
        f"# the Gaussian process's peak against `{ANCHOR}` on fresh seeds;"
        f" its sweep.json holds the\n# rule's final result. Stack and sim"
        f" parameters from {TEMPLATE}.\n\n"
    )
    with open(out, "w") as f:
        f.write(header)
        yaml.safe_dump(config, f, sort_keys=False, width=1000)
    print(f"Wrote {out}: the peak x {n_episodes} episodes")


def main():
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--config", help="the rule config (YAML)")
    what.add_argument("--final", help="a sweep's sweep.json, to run its GP peak")
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
    ap.add_argument("--out", help="the sim config to write (one part only)")
    args = ap.parse_args()
    if args.root:
        ROOT = Path(args.root).resolve()
    if args.final:
        write_final(args)
        return
    args.n_episodes = args.n_episodes or N_EPISODES
    args.seed = SEED if args.seed is None else args.seed
    if args.out and args.n_parts != 1:
        sys.exit("--out writes one config; leave it out with --n-parts")

    try:
        design = validate_rule(
            args.config, args.min_params, args.max_params, args.sobol_points
        )
        configs = build_sim_configs(
            args.config, design, args.n_episodes, args.seed, args.n_parts
        )
    except ValueError as e:
        sys.exit(f"Invalid: {e}")

    stem = Path(args.config).stem
    for suffix, config in configs:
        out = Path(args.out or ROOT / CONFIG_DIR / f"{stem}_sweep{suffix}.yml")
        out.parent.mkdir(parents=True, exist_ok=True)
        n_points = len(config["pairings"])
        header = (
            f"# Sweep of {args.config} (#227), written by"
            f" scripts/policy_finder/generate_sim_config.py:\n"
            f"# {n_points} of {len(design)} distinct design points"
            f" ({args.sobol_points} Sobol points), each against `{ANCHOR}`;"
            f" stack and sim parameters from\n# {TEMPLATE}.\n\n"
        )
        with open(out, "w") as f:
            f.write(header)
            yaml.safe_dump(config, f, sort_keys=False, width=1000)
        print(
            f"Wrote {out}: {n_points} design points x {args.n_episodes} episodes"
            f" = {n_points * args.n_episodes} episodes"
        )


if __name__ == "__main__":
    main()
