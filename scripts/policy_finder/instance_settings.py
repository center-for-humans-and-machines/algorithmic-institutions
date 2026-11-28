"""Write a policy-finder instance's settings for one version (#236, #244).

new_instance.sh runs it once per version of an instance: version 1 when the
instance is made, version k when it iterates on its own branch. It writes
the instance's gitignored .claude/policy_finder.json: the instance `name`,
its `version`, the version's `rule_name` (the stem of its rule, notes and
sweep: `<name>` for version 1, `<name>-v<k>` after), `start_commit` (the
commit the version starts from, which check_instance.sh checks the changes
since), `min_params`, `max_params` (null: no upper bound), `sobol_points` and
`python` (the interpreter the agent and the check run).

A single-idea instance (`--mode single_idea`) also gets `mode`
and, from version 2 on, `base_version`: the stem of the earlier version with
the highest final result (the GP peak's `pool` in its final `sweep.json`; a
tie goes to the earlier version; version 1 if none has a final), which the
version changes one idea of. Its version 1 is the seed rule of
scripts/policy_finder/seeds/, written by the pipeline, not by an agent.

The write paths are the version's rule and notes and the instance's scripts
folder, shared by every version. The agent runs in its own container
(agent_runner.py): its copy of the instance (export_instance.py) is mounted
read-only except these paths, and the container is the only sandbox. No
Claude Code permissions, sandbox settings or hooks are written.

An instance whose agent runs on OpenCode (`--harness opencode`, see
agent_runner.py) also gets `harness`, `model` (an OpenRouter model id) and,
if given, `providers` (OpenRouter's upstream providers to route to, in
order, with no fallback), which export_instance.py turns into the agent's
OpenCode config. A later version keeps them unless they are given again.
With `--base-url` (an OpenAI-compatible server of our own, such as the DAIS
job's vLLM; `{env:PF_LLM_BASE_URL}` lets OpenCode read it from the agent's
environment, so a resumed job's new port needs no new export), `model` is
the served model's name and `base_url` is kept as well.
The default harness, claude, writes none of these keys.

Usage:
    python scripts/policy_finder/instance_settings.py <worktree> <name>
        --version K --start-commit SHA --min-params N --max-params N|none
        --sobol-points N --python PATH [--mode free|single_idea]
        [--harness claude|opencode --model ID [--providers A,B] [--base-url URL]]
"""

import argparse
import json
import os


#: The settings with a single-idea agent (export_instance.MODE_AGENTS)
SINGLE_IDEA_SETTINGS = ("exp2", "exp1")


def rule_name(name: str, version: int) -> str:
    """The stem of a version's rule, notes and sweep."""
    return name if version == 1 else f"{name}-v{version}"


def write_paths(name: str, version: int) -> tuple:
    """The version's rule and notes, and the instance's scripts folder."""
    stem = rule_name(name, version)
    return (
        f"configs/managers/rule_based/{stem}.yml",
        f"notes/policy_finder/{stem}.md",
        f"scripts/policy_finder/{name}",
    )


def final_pool(wt: str, stem: str):
    """The GP peak's pool in a version's final run, or None without one."""
    path = os.path.join(
        wt, "plots/simulation/policy_finder", f"{stem}_sweep", "final", "sweep.json"
    )
    try:
        with open(path) as f:
            final = json.load(f)
        points = {p["name"]: p for p in final["points"]}
        return float(points[final["best_name"]]["pool"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def base_version(wt: str, name: str, version: int) -> str:
    """The earlier version with the highest final result (ties: the earlier)."""
    best, best_pool = rule_name(name, 1), None
    for v in range(1, version):
        pool = final_pool(wt, rule_name(name, v))
        if pool is not None and (best_pool is None or pool > best_pool):
            best, best_pool = rule_name(name, v), pool
    return best


def harness(args, previous):
    """The harness keys of the instance file: given, else the previous version's."""
    if args.harness is None:
        return {k: previous[k] for k in ("harness", "model", "providers", "base_url")
                if k in previous}
    if args.harness == "claude":
        if args.model or args.providers or args.base_url:
            raise SystemExit("--model, --providers and --base-url are for --harness opencode")
        return {}
    if not args.model:
        raise SystemExit("--harness opencode needs --model (an OpenRouter model id)")
    out = {"harness": args.harness, "model": args.model}
    if args.providers:
        out["providers"] = [p for p in args.providers.split(",") if p]
    if args.base_url:
        out["base_url"] = args.base_url
    return out


def write(args):
    wt = os.path.realpath(args.worktree)
    path = os.path.join(wt, ".claude", "policy_finder.json")
    previous = {}
    if os.path.isfile(path):
        with open(path) as f:
            previous = json.load(f)
    rule, notes, scripts = write_paths(args.name, args.version)
    for directory in (os.path.dirname(rule), os.path.dirname(notes), scripts):
        os.makedirs(os.path.join(wt, directory), exist_ok=True)
    instance = {
        "name": args.name,
        "version": args.version,
        "rule_name": rule_name(args.name, args.version),
        "start_commit": args.start_commit,
        "min_params": args.min_params,
        "max_params": args.max_params,
        "sobol_points": args.sobol_points,
        "python": args.python,
    }
    # the game (manager/rule.py SETTINGS); absent means exp2
    if args.setting != "exp2":
        instance["setting"] = args.setting
    if args.mode == "single_idea":
        if args.setting not in SINGLE_IDEA_SETTINGS:
            raise SystemExit("--mode single_idea has an agent for --setting "
                             + " and ".join(SINGLE_IDEA_SETTINGS) + " only")
        instance["mode"] = args.mode
        if args.version > 1:
            instance["base_version"] = base_version(wt, args.name, args.version)
    instance.update(harness(args, previous))
    os.makedirs(os.path.join(wt, ".claude"), exist_ok=True)
    with open(path, "w") as f:
        json.dump(instance, f, indent=2)
        f.write("\n")


def bound(value: str):
    return None if value == "none" else int(value)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("worktree")
    ap.add_argument("name")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--start-commit", required=True)
    ap.add_argument("--min-params", type=int, default=1)
    ap.add_argument("--max-params", type=bound, default=4, help="a number or none")
    ap.add_argument("--sobol-points", type=int, default=256)
    ap.add_argument("--python", default="python3")
    ap.add_argument("--setting", default="exp2", choices=("exp2", "exp1", "exp2_multi", "exp2_diff"))
    ap.add_argument("--mode", default="free", choices=("free", "single_idea"))
    ap.add_argument("--harness", choices=("claude", "opencode"),
                    help="default: the previous version's, else claude")
    ap.add_argument("--model", help="harness opencode: an OpenRouter model id")
    ap.add_argument("--providers", help="harness opencode: upstream providers, a,b")
    ap.add_argument("--base-url", help="harness opencode: an OpenAI-compatible server "
                    "of our own instead of OpenRouter (model: its served name)")
    write(ap.parse_args())


if __name__ == "__main__":
    main()
