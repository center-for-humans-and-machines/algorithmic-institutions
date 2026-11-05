"""Write a policy-finder instance's settings for one version (#236, #244).

new_instance.sh runs it once per version of an instance: version 1 when the
instance is made, version k when it iterates on its own branch. It writes
the instance's gitignored .claude/policy_finder.json: the instance `name`,
its `version`, the version's `rule_name` (the stem of its rule, notes and
sweep: `<name>` for version 1, `<name>-v<k>` after), `start_commit` (the
commit the version starts from, which check_instance.sh checks the changes
since), `min_params`, `max_params` (null: no upper bound), `sobol_points` and
`python` (the interpreter the agent and the check run).

The write paths are the version's rule and notes and the instance's scripts
folder, shared by every version. The agent runs in its own container
(agent_runner.py): its copy of the instance (export_instance.py) is mounted
read-only except these paths, and the container is the only sandbox. No
Claude Code permissions, sandbox settings or hooks are written.

Usage:
    python scripts/policy_finder/instance_settings.py <worktree> <name>
        --version K --start-commit SHA --min-params N --max-params N|none
        --sobol-points N --python PATH
"""

import argparse
import json
import os


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


def write(args):
    wt = os.path.realpath(args.worktree)
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
    os.makedirs(os.path.join(wt, ".claude"), exist_ok=True)
    with open(os.path.join(wt, ".claude", "policy_finder.json"), "w") as f:
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
    ap.add_argument("--setting", default="exp2", choices=("exp2", "exp1"))
    write(ap.parse_args())


if __name__ == "__main__":
    main()
