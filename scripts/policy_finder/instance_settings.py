"""Write a policy-finder instance's settings for one version (#236, #244).

new_instance.sh runs it once per version of an instance: version 1 when the
instance is made, version k when it iterates on its own branch. It writes
the instance's gitignored files:

  - .claude/policy_finder.json: the instance `name`, its `version`, the
    version's `rule_name` (the stem of its rule, notes and sweep: `<name>`
    for version 1, `<name>-v<k>` after), `start_commit` (the commit the
    version starts from, which check_instance.sh checks the changes since),
    `min_params`, `max_params` (null: no upper bound), `sobol_points` and
    `python`
  - .claude/settings.local.json: the permissions, the Bash sandbox and the
    guard hook, all bound to the version's write paths

The write paths are the version's rule and notes and the instance's scripts
folder, shared by every version. Everything else, earlier versions' rules,
notes and results included, is read-only:

  - permissions: Edit only on the write paths; Read denied on .git, the main
    checkout and ~/.claude; WebFetch and WebSearch denied
  - sandbox: Bash, and so any Python it runs, reads the worktree minus .git
    (plus the Python install, the main checkout's .venv and the session's
    own temp dir, where the Bash tool collects output) and modifies nothing
    outside the write paths; no network; no unsandboxed escape.
    A sandbox write deny beats any narrower allow, so writes are denied by
    complement: every existing entry beside the write paths, level by level
    (an earlier version's rule and notes among them). New files can still be
    created at those levels; check_instance.sh flags them. Claude Code itself
    makes the main checkout's .git writable for a worktree (it guards only
    .git/hooks), and a write to .git/config could run commands outside the
    sandbox: it is denied too.
  - auto memory off (a worktree shares the main checkout's memory)
  - the guard hook (.claude/hooks/policy_finder_guard.py) on every tool call

Usage:
    python scripts/policy_finder/instance_settings.py <worktree> <main checkout>
        <name> --version K --start-commit SHA --min-params N --max-params N|none
        --sobol-points N --python PATH --uv-python DIR
"""

import argparse
import json
import os
import re


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


def complement(wt: str, path: str) -> list:
    """Every existing entry beside `path`, at each level from the worktree down."""
    denied, parent = [], wt
    for part in path.split("/"):
        if not os.path.isdir(parent):
            break
        denied += [
            os.path.join(parent, e) for e in sorted(os.listdir(parent)) if e != part
        ]
        parent = os.path.join(parent, part)
    return denied


def settings(wt, main, rule, notes, scripts, uv_python):
    """settings.local.json for the write paths `rule`, `notes`, `scripts`."""
    home = os.path.expanduser("~")
    # the Bash tool writes command output here; Python drops stdout it cannot stat
    session_tmp = f"/private/tmp/claude-{os.getuid()}/" + re.sub(
        r"[^A-Za-z0-9]", "-", wt
    )
    # at the top level each chain lists the others' first entries; keep them
    keep = {os.path.join(wt, p.split("/")[0]) for p in (rule, notes, scripts)}
    deny_write = sorted(
        set(complement(wt, rule) + complement(wt, notes) + complement(wt, scripts))
        - keep
    )
    guard = f"{wt}/.claude/hooks/policy_finder_guard.py"
    return {
        "autoMemoryEnabled": False,
        "permissions": {
            # `/x` is relative to the worktree, `//x` is an absolute path
            "allow": [f"Edit(/{rule})", f"Edit(/{notes})", f"Edit(/{scripts}/**)"],
            "deny": [
                "Read(/.git)",
                "Read(/.git/**)",
                f"Read(/{main}/**)",
                f"Read(/{home}/.claude/**)",
                "WebFetch",
                "WebSearch",
            ],
        },
        "sandbox": {
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
            "filesystem": {
                # a narrower path wins: the worktree is readable inside the
                # denied home, its .git is not
                "denyRead": [home, "/private/tmp", "/Volumes", f"{wt}/.git"],
                "allowRead": [wt, uv_python, f"{main}/.venv", session_tmp],
                "denyWrite": deny_write + [f"{main}/.git"],
            },
        },
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python3 {guard}",
                        }
                    ],
                }
            ],
        },
    }


def write(args):
    wt = os.path.realpath(args.worktree)
    rule, notes, scripts = write_paths(args.name, args.version)
    os.makedirs(os.path.join(wt, scripts), exist_ok=True)
    os.makedirs(os.path.join(wt, os.path.dirname(notes)), exist_ok=True)
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
    files = {
        "policy_finder.json": instance,
        "settings.local.json": settings(
            wt, args.main, rule, notes, scripts, args.uv_python
        ),
    }
    for file, content in files.items():
        with open(os.path.join(wt, ".claude", file), "w") as f:
            json.dump(content, f, indent=2)
            f.write("\n")


def bound(value: str):
    return None if value == "none" else int(value)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("worktree")
    ap.add_argument("main", help="the main checkout")
    ap.add_argument("name")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--start-commit", required=True)
    ap.add_argument("--min-params", type=int, default=1)
    ap.add_argument("--max-params", type=bound, default=4, help="a number or none")
    ap.add_argument("--sobol-points", type=int, default=256)
    ap.add_argument("--python", required=True)
    ap.add_argument("--uv-python", required=True)
    write(ap.parse_args())


if __name__ == "__main__":
    main()
