#!/usr/bin/env bash
# Create and start a policy-finder instance (#236).
#
# Usage:
#   scripts/policy_finder/new_instance.sh <name> [--max-params N] [--min-params N]
#       [--sobol-points N] [--no-start | --headless [--prompt TEXT]]
#
# Makes branch policy-finder/<name> off policy-finder-base, checked out as its
# own worktree in ../policy-finder-worktrees/<name> (outside this checkout;
# override with PF_WORKTREE_ROOT), pulls the LFS data the agent reads, writes
# the instance's gitignored .claude/policy_finder.json and
# .claude/settings.local.json, and starts `claude --agent policy-finder-host`
# there with no MCP servers. --no-start stops before starting the session.
# --headless runs the session non-interactively instead (`claude -p`, Bash
# pre-allowed: the sandbox and the guard are the boundary) on --prompt
# (default: design your rule) and saves the agent's final report to
# ../policy-finder-worktrees/<name>.report.md, outside the worktree; then
# checks and commits the instance (check_instance.sh --commit) and runs its
# sweep on Raven (run_sweep.sh, #241).
# The rule declares between --min-params (default 1) and --max-params
# (default 4) params; equal values ask for exactly that many. --sobol-points
# (a power of two, default 256) sets the size of the sweep's design, which
# validate-rule, check_instance.sh and run_sweep.sh read from the instance.
# PF_BASE overrides the base branch, to try a branch before it lands on
# policy-finder-base.
#
# The instance settings bind only that worktree:
#   - permissions: Edit only on the write paths; Read denied on .git, this
#     checkout and ~/.claude; WebFetch and WebSearch denied
#   - sandbox: Bash, and so any Python it runs, reads the worktree minus .git
#     (plus the Python install, this checkout's .venv and the session's own
#     temp dir, where the Bash tool collects output) and modifies nothing
#     outside the write paths; no network; no unsandboxed escape.
#     A sandbox write deny beats any narrower allow, so writes are denied by
#     complement: every existing entry beside the write paths, level by
#     level. New files can still be created at those levels;
#     check_instance.sh flags them. Claude Code itself makes this checkout's
#     .git writable for a worktree (it guards only .git/hooks), and a write to
#     .git/config could run commands outside the sandbox: it is denied too.
#   - auto memory off (a worktree shares this checkout's memory)
#   - the guard hook (.claude/hooks/policy_finder_guard.py) on every tool call
#
# Check and commit an instance's work with scripts/policy_finder/check_instance.sh.

set -euo pipefail

usage() {
    sed -n '4,6p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

NAME=""
MAX_PARAMS=4
MIN_PARAMS=1
SOBOL_POINTS=256
START=1
HEADLESS=0
PROMPT="Design your rule."
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-params) MAX_PARAMS="${2:-}"; shift 2 ;;
        --min-params) MIN_PARAMS="${2:-}"; shift 2 ;;
        --sobol-points) SOBOL_POINTS="${2:-}"; shift 2 ;;
        --no-start) START=0; shift ;;
        --headless) HEADLESS=1; shift ;;
        --prompt) PROMPT="${2:-}"; shift 2 ;;
        -h|--help) usage ;;
        -*) echo "unknown option: $1" >&2; usage ;;
        *) [[ -z "$NAME" ]] || usage; NAME="$1"; shift ;;
    esac
done

[[ -n "$NAME" ]] || usage
if [[ ! "$NAME" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
    echo "name must match [a-z0-9][a-z0-9_-]*: $NAME" >&2
    exit 2
fi
for n in "$MAX_PARAMS" "$MIN_PARAMS"; do
    if [[ ! "$n" =~ ^[1-9][0-9]*$ ]]; then
        echo "--max-params and --min-params must be positive integers: $n" >&2
        exit 2
    fi
done
if [[ ! "$SOBOL_POINTS" =~ ^[1-9][0-9]*$ ]] || (( SOBOL_POINTS & (SOBOL_POINTS - 1) )); then
    echo "--sobol-points must be a power of two: $SOBOL_POINTS" >&2
    exit 2
fi
if (( MIN_PARAMS > MAX_PARAMS )); then
    echo "--min-params $MIN_PARAMS is above --max-params $MAX_PARAMS" >&2
    exit 2
fi
if [[ "$HEADLESS" == 1 && "$START" == 0 ]]; then
    echo "--headless and --no-start exclude each other" >&2
    exit 2
fi

MAIN="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
BASE="${PF_BASE:-policy-finder-base}"
BRANCH="policy-finder/$NAME"
WT_ROOT="${PF_WORKTREE_ROOT:-$(dirname "$MAIN")/policy-finder-worktrees}"
WT="$WT_ROOT/$NAME"
PYTHON="$MAIN/.venv/bin/python"
UV_PYTHON_DIR="$HOME/.local/share/uv/python"

git -C "$MAIN" rev-parse --verify --quiet "$BASE" >/dev/null \
    || { echo "no branch $BASE" >&2; exit 1; }
if git -C "$MAIN" rev-parse --verify --quiet "$BRANCH" >/dev/null; then
    echo "branch $BRANCH already exists" >&2
    exit 1
fi
[[ ! -e "$WT" ]] || { echo "$WT already exists" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "no Python at $PYTHON (run uv sync)" >&2; exit 1; }

mkdir -p "$WT_ROOT"
git -C "$MAIN" worktree add -b "$BRANCH" "$WT" "$BASE"
WT="$(cd "$WT" && pwd -P)"

# the data the agent reads; everything else may stay an LFS pointer
git -C "$WT" lfs pull --include \
    "experiments/2group_8agent_50ep.csv,plots/simulation/25_LEVIN_run1_*/**"

"$PYTHON" - "$WT" "$MAIN" "$NAME" "$MIN_PARAMS" "$MAX_PARAMS" "$PYTHON" "$UV_PYTHON_DIR" \
    "$SOBOL_POINTS" <<'EOF'
import json
import os
import re
import sys

wt, main, name, min_params, max_params, python, uv_python, sobol_points = sys.argv[1:]
home = os.path.expanduser("~")
rule = f"configs/managers/rule_based/{name}.yml"
notes = f"notes/policy_finder/{name}.md"
scripts = f"scripts/policy_finder/{name}"
os.makedirs(os.path.join(wt, scripts), exist_ok=True)
os.makedirs(os.path.join(wt, os.path.dirname(notes)), exist_ok=True)
# the Bash tool writes command output here; Python drops stdout it cannot stat
session_tmp = f"/private/tmp/claude-{os.getuid()}/" + re.sub(r"[^A-Za-z0-9]", "-", wt)


def complement(path):
    """Every existing entry beside `path`, at each level from the worktree down."""
    denied, parent = [], wt
    for part in path.split("/"):
        denied += [
            os.path.join(parent, e) for e in sorted(os.listdir(parent)) if e != part
        ]
        parent = os.path.join(parent, part)
    return denied


# at the top level each chain lists the others' first entries; keep them
keep = {os.path.join(wt, p.split("/")[0]) for p in (rule, notes, scripts)}
deny_write = sorted(
    set(complement(rule) + complement(notes) + complement(scripts)) - keep
)

instance = {
    "name": name,
    "min_params": int(min_params),
    "max_params": int(max_params),
    "sobol_points": int(sobol_points),
    "python": python,
}

settings = {
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
            # a narrower path wins: the worktree is readable inside the denied
            # home, its .git is not
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
                        "command": f"python3 {wt}/.claude/hooks/policy_finder_guard.py",
                    }
                ],
            }
        ],
    },
}

with open(os.path.join(wt, ".claude", "policy_finder.json"), "w") as f:
    json.dump(instance, f, indent=2)
    f.write("\n")
with open(os.path.join(wt, ".claude", "settings.local.json"), "w") as f:
    json.dump(settings, f, indent=2)
    f.write("\n")
EOF

echo "instance $NAME: branch $BRANCH, worktree $WT," \
    "params $MIN_PARAMS-$MAX_PARAMS, $SOBOL_POINTS Sobol points"
[[ "$START" == 1 ]] || exit 0
cd "$WT"
if [[ "$HEADLESS" == 1 ]]; then
    REPORT="$WT_ROOT/$NAME.report.md"
    claude -p "$PROMPT" --agent policy-finder-host --strict-mcp-config \
        --allowedTools=Bash < /dev/null > "$REPORT"
    echo "report: $REPORT"
    "$MAIN/scripts/policy_finder/check_instance.sh" "$NAME" --commit
    "$MAIN/scripts/policy_finder/run_sweep.sh" "$NAME"
else
    exec claude --agent policy-finder-host --strict-mcp-config
fi
