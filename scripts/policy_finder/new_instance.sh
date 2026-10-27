#!/usr/bin/env bash
# Create and start a policy-finder instance (#236).
#
# Usage:
#   scripts/policy_finder/new_instance.sh <name> [--max-params N] [--min-params N]
#       [--sobol-points N] [--no-start | --headless [--prompt TEXT]]
#   scripts/policy_finder/new_instance.sh --iterate-on <name> [the same options]
#
# Makes branch policy-finder/<name> off policy-finder-base, checked out as its
# own worktree in ../policy-finder-worktrees/<name> (outside this checkout;
# override with PF_WORKTREE_ROOT), pulls the LFS data the agent reads, writes
# the instance's gitignored .claude/policy_finder.json and
# .claude/settings.local.json (instance_settings.py), and starts
# `claude --agent policy-finder-host` there with no MCP servers. --no-start stops before starting the session.
# --headless runs the session non-interactively instead (`claude -p`, Bash
# pre-allowed: the sandbox and the guard are the boundary) on --prompt
# (default: design your rule) and saves the agent's final report to
# ../policy-finder-worktrees/<name>.report.md, outside the worktree; then
# checks and commits the instance (check_instance.sh --commit), runs its
# sweep on Raven (run_sweep.sh, #241; with its Gaussian process fit and
# the final run of its peak) and updates the observation app's
# Summary with a headless /policy-finder-summarizer (its report in
# ../policy-finder-worktrees/<name>.summary.md).
# The rule declares between --min-params (default 1) and --max-params
# (default 4) params; equal values ask for exactly that many. --sobol-points
# (a power of two, default 256) sets the size of the sweep's design, which
# validate-rule, check_instance.sh and run_sweep.sh read from the instance.
# PF_BASE overrides the base branch, to try a branch before it lands on
# policy-finder-base.
#
# --iterate-on <name> starts the next version of an existing instance (#244)
# on its own branch and worktree instead: the worktree must be clean and its
# current version's final run done (the final best the new one is compared with).
# It merges policy-finder-base into the branch, so the version runs on the
# latest code and agent definition, and writes the settings of version k:
# its rule, notes, sweep, report and summary are named <name>-v<k>, earlier
# versions' files are read-only and the scripts folder is shared. Without
# --max-params and --min-params the rule may declare any number of params
# from 1; --sobol-points defaults to the previous version's.
#
# The instance settings (instance_settings.py) bind only that worktree and the
# version's write paths: permissions, the Bash sandbox, no network, the guard
# hook (.claude/hooks/policy_finder_guard.py) on every tool call.
#
# Check and commit an instance's work with scripts/policy_finder/check_instance.sh.

set -euo pipefail

usage() {
    sed -n '4,7p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

NAME=""
MAX_PARAMS=4
MIN_PARAMS=1
SOBOL_POINTS=256
START=1
HEADLESS=0
PROMPT=""
ITERATE=0
PARAMS_GIVEN=0
SOBOL_GIVEN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --iterate-on) [[ -z "$NAME" ]] || usage; ITERATE=1; NAME="${2:-}"; shift 2 ;;
        --max-params) MAX_PARAMS="${2:-}"; PARAMS_GIVEN=1; shift 2 ;;
        --min-params) MIN_PARAMS="${2:-}"; PARAMS_GIVEN=1; shift 2 ;;
        --sobol-points) SOBOL_POINTS="${2:-}"; SOBOL_GIVEN=1; shift 2 ;;
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
# --iterate-on without them: any number of params from 1 (set below)
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
[[ -x "$PYTHON" ]] || { echo "no Python at $PYTHON (run uv sync)" >&2; exit 1; }
# the data the agent reads; everything else may stay an LFS pointer
LFS_DATA="experiments/2group_8agent_50ep.csv,plots/simulation/25_LEVIN_run1_*/**"

if [[ "$ITERATE" == 1 ]]; then
    # the next version of an existing instance, on its branch and worktree
    git -C "$MAIN" rev-parse --verify --quiet "$BRANCH" >/dev/null \
        || { echo "no instance $NAME: no branch $BRANCH" >&2; exit 1; }
    [[ -f "$WT/.claude/policy_finder.json" ]] \
        || { echo "no instance $NAME: no worktree at $WT" >&2; exit 1; }
    WT="$(cd "$WT" && pwd -P)"
    [[ -z "$(git -C "$WT" status --porcelain)" ]] \
        || { echo "$WT has uncommitted changes: check and commit them first" >&2; exit 1; }
    { read -r PREV_VERSION; read -r PREV_STEM; read -r PREV_SOBOL; } < <(
        python3 -c 'import json, sys
c = json.load(open(sys.argv[1]))
print(c.get("version", 1), c.get("rule_name") or sys.argv[2],
      c.get("sobol_points", 256), sep="\n")' "$WT/.claude/policy_finder.json" "$NAME"
    )
    PREV_FINAL="plots/simulation/policy_finder/${PREV_STEM}_sweep/final/sweep.json"
    [[ -f "$WT/$PREV_FINAL" ]] || {
        echo "$PREV_STEM has no final run ($PREV_FINAL): finish it first" \
            "(run_sweep.sh or run_final.sh $NAME)" >&2
        exit 1
    }
    VERSION=$(( PREV_VERSION + 1 ))
    if ! git -C "$WT" merge -q --no-edit \
        -m "policy-finder $NAME: $BASE for version $VERSION" "$BASE"; then
        git -C "$WT" merge --abort || true
        echo "cannot merge $BASE into $BRANCH" >&2
        exit 1
    fi
    if [[ "$PARAMS_GIVEN" == 0 ]]; then
        MIN_PARAMS=1
        MAX_PARAMS=none
    fi
    [[ "$SOBOL_GIVEN" == 1 ]] || SOBOL_POINTS="$PREV_SOBOL"
    [[ -n "$PROMPT" ]] || PROMPT="Design version $VERSION of your rule."
else
    if git -C "$MAIN" rev-parse --verify --quiet "$BRANCH" >/dev/null; then
        echo "branch $BRANCH already exists" >&2
        exit 1
    fi
    [[ ! -e "$WT" ]] || { echo "$WT already exists" >&2; exit 1; }
    mkdir -p "$WT_ROOT"
    git -C "$MAIN" worktree add -b "$BRANCH" "$WT" "$BASE"
    WT="$(cd "$WT" && pwd -P)"
    VERSION=1
    [[ -n "$PROMPT" ]] || PROMPT="Design your rule."
fi
git -C "$WT" lfs pull --include "$LFS_DATA"

"$PYTHON" "$MAIN/scripts/policy_finder/instance_settings.py" "$WT" "$MAIN" "$NAME" \
    --version "$VERSION" --start-commit "$(git -C "$WT" rev-parse HEAD)" \
    --min-params "$MIN_PARAMS" --max-params "$MAX_PARAMS" \
    --sobol-points "$SOBOL_POINTS" --python "$PYTHON" --uv-python "$UV_PYTHON_DIR"
STEM="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["rule_name"])' \
    "$WT/.claude/policy_finder.json")"

echo "instance $NAME, version $VERSION ($STEM): branch $BRANCH, worktree $WT," \
    "params $MIN_PARAMS-$MAX_PARAMS, $SOBOL_POINTS Sobol points"
[[ "$START" == 1 ]] || exit 0
cd "$WT"
if [[ "$HEADLESS" == 1 ]]; then
    REPORT="$WT_ROOT/$STEM.report.md"
    claude -p "$PROMPT" --agent policy-finder-host --strict-mcp-config \
        --allowedTools=Bash < /dev/null > "$REPORT"
    echo "report: $REPORT"
    "$MAIN/scripts/policy_finder/check_instance.sh" "$NAME" --commit
    "$MAIN/scripts/policy_finder/run_sweep.sh" "$NAME"
    # place the rule in the observation app's Summary and republish the shared
    # page: a headless /policy-finder-summarizer in this checkout, its report
    # beside the instance's; a failure here leaves the run as it is
    SUMMARY_LOG="$WT_ROOT/$STEM.summary.md"
    if (cd "$MAIN" && claude -p "/policy-finder-summarizer" --strict-mcp-config \
        --add-dir "$WT_ROOT" --allowedTools=Bash,Read,Artifact \
        < /dev/null > "$SUMMARY_LOG"); then
        echo "summary: $SUMMARY_LOG"
    else
        echo "summary not updated (see $SUMMARY_LOG): run /policy-finder-summarizer" >&2
    fi
else
    exec claude --agent policy-finder-host --strict-mcp-config
fi
