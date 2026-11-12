#!/usr/bin/env bash
# Create and run a policy-finder instance (#236), or its next version (#244).
#
# Usage:
#   scripts/policy_finder/new_instance.sh <name> [--max-params N] [--min-params N]
#       [--sobol-points N] [--prompt TEXT] [--setting exp2|exp1|exp2_multi|exp2_diff]
#   scripts/policy_finder/new_instance.sh --iterate-on <name> [--after-failure]
#       [the same options]
#
# Makes branch policy-finder/<name> off yannik/handover (PF_BASE
# overrides), checked out as its own worktree in $PF_RUNS/worktrees/<name>
# (PF_WORKTREE_ROOT overrides) with only the paths of
# scripts/policy_finder/instance_paths (the game rules, code, configs and
# reference sims; no reports or analyses of earlier work), writes the
# instance's gitignored .claude/policy_finder.json (instance_settings.py),
# and runs the version: scripts/policy_finder/pipeline.py run <name>. That
# runs the agent in its own container through the container spawner on a copy
# of the worktree (export_instance.py, agent_runner.py), checks and commits
# its work (check_instance.sh --commit), sweeps the rule on Raven through the
# command server, fits the Gaussian process and runs the final run of its
# peak and the sweep's 5 best points (the rule's final result). A version
# that stopped resumes with `pipeline.py run <name>`.
#
# The rule declares between --min-params (default 1) and --max-params
# (default 4) params; equal values ask for exactly that many. --sobol-points
# (a power of two, default 256) sets the size of the sweep's design, which
# validate-rule, check_instance.sh and the pipeline read from the instance.
# --prompt replaces the agent's prompt (default: design your rule).
#
# --iterate-on <name> starts the next version of an existing instance on its
# own branch and worktree instead: the worktree must be clean and its current
# version's final run done (the final best the new one is compared with). It
# merges the base into the branch, so the version runs on the latest code and
# agent definition, and writes the settings of version k: its rule, notes,
# sweep and report are named <name>-v<k>, earlier versions' files stay as they
# are and the scripts folder is shared. Without --max-params and --min-params
# the rule may declare any number of params from 1; --sobol-points defaults
# to the previous version's. --after-failure also iterates on a version whose
# pipeline failed (its state file says so): its uncommitted files are
# discarded, and what it committed stays as an earlier version without a
# final run.
#
# The agent's container is its only sandbox: it sees its copy of the instance
# (read-only but for the version's rule, notes and scripts folder) and has no
# network but the Anthropic API. PF_RUNS must be the folder whose exports/ is
# the container spawner's pf_exports root.

set -euo pipefail

usage() {
    sed -n '4,7p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

NAME=""
MAX_PARAMS=4
MIN_PARAMS=1
SOBOL_POINTS=256
SETTING=exp2
SETTING_GIVEN=0
PROMPT=""
ITERATE=0
PARAMS_GIVEN=0
SOBOL_GIVEN=0
AFTER_FAILURE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --iterate-on) [[ -z "$NAME" ]] || usage; ITERATE=1; NAME="${2:-}"; shift 2 ;;
        --max-params) MAX_PARAMS="${2:-}"; PARAMS_GIVEN=1; shift 2 ;;
        --min-params) MIN_PARAMS="${2:-}"; PARAMS_GIVEN=1; shift 2 ;;
        --sobol-points) SOBOL_POINTS="${2:-}"; SOBOL_GIVEN=1; shift 2 ;;
        --prompt) PROMPT="${2:-}"; shift 2 ;;
        --setting) SETTING="${2:-}"; SETTING_GIVEN=1; shift 2 ;;
        --after-failure) AFTER_FAILURE=1; shift ;;
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

MAIN="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
BASE="${PF_BASE:-yannik/handover}"
BRANCH="policy-finder/$NAME"
[[ -n "${PF_RUNS:-}" ]] || { echo "set PF_RUNS (see the header)" >&2; exit 2; }
export PF_RUNS
WT_ROOT="${PF_WORKTREE_ROOT:-$PF_RUNS/worktrees}"
export PF_WORKTREE_ROOT="$WT_ROOT"
WT="$WT_ROOT/$NAME"
PYTHON="${PF_PYTHON:-python3}"

git -C "$MAIN" rev-parse --verify --quiet "$BASE" >/dev/null \
    || { echo "no branch $BASE" >&2; exit 1; }

if [[ "$ITERATE" == 1 ]]; then
    # the next version of an existing instance, on its branch and worktree
    git -C "$MAIN" rev-parse --verify --quiet "$BRANCH" >/dev/null \
        || { echo "no instance $NAME: no branch $BRANCH" >&2; exit 1; }
    [[ -f "$WT/.claude/policy_finder.json" ]] \
        || { echo "no instance $NAME: no worktree at $WT" >&2; exit 1; }
    WT="$(cd "$WT" && pwd -P)"
    git -C "$MAIN" worktree lock --reason "policy finder: outside the command server's view" "$WT" 2>/dev/null || true
    { read -r PREV_VERSION; read -r PREV_STEM; read -r PREV_SOBOL; read -r PREV_SETTING; } < <(
        python3 -c 'import json, sys
c = json.load(open(sys.argv[1]))
print(c.get("version", 1), c.get("rule_name") or sys.argv[2],
      c.get("sobol_points", 256), c.get("setting", "exp2"), sep="\n")' "$WT/.claude/policy_finder.json" "$NAME"
    )
    # an instance keeps its game across versions
    if [[ "$SETTING_GIVEN" == 1 && "$SETTING" != "$PREV_SETTING" ]]; then
        echo "$NAME is a $PREV_SETTING instance: --setting $SETTING does not apply" >&2
        exit 2
    fi
    SETTING="$PREV_SETTING"
    PREV_FINAL="plots/simulation/policy_finder/${PREV_STEM}_sweep/final/sweep.json"
    PREV_STATE="$WT_ROOT/$PREV_STEM.state.json"
    if [[ ! -f "$WT/$PREV_FINAL" && "$AFTER_FAILURE" == 1 ]] \
        && grep -q '"status": "failed"' "$PREV_STATE" 2>/dev/null; then
        echo "$PREV_STEM failed: discarding its uncommitted files" >&2
        git -C "$WT" reset -q --hard HEAD
        git -C "$WT" clean -q -fd
    else
        [[ -f "$WT/$PREV_FINAL" ]] || {
            echo "$PREV_STEM has no final run ($PREV_FINAL): finish it first" \
                "(pipeline.py run $NAME), or pass --after-failure if it failed" >&2
            exit 1
        }
    fi
    [[ -z "$(git -C "$WT" status --porcelain)" ]] \
        || { echo "$WT has uncommitted changes: check and commit them first" >&2; exit 1; }
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
else
    if git -C "$MAIN" rev-parse --verify --quiet "$BRANCH" >/dev/null; then
        echo "branch $BRANCH already exists" >&2
        exit 1
    fi
    [[ ! -e "$WT" ]] || { echo "$WT already exists" >&2; exit 1; }
    mkdir -p "$WT_ROOT"
    git -C "$MAIN" worktree add -q -b "$BRANCH" "$WT" "$BASE"
    WT="$(cd "$WT" && pwd -P)"
    # the command server mounts this repo without $WT_ROOT and runs
    # `git worktree prune` after each snapshot: a locked worktree is kept
    git -C "$MAIN" worktree lock --reason "policy finder: outside the command server's view" "$WT"
    VERSION=1
fi
# only what the agent may see (an earlier version's worktree is stripped too)
grep -v '^#' "$MAIN/scripts/policy_finder/instance_paths" \
    | git -C "$WT" sparse-checkout set --no-cone --stdin

"$PYTHON" "$MAIN/scripts/policy_finder/instance_settings.py" "$WT" "$NAME" \
    --version "$VERSION" --start-commit "$(git -C "$WT" rev-parse HEAD)" \
    --min-params "$MIN_PARAMS" --max-params "$MAX_PARAMS" \
    --sobol-points "$SOBOL_POINTS" --python "$PYTHON" --setting "$SETTING"
STEM="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["rule_name"])' \
    "$WT/.claude/policy_finder.json")"

echo "instance $NAME, version $VERSION ($STEM): branch $BRANCH, worktree $WT," \
    "params $MIN_PARAMS-$MAX_PARAMS, $SOBOL_POINTS Sobol points, setting $SETTING"
exec "$PYTHON" "$MAIN/scripts/policy_finder/pipeline.py" run "$NAME" \
    ${PROMPT:+--prompt "$PROMPT"}
