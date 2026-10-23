#!/usr/bin/env bash
# Acceptance check of a policy-finder instance (#236).
#
# Usage:
#   scripts/policy_finder/check_instance.sh <name> [--commit]
#
# Checks the instance's current version (#244), read from its
# .claude/policy_finder.json. Passes when the version changed nothing outside
# its write paths, configs/managers/rule_based/<rule_name>.yml,
# notes/policy_finder/<rule_name>.md and scripts/policy_finder/<name>/
# (<rule_name> is <name> for version 1, <name>-v<k> after):
# neither on branch policy-finder/<name> since the version's start_commit
# (since policy-finder-base for an instance from before versions) nor in its
# worktree (uncommitted or untracked files), its sweep (run_sweep.sh) aside;
# so an earlier version's rule, notes or results changed fail it. It also
# needs the rule and its notes, with the notes' sections in order:
# Explorations, Key findings, Hypothesis and its Justification, and the rule
# passing `validate-rule` (#227) within the instance's min_params and
# max_params (null: no upper bound) on its sobol_points design, run with the
# instance's python. Prints what is wrong otherwise.
# --commit then commits the write paths on the instance's branch, since the
# agent has no git. PF_BASE and PF_WORKTREE_ROOT as in new_instance.sh.

set -euo pipefail

usage() {
    sed -n '4,5p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

NAME=""
COMMIT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --commit) COMMIT=1; shift ;;
        -h|--help) usage ;;
        -*) echo "unknown option: $1" >&2; usage ;;
        *) [[ -z "$NAME" ]] || usage; NAME="$1"; shift ;;
    esac
done
[[ -n "$NAME" ]] || usage

MAIN="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
BASE="${PF_BASE:-policy-finder-base}"
BRANCH="policy-finder/$NAME"
WT="${PF_WORKTREE_ROOT:-$(dirname "$MAIN")/policy-finder-worktrees}/$NAME"

git -C "$MAIN" rev-parse --verify --quiet "$BRANCH" >/dev/null \
    || { echo "no branch $BRANCH" >&2; exit 1; }
[[ -d "$WT" ]] || { echo "no worktree at $WT" >&2; exit 1; }

# the current version, from the instance file (instance_settings.py)
CONF="$WT/.claude/policy_finder.json"
{ read -r PY; read -r STEM; read -r START; read -r MIN_PARAMS; read -r MAX_PARAMS
  read -r SOBOL_POINTS; } < <(
    python3 -c 'import json, sys
c = json.load(open(sys.argv[1]))
print(c["python"], c.get("rule_name") or sys.argv[2], c.get("start_commit") or "",
      c["min_params"], "none" if c["max_params"] is None else c["max_params"],
      c.get("sobol_points", 256), sep="\n")' "$CONF" "$NAME"
) || { echo "FAIL: cannot read the instance from $CONF" >&2; exit 1; }
RULE="configs/managers/rule_based/$STEM.yml"
NOTES="notes/policy_finder/$STEM.md"
SCRIPTS="scripts/policy_finder/$NAME/"
# what run_sweep.sh (#241) commits once the check passed
SWEEP_CONFIGS="configs/simulation/policy_finder/${STEM}_sweep"
SWEEP="plots/simulation/policy_finder/${STEM}_sweep"
# the version's changes: since its start, or since the base for an instance
# from before versions
SINCE="${START:+$START $BRANCH}"
SINCE="${SINCE:-$BASE...$BRANCH}"

changed="$(
    {
        git -C "$MAIN" diff --name-only --no-renames $SINCE
        git -C "$WT" diff --name-only --no-renames HEAD
        git -C "$WT" ls-files --others --exclude-standard
    } | sort -u
)"

bad=0
while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    if [[ "$path" != "$RULE" && "$path" != "$NOTES" && "$path" != "$SCRIPTS"* \
        && "$path" != "$SWEEP_CONFIGS"* && "$path" != "$SWEEP/"* ]]; then
        echo "outside the write paths: $path" >&2
        bad=1
    fi
done <<< "$changed"
if [[ "$bad" == 1 ]]; then
    echo "FAIL: $BRANCH changed files outside $RULE, $NOTES and $SCRIPTS" >&2
    exit 1
fi

for path in "$RULE" "$NOTES"; do
    [[ -f "$WT/$path" ]] || { echo "FAIL: no $path" >&2; exit 1; }
done

# the notes' required headings, each after the previous one
last=0
for heading in "## Explorations" "## Key findings" "## Hypothesis" \
    "### Justification"; do
    line="$(grep -n -x -F -- "$heading" "$WT/$NOTES" | head -1 | cut -d: -f1 || true)"
    if [[ -z "$line" || "$line" -le "$last" ]]; then
        echo "FAIL: $NOTES needs the section '$heading', in order" >&2
        exit 1
    fi
    last="$line"
done

# the rule passes validate-rule (#227) within the instance's limits
if ! (cd "$WT" && PYTHONPATH="$WT/src" "$PY" -m aimanager validate-rule "$RULE" \
    --min-params "$MIN_PARAMS" --max-params "$MAX_PARAMS" \
    --sobol-points "$SOBOL_POINTS"); then
    echo "FAIL: $RULE does not pass validate-rule" >&2
    exit 1
fi

echo "OK: $BRANCH touches only $RULE, $NOTES and $SCRIPTS"

if [[ "$COMMIT" == 1 ]]; then
    git -C "$WT" add -- "$RULE" "$NOTES"
    if [[ -n "$(find "$WT/$SCRIPTS" -type f 2>/dev/null)" ]]; then
        git -C "$WT" add -- "$SCRIPTS"
    fi
    if git -C "$WT" diff --cached --quiet; then
        echo "nothing to commit"
    else
        git -C "$WT" commit -q -m "policy-finder $STEM: rule, notes and analysis"
        git -C "$WT" log --oneline -1
    fi
fi
