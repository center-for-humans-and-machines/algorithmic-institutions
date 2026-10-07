#!/usr/bin/env bash
# Acceptance check of a policy-finder instance (#236).
#
# Usage:
#   scripts/policy_finder/check_instance.sh <name> [--commit]
#
# Passes when the instance changed nothing outside its two write paths,
# configs/managers/rule_based/<name>.yml and scripts/policy_finder/<name>/:
# neither on branch policy-finder/<name> since policy-finder-base
# (git diff policy-finder-base...policy-finder/<name>) nor in its worktree
# (uncommitted or untracked files). Prints the offending paths otherwise.
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
RULE="configs/managers/rule_based/$NAME.yml"
SCRIPTS="scripts/policy_finder/$NAME/"

git -C "$MAIN" rev-parse --verify --quiet "$BRANCH" >/dev/null \
    || { echo "no branch $BRANCH" >&2; exit 1; }
[[ -d "$WT" ]] || { echo "no worktree at $WT" >&2; exit 1; }

changed="$(
    {
        git -C "$MAIN" diff --name-only --no-renames "$BASE...$BRANCH"
        git -C "$WT" diff --name-only --no-renames HEAD
        git -C "$WT" ls-files --others --exclude-standard
    } | sort -u
)"

bad=0
while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    if [[ "$path" != "$RULE" && "$path" != "$SCRIPTS"* ]]; then
        echo "outside the write paths: $path" >&2
        bad=1
    fi
done <<< "$changed"
if [[ "$bad" == 1 ]]; then
    echo "FAIL: $BRANCH changed files outside $RULE and $SCRIPTS" >&2
    exit 1
fi

if [[ ! -f "$WT/$RULE" ]]; then
    echo "FAIL: no rule at $RULE" >&2
    exit 1
fi

# #227 adds the rule validation here
echo "OK: $BRANCH touches only $RULE and $SCRIPTS"

if [[ "$COMMIT" == 1 ]]; then
    git -C "$WT" add -- "$RULE"
    if [[ -n "$(find "$WT/$SCRIPTS" -type f 2>/dev/null)" ]]; then
        git -C "$WT" add -- "$SCRIPTS"
    fi
    if git -C "$WT" diff --cached --quiet; then
        echo "nothing to commit"
    else
        git -C "$WT" commit -q -m "policy-finder $NAME: rule and analysis"
        git -C "$WT" log --oneline -1
    fi
fi
