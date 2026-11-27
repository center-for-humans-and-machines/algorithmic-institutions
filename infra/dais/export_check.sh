#!/bin/bash
# Build a fresh agent export for one instance of each setting given (default
# exp2 exp1) from this checkout, as new_instance.sh and pipeline.py do, and
# check that every path its agent definition names (in backticks, without
# placeholders) is in the export as a file or a non-empty folder. Leaves the
# checkout as it was (a detached worktree in a temp folder, removed after).
#
#   bash infra/dais/export_check.sh [setting ...]
#
# Needs the driver's python (PF_PYTHON, default the scratch venv).
set -euo pipefail
MAIN="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PY="${PF_PYTHON:-/dais/fs/scratch/$USER/policy-finder-agents/venv/bin/python}"
TMP=$(mktemp -d)
trap 'for w in "$TMP"/wt-*; do git -C "$MAIN" worktree remove --force "$w" 2>/dev/null || true; done; rm -rf "$TMP"' EXIT
fails=0
for setting in "${@:-exp2 exp1}"; do
  for s in $setting; do
    name="pfcheck$(echo "$s" | tr -d _)-1"
    wt="$TMP/wt-$name"
    git -C "$MAIN" worktree add -q --detach "$wt" HEAD
    grep -v '^#' "$MAIN/scripts/policy_finder/instance_paths" \
      | git -C "$wt" sparse-checkout set --no-cone --stdin
    "$PY" "$MAIN/scripts/policy_finder/instance_settings.py" "$wt" "$name" \
      --version 1 --start-commit "$(git -C "$wt" rev-parse HEAD)" --min-params 1 \
      --max-params 5 --sobol-points 256 --python "$PY" --setting "$s" --mode free \
      --harness opencode --model Qwen/Qwen3-Coder-Next --base-url '{env:PF_LLM_BASE_URL}'
    export_dir=$("$PY" "$MAIN/scripts/policy_finder/export_instance.py" build "$wt" "$MAIN" "$TMP/exports" | tail -1)
    [ -d "$export_dir" ] || export_dir="$TMP/exports/$name"
    agent="$export_dir/.opencode/agents/policy-finder.md"
    echo "== $s: export $export_dir"
    while read -r path; do
      p="${path%/}"
      if [ -f "$export_dir/$p" ] || { [ -d "$export_dir/$p" ] && [ -n "$(ls -A "$export_dir/$p")" ]; }; then
        echo "  ok      $path"
      else
        echo "  MISSING $path"; fails=$((fails + 1))
      fi
    done < <(grep -o '`[^` ]*/[^` ]*`' "$agent" | tr -d '`' | grep -v '[<>*{}]' \
             | grep -v '^/' | grep -v '^\.\.' | sort -u)
    for f in $(cd "$export_dir" && ls plots/simulation/*/per_round*.parquet experiments/*.csv 2>/dev/null); do
      echo "  data    $f $(stat -c %s "$export_dir/$f") bytes"
    done
  done
done
echo "$fails missing"
[ "$fails" = 0 ]
