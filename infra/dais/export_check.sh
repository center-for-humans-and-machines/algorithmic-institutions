#!/bin/bash
# Build a fresh agent export for one instance of each setting given (default
# exp2 exp1) from this checkout, as new_instance.sh and pipeline.py do, and
# check that every path its agent definition names (in backticks, without
# placeholders) is in the export as a file or a non-empty folder. Leaves the
# checkout as it was (a detached worktree in a temp folder, removed after).
#
#   bash infra/dais/export_check.sh [setting ...]
#
# Then runs, inside the agent's container (AGENT_SIF, as agent_apptainer.py
# starts it), the validate-rule command the definition tells the agent to run,
# with the instance file's own values (its `python` among them), on a rule of
# the reference campaign copied in as the instance's rule; it must pass.
#
# Needs the driver's python (PF_PYTHON, default the scratch venv).
set -euo pipefail
module load apptainer/1.5.2 >/dev/null 2>&1 || true
AGENT_SIF="${AGENT_SIF:-/dais/fs/scratch/$USER/policy-finder-agents/images/algoinst_claude_opencode.sif}"
MAIN="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
PY="${PF_PYTHON:-/dais/fs/scratch/$USER/policy-finder-agents/venv/bin/python}"
TMP=$(mktemp -d)
trap 'for w in "$TMP"/wt-*; do git -C "$MAIN" worktree remove --force "$w" 2>/dev/null || true; done; rm -rf "$TMP"' EXIT
fails=0
# named by a definition but not in a fresh version 1's export, by design:
# `final/` (a sweep's subfolder, from version 2 on), and exp1's mention of
# basics.md ("if present, is a different game"; the export denies it)
EXPECTED_ABSENT="exp2:final exp1:final exp1:docs/reports/basics.md"
for setting in "${@:-exp2 exp1}"; do
  for s in $setting; do
    name="pfcheck$(echo "$s" | tr -d _)-1"
    wt="$TMP/wt-$name"
    git -C "$MAIN" worktree add -q --detach "$wt" HEAD
    grep -v '^#' "$MAIN/scripts/policy_finder/instance_paths" \
      | git -C "$wt" sparse-checkout set --no-cone --stdin
    "$PY" "$MAIN/scripts/policy_finder/instance_settings.py" "$wt" "$name" \
      --version 1 --start-commit "$(git -C "$wt" rev-parse HEAD)" --min-params 1 \
      --max-params none --sobol-points 256 --python "${PF_AGENT_PYTHON:-python3}" \
      --setting "$s" --mode free \
      --harness opencode --model Qwen/Qwen3-Coder-Next --base-url '{env:PF_LLM_BASE_URL}'
    export_dir=$("$PY" "$MAIN/scripts/policy_finder/export_instance.py" build "$wt" "$MAIN" "$TMP/exports" | tail -1)
    [ -d "$export_dir" ] || export_dir="$TMP/exports/$name"
    agent="$export_dir/.opencode/agents/policy-finder.md"
    echo "== $s: export $export_dir"
    while read -r path; do
      p="${path%/}"
      found=""
      for q in "$p" "src/aimanager/$p"; do  # the definitions name some paths under src/aimanager
        if [ -f "$export_dir/$q" ] || { [ -d "$export_dir/$q" ] && [ -n "$(ls -A "$export_dir/$q")" ]; }; then
          found="$q"; break
        fi
      done
      if [ -n "$found" ]; then
        echo "  ok      $path"
      elif [[ " $EXPECTED_ABSENT " == *" $s:$p "* ]]; then
        echo "  absent  $path (as intended)"
      else
        echo "  MISSING $path"; fails=$((fails + 1))
      fi
    done < <(grep -o '`[^` ]*/[^` ]*`' "$agent" | tr -d '`' | grep -v '[<>*{}]' \
             | grep -v '^/' | grep -v '^\.\.' | sort -u)
    # the agent's validate-rule, in its container, with the instance file's values
    read -r ipy rule mn mx sp < <("$PY" -c 'import json, sys
c = json.load(open(sys.argv[1]))
print(c["python"], c["rule_name"], c["min_params"],
      "none" if c["max_params"] is None else c["max_params"], c["sobol_points"])' \
      "$export_dir/.claude/policy_finder.json")
    ref=$([ "$s" = exp1 ] && echo pf-20261105-7-v3 || echo pf-20261104-12-v7)
    cp "$MAIN/configs/managers/rule_based/$ref.yml" "$export_dir/configs/managers/rule_based/$rule.yml"
    mkdir -p "$TMP/home-$name" "$TMP/tmp-$name"
    cmd="PYTHONPATH=src $ipy -m aimanager validate-rule configs/managers/rule_based/$rule.yml --min-params $mn --max-params $mx --sobol-points $sp --setting $s"
    if apptainer exec --containall --cleanenv --no-mount cwd,tmp,bind-paths \
        --home "$TMP/home-$name:/home/claude" --bind "$TMP/tmp-$name:/tmp" \
        --bind "$export_dir:/opt/algorithmic-institutions:ro" --pwd /opt/algorithmic-institutions \
        "$AGENT_SIF" /bin/bash -c "$cmd" > "$TMP/validate-$name.log" 2>&1; then
      echo "  ok      in the container: $cmd"
    else
      echo "  FAILED  in the container: $cmd"; tail -5 "$TMP/validate-$name.log" | sed 's/^/          /'
      fails=$((fails + 1))
    fi
    echo "  instance file: $(tr -d '\n' < "$export_dir/.claude/policy_finder.json")"
    for f in $(cd "$export_dir" && ls plots/simulation/*/per_round*.parquet experiments/*.csv 2>/dev/null); do
      echo "  data    $f $(stat -c %s "$export_dir/$f") bytes"
    done
  done
done
echo "$fails missing"
[ "$fails" = 0 ]
