#!/usr/bin/env bash
# Run a policy-finder instance's sweep on Raven (#241).
#
# Usage:
#   scripts/policy_finder/run_sweep.sh <name>
#
# Run once the instance passed check_instance.sh, outside the agent's sandbox
# (new_instance.sh --headless calls it). From the instance worktree: writes
# the sweep sim configs of its rule (generate_sim_config.py, split into parts
# of at most MAX_JOB_EPISODES episodes, what one 16 GB job holds), syncs the
# worktree to its own Raven dir (its .raven_remote_dir:
# ~/ai-isolated/policy-finder--<name>), submits every part, waits for the
# jobs, fetches each part's sweep.json and best-point plots (never
# per_round.parquet) and merges them into
# plots/simulation/policy_finder/<name>_sweep/ in the worktree.
# Needs the SSH ControlMaster to Raven (`ssh raven`).
# PF_WORKTREE_ROOT as in new_instance.sh.

set -euo pipefail

SOBOL_POINTS=256
N_EPISODES=500
MAX_JOB_EPISODES=125000
POLL_SECONDS=60

NAME="${1:-}"
[[ -n "$NAME" && $# -eq 1 ]] || { sed -n '4,5p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

MAIN="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
WT_ROOT="${PF_WORKTREE_ROOT:-$(dirname "$MAIN")/policy-finder-worktrees}"
WT="$WT_ROOT/$NAME"
CONF="$WT/.claude/policy_finder.json"
RULE="configs/managers/rule_based/$NAME.yml"
OUT_DIR="plots/simulation/policy_finder"
[[ -f "$CONF" ]] || { echo "no instance $NAME: $CONF missing" >&2; exit 1; }
{ read -r PY; read -r MIN_PARAMS; read -r MAX_PARAMS; } < <(
    python3 -c 'import json, sys
c = json.load(open(sys.argv[1]))
print(c["python"], c["min_params"], c["max_params"], sep="\n")' "$CONF"
)

cd "$WT"
source scripts/raven_remote_dir.sh
REMOTE="$(raven_remote_dir "$WT" "~/algorithmic-institutions")"
ssh -O check raven 2>/dev/null \
    || { echo "no SSH ControlMaster to raven: run 'ssh raven' first" >&2; exit 2; }

# 1. the sim configs, one per part
N_PARTS=$(( (SOBOL_POINTS * N_EPISODES + MAX_JOB_EPISODES - 1) / MAX_JOB_EPISODES ))
PYTHONPATH="$WT/src" "$PY" scripts/policy_finder/generate_sim_config.py \
    --config "$RULE" --sobol-points "$SOBOL_POINTS" --n-episodes "$N_EPISODES" \
    --n-parts "$N_PARTS" --min-params "$MIN_PARAMS" --max-params "$MAX_PARAMS"
if (( N_PARTS == 1 )); then
    PARTS=("${NAME}_sweep")
else
    PARTS=()
    for (( k = 1; k <= N_PARTS; k++ )); do PARTS+=("${NAME}_sweep_p${k}of${N_PARTS}"); done
fi

# 2. sync once, submit every part
scripts/simulate_cluster.sh --sync-only
JOBS=()
for part in "${PARTS[@]}"; do
    out="$(scripts/simulate_cluster.sh --no-sync "configs/simulation/policy_finder/$part.yml")"
    echo "$out"
    job="$(sed -n 's/^Submitted batch job \([0-9]*\).*/\1/p' <<< "$out")"
    [[ -n "$job" ]] || { echo "no job id for $part" >&2; exit 1; }
    JOBS+=("$job")
done

# 3. wait until no job is left in the queue
echo "waiting for jobs ${JOBS[*]}"
while true; do
    queue="$(ssh raven 'squeue -h -u $USER -o %i')"
    left="$(grep -cxF -f <(printf '%s\n' "${JOBS[@]}") <<< "$queue" || true)"
    (( left > 0 )) || break
    sleep "$POLL_SECONDS"
done

# 4. fetch each part's results; a part without sweep.json failed
for part in "${PARTS[@]}"; do
    mkdir -p "$OUT_DIR/$part"
    rsync -az --exclude=per_round.parquet \
        "raven:$REMOTE/$OUT_DIR/$part/" "$OUT_DIR/$part/" \
        || { echo "FAIL: cannot fetch $part (see .log/ in $REMOTE)" >&2; exit 1; }
    [[ -f "$OUT_DIR/$part/sweep.json" ]] \
        || { echo "FAIL: $part wrote no sweep.json (see .log/ in $REMOTE)" >&2; exit 1; }
done

# 5. merge the parts
if (( N_PARTS > 1 )); then
    dirs=()
    for part in "${PARTS[@]}"; do dirs+=("$OUT_DIR/$part"); done
    PYTHONPATH="$WT/src" "$PY" scripts/policy_finder/merge_sweeps.py "${dirs[@]}"
fi
echo "sweep of $NAME: $WT/$OUT_DIR/${NAME}_sweep"
