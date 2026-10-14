#!/usr/bin/env bash
# Run a policy-finder instance's validation sim on Raven (#238).
#
# Usage:
#   scripts/policy_finder/run_validate.sh <name>
#
# Run once the instance's sweep is committed with its landscapes.json
# (run_sweep.sh calls it). From the instance worktree, with this checkout's
# scripts (an older instance's own copies predate it): writes the
# validation sim config (generate_sim_config.py --validate: the Gaussian
# process's peak and the top measured points, fresh seeds, more episodes),
# syncs the worktree to its own Raven dir, submits the sim, waits for it,
# fetches its sweep.json, best-point plots and per-round data into
# plots/simulation/policy_finder/<name>_sweep/validate/ and commits that and
# the config on policy-finder/<name>. Its sweep.json's `best` is the rule's
# final result. Needs the SSH ControlMaster to Raven (`ssh raven`).
# PF_WORKTREE_ROOT as in new_instance.sh.

set -euo pipefail

POLL_SECONDS=60

NAME="${1:-}"
[[ -n "$NAME" && $# -eq 1 ]] || { sed -n '4,5p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

MAIN="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
WT_ROOT="${PF_WORKTREE_ROOT:-$(dirname "$MAIN")/policy-finder-worktrees}"
WT="$WT_ROOT/$NAME"
CONF="$WT/.claude/policy_finder.json"
SWEEP="plots/simulation/policy_finder/${NAME}_sweep"
VALIDATE="$SWEEP/validate"
CONFIG="configs/simulation/policy_finder/${NAME}_sweep_validate.yml"
[[ -f "$CONF" ]] || { echo "no instance $NAME: $CONF missing" >&2; exit 1; }
PY="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["python"])' "$CONF")"

cd "$WT"
[[ -f "$SWEEP/landscapes.json" ]] \
    || { echo "no $SWEEP/landscapes.json: run run_sweep.sh (or fit_surrogate.py) first" >&2; exit 1; }
source "$MAIN/scripts/raven_remote_dir.sh"
REMOTE="$(raven_remote_dir "$WT" "~/algorithmic-institutions")"
ssh -O check raven 2>/dev/null \
    || { echo "no SSH ControlMaster to raven: run 'ssh raven' first" >&2; exit 2; }

# 1. the sim config
PYTHONPATH="$WT/src" "$PY" "$MAIN/scripts/policy_finder/generate_sim_config.py" \
    --validate "$SWEEP/sweep.json" --root "$WT"

# 2. sync, submit
scripts/simulate_cluster.sh --sync-only
out="$(scripts/simulate_cluster.sh --no-sync "$CONFIG")"
echo "$out"
job="$(sed -n 's/^Submitted batch job \([0-9]*\).*/\1/p' <<< "$out")"
if [[ -z "$job" ]]; then
    # an instance from before #241 does not print sbatch's output: find the
    # job by its name, the run id in its script's path
    run_id="$(sed -n 's|^sbatch .*/\([0-9a-f]\{8\}\)/run\.sh$|\1|p' <<< "$out")"
    [[ -n "$run_id" ]] && job="$(ssh raven "squeue -h -u \$USER -n $run_id -o %i")"
fi
[[ -n "$job" ]] || { echo "no job id for $CONFIG" >&2; exit 1; }

# 3. wait until the job has left the queue
echo "waiting for job $job"
while ssh raven 'squeue -h -u $USER -o %i' | grep -qxF "$job"; do
    sleep "$POLL_SECONDS"
done

# 4. fetch; no sweep.json means the sim failed
mkdir -p "$VALIDATE"
rsync -az "raven:$REMOTE/$VALIDATE/" "$VALIDATE/" \
    || { echo "FAIL: cannot fetch $VALIDATE (see .log/ in $REMOTE)" >&2; exit 1; }
[[ -f "$VALIDATE/sweep.json" ]] \
    || { echo "FAIL: the validation wrote no sweep.json (see .log/ in $REMOTE)" >&2; exit 1; }

# 5. commit on the instance branch, never on policy-finder-base
git add -- "$VALIDATE" "$CONFIG"
git commit -q -m "policy-finder $NAME: validation"
git log --oneline -1
"$PY" -c 'import json, sys
v = json.load(open(sys.argv[1]))
b = next(p for p in v["points"] if p["name"] == v["best_name"])
print("final best %s: %.1f +- %.1f, %s" % (b["name"], b["pool"], b["pool_se"], v["best"]))' \
    "$VALIDATE/sweep.json"
