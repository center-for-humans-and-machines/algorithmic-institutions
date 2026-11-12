#!/bin/bash
# Submit one serve-and-run job.
#   scripts/annotation/cluster/submit.sh <viper|dais> <run_id> "<command>" [CFG_KEY=VALUE ...]
#       [--no-sync] [--dry-run] [--dependency=afterok:<jobid>]
# Example:
#   submit.sh viper r1 "python -m aimanager.annotation stage A --run r1"
#   submit.sh dais smoke "python scripts/annotation/cluster/probe.py" \
#       CFG_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CFG_TP=1 CFG_GPUS=1 \
#       CFG_PARTITION=gpudev CFG_TIME=00:15:00
# Syncs code and the local run dir first (sync_code.sh). The config is
# resolved on the login node from config.sh plus the overrides, written to
# <run>/jobs/<stamp>.env (the job sources it, so values with commas or quotes
# survive) and the sbatch header values are passed on the command line.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
usage="usage: submit.sh <viper|dais> <run_id> \"<command>\" [CFG_KEY=VALUE ...] [--no-sync] [--dry-run]"
cluster=${1:?$usage}
run_id=${2:?$usage}
cmd=${3:?$usage}
shift 3
[[ $run_id =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad run_id: $run_id" >&2; exit 2; }
sync=1 extra=()
for a in "$@"; do
    case "$a" in
    --no-sync) sync=0 ;;
    --dry-run | --dependency=* | --begin=* | --nodelist=* | --exclude=*) extra+=("$a") ;;
    CFG_*=*) extra+=("$a") ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done
# shellcheck source=config.sh
source "$here/config.sh"
P=$(cluster_config "$cluster" && echo "$CFG_PROJECT")
[ "$sync" = 1 ] && "$here/sync_code.sh" "$cluster" "$run_id"

# ssh hands the remote shell one string; %q quoting survives that re-parse
ssh -o BatchMode=yes "$cluster" "bash -s -- $(printf '%q ' "$P" "$cluster" "$run_id" "$cmd" ${extra[@]+"${extra[@]}"})" <<'REMOTE'
set -euo pipefail
P=$1 cluster=$2 run_id=$3 cmd=$4
shift 4
dry=0 sbatch_extra=()
for a in "$@"; do
    case "$a" in
    --dry-run) dry=1 ;;
    CFG_*=*) export "${a?}" ;;
    *) sbatch_extra+=("$a") ;;
    esac
done
source "$P/scripts/annotation/cluster/config.sh"
cluster_config "$cluster"
model_config
run_dir="$CFG_RUNS_ROOT/$run_id"
mkdir -p "$run_dir/jobs" "$run_dir/logs"
envf="$run_dir/jobs/$(date +%Y%m%dT%H%M%S)-$$.env"
{
    # shellcheck disable=SC2046
    declare -p $(compgen -v CFG_)
    printf 'export ANNOT_RUN_ID=%q\nexport ANNOT_CMD=%q\n' "$run_id" "$cmd"
} >"$envf"
if [ "$cluster" = dais ]; then gres="gpu:$CFG_GRES_TYPE:$CFG_GPUS"; else gres="gpu:$CFG_GPUS"; fi
# shellcheck disable=SC2206
args=(--parsable --job-name "annot-$run_id" --partition "$CFG_PARTITION"
    --gres "$gres" --cpus-per-task "$CFG_CPUS" --mem "$CFG_MEM" --time "$CFG_TIME"
    --signal "B:USR1@$((CFG_WALL_BUFFER_MIN * 60))"
    --output "$run_dir/logs/slurm-%j.out"
    $CFG_SBATCH_EXTRA "${sbatch_extra[@]}"
    --export "ALL,ANNOT_JOB_ENV=$envf"
    "$P/scripts/annotation/cluster/serve_and_run.$cluster.sbatch")
echo "sbatch ${args[*]}"
echo "model=$CFG_MODEL@$CFG_REVISION tp=$CFG_TP image=$CFG_IMAGE"
if [ "$dry" = 1 ]; then
    sbatch --test-only "${args[@]:1}"
else
    jid=$(sbatch "${args[@]}")
    echo "submitted job $jid; log $run_dir/logs/slurm-$jid.out, job logs $run_dir/logs/job-$jid/"
fi
REMOTE
