#!/bin/bash
# Submit one serve-and-run job.
#   scripts/annotation/cluster/submit.sh <viper|dais> <run_id> "<command>" [CFG_KEY=VALUE ...]
#       [--no-sync] [--dry-run] [--dependency=afterok:<jobid>]
# Example:
#   submit.sh viper r1 "python -m aimanager.annotation stage A --run r1"
#   submit.sh dais smoke "python scripts/annotation/cluster/probe.py" \
#       CFG_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CFG_TP=1 CFG_GPUS=1 \
#       CFG_PARTITION=gpudev CFG_TIME=00:15:00
# Syncs code and the local run dir first (sync_code.sh; not with --dry-run),
# then snapshots the synced code into <run>/jobs/<stamp>.code/, which the job
# runs from, so a later sync cannot change a queued or running job. The config is
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
sync=1 dry=0 extra=()
for a in "$@"; do
    case "$a" in
    --no-sync) sync=0 ;;
    --dry-run) dry=1 extra+=("$a") ;;
    --dependency=* | --begin=* | --nodelist=* | --exclude=*) extra+=("$a") ;;
    CFG_*=*) extra+=("$a") ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done
# shellcheck source=config.sh
source "$here/config.sh"
P=$(cluster_config "$cluster" && echo "$CFG_PROJECT")
if [ "$sync" = 1 ] && [ "$dry" = 0 ]; then "$here/sync_code.sh" "$cluster" "$run_id"; fi

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
if [ "$cluster" = dais ] && [ "${CFG_SKIP_VERIFY:-0}" != 1 ]; then
    # the image and the production weights must have been verified on disk
    [ -s "$CFG_IMAGE.sha256" ] || {
        echo "no $CFG_IMAGE.sha256: image pull unfinished or unverified (CFG_SKIP_VERIFY=1 overrides)" >&2
        exit 1
    }
    if [ "$CFG_MODEL" = "${CFG_PROD_MODEL:-}" ]; then
        v="$CFG_PROD_HF_HOME/annot_downloads/${CFG_MODEL//\//--}/$CFG_REVISION/VERIFIED"
        [ -f "$v" ] || {
            echo "no $v: weights not verified by download_model.sh (CFG_SKIP_VERIFY=1 overrides)" >&2
            exit 1
        }
    fi
fi
run_dir="$CFG_RUNS_ROOT/$run_id"
mkdir -p "$run_dir/jobs" "$run_dir/logs"
stamp="$(date +%Y%m%dT%H%M%S)-$$"
envf="$run_dir/jobs/$stamp.env"
CFG_CODE_DIR=$P
if [ "$dry" = 0 ]; then
    # per-job code snapshot: package, config, scripts and the synced version
    CFG_CODE_DIR="$run_dir/jobs/$stamp.code"
    mkdir -p "$CFG_CODE_DIR/src/aimanager" "$CFG_CODE_DIR/configs" "$CFG_CODE_DIR/scripts"
    cp -a "$P/src/aimanager/__init__.py" "$P/src/aimanager/annotation" "$CFG_CODE_DIR/src/aimanager/"
    cp -a "$P/configs/annotation" "$CFG_CODE_DIR/configs/"
    cp -a "$P/scripts/annotation" "$CFG_CODE_DIR/scripts/"
    cp -a "$P/CODE_VERSION" "$CFG_CODE_DIR/" 2>/dev/null || true
fi
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
    "$CFG_CODE_DIR/scripts/annotation/cluster/serve_and_run.$cluster.sbatch")
echo "sbatch ${args[*]}"
echo "model=$CFG_MODEL@$CFG_REVISION tp=$CFG_TP image=$CFG_IMAGE code=$CFG_CODE_DIR"
if [ "$dry" = 1 ]; then
    sbatch --test-only "${args[@]:1}"
else
    jid=$(sbatch "${args[@]}")
    echo "submitted job $jid; log $run_dir/logs/slurm-$jid.out, job logs $run_dir/logs/job-$jid/"
fi
REMOTE
