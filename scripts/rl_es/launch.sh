#!/usr/bin/env bash
#
# Launch the five evolution-strategies seeds on Raven, in this arm's own
# isolated remote dir.
#
# The sync is `train_cluster.sh --sync-only`, which ships the AH artifacts the
# configs reference but excludes artifacts/manager/ -- so re-running this after
# the jobs have started cannot delete their outputs.
#
# Usage:
#   scripts/rl_es/launch.sh              # sync, then submit all five seeds
#   scripts/rl_es/launch.sh --no-sync    # submit only
#   scripts/rl_es/launch.sh pilot        # submit the cost/guard pilot instead
#
set -euo pipefail

REMOTE_HOST="raven"
CANONICAL="~/algorithmic-institutions"
export AI_REMOTE_DIR="${AI_REMOTE_DIR:-~/repros/ai-runs/rl-es}"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"

DO_SYNC=true
TARGET="seeds"
for arg in "$@"; do
    case "$arg" in
        --no-sync) DO_SYNC=false ;;
        pilot) TARGET="pilot" ;;
        *) echo "unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [[ "${DO_SYNC}" == true ]]; then
    "${HERE}/scripts/train_cluster.sh" --sync-only
fi

env_prefix="export AIMANAGER_VENV=${CANONICAL}/.venv"
env_prefix+=" PYTHONPATH=${AI_REMOTE_DIR}/src SBATCH_EXPORT=ALL"

if [[ "${TARGET}" == "pilot" ]]; then
    ssh "${REMOTE_HOST}" "bash -l -c 'cd ${AI_REMOTE_DIR} && mkdir -p .log \
        && ${env_prefix} \
        && sbatch scripts/rl_es/pilot.slurm'"
    exit 0
fi

for seed in 42 43 44 45 46; do
    cfg="configs/training/rl_manager/rl_es_s${seed}.yml"
    echo "==> submitting ${cfg}"
    ssh "${REMOTE_HOST}" "bash -l -c 'cd ${AI_REMOTE_DIR} \
        && ${env_prefix} \
        && python src/aimanager/manager/run_es.py ${cfg}'"
done
