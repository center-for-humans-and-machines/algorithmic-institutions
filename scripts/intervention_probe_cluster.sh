#!/usr/bin/env bash
#
# Submit the focal intervention probe (aimanager.simulation.intervention_probe)
# as a SLURM job on Raven. Mirrors the sync-then-submit pattern of
# simulate_cluster.sh.
#
# Prerequisites:
#   - SSH ControlMaster active: ssh raven
#   - Remote .venv at ~/algorithmic-institutions/.venv
#
# Usage:
#   scripts/intervention_probe_cluster.sh <config> [--trace]
#
set -euo pipefail

REMOTE_HOST="raven"
REMOTE_PROJECT_DIR="~/algorithmic-institutions"
LOCAL_PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

CONFIG_FILE="${1:?Usage: $0 <config> [--trace]}"
shift
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --trace)
            EXTRA_ARGS="${EXTRA_ARGS} --trace"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

JOB_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR=".log/fidelity/${JOB_ID}"
LOG_FILE="${LOG_DIR}/log.log"

info()  { echo "==> $*"; }
error() { echo "ERROR: $*" >&2; }

if ! ssh -O check "${REMOTE_HOST}" 2>/dev/null; then
    error "No active SSH ControlMaster to ${REMOTE_HOST}."
    error "Run 'ssh raven' in a separate terminal first."
    exit 2
fi

info "Syncing files to ${REMOTE_HOST}:${REMOTE_PROJECT_DIR}..."
rsync -azP --delete \
    --filter='P /.env' \
    --filter=':- .gitignore' \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='.log/' \
    --exclude='artifacts/' \
    --exclude='plots/' \
    --exclude='notebooks/' \
    --exclude='temp/' \
    "${LOCAL_PROJECT_DIR}/" \
    "${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/"
info "Sync complete."

info "Submitting fidelity check on ${REMOTE_HOST} (job_id: ${JOB_ID})..."
info "Config: ${CONFIG_FILE}"
info "Log:    ${LOG_FILE}"

# sbatch --wrap embeds the work as a single shell command. CPU is enough
# (autoregressive punishment AH is small + one-round forward), but request
# the same gpu:a100 the simulate template uses for parity with the env
# torch installation.
ssh "${REMOTE_HOST}" bash -l <<EOSSH
set -e
cd ${REMOTE_PROJECT_DIR}
mkdir -p ${LOG_DIR}
sbatch \\
    --output=${LOG_FILE} \\
    --error=${LOG_FILE} \\
    --job-name=fidelity_${JOB_ID} \\
    --nodes=1 \\
    --tasks-per-node=1 \\
    --cpus-per-task=2 \\
    --mem=8GB \\
    --constraint=gpu \\
    --gres=gpu:a100:1 \\
    --time=00:30:00 \\
    --wrap="source .venv/bin/activate && module load cuda/11.4 && python -m aimanager.simulation.intervention_probe --config ${CONFIG_FILE} ${EXTRA_ARGS}"
EOSSH

info "Submitted. Fetch the log with:"
info "  scripts/fetch_cluster.sh ${LOG_DIR}"
