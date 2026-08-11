#!/usr/bin/env bash
#
# Remote training runner for algorithmic-institutions on Raven HPC.
#
# Prerequisites:
#   - SSH ControlMaster connection must be established first: ssh raven
#   - Remote .venv must exist at ~/algorithmic-institutions/.venv
#
# Usage:
#   scripts/train_cluster.sh ah <config>           # sync + train AH
#   scripts/train_cluster.sh manager <config>      # sync + train manager
#   scripts/train_cluster.sh --sync-only           # sync files only
#   scripts/train_cluster.sh --no-sync ah <config> # train without syncing
#
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
REMOTE_HOST="raven"
REMOTE_PROJECT_DIR="~/algorithmic-institutions"
LOCAL_PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── Parse arguments ──────────────────────────────────────────────────
DO_SYNC=true
DO_TRAIN=true
TRAIN_TYPE=""
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sync-only)
            DO_TRAIN=false
            shift
            ;;
        --no-sync)
            DO_SYNC=false
            shift
            ;;
        ah|manager)
            TRAIN_TYPE="$1"
            shift
            ;;
        *)
            CONFIG_FILE="$1"
            shift
            ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────
info()  { echo "==> $*"; }
error() { echo "ERROR: $*" >&2; }

# ── Step 1: Check SSH connection ─────────────────────────────────────
check_ssh() {
    info "Checking SSH connection to ${REMOTE_HOST}..."
    if ssh -O check "${REMOTE_HOST}" 2>/dev/null; then
        info "SSH ControlMaster connection is active."
    else
        error "No active SSH ControlMaster connection to ${REMOTE_HOST}."
        error "Please run 'ssh raven' in a separate terminal first."
        error "The connection will persist for 12 hours via ControlPersist."
        exit 2
    fi
}

# ── Step 2: Sync files ──────────────────────────────────────────────
sync_files() {
    info "Syncing files to ${REMOTE_HOST}:${REMOTE_PROJECT_DIR}..."
    rsync -azP --delete \
        --filter='P /.env' \
        --filter=':- .gitignore' \
        --exclude='.git/' \
        --exclude='.venv/' \
        --exclude='.log/' \
        --exclude='artifacts/' \
        --exclude='plots/' \
        --exclude='temp/' \
        "${LOCAL_PROJECT_DIR}/" \
        "${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/"
    info "Sync complete."
}

# ── Step 3: Run training remotely ────────────────────────────────────
run_training() {
    info "Running ${TRAIN_TYPE} training on ${REMOTE_HOST}..."
    info "Config: ${CONFIG_FILE}"

    local train_cmd="cd ${REMOTE_PROJECT_DIR}"
    train_cmd+=" && source .venv/bin/activate"

    case "${TRAIN_TYPE}" in
        ah)
            train_cmd+=" && uv run python"
            train_cmd+=" src/aimanager/artificial_humans/run.py"
            train_cmd+=" ${CONFIG_FILE}"
            ;;
        manager)
            train_cmd+=" && uv run python"
            train_cmd+=" src/aimanager/manager/run.py"
            train_cmd+=" ${CONFIG_FILE}"
            ;;
    esac

    ssh "${REMOTE_HOST}" "bash -l -c '${train_cmd}'"
}

# ── Main ─────────────────────────────────────────────────────────────
main() {
    if [[ "${DO_TRAIN}" == true ]]; then
        if [[ -z "${TRAIN_TYPE}" || -z "${CONFIG_FILE}" ]]; then
            error "Training type and config file are required."
            echo "Usage: scripts/train_cluster.sh [--sync-only|--no-sync] <ah|manager> <config>" >&2
            exit 1
        fi
    fi

    check_ssh

    if [[ "${DO_SYNC}" == true ]]; then
        sync_files
    fi

    if [[ "${DO_TRAIN}" == true ]]; then
        run_training
        info "Training complete."
    fi

    info "Done."
}

main
