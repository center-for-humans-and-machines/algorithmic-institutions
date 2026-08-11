#!/usr/bin/env bash
#
# Remote simulation runner for algorithmic-institutions on Raven HPC.
#
# Prerequisites:
#   - SSH ControlMaster connection must be established first: ssh raven
#   - Remote .venv must exist at ~/algorithmic-institutions/.venv
#
# Usage:
#   scripts/simulate_cluster.sh <config>           # sync + simulate
#   scripts/simulate_cluster.sh --sync-only        # sync files only
#   scripts/simulate_cluster.sh --no-sync <config> # simulate without syncing
#
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
REMOTE_HOST="raven"
REMOTE_PROJECT_DIR="~/algorithmic-institutions"
LOCAL_PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── Parse arguments ──────────────────────────────────────────────────
DO_SYNC=true
DO_SIMULATE=true
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sync-only)
            DO_SIMULATE=false
            shift
            ;;
        --no-sync)
            DO_SYNC=false
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

# ── Step 3: Run simulation remotely ─────────────────────────────────
run_simulation() {
    info "Running simulation on ${REMOTE_HOST}..."
    info "Config: ${CONFIG_FILE}"

    local sim_cmd="cd ${REMOTE_PROJECT_DIR}"
    sim_cmd+=" && source .venv/bin/activate"
    sim_cmd+=" && uv run python"
    sim_cmd+=" src/aimanager/simulation/run.py"
    sim_cmd+=" ${CONFIG_FILE}"

    ssh "${REMOTE_HOST}" "bash -l -c '${sim_cmd}'"
}

# ── Main ─────────────────────────────────────────────────────────────
main() {
    if [[ "${DO_SIMULATE}" == true ]]; then
        if [[ -z "${CONFIG_FILE}" ]]; then
            error "Config file is required."
            echo "Usage: scripts/simulate_cluster.sh [--sync-only|--no-sync] <config>" >&2
            exit 1
        fi
    fi

    check_ssh

    if [[ "${DO_SYNC}" == true ]]; then
        sync_files
    fi

    if [[ "${DO_SIMULATE}" == true ]]; then
        run_simulation
        info "Simulation complete."
    fi

    info "Done."
}

main
