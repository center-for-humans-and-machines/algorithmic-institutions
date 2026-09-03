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
# Set AI_REMOTE_DIR to sync and run in an isolated remote dir (e.g.
# ~/autoresearch/<slug>) instead of the shared checkout. Isolated dirs
# carry no venv: jobs use the shared checkout's venv and import their
# own code via PYTHONPATH.
#
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
REMOTE_HOST="raven"
CANONICAL_REMOTE_DIR="~/algorithmic-institutions"
REMOTE_PROJECT_DIR="${AI_REMOTE_DIR:-${CANONICAL_REMOTE_DIR}}"
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
    local artifacts_opts=(--exclude='artifacts/')
    if [[ "${REMOTE_PROJECT_DIR}" != "${CANONICAL_REMOTE_DIR}" ]]; then
        # Isolated experiment dir: create it and ship the small AH
        # artifacts its configs reference (manager artifacts not needed).
        ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_PROJECT_DIR} \
            && cp -n ${CANONICAL_REMOTE_DIR}/.env ${REMOTE_PROJECT_DIR}/ \
            2>/dev/null || true"
        artifacts_opts=(--exclude='artifacts/manager/')
    fi
    rsync -azP --delete \
        --filter='P /.env' \
        --filter=':- .gitignore' \
        --exclude='.git/' \
        --exclude='.venv/' \
        --exclude='.log/' \
        "${artifacts_opts[@]}" \
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
    sim_cmd+=" && source ${CANONICAL_REMOTE_DIR}/.venv/bin/activate"
    if [[ "${REMOTE_PROJECT_DIR}" == "${CANONICAL_REMOTE_DIR}" ]]; then
        sim_cmd+=" && uv run python"
    else
        # Shared venv + this dir's code. Raven's login shells set
        # SBATCH_EXPORT=NONE, so an exported variable does NOT reach the job
        # by itself -- SBATCH_EXPORT=ALL restores propagation, without which
        # run_simulation.sh falls back to the isolated dir's non-existent
        # .venv and python imports the SHARED checkout's editable install
        # instead of this dir's src/ (silently voiding the isolation).
        sim_cmd+=" && export AIMANAGER_VENV=${CANONICAL_REMOTE_DIR}/.venv"
        sim_cmd+=" PYTHONPATH=${REMOTE_PROJECT_DIR}/src"
        sim_cmd+=" SBATCH_EXPORT=ALL"
        sim_cmd+=" && python"
    fi
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
