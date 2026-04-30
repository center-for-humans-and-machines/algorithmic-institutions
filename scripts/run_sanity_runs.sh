#!/usr/bin/env bash
#
# Run the sanity-snapshot fixtures on Raven.
#
# Submits the 3 AH-training fixtures and 1 simulation fixture with
# AIM_SANITY_LOG=1 + AIM_SANITY_LOG_FILE=<path> set, so each run writes a
# JSONL log under src/aimanager/tests/fixtures/_logs/. After this finishes,
# pull the logs back with `/fetch-cluster src/aimanager/tests/fixtures/_logs`.
#
# Prerequisites:
#   - SSH ControlMaster connection: ssh raven
#   - Remote .venv at ~/algorithmic-institutions/.venv
#
# Usage:
#   scripts/run_sanity_runs.sh             # sync + run all 4 fixtures
#   scripts/run_sanity_runs.sh --no-sync   # skip sync
#
set -euo pipefail

REMOTE_HOST="raven"
REMOTE_PROJECT_DIR="~/algorithmic-institutions"
LOCAL_PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE_DIR="src/aimanager/tests/fixtures"
LOG_DIR="${FIXTURE_DIR}/_logs"

DO_SYNC=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-sync) DO_SYNC=false; shift ;;
        *) shift ;;
    esac
done

info()  { echo "==> $*"; }
error() { echo "ERROR: $*" >&2; }

check_ssh() {
    info "Checking SSH connection to ${REMOTE_HOST}..."
    if ssh -O check "${REMOTE_HOST}" 2>/dev/null; then
        info "SSH ControlMaster connection is active."
    else
        error "No active SSH ControlMaster connection to ${REMOTE_HOST}."
        error "Please run 'ssh raven' in a separate terminal first."
        exit 2
    fi
}

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
        --exclude='notebooks/' \
        --exclude='temp/' \
        "${LOCAL_PROJECT_DIR}/" \
        "${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/"
    info "Sync complete."
}

run_fixture() {
    local subcmd="$1"  # train-ah | simulate
    local name="$2"    # e.g. tiny_contribution
    local log_file="${LOG_DIR}/${name}.actual.jsonl"
    info "Running fixture ${name} (${subcmd})..."
    ssh "${REMOTE_HOST}" "bash -l -c '\
        cd ${REMOTE_PROJECT_DIR} && \
        source .venv/bin/activate && \
        mkdir -p ${LOG_DIR} && \
        rm -f ${log_file} && \
        AIM_SANITY_LOG=1 \
        AIM_SANITY_LOG_FILE=${log_file} \
        uv run python -m aimanager ${subcmd} ${FIXTURE_DIR}/${name}.yml \
    '"
}

main() {
    check_ssh
    if [[ "${DO_SYNC}" == true ]]; then
        sync_files
    fi
    run_fixture train-ah tiny_contribution
    run_fixture train-ah tiny_punishment
    run_fixture train-ah tiny_switch
    run_fixture simulate tiny_simulation
    info "All fixtures done."
    info "Fetch with: scripts/fetch_cluster.sh ${LOG_DIR}"
}

main
