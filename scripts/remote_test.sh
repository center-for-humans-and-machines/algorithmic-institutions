#!/usr/bin/env bash
#
# Remote test runner for algorithmic-institutions on Raven HPC.
#
# Prerequisites:
#   - SSH ControlMaster connection must be established first: ssh raven
#   - Remote .venv must exist at ~/algorithmic-institutions/.venv
#
# Usage:
#   scripts/remote_test.sh              # sync + test (all tests)
#   scripts/remote_test.sh --sync-only  # sync files only
#   scripts/remote_test.sh --test-only  # run tests only (no sync)
#   scripts/remote_test.sh -- -k test_encoder  # pass args to pytest
#   scripts/remote_test.sh --test-only -- -k "test_multi_group" -v
#
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
REMOTE_HOST="raven"
CANONICAL_REMOTE_DIR="~/algorithmic-institutions"
# Set AI_REMOTE_DIR to sync and test in an isolated experiment dir
# (shared venv + own code via PYTHONPATH; no venv of its own).
REMOTE_PROJECT_DIR="${AI_REMOTE_DIR:-${CANONICAL_REMOTE_DIR}}"
LOCAL_PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOCAL_PROJECT_DIR}/.claude/test-logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/test_${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/latest.log"

# ── Parse arguments ──────────────────────────────────────────────────
DO_SYNC=true
DO_TEST=true
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sync-only)
            DO_TEST=false
            shift
            ;;
        --test-only)
            DO_SYNC=false
            shift
            ;;
        --)
            shift
            PYTEST_ARGS=("$@")
            break
            ;;
        *)
            PYTEST_ARGS+=("$1")
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
    if [[ "${REMOTE_PROJECT_DIR}" != "${CANONICAL_REMOTE_DIR}" ]]; then
        ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_PROJECT_DIR}"
    fi
    rsync -azP --delete \
        --filter='P /.env' \
        --filter=':- .gitignore' \
        --exclude='.git/' \
        --exclude='.venv/' \
        --exclude='.log/' \
        --exclude='artifacts/' \
        --exclude='plots/' \
        --exclude='temp/' \
        --exclude='data/' \
        --exclude='__pycache__/' \
        --exclude='.claude/test-logs/' \
        "${LOCAL_PROJECT_DIR}/" \
        "${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/"
    info "Sync complete."
}

# ── Step 3: Run tests remotely ───────────────────────────────────────
run_tests() {
    info "Running tests on ${REMOTE_HOST}..."

    local pytest_cmd="cd ${REMOTE_PROJECT_DIR}"
    if [[ "${REMOTE_PROJECT_DIR}" == "${CANONICAL_REMOTE_DIR}" ]]; then
        pytest_cmd+=" && source .venv/bin/activate && python -m pytest"
    else
        pytest_cmd+=" && source ${CANONICAL_REMOTE_DIR}/.venv/bin/activate"
        pytest_cmd+=" && PYTHONPATH=${REMOTE_PROJECT_DIR}/src python -m pytest"
    fi

    if [[ ${#PYTEST_ARGS[@]} -eq 0 ]]; then
        pytest_cmd+=" src/ -v --tb=short"
    else
        pytest_cmd+=" ${PYTEST_ARGS[*]}"
    fi

    mkdir -p "${LOG_DIR}"

    local exit_code=0
    ssh "${REMOTE_HOST}" "bash -l -c '${pytest_cmd}'" 2>&1 | tee "${LOG_FILE}" || exit_code=${PIPESTATUS[0]}

    ln -sf "${LOG_FILE}" "${LATEST_LOG}"

    info "Test output saved to: ${LOG_FILE}"
    info "Latest log symlink: ${LATEST_LOG}"

    return ${exit_code}
}

# ── Main ─────────────────────────────────────────────────────────────
main() {
    check_ssh

    if [[ "${DO_SYNC}" == true ]]; then
        sync_files
    fi

    if [[ "${DO_TEST}" == true ]]; then
        run_tests
        exit_code=$?
        if [[ ${exit_code} -eq 0 ]]; then
            info "All tests passed."
        else
            error "Tests failed with exit code ${exit_code}."
        fi
        exit ${exit_code}
    fi

    info "Done."
}

main
