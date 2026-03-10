#!/usr/bin/env bash
#
# Fetch files/folders from Raven HPC cluster to local.
#
# Prerequisites:
#   - SSH ControlMaster connection must be established first: ssh raven
#
# Usage:
#   scripts/fetch_cluster.sh <remote_path>
#
# The remote_path is relative to ~/algorithmic-institutions on Raven.
# Files are fetched to the same relative path locally.
#
# Examples:
#   scripts/fetch_cluster.sh artifacts/model.pt
#   scripts/fetch_cluster.sh artifacts/
#   scripts/fetch_cluster.sh configs/training/
#
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
REMOTE_HOST="raven"
REMOTE_PROJECT_DIR="~/algorithmic-institutions"
LOCAL_PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── Validate arguments ───────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    echo "Usage: scripts/fetch_cluster.sh <remote_path>" >&2
    echo "  remote_path is relative to ${REMOTE_PROJECT_DIR}" >&2
    exit 1
fi

REMOTE_PATH="$1"

# ── Helpers ──────────────────────────────────────────────────────────
info()  { echo "==> $*"; }
error() { echo "ERROR: $*" >&2; }

# ── Check SSH connection ─────────────────────────────────────────────
info "Checking SSH connection to ${REMOTE_HOST}..."
if ssh -O check "${REMOTE_HOST}" 2>/dev/null; then
    info "SSH ControlMaster connection is active."
else
    error "No active SSH ControlMaster connection to ${REMOTE_HOST}."
    error "Please run 'ssh raven' in a separate terminal first."
    exit 2
fi

# ── Ensure local parent directory exists ─────────────────────────────
LOCAL_TARGET="${LOCAL_PROJECT_DIR}/${REMOTE_PATH}"
mkdir -p "$(dirname "${LOCAL_TARGET}")"

# ── Fetch from cluster ───────────────────────────────────────────────
info "Fetching ${REMOTE_PATH} from ${REMOTE_HOST}..."
rsync -azP \
    "${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/${REMOTE_PATH}" \
    "${LOCAL_TARGET}"

info "Done. Fetched to ${LOCAL_TARGET}"
