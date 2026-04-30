#!/usr/bin/env bash
#
# Submit the sanity-snapshot fixtures to Raven via the regular
# train_cluster.sh / simulate_cluster.sh entry points (i.e. real
# sbatch'd GPU jobs, exactly like /train and /simulate).
#
# Each fixture's exec.command sets AIM_SANITY_LOG=1 and
# AIM_SANITY_LOG_FILE=<path> so the sbatch'd python invocation writes
# a JSONL sanity log under src/aimanager/tests/fixtures/_logs/. After
# the SLURM jobs finish, pull the logs back with:
#   scripts/fetch_cluster.sh src/aimanager/tests/fixtures/_logs
#
# Prerequisites:
#   - SSH ControlMaster connection: ssh raven
#   - Remote .venv at ~/algorithmic-institutions/.venv
#
# Usage:
#   scripts/run_sanity_runs.sh             # sync once + submit all 4
#   scripts/run_sanity_runs.sh --no-sync   # submit only (no sync)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FIXTURE_DIR="src/aimanager/tests/fixtures"

DO_SYNC=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-sync) DO_SYNC=false; shift ;;
        *) shift ;;
    esac
done

info() { echo "==> $*"; }

if [[ "${DO_SYNC}" == true ]]; then
    info "Syncing once before submitting fixtures..."
    "${SCRIPT_DIR}/train_cluster.sh" --sync-only
fi

info "Submitting tiny_contribution (train-ah)..."
"${SCRIPT_DIR}/train_cluster.sh" --no-sync ah "${FIXTURE_DIR}/tiny_contribution.yml"

info "Submitting tiny_switch (train-ah)..."
"${SCRIPT_DIR}/train_cluster.sh" --no-sync ah "${FIXTURE_DIR}/tiny_switch.yml"

info "Submitting tiny_punishment (train-ah)..."
"${SCRIPT_DIR}/train_cluster.sh" --no-sync ah "${FIXTURE_DIR}/tiny_punishment.yml"

# tiny_simulation.yml is added in a follow-up commit alongside the
# simulation-side instrumentation. Skipped for now.
# info "Submitting tiny_simulation..."
# "${SCRIPT_DIR}/simulate_cluster.sh" --no-sync "${FIXTURE_DIR}/tiny_simulation.yml"

info "All fixtures submitted via sbatch. Watch with: ssh raven squeue -u certuer"
info "Once all SLURM jobs complete, fetch logs:"
info "  scripts/fetch_cluster.sh ${FIXTURE_DIR}/_logs"
