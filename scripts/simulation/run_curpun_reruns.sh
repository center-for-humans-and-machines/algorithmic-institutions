#!/usr/bin/env bash
#
# punisher-current-contribution re-baseline: submit, fetch and evaluate the
# `_curpun` reruns (cases a-e in
# notes/autoresearch_log/punisher-current-contribution-cases.md).
#
# Usage:
#   scripts/simulation/run_curpun_reruns.sh [--dry-run] submit
#   scripts/simulation/run_curpun_reruns.sh [--dry-run] fetch
#   scripts/simulation/run_curpun_reruns.sh [--dry-run] evaluate
#
# `submit` syncs and sbatches every case through scripts/simulate_cluster.sh
# (one SLURM job per config, sbatch returns at once); wait for the jobs
# (ssh raven squeue -u certuer), then `fetch` pulls the sim dirs back and
# `evaluate` scores them and builds the before/after table
# (scripts/data_analysis/curpun_rebaseline.py). `--dry-run` prints the
# commands instead of running them; `--skip-gmlp` runs cases a, b, e only.
#
# Two code lineages are involved. Cases a, b and e run from this checkout.
# Cases c and d (gaussian_mlp contributors) unpickle classes that exist only
# on the gmlp lineage (origin/auto/contribution-inflated-gmlp), so they run
# from a second local checkout, CURPUN_GMLP_DIR, which must already carry the
# punisher fix (stage A's commits cherry-picked onto that lineage). The
# retrained punisher artifact and the two configs are copied into it here.
# Each lineage gets its own isolated remote dir.
#
# Environment:
#   CURPUN_REMOTE_DIR  isolated remote dir (default ~/repros/ai-runs/punisher-current-contr);
#                      the gmlp lineage uses "${CURPUN_REMOTE_DIR}-gmlp"
#   CURPUN_GMLP_DIR    local checkout of the gmlp lineage with the punisher fix
#                      (required for submit/fetch unless --skip-gmlp)
#   PYTHON             interpreter for the local evaluate step (default: python)
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE_DIR="${CURPUN_REMOTE_DIR:-~/repros/ai-runs/punisher-current-contr}"
GMLP_DIR="${CURPUN_GMLP_DIR:-}"
PYTHON="${PYTHON:-python}"
CONFIG_DIR="configs/simulation/manager_testing"
PUNISHER_COPULA="artifacts/baselines/punishment_multinomial_current_contr_severity_copula.joblib"

# case id | config basename | lineage (main = this checkout, gmlp = CURPUN_GMLP_DIR)
CASES=(
    "a_vnode|23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch_curpun|main"
    "b_skip|23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_curpun|main"
    "c_infl|23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus_k_onehot_switch_curpun|gmlp"
    "d_kexo|23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus_k_onehot_switch_curpun|gmlp"
    "e_main|23_2g8a_self_gnn_contr_gnn_switch_curpun|main"
)

DRY_RUN=false
SKIP_GMLP=false
PHASE=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --skip-gmlp) SKIP_GMLP=true ;;
        submit|fetch|evaluate) PHASE="$arg" ;;
        *) echo "unknown argument: $arg" >&2; exit 1 ;;
    esac
done
[[ -n "$PHASE" ]] || { sed -n '2,30p' "$0"; exit 1; }

info() { echo "==> $*"; }
run() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "+ $*"
    else
        "$@"
    fi
}

lineage_dir() { [[ "$1" == gmlp ]] && echo "$GMLP_DIR" || echo "$ROOT"; }
lineage_remote() { [[ "$1" == gmlp ]] && echo "${REMOTE_DIR}-gmlp" || echo "$REMOTE_DIR"; }
use_case() { [[ "$SKIP_GMLP" == true && "$1" == gmlp ]] && return 1 || return 0; }

check_gmlp() {
    [[ "$SKIP_GMLP" == true ]] && return 0
    if [[ -z "$GMLP_DIR" || ! -f "$GMLP_DIR/scripts/simulate_cluster.sh" ]]; then
        echo "ERROR: CURPUN_GMLP_DIR='${GMLP_DIR}' is not a checkout of the gmlp lineage" >&2
        echo "       (origin/auto/contribution-inflated-gmlp + the punisher fix);" >&2
        echo "       set it, or pass --skip-gmlp to run cases a, b and e only." >&2
        [[ "$DRY_RUN" == true ]] || exit 1
        GMLP_DIR="${GMLP_DIR:-<CURPUN_GMLP_DIR>}"
    fi
}

stage_gmlp() {
    # The gmlp checkout needs the two configs and the retrained punisher.
    [[ "$SKIP_GMLP" == true ]] && return 0
    info "Staging configs and punisher artifact into ${GMLP_DIR}"
    for entry in "${CASES[@]}"; do
        IFS='|' read -r _ cfg lineage <<< "$entry"
        [[ "$lineage" == gmlp ]] || continue
        run cp "$ROOT/$CONFIG_DIR/$cfg.yml" "$GMLP_DIR/$CONFIG_DIR/$cfg.yml"
    done
    run cp "$ROOT/$PUNISHER_COPULA" "$GMLP_DIR/$PUNISHER_COPULA"
}

submit() {
    check_gmlp
    stage_gmlp
    local synced_main=false synced_gmlp=false
    for entry in "${CASES[@]}"; do
        IFS='|' read -r case_id cfg lineage <<< "$entry"
        use_case "$lineage" || continue
        local dir remote sync_flag=()
        dir="$(lineage_dir "$lineage")"
        remote="$(lineage_remote "$lineage")"
        if [[ "$lineage" == gmlp ]]; then
            [[ "$synced_gmlp" == true ]] && sync_flag=(--no-sync); synced_gmlp=true
        else
            [[ "$synced_main" == true ]] && sync_flag=(--no-sync); synced_main=true
        fi
        info "submit $case_id ($lineage lineage, remote $remote)"
        run env AI_REMOTE_DIR="$remote" "$dir/scripts/simulate_cluster.sh" \
            ${sync_flag[@]+"${sync_flag[@]}"} "$CONFIG_DIR/$cfg.yml"
    done
    info "Jobs submitted; check with: ssh raven squeue -u certuer"
}

fetch() {
    check_gmlp
    for entry in "${CASES[@]}"; do
        IFS='|' read -r case_id cfg lineage <<< "$entry"
        use_case "$lineage" || continue
        local dir remote
        dir="$(lineage_dir "$lineage")"
        remote="$(lineage_remote "$lineage")"
        info "fetch $case_id -> $ROOT/plots/simulation/$cfg"
        # explicit destination: results from both lineages land in this checkout
        run env AI_REMOTE_DIR="$remote" "$dir/scripts/fetch_cluster.sh" \
            "plots/simulation/$cfg" "$ROOT/plots/simulation"
    done
}

evaluate() {
    cd "$ROOT"
    for entry in "${CASES[@]}"; do
        IFS='|' read -r case_id cfg lineage <<< "$entry"
        use_case "$lineage" || continue
        if [[ "$DRY_RUN" == false && ! -f "plots/simulation/$cfg/per_round.parquet" ]]; then
            echo "skip $case_id: plots/simulation/$cfg/per_round.parquet missing" >&2
            continue
        fi
        info "evaluate $case_id"
        run $PYTHON -m aimanager evaluate "$CONFIG_DIR/$cfg.yml"
    done
    info "before/after table"
    run $PYTHON scripts/data_analysis/curpun_rebaseline.py
}

case "$PHASE" in
    submit) submit ;;
    fetch) fetch ;;
    evaluate) evaluate ;;
esac
