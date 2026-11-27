#!/bin/bash -l
# The policy-finder pipeline inside one CPU job, its sims in a local pool on
# the job's cores (PF_CLUSTER_BACKEND=local, scripts/policy_finder/local_cluster.py):
# one single-threaded sim per hyperthread, pinned, in the unchanged sim image.
#
#   cd <checkout>    # see "The pipeline on DAIS" in infra/dais/README.md
#   sbatch infra/dais/pipeline.slurm.sh instance <name> [new_instance.sh options]
#   sbatch infra/dais/pipeline.slurm.sh resume <name>
#   sbatch infra/dais/pipeline.slurm.sh campaign <campaign.yml>
#   sbatch infra/dais/pipeline.slurm.sh python <script> [args]   # e.g. rescore.py
#
# State (worktrees, state files, campaign state, snapshots, the pool's jobs)
# lives in PF_RUNS (default $PF_SCRATCH/pf_runs), so a later job resumes a
# version or campaign where an earlier one stopped. The first PF_RESERVE_CORES
# cores (default 2) run the driver (git, check_instance.sh, the GP fit); the
# rest are the pool. The job exits non-zero unless the version (or campaign,
# or script) succeeded.
#
#SBATCH -J pfpipe
#SBATCH -o /dais/fs/scratch/%u/policy-finder-pipe/logs/%x.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=small
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200000
#SBATCH --time=02:00:00

set -euo pipefail
MODE="${1:?instance, resume, campaign or python}"
shift
export PF_SCRATCH="${PF_SCRATCH:-/dais/fs/scratch/$USER/policy-finder-pipe}"
export PF_REPO="${PF_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PF_REPO/infra/dais/common.sh"
init_run
cd "$PF_REPO"

export PF_RUNS="${PF_RUNS:-$PF_SCRATCH/pf_runs}"
export PF_CLUSTER_BACKEND=local
export PF_SIM_SIF="$SIM_SIF"
export PF_ARTIFACTS="${PF_ARTIFACTS:-$PF_SCRATCH/artifacts}"
export PF_PYTHON="${PF_PYTHON:-$PF_SCRATCH/venv/bin/python}"
export PF_BASE="${PF_BASE:-$(git rev-parse --abbrev-ref HEAD)}"
export PF_POOL_LOCKS="$PF_RUN/tmp/pool-locks"
mkdir -p "$PF_RUNS"

[ -x "$PF_PYTHON" ] || { echo "FATAL: no driver python at $PF_PYTHON" >&2; exit 1; }
[ -d "$PF_ARTIFACTS/artificial_humans" ] \
  || { echo "FATAL: no model files in $PF_ARTIFACTS" >&2; exit 1; }
sha=$(sha256sum "$PF_SIM_SIF" | cut -d' ' -f1)
[ "$sha" = "$SIM_SIF_SHA256" ] || { echo "FATAL: $PF_SIM_SIF is $sha" >&2; exit 1; }

# the driver on the first cores; the pool on the rest, one slot per
# hyperthread (first one per core, then the cores' second ones)
RESERVE="${PF_RESERVE_CORES:-2}"
DRIVER_CPUS=$(python3 "$PF_HERE/cpu_slots.py" reserved "$RESERVE")
PF_POOL_CPUS=$( { python3 "$PF_HERE/cpu_slots.py" slots "$RESERVE" 1 | cut -d, -f1
                  python3 "$PF_HERE/cpu_slots.py" slots "$RESERVE" 1 | cut -s -d, -f2
                } | paste -sd, -)
export PF_POOL_CPUS
echo "job ${SLURM_JOB_ID:-local} on $(hostname), commit $(git rev-parse HEAD)"
echo "driver CPUs $DRIVER_CPUS; pool of $(tr ',' '\n' <<< "$PF_POOL_CPUS" | wc -l) slots;" \
  "PF_RUNS $PF_RUNS; base $PF_BASE"

case "$MODE" in
  instance) cmd=(scripts/policy_finder/new_instance.sh "$@") ;;
  resume) cmd=("$PF_PYTHON" scripts/policy_finder/pipeline.py run "$@") ;;
  campaign) cmd=("$PF_PYTHON" scripts/policy_finder/campaign.py run "$@") ;;
  python) cmd=("$PF_PYTHON" "$@") ;;
  *) echo "FATAL: mode is instance, resume, campaign or python, not $MODE" >&2; exit 2 ;;
esac
rc=0
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 taskset -c "$DRIVER_CPUS" "${cmd[@]}" || rc=$?
echo "exit $rc after $SECONDS s"
exit "$rc"
