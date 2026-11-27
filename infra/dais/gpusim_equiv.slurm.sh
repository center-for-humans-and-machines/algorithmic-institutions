#!/bin/bash -l
# Equivalence sims: each given config (infra/dais/gpusim_configs/) in one
# image, on the job's GPU or, without one, on its CPUs; per-episode scores
# to $GS_SCRATCH/runs/equiv/<image>_<device>/<config>/episodes.csv.
#
#   cd /u/$USER/algorithmic-institutions-gpusim
#   # new image on one H200
#   sbatch -p gpu1 --qos=g0001 --gres=gpu:h200:1 --cpus-per-task=16 \
#     infra/dais/gpusim_equiv.slurm.sh new infra/dais/gpusim_configs/final1_j0.yml ...
#   # old image on CPUs
#   sbatch -p small --qos=cpu --cpus-per-task=64 \
#     infra/dais/gpusim_equiv.slurm.sh old infra/dais/gpusim_configs/final1_j0.yml ...
#
#SBATCH -J gpusim_equiv
#SBATCH --account=mpib_inst
#SBATCH --ntasks=1
#SBATCH --mem=120000
#SBATCH --time=01:00:00

set -euo pipefail
IMAGE="${1:?old or new}"
shift
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/gpusim_common.sh"
use_image "$IMAGE"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
DEVICE=$([ -n "$CUDA_VISIBLE_DEVICES" ] && echo gpu || echo cpu)
cd "$PF_REPO"
echo "job $SLURM_JOB_ID on $(hostname): image $IMAGE ($SIM_SIF), device $DEVICE," \
  "$(nproc) cpus, commit $(git rev-parse HEAD)"
[ "$DEVICE" = gpu ] && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for cfg in "$@"; do
  run_config "$cfg" "$GS_SCRATCH/runs/equiv/${IMAGE}_$DEVICE/$(basename "$cfg" .yml)"
done
echo "EQUIV OK"
