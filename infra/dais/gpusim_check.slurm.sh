#!/bin/bash -l
# The GPU sim image (Dockerfile.gpu-sim) on one H200: torch, CUDA and PyG on
# the GPU, the trained models (saved under torch 1.11) loaded and run with
# finite outputs, and the smoke sweep of 2026-11-27 through the repo's own
# entry point. Fails on the first check that fails.
#
#   cd /u/$USER/algorithmic-institutions-gpusim
#   sbatch -o $GS_SCRATCH/logs/%x.%j.out infra/dais/gpusim_check.slurm.sh
#
#SBATCH -J gpusim_check
#SBATCH --account=mpib_inst
#SBATCH --partition=gpudev
#SBATCH --qos=g0001
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=00:15:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/gpusim_common.sh"
use_image new
cd "$PF_REPO"
OUT="$GS_SCRATCH/runs/check/$SLURM_JOB_ID"
mkdir -p "$OUT"
# the other DAIS checkout's smoke sweep (read only), as the pipeline ran it
SMOKE_REPO="${SMOKE_REPO:-/u/$USER/algorithmic-institutions-pf}"
SMOKE=pfsmoke-20261127-1

echo "== node and image"
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
sha256sum "$SIM_SIF"
git -C "$PF_REPO" rev-parse HEAD

echo "== torch, CUDA and PyG on the GPU"
in_sim_image python infra/dais/sim_variant.py check-gpu

echo "== the models on the GPU: a small sim's model inputs, replayed"
in_sim_image python infra/dais/gpusim_forward.py capture \
  infra/dais/gpusim_configs/final1_j0.yml "$OUT/capture.pkl" --episodes 64
for dev in cuda cpu; do
  in_sim_image python infra/dais/gpusim_forward.py replay \
    infra/dais/gpusim_configs/final1_j0.yml "$OUT/capture.pkl" \
    "$OUT/replay_$dev.npz" --device "$dev"
done
in_sim_image python infra/dais/gpusim_forward.py compare \
  "$OUT/replay_cuda.npz" "$OUT/replay_cpu.npz"

echo "== the smoke sweep (Raven, old image: s000 65.373 +- 10.063 at 20 episodes)"
cp "$SMOKE_REPO/configs/managers/rule_based/$SMOKE.yml" "$OUT/"
sed "s#configs/managers/rule_based/$SMOKE.yml#$OUT/$SMOKE.yml#" \
  "$SMOKE_REPO/configs/simulation/policy_finder/${SMOKE}_sweep.yml" > "$OUT/smoke.yml"
run_config "$OUT/smoke.yml" "$OUT/smoke"
cat "$OUT/smoke/sim/sweep.json"
echo "CHECK OK"
