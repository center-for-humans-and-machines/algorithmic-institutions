#!/bin/bash -l
# The GPU sim image (Dockerfile.gpu-sim) on one H200: torch, CUDA and PyG on
# the GPU, the trained models (saved under torch 1.11) loaded and run with
# finite outputs, and the smoke sweep of 2026-11-27 through the repo's own
# entry point. Fails on the first check that fails.
#
#   cd /u/$USER/algorithmic-institutions-gpusim
#   sbatch -o $GS_SCRATCH/logs/%x.%j.out infra/dais/gpusim_check.slurm.sh
#   EXTRA_CAPTURE=<capture.pkl from another image> sbatch ...  (optional)
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

echo "== the models: a small sim's model inputs, replayed in both images"
# inputs captured here (new image, GPU) and, if given, elsewhere (EXTRA_CAPTURE)
CFG=infra/dais/gpusim_configs/final1_j0.yml
in_sim_image python infra/dais/gpusim_forward.py capture "$CFG" \
  "$OUT/capture_here.pkl" --episodes 64
captures=("$OUT/capture_here.pkl")
if [ -n "${EXTRA_CAPTURE:-}" ]; then
  cp "$EXTRA_CAPTURE" "$OUT/capture_extra.pkl"
  captures+=("$OUT/capture_extra.pkl")
fi
for cap in "${captures[@]}"; do
  r="${cap%.pkl}"
  use_image new
  for dev in cuda cpu; do
    in_sim_image python infra/dais/gpusim_forward.py replay "$CFG" "$cap" \
      "${r}_new_$dev.npz" --device "$dev"
  done
  use_image old  # torch 1.11, on this node's CPU
  in_sim_image python infra/dais/gpusim_forward.py replay "$CFG" "$cap" \
    "${r}_old_cpu.npz" --device cpu
  for pair in "old_cpu new_cpu" "new_cpu new_cuda" "old_cpu new_cuda"; do
    set -- $pair
    echo "-- $(basename "$r"): $1 vs $2"
    in_sim_image python infra/dais/gpusim_forward.py compare \
      "${r}_$1.npz" "${r}_$2.npz"
  done
done
use_image new

echo "== the smoke sweep (Raven, old image: s000 65.373 +- 10.063 at 20 episodes)"
cp "$SMOKE_REPO/configs/managers/rule_based/$SMOKE.yml" "$OUT/"
sed "s#configs/managers/rule_based/$SMOKE.yml#$OUT/$SMOKE.yml#" \
  "$SMOKE_REPO/configs/simulation/policy_finder/${SMOKE}_sweep.yml" > "$OUT/smoke.yml"
run_config "$OUT/smoke.yml" "$OUT/smoke"
cat "$OUT/smoke/sim/sweep.json"
echo "CHECK OK"
