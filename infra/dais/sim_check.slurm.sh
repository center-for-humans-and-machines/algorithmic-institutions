#!/bin/bash -l
# The sim image on one DAIS GPU: network and Apptainer checks, torch and PyG
# on the GPU, the two smoke reference sims of 2026-11-27 with the repo's own
# entry point (compare with Raven), then GPU memory and time for a few
# episode_batch_size values.
#
#   cd /u/$USER/algorithmic-institutions-pf   # the smoke commit + infra/dais
#   sbatch infra/dais/sim_check.slurm.sh                # H200
#   sbatch --gres=gpu:b200:1 infra/dais/sim_check.slurm.sh
#
#SBATCH -J pf_simcheck
#SBATCH -o /dais/fs/scratch/levinb/policy-finder/logs/sim_check.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=gpudev
#SBATCH --qos=g0001
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=00:15:00

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/common.sh"
cd "$PF_REPO"
OUT="$PF_SCRATCH/sim_check/$SLURM_JOB_ID"
mkdir -p "$OUT"
SWEEP=configs/simulation/policy_finder/pfsmoke-20261127-1_sweep.yml

echo "== node"
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== network from the compute node"
for u in https://huggingface.co https://openrouter.ai/api/v1/models https://pypi.org/simple/; do
  curl -sS -m 15 -o /dev/null -w "$u %{http_code}\n" "$u" || echo "$u FAILED"
done
echo "== apptainer: network namespace and cgroup limits inside a job"
for flags in "--net --network none" "--cpus 2" "--memory 2G" "--pids-limit 500"; do
  if err=$(apptainer exec $flags "$SIM_SIF" true 2>&1); then
    echo "$flags: works"
  else
    echo "$flags: refused: $(echo "$err" | grep -E 'ERROR|FATAL' | head -1)"
  fi
done

echo "== image checksum"
sha256sum "$SIM_SIF" | grep -q "$SIM_SIF_SHA256" && echo "matches Raven" || echo "DIFFERS"

echo "== torch and PyG on the GPU"
if ! in_sim_image python infra/dais/sim_variant.py check-gpu; then
  # torch 1.11+cu113 ships kernels up to sm_86 and no PTX; if this GPU is
  # newer, check the sims on the CPU instead (aimanager uses the GPU only
  # when torch sees one)
  echo "GPU check failed: the reference sims run on the CPU instead"
  export CUDA_VISIBLE_DEVICES=""
fi

echo "== reference sims, as the pipeline runs them (Raven: s000 65.373; final 55.313, 55.313)"
for cfg in pfsmoke-20261127-1_sweep pfsmoke-20261127-1_sweep_final; do
  simulate "configs/simulation/policy_finder/$cfg.yml" "$cfg"
done
cat plots/simulation/policy_finder/pfsmoke-20261127-1_sweep/sweep.json \
  plots/simulation/policy_finder/pfsmoke-20261127-1_sweep/final/sweep.json
cp -r plots/simulation/policy_finder/pfsmoke-20261127-1_sweep "$OUT/"

echo "== memory and time per batch size (points x episodes = one batch, then a real sweep)"
variant() {  # <label> <points> <episodes> <batch>
  in_sim_image python infra/dais/sim_variant.py config "$SWEEP" "$OUT/$1.yml" \
    --points "$2" --episodes "$3" --batch "$4" --output-dir "$OUT/$1"
  simulate "$OUT/$1.yml" "$1"
}
variant b100 2 50 100
variant b1000 2 500 1000
variant b4000 8 500 4000
variant sweep16x500_b1000 16 500 1000
