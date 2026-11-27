#!/bin/bash -l
# The sim image on a DAIS node: network and Apptainer checks, torch and PyG on
# the GPU, then the two smoke reference sims of 2026-11-27 with the repo's own
# entry point on the CPUs (compare with Raven's committed sweep.json files;
# outputs go to the run folder, never into the checkout).
#
#   cd /u/$USER/algorithmic-institutions-pf   # the smoke commit + infra/dais
#   sbatch infra/dais/sim_check.slurm.sh
#
# Job 986771 (2026-11-27) ran an earlier version of this script, which also
# timed a few episode_batch_size values with the smoke rule (punishes
# nothing) and wrote the reference outputs into the checkout.
#
#SBATCH -J pf_simcheck
#SBATCH -o /dais/fs/scratch/%u/policy-finder/logs/%x.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=gpudev
#SBATCH --qos=g0001
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=00:15:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/common.sh"
init_run
cd "$PF_REPO"

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

echo "== torch and PyG on the GPU (expected to fail on H200/B200)"
SIM_GPU=0 in_sim_image python /opt/dais/sim_variant.py check-gpu || echo "GPU check failed"

echo "== reference sims on the CPUs (Raven: s000 65.373; final gp 55.313, s000 55.313)"
FAILS=0
for cfg in pfsmoke-20261127-1_sweep pfsmoke-20261127-1_sweep_final; do
  in_sim_image python /opt/dais/sim_variant.py config \
    "configs/simulation/policy_finder/$cfg.yml" "$PF_RUN/$cfg.yml" \
    --output-dir "$PF_RUN/out/$cfg"
  simulate "$PF_RUN/$cfg.yml" "$cfg" || FAILS=$((FAILS + 1))
  cat "$PF_RUN/out/$cfg/sweep.json" || FAILS=$((FAILS + 1))
done
[ "$FAILS" = 0 ]
