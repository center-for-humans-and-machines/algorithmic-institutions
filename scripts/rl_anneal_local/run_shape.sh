#!/bin/bash -l
#
# The policy-shape guard for both pilots, as a batch job.
#
# A batch-1000 rollout against the GNN clones is a GPU job; on the login node
# it would take longer than the training that produced the manager.
#
#   sbatch scripts/rl_anneal_local/run_shape.sh
#
#SBATCH --chdir=.
#SBATCH --output=.log/rl_anneal_local_shape.log
#SBATCH --error=.log/rl_anneal_local_shape.log
#SBATCH --job-name=anneal_shape
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem 16GB
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --time=00:30:00

set -e

source "${AIMANAGER_VENV:-.venv}/bin/activate"
module load cuda/11.4

OUT=plots/data_analysis/rl_anneal_local

for job in rl_anneal_local_pilot rl_anneal_control_pilot; do
    python scripts/rl_anneal_local/guard.py shape \
        "configs/training/rl_manager/${job}.yml" \
        "artifacts/manager/${job}/model/${job}_manager.pt" \
        --device cuda \
        --out "${OUT}/shape_${job}.csv"
done
