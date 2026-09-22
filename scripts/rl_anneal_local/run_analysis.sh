#!/bin/bash -l
#
# Policy shape and the leaver/stayer targeting diagnostic, for all ten
# finished managers: five arm seeds and five control seeds.
#
# Both come out of one deterministic batch-1000 rollout per manager, so the
# targeting number and the shape table describe the same trajectories.
#
#   sbatch scripts/rl_anneal_local/run_analysis.sh
#
#SBATCH --chdir=.
#SBATCH --output=.log/rl_anneal_local_analysis.log
#SBATCH --error=.log/rl_anneal_local_analysis.log
#SBATCH --job-name=anneal_analysis
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem 16GB
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --time=00:20:00

set -e

source "${AIMANAGER_VENV:-.venv}/bin/activate"
module load cuda/11.4

OUT=plots/data_analysis/rl_anneal_local

for job in \
    rl_anneal_local_s42 rl_anneal_local_s43 rl_anneal_local_s44 \
    rl_anneal_local_s45 rl_anneal_local_s46 \
    rl_new_clones_s42 rl_new_clones_s43 rl_new_clones_s44 \
    rl_new_clones_s45 rl_new_clones_s46; do
    echo "=== ${job} ==="
    python scripts/rl_anneal_local/guard.py shape \
        "configs/training/rl_manager/${job}.yml" \
        "artifacts/manager/${job}/model/${job}_manager.pt" \
        --device cuda \
        --out "${OUT}/shape_${job}.csv" \
        --targeting-out "${OUT}/targeting_${job}.csv"
done
