#!/bin/bash -l
#
#SBATCH --chdir=.
#SBATCH --output=.log/rl_es_pilot.log
#SBATCH --error=.log/rl_es_pilot.log
#SBATCH --job-name=rl_es_pilot
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task 2
#SBATCH --mem 16GB
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --time=01:00:00

set -e

source "${AIMANAGER_VENV:-.venv}/bin/activate"
module load cuda/11.4

python scripts/rl_es/pilot.py configs/training/rl_manager/rl_es_pilot.yml \
    --mode all \
    --out plots/data_analysis/evaluation/rl_manager_es
