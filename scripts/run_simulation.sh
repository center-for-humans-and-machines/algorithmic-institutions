#!/bin/bash -l
#
#SBATCH --chdir=.
#SBATCH --output=.log/aimanager_simulation_%j.out
#SBATCH --error=.log/aimanager_simulation_%j.out
#SBATCH --job-name=aimanager_simulation
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#
#SBATCH --cpus-per-task 2
#SBATCH --mem 16GB
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --time=01:00:00

CONFIG_PATH=${1:-"notebooks/test_manager/simulate_mixed/01_compare.yml"}

set -e

source .venv/bin/activate

module load cuda/11.4

python run.py run $CONFIG_PATH
