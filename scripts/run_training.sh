#!/bin/bash -l
#
#SBATCH --chdir=.
#SBATCH --output=.log/aimanager_training_%j.out
#SBATCH --error=.log/aimanager_training_%j.out
#SBATCH --job-name=aimanager_training
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
#SBATCH --time=03:00:00

CONFIG_PATH=${1:-"src/aimanager/rl_manager.yml"}

set -e

source .venv/bin/activate

module load cuda/11.4

python src/aimanager/rl_manager.py $CONFIG_PATH