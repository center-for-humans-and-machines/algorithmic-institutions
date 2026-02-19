#!/bin/bash -l
#
#SBATCH --chdir=.
#SBATCH --output={log_file}
#SBATCH --error={log_file}
#SBATCH --job-name={job_id}
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#
#SBATCH --cpus-per-task {cores}
#SBATCH --mem {memory}GB
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --time=10:00:00

set -e

source .venv/bin/activate

module load cuda/11.4

# python src/aimanager/artificial_humans/train.py $CONFIG_PATH
{command}
