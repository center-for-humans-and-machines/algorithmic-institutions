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
#SBATCH --cpus-per-task 2
#SBATCH --mem 16GB
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --time=20:00:00

set -e

source .venv/bin/activate

module load cuda/11.4

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi
export WANDB_RUN_GROUP={experiment_name}
export WANDB_NAME={job_id}

python -m aimanager train-manager {config_path}
