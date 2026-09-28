#!/bin/bash -l
#
# The episode-budget guard, as a batch job.
#
# It is a count, not a benchmark -- the number of environment rollouts a run
# consumes does not depend on the device or the batch size -- but the login
# node routinely sits at load 35 with a hundred users on it, where six update
# steps take over half an hour. A compute node takes under a minute.
#
#   sbatch scripts/rl_anneal_local/run_budget.sh
#
#SBATCH --chdir=.
#SBATCH --output=.log/rl_anneal_local_budget.log
#SBATCH --error=.log/rl_anneal_local_budget.log
#SBATCH --job-name=anneal_budget
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

python scripts/rl_anneal_local/guard.py budget \
    configs/training/rl_manager/rl_anneal_local_s42.yml \
    --steps 6 --eval-period 2 --batch-size 4 \
    --out plots/data_analysis/rl_anneal_local/budget.json
