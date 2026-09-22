#!/bin/bash -l
#
# The episode-budget verification. Instruments ArtificialHumanEnv.step over a
# real (shortened) training call and divides by n_rounds -- counting `reset`
# would overstate the budget by one rollout per run, because the env's
# constructor resets once without playing an episode.
#
#SBATCH --chdir=.
#SBATCH --output=.log/rl_es_budget.log
#SBATCH --error=.log/rl_es_budget.log
#SBATCH --job-name=rl_es_budget
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task 2
#SBATCH --mem 16GB
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --time=00:30:00

set -e

source "${AIMANAGER_VENV:-.venv}/bin/activate"
module load cuda/11.4

python scripts/rl_es/pilot.py configs/training/rl_manager/rl_es_s42.yml \
    --mode budget \
    --budget-generations 6 \
    --out plots/data_analysis/evaluation/rl_manager_es
