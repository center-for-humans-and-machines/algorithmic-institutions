#!/bin/bash -l
#
# The plateau probe. The 40-generation smoke run drove mean punishment to
# exactly 0.000 by generation 10 and the whole population went with it, so the
# question that decides whether the full runs are worth 5 A100-hours is
# whether a perturbation can still move the policy off zero at THAT point --
# not at the initialisation, which is what pilot.py measured first.
#
#SBATCH --chdir=.
#SBATCH --output=.log/rl_es_plateau.log
#SBATCH --error=.log/rl_es_plateau.log
#SBATCH --job-name=rl_es_plateau
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
    --mode noise,shape \
    --suffix _plateau \
    --sigmas 0.02,0.05,0.1,0.2,0.4 \
    --noise-repeats 1 \
    --theta-from artifacts/manager/rl_es_pilot/model/rl_es_pilot_manager.pt \
    --out plots/data_analysis/evaluation/rl_manager_es
