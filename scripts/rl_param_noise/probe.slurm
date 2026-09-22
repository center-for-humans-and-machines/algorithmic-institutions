#!/bin/bash -l
#
# The pre-launch probe (scripts/rl_param_noise/probe.py) as a batch job.
#
# It runs the real stack -- three GNN artificial humans, the linear opponent
# and the manager -- over a sigma sweep, which is far too slow on a shared
# login node. Short: minutes, not hours.
#
#   sbatch scripts/rl_param_noise/probe.slurm <config> <out.json>
#
#SBATCH --chdir=.
#SBATCH --output=temp/probe.%j.log
#SBATCH --error=temp/probe.%j.log
#SBATCH --job-name=pnoise_probe
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --time=00:30:00

set -e

source "${AIMANAGER_VENV:-.venv}/bin/activate"
module load cuda/11.4

python scripts/rl_param_noise/probe.py "$1" \
    --batch-size 256 --device cuda --out "$2"
