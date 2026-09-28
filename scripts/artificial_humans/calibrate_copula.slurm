#!/bin/bash -l
#
# CPU-only SLURM job for the contribution copula calibration.
# Usage: sbatch scripts/artificial_humans/calibrate_copula.slurm <model.pt> <out.pt>
#
#SBATCH --chdir=.
#SBATCH --output=temp/calibrate_copula_%j.log
#SBATCH --error=temp/calibrate_copula_%j.log
#SBATCH --job-name=cg_copula_rho
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#
#SBATCH --time=02:00:00

set -e

source .venv/bin/activate

python scripts/artificial_humans/contribution_copula_rho.py \
    --model "$1" --out "$2" --preflight --roundtrip
