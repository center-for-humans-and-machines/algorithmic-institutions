#!/bin/bash -l
#
# CPU-only SLURM job for the contribution group-latent persistence diagnostic.
# Usage: sbatch scripts/artificial_humans/latent_phi.slurm <model.pt> <out.json>
#
#SBATCH --chdir=.
#SBATCH --output=temp/latent_phi_%j.log
#SBATCH --error=temp/latent_phi_%j.log
#SBATCH --job-name=cg_latent_phi
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#
#SBATCH --time=03:00:00

set -e

source .venv/bin/activate

python scripts/artificial_humans/contribution_latent_phi.py \
    --model "$1" --out "$2" --holdout --roundtrip
