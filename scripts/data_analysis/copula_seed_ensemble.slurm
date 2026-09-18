#!/bin/bash -l
#
# CPU-only SLURM job for the seed-ensemble vs copula measurement.
# Wraps scripts/data_analysis/copula_seed_ensemble.py: teacher-forced forward
# passes of the five seed trunks plus the shipped one over the human episodes,
# then the pairwise-likelihood copula fits (real compute, so batch system).
# Usage (from the isolated remote dir): sbatch scripts/data_analysis/copula_seed_ensemble.slurm
#
#SBATCH --chdir=.
#SBATCH --output=temp/copula_seed_ensemble_%j.log
#SBATCH --error=temp/copula_seed_ensemble_%j.log
#SBATCH --job-name=cop_ens
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#
#SBATCH --time=01:30:00

set -e

PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p temp

echo "cwd        $PWD"
"$PY" -c 'import aimanager; print("aimanager ", aimanager.__file__)'

"$PY" scripts/data_analysis/copula_seed_ensemble.py "$@"
