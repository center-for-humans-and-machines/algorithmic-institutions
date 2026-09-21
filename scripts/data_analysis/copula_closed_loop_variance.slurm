#!/bin/bash -l
#
# CPU-only SLURM job for the closed-loop variance experiment's Raven stage:
# teacher-forces the bare stimulus-skip trunk over the human data and over
# the realised trajectories of the three arms (A copula, B none, C phi = 0)
# and writes tf_<arm>.parquet under
# plots/data_analysis/evaluation/copula_closed_loop/. Four unpickled GNN
# forward passes, so it goes through the batch system, not a login node.
# Usage: sbatch scripts/data_analysis/copula_closed_loop_variance.slurm
#
#SBATCH --chdir=.
#SBATCH --output=temp/copula_cl_tf_%j.log
#SBATCH --error=temp/copula_cl_tf_%j.log
#SBATCH --job-name=copula_cl_tf
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#
#SBATCH --time=00:30:00

set -e

# Provenance-isolated run: submitted from, and --chdir=. run in, its own
# remote dir (~/repros/ai-runs/copula-cl-variance), which carries no venv.
# Borrow the shared checkout's interpreter, keep THIS tree's code first.
PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "cwd        $PWD"
"$PY" -c 'import aimanager; print("aimanager ", aimanager.__file__)'
"$PY" scripts/data_analysis/copula_closed_loop_variance.py --teacher-force
