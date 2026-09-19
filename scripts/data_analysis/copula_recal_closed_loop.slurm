#!/bin/bash -l
#
# Stage 1 (Raven) of the state-spread diagnostic for the copula
# recalibration: teacher-force the BARE stimulus-skip trunk over the human
# histories and over each arm's realised trajectories, writing tf_<arm>.parquet
# for the local --analyse pass. Real compute (unpickled GNN forward passes over
# 100-episode trajectories), so it goes through the batch system.
# Usage: sbatch scripts/data_analysis/copula_recal_closed_loop.slurm
#
#SBATCH --chdir=.
#SBATCH --output=temp/copula_recal_cl_%j.log
#SBATCH --error=temp/copula_recal_cl_%j.log
#SBATCH --job-name=copula_recal_cl
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#
# Pure CPU work, but the mpib_cpu QOS group limit was saturated for the whole
# session, so this runs on the gpu partition where the account had capacity.
# The GPU is unused.
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --time=00:40:00

set -e

PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "cwd        $PWD"
"$PY" -c 'import aimanager; print("aimanager ", aimanager.__file__)'

"$PY" -u scripts/data_analysis/copula_recal_closed_loop_variance.py --teacher-force
