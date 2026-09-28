#!/bin/bash -l
#
# CPU-only SLURM job for the RCB step-0 measurement.
# Wraps scripts/data_analysis/rcb_teacher_forced.py, which unpickles the
# parent's bare GNN trunk (torch_geometric) and runs one teacher-forced
# forward pass over the human episodes -- real compute, so it goes through
# the batch system rather than a login node.
# Usage: sbatch scripts/data_analysis/rcb_teacher_forced.slurm
#
#SBATCH --chdir=.
#SBATCH --output=temp/rcb_teacher_forced_%j.log
#SBATCH --error=temp/rcb_teacher_forced_%j.log
#SBATCH --job-name=rcb_tf
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#
#SBATCH --time=00:20:00

set -e

# Provenance-isolated run: this job is submitted from, and --chdir=. runs it
# in, its own remote dir (~/autoresearch/contribution-punish-onehot), which
# carries no venv of its own. Borrow the shared checkout's interpreter but
# keep THIS tree's code ahead of the editable install on sys.path (the
# script itself also does sys.path.insert(0, ROOT / "src") off its own
# __file__).
PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "cwd        $PWD"
"$PY" -c 'import aimanager; print("aimanager ", aimanager.__file__)'

"$PY" scripts/data_analysis/rcb_teacher_forced.py
