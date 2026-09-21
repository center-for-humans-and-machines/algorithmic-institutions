#!/bin/bash -l
#
# CPU-only SLURM job for the shared-signal interventional probe.
# Wraps `scripts/data_analysis/shared_signal_gain.py probe`, which unpickles
# the bare GNN contribution trunks (torch_geometric) and runs ~100 teacher-
# forced forward passes per model over the 50 human games -- real compute, so
# it goes through the batch system rather than a login node.
#
# Usage:
#   sbatch scripts/data_analysis/shared_signal_gain.slurm --seed-model-dir seedmodels
#   sbatch scripts/data_analysis/shared_signal_gain.slurm --one-round
#
#SBATCH --chdir=.
#SBATCH --output=temp/shared_signal_probe_%j.log
#SBATCH --error=temp/shared_signal_probe_%j.log
#SBATCH --job-name=shared_signal
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#
#SBATCH --time=02:00:00

set -e

# Provenance-isolated run: --chdir=. runs this in its own remote dir, which
# carries no venv. Borrow the shared checkout's interpreter but keep THIS
# tree's code ahead of the editable install on sys.path.
PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "cwd        $PWD"
echo "args       $*"
"$PY" -c 'import aimanager; print("aimanager ", aimanager.__file__)'

"$PY" scripts/data_analysis/shared_signal_gain.py probe "$@"
