#!/bin/bash -l
#
# CPU-only SLURM job for stamping the recalibrated switch-herding-copula grid
# artifacts. Wraps make_switch_copula_recal_artifacts.py, which unpickles the
# base GNN (torch_geometric) and verifies each of the frozen grid's four arms
# -- real compute, so it goes through the batch system rather than a login
# node.
# Usage: sbatch scripts/artificial_humans/stamp_copula_recal.slurm
#
#SBATCH --chdir=.
#SBATCH --output=temp/stamp_copula_recal_%j.log
#SBATCH --error=temp/stamp_copula_recal_%j.log
#SBATCH --job-name=switch_copula_recal_stamp
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#
#SBATCH --time=01:00:00

set -e

# Provenance-isolated run: this job is submitted from, and --chdir=. runs it
# in, its own remote dir (~/autoresearch/switch-herding-copula-recal), which
# carries no venv of its own. Borrow the shared checkout's interpreter but
# keep THIS tree's code ahead of the editable install on sys.path (the
# script itself also does sys.path.insert(0, ROOT / "src") off its own
# __file__).
PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "cwd        $PWD"
"$PY" -c 'import aimanager; print("aimanager ", aimanager.__file__)'

"$PY" scripts/artificial_humans/make_switch_copula_recal_artifacts.py
