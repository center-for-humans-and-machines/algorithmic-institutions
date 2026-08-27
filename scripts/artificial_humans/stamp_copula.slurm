#!/bin/bash -l
#
# CPU-only SLURM job for stamping the contribution copula artifact.
# Wraps make_contribution_copula_artifact.py, which unpickles the base GNN
# (torch_geometric) and runs the teacher-forced honesty check over the human
# train split -- real compute, so it goes through the batch system rather
# than a login node.
# Usage: sbatch scripts/artificial_humans/stamp_copula.slurm
#
#SBATCH --chdir=.
#SBATCH --output=temp/stamp_copula_%j.log
#SBATCH --error=temp/stamp_copula_%j.log
#SBATCH --job-name=cg_copula_stamp
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#
#SBATCH --time=01:00:00

set -e

# Provenance-isolated run: this job is submitted from, and --chdir=. runs it
# in, its own remote dir (~/iso_contr_herdcopar1_v2), which carries no venv of
# its own. Borrow the shared checkout's interpreter but keep THIS tree's code
# ahead of the editable install on sys.path (the script itself also does
# sys.path.insert(0, ROOT / "src") off its own __file__).
PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "cwd        $PWD"
"$PY" -c 'import aimanager; print("aimanager ", aimanager.__file__)'

"$PY" scripts/artificial_humans/make_contribution_copula_artifact.py
