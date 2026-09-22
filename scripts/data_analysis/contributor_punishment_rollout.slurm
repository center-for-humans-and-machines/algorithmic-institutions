#!/bin/bash -l
#
# GPU SLURM job for the own-path rollout companion of the intervention probe.
# Wraps scripts/data_analysis/contributor_punishment_rollout.py, which
# forces every (contribution, punishment) cell on every context and needs one
# forward per (context, cell) batch -- real compute, so it goes through the
# batch system rather than a login node.
# Usage: sbatch scripts/data_analysis/contributor_punishment_intervention.slurm
#
#SBATCH --chdir=.
#SBATCH --output=temp/contrib_rollout_%j.log
#SBATCH --error=temp/contrib_rollout_%j.log
#SBATCH --job-name=contrib_ro
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#
#SBATCH --time=01:00:00

set -e

# Provenance-isolated run: submitted from, and --chdir=. runs it in, its own
# remote dir (~/autoresearch/contributor-punishment-targeting), which carries
# no venv of its own. Borrow the shared checkout's interpreter but keep THIS
# tree's code ahead of the editable install on sys.path.
PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "cwd        $PWD"
"$PY" -c 'import aimanager; print("aimanager ", aimanager.__file__)'

"$PY" scripts/data_analysis/contributor_punishment_rollout.py \
    --output-dir plots/data_analysis/contributor_punishment_targeting \
    "$@"
