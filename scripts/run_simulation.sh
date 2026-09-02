#!/bin/bash -l
#
#SBATCH --chdir=.
#SBATCH --output={log_file}
#SBATCH --error={log_file}
#SBATCH --job-name={job_id}
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#
#SBATCH --cpus-per-task 2
#SBATCH --mem 16GB
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --time=01:00:00

set -e

source .venv/bin/activate

# Isolated experiment dirs share the canonical checkout's venv, whose editable
# install resolves `aimanager` to THAT checkout -- so without this line a job
# submitted from an isolated dir silently runs the shared tree's code instead
# of the experiment's. --chdir=. puts us in the submitting dir, so $PWD/src is
# this experiment's source. Mirrors the preamble the stamping SLURM templates
# already use. No effect in the canonical checkout, where the two coincide.
export PYTHONPATH="$PWD/src${{PYTHONPATH:+:$PYTHONPATH}}"

module load cuda/11.4

python -m aimanager simulate {config_path}
