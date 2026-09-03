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
#SBATCH --cpus-per-task {cores}
#SBATCH --mem {memory}GB
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --time=10:00:00

set -e

source "${{AIMANAGER_VENV:-.venv}}/bin/activate"

# An isolated experiment dir shares the canonical checkout's venv, whose
# editable install resolves `aimanager` to THAT checkout, so without this
# the job would silently run the shared tree's code instead of this
# branch's. --chdir=. puts the job in the submitting dir, so $PWD/src is
# this experiment's source; no effect in the canonical checkout, where the
# two coincide.
export PYTHONPATH="$PWD/src${{PYTHONPATH:+:$PYTHONPATH}}"

module load cuda/11.4

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi
export WANDB_RUN_GROUP={experiment_name}
export WANDB_NAME={job_id}

# python src/aimanager/artificial_humans/train.py $CONFIG_PATH
python -c "import sys, aimanager; print('PROVENANCE', sys.executable, aimanager.__file__)"
{command}
