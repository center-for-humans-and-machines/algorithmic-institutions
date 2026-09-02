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

source "${{AIMANAGER_VENV:-.venv}}/bin/activate"

module load cuda/11.4

# Provenance, one line in every job log: which interpreter and which src/
# actually ran. An isolated experiment dir shares the canonical checkout's
# venv, whose editable install resolves `aimanager` to THAT checkout, so a
# job can silently simulate the shared tree's code instead of the branch's
# (PR #167 note 8; four of PR #168's runs). A print cannot move a score.
python -c "import sys, aimanager; print('PROVENANCE', sys.executable, aimanager.__file__)"

python -m aimanager simulate {config_path}
