#!/bin/bash -l
#
#SBATCH --workdir=.
#SBATCH --output={log_file}
#SBATCH --job-name={job_name}
#SBATCH --cpus-per-task 2
#SBATCH --mem 16GB
#SBATCH --gres=gpu
#SBATCH --partition gpu

set -e

module load python/3.10
module load cuda

source .venv/bin/activate

echo "Entered environment"

wandb agent {sweep_id} --project {project_name} --entity chm-hci
