#!/bin/bash -l
#
# CPU-only SLURM job for the contribution copula REFIT on the corrected
# serving path (auto/contribution-copula-recalibrated, step 2).
#
# Same script, same base trunk, same flags and same seed as
# calibrate_copula_stimulus_skip.slurm, which produced the shipped
# rho = 0.03949863621805423; only the tree it runs in differs (the parent's
# serving-path fix is in it) and the params JSON it writes.
# Estimation only -- no artifact is written here.
# Usage: sbatch scripts/artificial_humans/calibrate_copula_stimulus_skip_recal.slurm
#
#SBATCH --chdir=.
#SBATCH --output=temp/calibrate_copula_recal_%j.log
#SBATCH --error=temp/calibrate_copula_recal_%j.log
#SBATCH --job-name=cg_copula_rho_recal
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#
# The estimator is pure CPU (the model is loaded with device="cpu"), but the
# mpib_cpu QOS group limit was saturated for the whole session, so this runs
# on the gpu partition, where the account had capacity. The GPU is unused;
# only the allocation's 4 CPUs matter, exactly as on the `small` partition.
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --time=02:00:00

set -e

# Provenance-isolated run: this job is submitted from, and --chdir=. runs it
# in, its own remote dir (~/repros/ai-runs/copula-recal), which carries no
# venv of its own. Borrow the shared checkout's interpreter but keep THIS
# tree's code ahead of the editable install on sys.path.
PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "cwd        $PWD"
"$PY" -c 'import aimanager; print("aimanager ", aimanager.__file__)'

BASE=artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt
PARAMS=artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula_recal/calibration/copula_params.json

"$PY" scripts/artificial_humans/contribution_copula_rho.py \
    --model "$BASE" \
    --roundtrip \
    --preflight \
    --write-params "$PARAMS"
