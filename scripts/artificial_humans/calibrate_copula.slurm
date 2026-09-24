#!/bin/bash -l
#
# CPU-only SLURM job for the contribution copula calibration (rho and phi).
# Estimation only -- no artifact is written; the copula fields are stamped
# onto a copy of the base model by stamp_copula.slurm.
# Usage: sbatch scripts/artificial_humans/calibrate_copula.slurm [BASE] [PARAMS]
#   BASE    base contributor .pt   (default: the frontier group_vnode trunk)
#   PARAMS  copula_params.json to write (default: the frontier's)
# Refitting rho alone? Run freeze_phi_in_params.py on the new PARAMS before
# stamping, or the frozen phi = 1.0 is silently replaced by the lag-1 estimate.
#
#SBATCH --chdir=.
#SBATCH --output=temp/calibrate_copula_%j.log
#SBATCH --error=temp/calibrate_copula_%j.log
#SBATCH --job-name=cg_copula_rho
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#
#SBATCH --time=02:00:00

set -e

# Isolated remote dirs carry no venv of their own: borrow the shared
# checkout's interpreter but keep THIS tree's code first on sys.path.
PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "cwd        $PWD"
"$PY" -c 'import aimanager; print("aimanager ", aimanager.__file__)'

FRONTIER=artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode_herding_copula
BASE=${1:-artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt}
PARAMS=${2:-$FRONTIER/calibration/copula_params.json}
echo "base       $BASE"
echo "params     $PARAMS"

"$PY" scripts/artificial_humans/contribution_copula_rho.py \
    --model "$BASE" \
    --roundtrip \
    --preflight \
    --write-params "$PARAMS"
