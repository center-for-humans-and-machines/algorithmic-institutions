#!/bin/bash -l
#
# CPU-only SLURM job for stamping the contribution copula artifact.
# Wraps make_contribution_copula_artifact.py, which unpickles the base GNN
# (torch_geometric) and runs the teacher-forced honesty check over the human
# train split -- real compute, so it goes through the batch system rather
# than a login node.
# Usage: sbatch scripts/artificial_humans/stamp_copula.slurm [BASE] [PARAMS] [OUT]
#   BASE    base contributor .pt       (default: the frontier group_vnode trunk)
#   PARAMS  copula_params.json to read (default: the frontier's)
#   OUT     stamped .pt to write       (default: the frontier artifact)
#
#SBATCH --chdir=.
#SBATCH --output=temp/stamp_copula_%j.log
#SBATCH --error=temp/stamp_copula_%j.log
#SBATCH --job-name=cg_copula_stamp
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#
#SBATCH --time=00:20:00

set -e

# Isolated remote dirs carry no venv of their own: borrow the shared
# checkout's interpreter but keep THIS tree's code first on sys.path.
PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "cwd        $PWD"
"$PY" -c 'import aimanager; print("aimanager ", aimanager.__file__)'

FRONTIER=artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode_herding_copula
ARTIFACT=architecture_node+edge+rnn__dataset_50ep__epochs_575.pt
BASE=${1:-artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode/model/$ARTIFACT}
PARAMS=${2:-$FRONTIER/calibration/copula_params.json}
OUT=${3:-$FRONTIER/model/$ARTIFACT}
echo "base       $BASE"
echo "params     $PARAMS"
echo "out        $OUT"

"$PY" scripts/artificial_humans/make_contribution_copula_artifact.py \
    --params "$PARAMS" \
    --base "$BASE" \
    --out "$OUT"
