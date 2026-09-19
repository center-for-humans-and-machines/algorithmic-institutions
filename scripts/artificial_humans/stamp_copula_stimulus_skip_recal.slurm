#!/bin/bash -l
#
# CPU-only SLURM job for stamping the RECALIBRATED contribution copula
# artifact (auto/contribution-copula-recalibrated, step 3).
# Wraps make_contribution_copula_artifact.py, which copies the base GNN's
# artifact dict, inserts the three copula fields, and verifies every
# pre-existing key bit-identically plus the teacher-forced train-split
# probabilities -- real compute, so it goes through the batch system.
# Usage: sbatch scripts/artificial_humans/stamp_copula_stimulus_skip_recal.slurm
#
#SBATCH --chdir=.
#SBATCH --output=temp/stamp_copula_recal_%j.log
#SBATCH --error=temp/stamp_copula_recal_%j.log
#SBATCH --job-name=cg_copula_stamp_recal
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#
# Pure CPU work, but the mpib_cpu QOS group limit was saturated for the whole
# session, so this runs on the gpu partition where the account had capacity.
# The GPU is unused.
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --time=00:20:00

set -e

PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "cwd        $PWD"
"$PY" -c 'import aimanager; print("aimanager ", aimanager.__file__)'

DIR=artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula_recal
BASE=artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt
PARAMS=$DIR/calibration/copula_params.json
OUT=$DIR/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt

"$PY" scripts/artificial_humans/make_contribution_copula_artifact.py \
    --params "$PARAMS" \
    --base "$BASE" \
    --out "$OUT"

echo "=== stamped vs the shipped frontier artifact ==="
sha256sum "$OUT" \
    artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt
