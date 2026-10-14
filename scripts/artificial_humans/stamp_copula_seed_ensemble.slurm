#!/bin/bash -l
#
# CPU-only SLURM job: carry the frozen stimulus-skip contribution copula
# (rho 0.0394986, phi_final 1.0, switch_every 1) onto each of PR #188's five
# seed-ensemble members, so every arm of the seed-spread measurement differs
# from the shipped frontier contributor in its trained weights and in nothing
# else. Nothing is recalibrated -- the parameters are frozen per model family
# (notes/autoresearch.md is silent, the punisher side does the same through
# `punishment_copula_rho.py --stamp-rho`).
# Usage: sbatch scripts/artificial_humans/stamp_copula_seed_ensemble.slurm
#
#SBATCH --chdir=.
#SBATCH --output=temp/stamp_copula_seed_ensemble_%j.log
#SBATCH --error=temp/stamp_copula_seed_ensemble_%j.log
#SBATCH --job-name=cg_copula_stamp_seeds
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#
#SBATCH --time=00:30:00

set -e

PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "cwd        $PWD"
"$PY" -c 'import aimanager; print("aimanager ", aimanager.__file__)'

ENS=artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_seed_ensemble
OUTDIR=artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_seed_ensemble_copula
SRC_PARAMS=artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/calibration/copula_params.json
STEM=architecture_node+edge+rnn__dataset_50ep__epochs_575

mkdir -p "$OUTDIR/model" "$OUTDIR/calibration"

for k in 1 2 3 4 5; do
    BASE="$ENS/model/${STEM}__seed_${k}.pt"
    PARAMS="$OUTDIR/calibration/copula_params__seed_${k}.json"
    OUT="$OUTDIR/model/${STEM}__seed_${k}.pt"
    echo "===== seed $k ====="
    "$PY" scripts/artificial_humans/carry_contribution_copula_params.py \
        --from "$SRC_PARAMS" --base "$BASE" --out "$PARAMS"
    "$PY" scripts/artificial_humans/make_contribution_copula_artifact.py \
        --params "$PARAMS" --base "$BASE" --out "$OUT"
done

echo "===== all five members stamped ====="

"$PY" scripts/data_analysis/seed_spread_verify_members.py
