#!/bin/bash -l
#
# CPU-only SLURM job for the contribution copula calibration (rho and phi).
# Estimation only -- no artifact is written; the copula fields are stamped
# onto a copy of the base model by make_contribution_copula_artifact.py.
# Usage: sbatch scripts/artificial_humans/calibrate_copula.slurm
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

# Provenance-isolated run: this job is submitted from, and --chdir=. runs it
# in, its own remote dir (~/iso_contr_herdcopar1_v2), which carries no venv of
# its own. Borrow the shared checkout's interpreter but keep THIS tree's code
# ahead of the editable install on sys.path (the script itself also does
# sys.path.insert(0, ROOT / "src") off its own __file__).
PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "cwd        $PWD"
"$PY" -c 'import aimanager; print("aimanager ", aimanager.__file__)'

BASE=artifacts/artificial_humans/group_switching_contribution_50ep/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt
PARAMS=artifacts/artificial_humans/group_switching_contribution_50ep_herding_copula_v2/calibration/copula_params.json

"$PY" scripts/artificial_humans/contribution_copula_rho.py \
    --model "$BASE" \
    --roundtrip \
    --preflight \
    --write-params "$PARAMS"
