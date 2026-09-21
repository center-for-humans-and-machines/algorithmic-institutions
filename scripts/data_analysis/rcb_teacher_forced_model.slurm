#!/bin/bash -l
#
# `rcb_teacher_forced.slurm` with the model path taken as an argument, so a
# candidate trunk can be screened against the current one without editing a
# script between the two runs. An optional second argument is passed through
# as `--sim-parquet`, the script's sim mode. Everything else -- the
# interpreter, the PYTHONPATH wiring for an isolated remote dir, the
# measurement itself -- is identical to the sibling job.
# Usage: sbatch scripts/data_analysis/rcb_teacher_forced_model.slurm \
#            <model.pt> [per_round.parquet]
#
# NOTE on sim mode: the script's (C) self-check compares the OBSERVED sim
# slopes against PR #179's parent run, hard-coded in `SIM_REF_SLOPES`, and
# exits 2 when they differ. Screening over any other simulation's parquet
# therefore exits 2 by construction; the measurement above it is still valid
# and the observed column is the one to read.
#
#SBATCH --chdir=.
#SBATCH --output=temp/rcb_teacher_forced_%j.log
#SBATCH --error=temp/rcb_teacher_forced_%j.log
#SBATCH --job-name=rcb_tf
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#
#SBATCH --time=00:20:00

set -e

MODEL="$1"
SIM="${2:-}"
if [[ -z "$MODEL" ]]; then
    echo "usage: sbatch $0 <model.pt> [per_round.parquet]" >&2
    exit 2
fi

PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "cwd        $PWD"
echo "model arg  $MODEL"
echo "sim arg    ${SIM:-<human data>}"
"$PY" -c 'import aimanager; print("aimanager ", aimanager.__file__)'

if [[ -n "$SIM" ]]; then
    "$PY" scripts/data_analysis/rcb_teacher_forced.py \
        --model "$MODEL" --sim-parquet "$SIM"
else
    "$PY" scripts/data_analysis/rcb_teacher_forced.py --model "$MODEL"
fi
