#!/bin/bash -l
#
# CPU array job for one sigmoid-rule sweep.
#
# The GPU is not what makes this sweep fast. A rollout's cost is dominated by
# per-round Python and model-call overhead -- 24 rounds times a handful of
# forward passes -- and barely moves with the batch size, because the models
# are small (the contribution GNN is 35 KB). Measured on a login node: 192
# episodes in one rollout, 3 seconds, four threads. So the sweep runs on the
# `small` partition, where hundreds of nodes sit idle, instead of queueing
# behind the GPU arms this project also has running.
#
# Each array task takes one contiguous slice of the design (--part / --n-parts)
# and writes its own `episodes_p<part>_shard<k>.parquet` into the shared out
# dir; `aggregate.load_sweep` globs them all back together.
#
# Usage: sbatch --array=0-7 scripts/rule_sigmoid/sweep_cpu.slurm \
#                  <design.csv> <out_dir> <seeds> <episodes> \
#                  [rival] [shard] [chunk] [n_parts]
#
#SBATCH --chdir=.
#SBATCH --output=temp/sigcpu_%A_%a.log
#SBATCH --error=temp/sigcpu_%A_%a.log
#SBATCH --job-name=sigcpu
#
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#
#SBATCH --time=04:00:00

set -e

PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

DESIGN="${1:?usage: sbatch sweep_cpu.slurm <design.csv> <out> <seeds> <episodes> ...}"
OUT="${2:?missing out dir}"
SEEDS="${3:-42,43}"
EPISODES="${4:-512}"
RIVAL="${5:-clone}"
SHARD="${6:-128}"
CHUNK="${7:-16}"
NPARTS="${8:-${SLURM_ARRAY_TASK_COUNT:-1}}"
PART="${SLURM_ARRAY_TASK_ID:-0}"

echo "===== PROVENANCE ====="
echo "cwd        $PWD"
echo "job        ${SLURM_JOB_ID:-none} array ${SLURM_ARRAY_JOB_ID:-none}[$PART/$NPARTS]"
echo "host       $(hostname)  threads $OMP_NUM_THREADS"
echo "design     $DESIGN -> $OUT  seeds=$SEEDS episodes=$EPISODES rival=$RIVAL"
md5sum src/aimanager/manager/sigmoid_rule.py \
       src/aimanager/manager/paired_rollout.py \
       src/aimanager/manager/linear_opponent.py \
       src/aimanager/manager/environment.py \
       scripts/rule_sigmoid/sweep.py \
       "$DESIGN"

"$PY" - <<'PYEOF'
import hashlib
import sys

sys.path.insert(0, "scripts/rule_sigmoid")
from sweep import DEFAULT_STACK  # noqa: E402

print("--- artifact sha256 ---")
for label, p in DEFAULT_STACK.items():
    with open(p, "rb") as f:
        print(f"{hashlib.sha256(f.read()).hexdigest()}  {label}  {p}")
PYEOF

echo "===== SWEEP part $PART of $NPARTS ====="
"$PY" scripts/rule_sigmoid/sweep.py \
    --design "$DESIGN" --out "$OUT" --seeds "$SEEDS" --episodes "$EPISODES" \
    --rival "$RIVAL" --shard-size "$SHARD" --chunk "$CHUNK" \
    --part "$PART" --n-parts "$NPARTS" --device cpu
