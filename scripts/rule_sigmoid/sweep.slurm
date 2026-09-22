#!/bin/bash -l
#
# One GPU job for one sigmoid-rule sweep, run provenance-isolated.
#
# --chdir=. runs the job in the directory it was submitted from, which for
# this arm is an isolated remote dir carrying no venv of its own: the shared
# checkout supplies only the interpreter, THIS tree supplies the code.
#
# PYTHONPATH=$PWD/src is not optional: without it the interpreter imports
# `aimanager` from the shared checkout and the job silently measures someone
# else's code.
#
# Usage: sbatch scripts/rule_sigmoid/sweep.slurm <design.csv> <out_dir> \
#                                                <seeds> <episodes> \
#                                                [rival] [shard] [chunk]
#
#SBATCH --chdir=.
#SBATCH --output=temp/sigsweep_%j.log
#SBATCH --error=temp/sigsweep_%j.log
#SBATCH --job-name=sigsweep
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --time=12:00:00

set -e

module load cuda/11.4

PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

DESIGN="${1:?usage: sbatch sweep.slurm <design.csv> <out> <seeds> <episodes> ...}"
OUT="${2:?missing out dir}"
SEEDS="${3:-42,43}"
EPISODES="${4:-512}"
RIVAL="${5:-clone}"
SHARD="${6:-128}"
CHUNK="${7:-16}"

echo "===== PROVENANCE ====="
echo "cwd        $PWD"
echo "job        ${SLURM_JOB_ID:-none} on $(hostname)"
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

import torch as th

sys.path.insert(0, "scripts/rule_sigmoid")
from sweep import DEFAULT_STACK  # noqa: E402

import aimanager  # noqa: E402

print("aimanager  ", aimanager.__file__)
print("torch      ", th.__version__, "cuda", th.cuda.is_available())
if th.cuda.is_available():
    print("gpu        ", th.cuda.get_device_name(0))
print("--- artifact sha256 ---")
for label, p in DEFAULT_STACK.items():
    with open(p, "rb") as f:
        print(f"{hashlib.sha256(f.read()).hexdigest()}  {label}  {p}")
PYEOF

echo "===== SWEEP ====="
"$PY" scripts/rule_sigmoid/sweep.py \
    --design "$DESIGN" --out "$OUT" --seeds "$SEEDS" --episodes "$EPISODES" \
    --rival "$RIVAL" --shard-size "$SHARD" --chunk "$CHUNK" --device cuda
