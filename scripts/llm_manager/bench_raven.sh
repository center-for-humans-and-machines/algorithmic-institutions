#!/usr/bin/env bash
#
# Submit the LLM manager's throughput measurement on Raven: N vLLM servers on
# the job's own node, then the benchmark against them.
#
# Run this ON Raven, from the checkout you want measured. It submits
# `serve_vllm.slurm.sh` with the benchmark as its command, and passes every
# LLM_* setting through `--export` rather than relying on the submitting
# environment being inherited (isolated-dir env propagation into sbatch has
# bitten this repo before).
#
#   scripts/llm_manager/bench_raven.sh                       # 8B on 4 cards
#   MODEL=Qwen/Qwen3-32B TP=2 scripts/llm_manager/bench_raven.sh   # 32B, 2 servers
#
# Knobs, all optional:
#   MODEL      HF id                     (Qwen/Qwen3-8B)
#   TP         GPUs per server           (1)
#   GPUS       cards to request          (4)
#   TIME       wall clock limit          (04:00:00)
#   EPISODES   rollout widths            ("200 1000 3000")
#   ROUNDS     rounds per rollout        (24)
#   WORK       scratch + model cache     (/ptmp/$USER/llm-manager-2026-09-22)
#   TAG        names the output files    (derived from MODEL)

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
TP="${TP:-1}"
GPUS="${GPUS:-4}"
TIME="${TIME:-04:00:00}"
EPISODES="${EPISODES:-200 1000 3000}"
ROUNDS="${ROUNDS:-24}"
SATURATION="${SATURATION:-16 64 128 256 512 1024 2048}"
WORK="${WORK:-/ptmp/$USER/llm-manager-2026-09-22}"
TAG="${TAG:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$HOME/algorithmic-institutions/.venv/bin/python}"
RESULTS="$WORK/results"
mkdir -p "$RESULTS" "$REPO/temp"

# THE TRAP: without this the package resolves to the shared checkout and the
# measurement is of code other than this one.
export PYTHONPATH="$REPO/src"

echo "[bench] repo   $REPO"
echo "[bench] model  $MODEL  tp=$TP  gpus=$GPUS"
echo "[bench] out    $RESULTS/bench-$TAG.json"

sbatch \
    --chdir="$REPO" \
    --gres="gpu:a100:$GPUS" \
    --time="$TIME" \
    --job-name="llmmgr-$TAG" \
    --output="$REPO/temp/bench-$TAG-%j.log" \
    --error="$REPO/temp/bench-$TAG-%j.log" \
    --export="ALL,PYTHONPATH=$REPO/src,LLM_MODEL=$MODEL,LLM_TP_SIZE=$TP,LLM_WORK=$WORK,LLM_HF_HOME=$WORK/hf_cache,LLM_CONTAINER=/ptmp/$USER/containers/vllm.sif" \
    "$REPO/scripts/llm_manager/serve_vllm.slurm.sh" \
    "$PY" "$REPO/scripts/llm_manager/throughput_benchmark.py" \
        --episodes $EPISODES \
        --rounds "$ROUNDS" \
        --saturation $SATURATION \
        --log-dir "$RESULTS/calls-$TAG" \
        --out "$RESULTS/bench-$TAG.json"
