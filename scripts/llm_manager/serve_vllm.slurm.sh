#!/bin/bash -l
#
# Serve a language model with vLLM on the job's own compute node, then run a
# command against it. The servers live and die with the job, so a run has no
# external dependency and no endpoint that can be reconfigured under it -- the
# reason this is preferred over the MPCDF hosted service, whose model is fixed
# at launch time and shared with other users.
#
# DATA PARALLEL BY DEFAULT. This workload is throughput-bound, not
# memory-bound: prompts are a couple of thousand tokens of accumulated trace
# and completions are a handful of tokens under a decode constraint. An 8B
# model fits on one A100 with room for a large KV cache, so N independent
# servers beat one N-way tensor-parallel instance, which for a small model
# mostly buys communication overhead. The script therefore starts
# (GPUs / LLM_TP_SIZE) servers, one per GPU group, and hands the client every
# endpoint; `ChatClient` shards episodes across them with a fixed assignment.
#
# For a model too big for one card, raise LLM_TP_SIZE: Qwen3-32B at
# LLM_TP_SIZE=2 on four A100s gives two instances.
#
# vLLM exposes an OpenAI-compatible API. `aimanager.manager.llm_client` talks
# to it directly, and LiteLLM's `hosted_vllm/` provider talks to the same
# endpoints, so HOSTED_VLLM_API_BASE / HOSTED_VLLM_API_KEY are exported under
# the names both conventions read. HOSTED_VLLM_API_BASE is a COMMA-SEPARATED
# list when there is more than one server.
#
# Usage:
#   sbatch --gres=gpu:a100:4 scripts/llm_manager/serve_vllm.slurm.sh <command...>
#
#   LLM_MODEL=Qwen/Qwen3-32B LLM_TP_SIZE=2 \
#     sbatch --gres=gpu:a100:4 scripts/llm_manager/serve_vllm.slurm.sh <command...>
#
# Everything is set by environment, so moving from 8B to 32B is a config
# change and never an edit to this file:
#
#   LLM_MODEL          HF model id                 (Qwen/Qwen3-8B)
#   LLM_TP_SIZE        GPUs per server             (1)
#   LLM_N_SERVERS      servers to start            (GPUs / LLM_TP_SIZE)
#   LLM_PORT           first server's port         (8000)
#   LLM_MAX_MODEL_LEN  context window              (8192)
#   LLM_GPU_UTIL       fraction of GPU memory      (0.90)
#   LLM_HF_HOME        model cache                 (/ptmp/$USER/llm-manager/hf_cache)
#   LLM_WORK           writable scratch            (/ptmp/$USER/llm-manager)
#   LLM_CONTAINER      vLLM apptainer image        (/ptmp/$USER/containers/vllm.sif)
#   LLM_EXTRA_ARGS     appended to `vllm serve`    (empty)
#   LLM_STARTUP_TRIES  health polls, 3 s apart     (600)
#
#SBATCH --job-name=llm-manager
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=480000
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-llm-manager-%j.out
#SBATCH --error=slurm-llm-manager-%j.out

set -euo pipefail

MODEL="${LLM_MODEL:-Qwen/Qwen3-8B}"
TP_SIZE="${LLM_TP_SIZE:-1}"
FIRST_PORT="${LLM_PORT:-8000}"
MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${LLM_GPU_UTIL:-0.90}"
WORK="${LLM_WORK:-/ptmp/$USER/llm-manager}"
HF_CACHE="${LLM_HF_HOME:-$WORK/hf_cache}"
CONTAINER="${LLM_CONTAINER:-/ptmp/$USER/containers/vllm.sif}"
EXTRA_ARGS="${LLM_EXTRA_ARGS:-}"
STARTUP_TRIES="${LLM_STARTUP_TRIES:-600}"

# How many GPUs the job was actually given.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    N_GPUS=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
else
    N_GPUS=1
fi
N_SERVERS="${LLM_N_SERVERS:-$((N_GPUS / TP_SIZE))}"
if [ "$N_SERVERS" -lt 1 ]; then
    echo "[serve] FAIL: $N_GPUS GPU(s) cannot host one server at tp=$TP_SIZE" >&2
    exit 1
fi

mkdir -p "$WORK/logs" "$HF_CACHE"

module purge
module load apptainer/1.4.3

# The servers run inside the container; APPTAINERENV_* is how a variable
# crosses that boundary. HF_HUB_OFFLINE keeps a compute node from reaching for
# the network: the weights must already be in the cache.
export APPTAINERENV_HF_HOME="$HF_CACHE"
export APPTAINERENV_HF_HUB_OFFLINE="${LLM_HF_OFFLINE:-1}"
export APPTAINERENV_VLLM_WORKER_MULTIPROC_METHOD=spawn
export APPTAINERENV_PYTHONUNBUFFERED=1

echo "[serve] job=${SLURM_JOB_ID:-none} node=$(hostname)"
echo "[serve] model=$MODEL gpus=$N_GPUS servers=$N_SERVERS tp=$TP_SIZE"

PIDS=()
PORTS=()
cleanup() {
    for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT

for i in $(seq 0 $((N_SERVERS - 1))); do
    PORT=$((FIRST_PORT + i))
    # Contiguous slice of the job's GPUs, one slice per server. Indices are
    # relative to CUDA_VISIBLE_DEVICES, which SLURM has already narrowed to
    # this job's cards.
    FIRST_GPU=$((i * TP_SIZE))
    SLICE=$(seq -s, "$FIRST_GPU" $((FIRST_GPU + TP_SIZE - 1)))
    LOG="$WORK/logs/vllm-${SLURM_JOB_ID:-local}-p${PORT}.log"
    echo "[serve] server $i: port=$PORT gpus=$SLICE log=$LOG"

    # Each server needs its own compile cache, or they race over one directory.
    APPTAINERENV_CUDA_VISIBLE_DEVICES="$SLICE" \
    APPTAINERENV_VLLM_CACHE_ROOT="$WORK/vllm-cache/p${PORT}" \
    apptainer exec --nv --bind /ptmp:/ptmp "$CONTAINER" \
        vllm serve "$MODEL" \
            --port "$PORT" --host 127.0.0.1 \
            --tensor-parallel-size "$TP_SIZE" \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$GPU_UTIL" \
            $EXTRA_ARGS \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    PORTS+=("$PORT")
done

echo "[serve] waiting for all $N_SERVERS servers ..."
for idx in "${!PORTS[@]}"; do
    PORT="${PORTS[$idx]}"
    PID="${PIDS[$idx]}"
    ready=0
    for _ in $(seq 1 "$STARTUP_TRIES"); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
            ready=1; break
        fi
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "[serve] FAIL: server on :$PORT exited during startup"
            tail -80 "$WORK/logs/vllm-${SLURM_JOB_ID:-local}-p${PORT}.log"
            exit 1
        fi
        sleep 3
    done
    if [ "$ready" -ne 1 ]; then
        echo "[serve] FAIL: server on :$PORT never became healthy"
        tail -80 "$WORK/logs/vllm-${SLURM_JOB_ID:-local}-p${PORT}.log"
        exit 1
    fi
    echo "[serve] server on :$PORT ready"
done

API_BASE=""
for PORT in "${PORTS[@]}"; do
    [ -n "$API_BASE" ] && API_BASE="$API_BASE,"
    API_BASE="${API_BASE}http://127.0.0.1:$PORT/v1"
done

export HOSTED_VLLM_API_BASE="$API_BASE"
export HOSTED_VLLM_API_KEY="${LLM_API_KEY:-EMPTY}"
export LLM_MANAGER_MODEL="$MODEL"
export LLM_MANAGER_N_SERVERS="$N_SERVERS"
export LLM_MANAGER_N_GPUS="$N_GPUS"

echo "[serve] endpoints: $HOSTED_VLLM_API_BASE"

if [ "$#" -eq 0 ]; then
    echo "[serve] no command given; holding the servers until the job ends"
    wait "${PIDS[0]}"
    exit 0
fi

echo "[serve] running: $*"
"$@"
