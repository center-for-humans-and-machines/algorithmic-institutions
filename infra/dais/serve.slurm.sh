#!/bin/bash -l
# Serve Qwen3-Coder-Next with vLLM on 2 H200 (tensor parallel), with tool
# calling, offline from the pinned weights in $HF_HOME. MODE=probe (default)
# checks the endpoint with probe_endpoint.py and ends the job; MODE=serve
# keeps serving until the time limit (an agent on this node uses
# http://127.0.0.1:8000/v1).
#
#   cd /u/$USER/algorithmic-institutions-pf
#   sbatch infra/dais/serve.slurm.sh
#   sbatch --export=ALL,MODE=serve,GPU_UTIL=0.88 --time=08:00:00 infra/dais/serve.slurm.sh
#
#SBATCH -J pf_serve
#SBATCH -o /dais/fs/scratch/levinb/policy-finder/logs/serve.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=gpu1
#SBATCH --qos=g0002
#SBATCH --gres=gpu:h200:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=250000
#SBATCH --time=01:00:00

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/common.sh"
MODE="${MODE:-probe}"
GPU_UTIL="${GPU_UTIL:-0.88}"
PORT=8000
LOG="$PF_LOGS/vllm.$SLURM_JOB_ID.log"

assert_vllm_image
nvidia-smi --query-gpu=index,name,memory.total --format=csv
gpu_mem_log "$PF_LOGS/gpu_mem.$SLURM_JOB_ID.csv" &
MEMLOG=$!
trap 'kill $SERVER_PID $MEMLOG 2>/dev/null' EXIT

serve_model "$PORT" "$GPU_UTIL" 2 "$LOG"
wait_ready "$PORT" 1800 || { tail -50 "$LOG"; exit 1; }
grep -E "Model loading took|model weights took|KV cache|Maximum concurrency|Available KV" "$LOG" | tail -8
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

if [ "$MODE" = probe ]; then
  apptainer exec --cleanenv "$VLLM_SIF" python3 "$PF_REPO/infra/dais/probe_endpoint.py" \
    --base "http://127.0.0.1:$PORT/v1" --model "$MODEL"
  exit $?
fi
wait "$SERVER_PID"
