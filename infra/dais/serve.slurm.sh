#!/bin/bash -l
# Serve Qwen3-Coder-Next with vLLM on 2 H200 (tensor parallel), with tool
# calling, offline from the pinned weights in $HF_HOME, on a free port with an
# API key of the job's own ($PF_RUN/api_key, mode 600). MODE=probe (default)
# checks the endpoint with probe_endpoint.py and ends the job, failing if a
# check fails; MODE=serve keeps serving until the time limit and writes the
# URL to $PF_RUN/url for a client on the same node.
#
#   cd /u/$USER/algorithmic-institutions-pf
#   sbatch infra/dais/serve.slurm.sh
#   sbatch --export=ALL,MODE=serve --time=08:00:00 infra/dais/serve.slurm.sh
#   # another model: its profile (infra/dais/model_profiles/<name>.sh), the GPUs it needs
#   sbatch --export=MODEL_PROFILE=deepseek-v4.1-flash-1017,PF_SCRATCH=... \
#     -o <log> --qos=g0004 --gres=gpu:b200:4 --cpus-per-task=48 --mem=1000000 \
#     infra/dais/serve.slurm.sh
#
# This version ran as job 987950 (2026-11-27, `-p gpudev`, MODE=probe). The
# default gpu1/g0002 path and MODE=serve have not been run.
#
#SBATCH -J pf_serve
#SBATCH -o /dais/fs/scratch/%u/policy-finder/logs/%x.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=gpu1
#SBATCH --qos=g0002
#SBATCH --gres=gpu:h200:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=500000
#SBATCH --time=01:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/common.sh"
init_run
MODE="${MODE:-probe}"
GPU_UTIL="${GPU_UTIL:-0.90}"

assert_vllm_image
nvidia-smi --query-gpu=index,name,memory.total --format=csv
# SERVER_GPUS (default 2; a model profile may set more): tensor parallel
serve_model "$GPU_UTIL" "${SERVER_GPUS:-2}" "$PF_RUN/vllm.log" ${VLLM_EXTRA:-}
wait_ready 1800 || { tail -50 "$PF_RUN/vllm.log"; exit 1; }
grep -E "Model loading took|KV cache|Maximum concurrency" "$PF_RUN/vllm.log" | tail -4
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

if [ "$MODE" = probe ]; then
  in_vllm_image python3 /opt/dais/probe_endpoint.py \
    --base "$SERVER_URL" --key-file "$SERVER_KEY_FILE" --model "$MODEL"
  exit $?
fi
echo "$SERVER_URL" > "$PF_RUN/url"
wait "$SERVER_PID"
