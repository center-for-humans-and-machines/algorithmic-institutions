#!/bin/bash -l
# vLLM and simulations on the same GPUs. On 2 H200:
#   1. a sim alone on GPU 0 (no server yet): time and memory
#   2. vLLM (tp 2) with --gpu-memory-utilization GPU_UTIL, so that a fixed
#      slice of each GPU stays free; load time and memory per GPU;
#      probe_endpoint.py (chat, tool calls, streaming, usage)
#   3. 5 concurrent long-context sessions, alone (bench_load.py)
#   4. the same while a sim runs on GPU 0; the sim's time against step 1
#   (the sims run on the CPUs instead when the sim image has no kernels for
#   the GPU, as with algoinst.sif on H200 and B200)
#
#   cd /u/$USER/algorithmic-institutions-pf     # the smoke commit + infra/dais
#   sbatch infra/dais/share_bench.slurm.sh
#   sbatch -p gpudev --time=00:15:00 --export=ALL,SIM_POINTS=4,CONTEXTS=32000 \
#     infra/dais/share_bench.slurm.sh          # a short version that fits gpudev
#
#SBATCH -J pf_share
#SBATCH -o /dais/fs/scratch/levinb/policy-finder/logs/share_bench.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=gpu1
#SBATCH --qos=g0002
#SBATCH --gres=gpu:h200:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250000
#SBATCH --time=01:30:00

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/common.sh"
GPU_UTIL="${GPU_UTIL:-0.88}"
PORT=8000
OUT="$PF_SCRATCH/share_bench/$SLURM_JOB_ID"
LOG="$OUT/vllm.log"
mkdir -p "$OUT"
cd "$PF_REPO"
# a real sweep's shape: 16 points x 500 episodes in batches of 1000
SWEEP=configs/simulation/policy_finder/pfsmoke-20261127-1_sweep.yml
CONTEXTS="${CONTEXTS:-32000 100000}"   # prompt tokens per session
ALL_GPUS="$CUDA_VISIBLE_DEVICES"
GPU0="${ALL_GPUS%%,*}"

assert_vllm_image
nvidia-smi --query-gpu=index,name,memory.total --format=csv
gpu_mem_log "$OUT/gpu_mem.csv" &
MEMLOG=$!
SERVER_PID=""
trap 'kill $SERVER_PID $MEMLOG 2>/dev/null' EXIT

SIM_GPU="$GPU0"   # emptied below if the sim image has no kernels for this GPU
sim() {  # <label>: the sweep-shaped sim on SIM_GPU (or the CPUs), own output dir
  CUDA_VISIBLE_DEVICES="$SIM_GPU" in_sim_image python infra/dais/sim_variant.py config \
    "$SWEEP" "$OUT/$1.yml" --points "${SIM_POINTS:-16}" --episodes 500 --batch 1000 \
    --output-dir "$OUT/$1"
  CUDA_VISIBLE_DEVICES="$SIM_GPU" simulate "$OUT/$1.yml" "$1"
}
if ! CUDA_VISIBLE_DEVICES="$GPU0" in_sim_image python infra/dais/sim_variant.py check-gpu \
    > "$OUT/check_gpu.log" 2>&1; then
  echo "== the sim image has no kernels for this GPU: sims run on the CPUs"
  SIM_GPU=""
fi
echo "== 1. sim alone on ${SIM_GPU:+GPU }${SIM_GPU:-the CPUs}, $(date +%T)"
sim sim_alone

echo "== 2. vLLM, util $GPU_UTIL, $(date +%T)"
serve_model "$PORT" "$GPU_UTIL" 2 "$LOG"
wait_ready "$PORT" 1800 || { tail -50 "$LOG"; exit 1; }
grep -E "weights took|Loading weights took|KV cache|concurrency|CUDA graph|init engine" "$LOG" | tail -10
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
apptainer exec --cleanenv "$VLLM_SIF" python3 "$PF_REPO/infra/dais/probe_endpoint.py" \
  --base "http://127.0.0.1:$PORT/v1" --model "$MODEL"

bench() {
  apptainer exec --cleanenv "$VLLM_SIF" python3 "$PF_REPO/infra/dais/bench_load.py" \
    --base "http://127.0.0.1:$PORT/v1" --model "$MODEL" --sessions 5 "$@"
}
echo "== 3. 5 sessions alone, $(date +%T)"
for c in $CONTEXTS; do bench --context "$c" --gen 1024 --tag "alone$c"; done

echo "== 4. 5 sessions while a sim runs on ${SIM_GPU:+GPU }${SIM_GPU:-the CPUs}, $(date +%T)"
# sims back to back until the benchmark is done, so that every request overlaps one
(i=0; while [ ! -e "$OUT/stop" ]; do sim "sim_shared_$i"; i=$((i + 1)); done) \
  > "$OUT/sim_shared.log" 2>&1 &
SIM_PID=$!
sleep 30   # let the first sim load its models and start its first batch
for c in $CONTEXTS; do bench --context "$c" --gen 1024 --tag "shared$c"; done
touch "$OUT/stop"
wait "$SIM_PID"
grep "^SIM " "$OUT/sim_shared.log"
echo "== 5. sim next to an idle server, $(date +%T)"
sim sim_idle_server

echo "== peak memory per GPU (MiB)"
for g in ${ALL_GPUS//,/ }; do
  awk -F', ' -v g="$g" '$2==g {if ($3>m) m=$3} END {print "GPU " g ": " m}' "$OUT/gpu_mem.csv"
done
