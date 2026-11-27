#!/bin/bash -l
# vLLM and sweeps on one allocation (2 H200, 24 cores), each on its own cores.
#
#   B3b  vLLM (tp 2) pinned to SERVER_CORES cores, then probe_endpoint.py
#   B5   aggregate and per-session decode speed and time to first token for
#        5, 10, 20, 40, 80 concurrent sessions (warm), 30k-token prompts
#   B3d  steady generation (5 warm long-context sessions) while K sweep slices
#        run, pinned one per core on the other cores
#   B3c  the same steady generation alone, for as long as B3d took
#   B3a  the same K slices alone, after the server is stopped
#
#   cd /u/$USER/algorithmic-institutions-pf
#   sbatch infra/dais/serve_bench.slurm.sh
#   sbatch -p gpudev --time=00:15:00 infra/dais/serve_bench.slurm.sh   # if it fits
#
#SBATCH -J pf_servebench
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
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "CPUs allowed: $(nproc)"
SERVER_CORES="${SERVER_CORES:-4}"   # vLLM and the benchmark clients
K="${K:-16}"                        # sweep slices beside the server
export SIM_THREADS="${SIM_THREADS:-2}"
SERVER_CPUS=$(python3 "$PF_HERE/cpu_slots.py" reserved "$SERVER_CORES")
mapfile -t SLOTS < <(python3 "$PF_HERE/cpu_slots.py" slots "$SERVER_CORES" 1 "$K")
echo "server CPUs: $SERVER_CPUS; ${#SLOTS[@]} slice slots of one core"
assert_vllm_image
FAILS=0

make_sweep_parts 32 > "$PF_RUN/generate.log" 2>&1 || { tail "$PF_RUN/generate.log"; exit 1; }

slices() {  # <tag>: K slices at once, pinned; prints the round's wall time
  local tag="$1" t0=$SECONDS i cfg pids=() fails=0
  for i in $(seq 1 "$K"); do
    cfg=$(part_config "$i" 32 "${tag}_$i")
    SIM_CPUS=${SLOTS[$((i - 1))]} simulate "$cfg" "${tag}_$i" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || fails=$((fails + 1)); done
  SLICES_WALL=$(( SECONDS - t0 ))
  echo "B3 slices=$tag K=$K wall=$SLICES_WALL failures=$fails"
  [ "$fails" = 0 ]
}

bench() {  # bench_load.py on the server's cores
  taskset -c "$SERVER_CPUS" apptainer exec --containall --cleanenv \
    --bind "$PF_RUN" --bind "$PF_HERE:/opt/dais:ro" "$VLLM_SIF" \
    python3 /opt/dais/bench_load.py --base "$SERVER_URL" --key-file "$SERVER_KEY_FILE" \
    --model "$MODEL" "$@"
}

echo "== B3b: vLLM on CPUs $SERVER_CPUS, $(date +%T)"
SERVER_CPUS="$SERVER_CPUS" serve_model 0.90 2 "$PF_RUN/vllm.log"
wait_ready 1800 || { tail -50 "$PF_RUN/vllm.log"; exit 1; }
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
taskset -c "$SERVER_CPUS" apptainer exec --containall --cleanenv \
  --bind "$PF_RUN" --bind "$PF_HERE:/opt/dais:ro" "$VLLM_SIF" \
  python3 /opt/dais/probe_endpoint.py --base "$SERVER_URL" --key-file "$SERVER_KEY_FILE" \
  --model "$MODEL" --trials "${TRIALS:-3}" || FAILS=$((FAILS + 1))

echo "== B5: concurrency, $(date +%T)"
for n in ${CONCURRENCY:-5 10 20 40 80}; do
  bench --sessions "$n" --context 30000 --gen 512 --tag "conc$n" || FAILS=$((FAILS + 1))
done

echo "== B3d: steady generation while $K slices run, $(date +%T)"
rm -f "$PF_RUN/stop"
(set +e; slices shared; rc=$?; echo "$SLICES_WALL" > "$PF_RUN/slices_wall"
 touch "$PF_RUN/stop"; exit $rc) &
SLICES_PID=$!
BG_PIDS+=("$SLICES_PID")
# generation starts with the slices and stops when the last one ends
bench --sessions 5 --context 30000 --gen 1024 --tag steady_shared \
  --duration 3600 --stop-file "$PF_RUN/stop" || FAILS=$((FAILS + 1))
wait "$SLICES_PID" || FAILS=$((FAILS + 1))
SHARED_WALL=$(cat "$PF_RUN/slices_wall")

echo "== B3c: steady generation alone for ${SHARED_WALL}s, $(date +%T)"
bench --sessions 5 --context 30000 --gen 1024 --tag steady_alone --duration "$SHARED_WALL" \
  || FAILS=$((FAILS + 1))

echo "== B3a: the same $K slices alone, server stopped, $(date +%T)"
kill "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true
slices alone || FAILS=$((FAILS + 1))

grep -E "KV cache size|Maximum concurrency" "$PF_RUN/vllm.log" | tail -2
echo "done $(date +%T), failures: $FAILS"
[ "$FAILS" = 0 ]
