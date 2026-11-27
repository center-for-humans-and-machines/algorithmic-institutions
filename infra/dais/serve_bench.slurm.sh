#!/bin/bash -l
# The recommended layout on one allocation (2 GPUs, 24 cores of an H200 or
# B200 node), each part on its own cores:
#   SERVER_CORES (4)  vLLM, tp 2
#   CLIENT_CORES (2)  the benchmark client
#   the rest          a pool of single-threaded sweep slices, one per
#                     hyperthread, all launched at once (configs prepared
#                     before any timing)
#
#   B3 (REPEATS times, default 3): (a) the pool alone, server loaded but idle;
#      (b) steady generation alone, SESSIONS (20) warm long-context sessions,
#      for as long as (a) took; (c) the pool while the same steady generation
#      runs (the pool starts once the generation is steady)
#   B5 sessions 20, 40, 80: one cold round, then WARM_ROUNDS (3) warm rounds
#
#   cd /u/$USER/algorithmic-institutions-pf
#   sbatch infra/dais/serve_bench.slurm.sh                       # gpu1
#   sbatch -p gpudev --time=00:15:00 infra/dais/serve_bench.slurm.sh
#
# Job 987430 ran an earlier version (5 sessions, 2-thread slices started one by
# one, client on the server's cores).
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
SERVER_CORES="${SERVER_CORES:-4}"
CLIENT_CORES="${CLIENT_CORES:-2}"
SESSIONS="${SESSIONS:-20}"
REPEATS="${REPEATS:-3}"
FAILS=0
SERVER_CPUS=$(python3 "$PF_HERE/cpu_slots.py" reserved "$SERVER_CORES")
CLIENT_CPUS=$(python3 "$PF_HERE/cpu_slots.py" slots "$SERVER_CORES" "$CLIENT_CORES" 1)
mapfile -t POOL < <(python3 "$PF_HERE/cpu_slots.py" slots $((SERVER_CORES + CLIENT_CORES)) 1 \
  | tr , '\n')
echo "server CPUs $SERVER_CPUS; client CPUs $CLIENT_CPUS; pool of ${#POOL[@]} CPUs"
assert_vllm_image

# every slice config of every round, before anything is timed
make_sweep_parts 32 > "$PF_RUN/generate.log" 2>&1 || { tail "$PF_RUN/generate.log"; exit 1; }
for k in $(seq 1 32); do part_config "$k" 32 "part$k" > /dev/null; done
labels() {  # <tag>: one config per pool CPU, part ((i-1) % 32) + 1
  local tag="$1" i part
  for i in $(seq 1 "${#POOL[@]}"); do
    part=$(( (i - 1) % 32 + 1 ))
    sed "s#/out/part$part\$#/out/${tag}_$i#" "$PF_RUN/cfg/part$part.yml" \
      > "$PF_RUN/cfg/${tag}_$i.yml"
    grep -q "/out/${tag}_$i\$" "$PF_RUN/cfg/${tag}_$i.yml"
  done
}
for r in $(seq 1 "$REPEATS"); do labels "alone$r"; labels "shared$r"; done

pool() {  # <tag>: every pool CPU runs one slice, all launched at once
  local tag="$1" t0=$SECONDS i pids=() fails=0
  for i in $(seq 1 "${#POOL[@]}"); do
    SIM_THREADS=1 SIM_CPUS=${POOL[$((i - 1))]} simulate "$PF_RUN/cfg/${tag}_$i.yml" "${tag}_$i" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || fails=$((fails + 1)); done
  echo "POOL $tag slices=${#POOL[@]} episodes=$(( ${#POOL[@]} * 4000 )) wall=$(( SECONDS - t0 )) failures=$fails"
  echo $(( SECONDS - t0 )) > "$PF_RUN/$tag.wall"
  [ "$fails" = 0 ]
}

bench() {
  taskset -c "$CLIENT_CPUS" apptainer exec --containall --cleanenv \
    --bind "$PF_RUN" --bind "$PF_HERE:/opt/dais:ro" "$VLLM_SIF" \
    python3 /opt/dais/bench_load.py --base "$SERVER_URL" --key-file "$SERVER_KEY_FILE" \
    --model "$MODEL" "$@"
}

echo "== server, $(date +%T)"
SERVER_CPUS="$SERVER_CPUS" serve_model 0.90 2 "$PF_RUN/vllm.log"
wait_ready 1800 || { tail -50 "$PF_RUN/vllm.log"; exit 1; }

for r in $(seq 1 "$REPEATS"); do
  echo "== B3 repeat $r (a): pool alone, $(date +%T)"
  pool "alone$r" || FAILS=$((FAILS + 1))
  echo "== B3 repeat $r (b): $SESSIONS sessions alone, $(date +%T)"
  bench --sessions "$SESSIONS" --gen 1024 --tag "steady_alone$r" \
    --duration "$(cat "$PF_RUN/alone$r.wall")" || FAILS=$((FAILS + 1))
  echo "== B3 repeat $r (c): pool while $SESSIONS sessions generate, $(date +%T)"
  rm -f "$PF_RUN/stop" "$PF_RUN/ready"
  bench --sessions "$SESSIONS" --gen 1024 --tag "steady_shared$r" --duration 1800 \
    --stop-file "$PF_RUN/stop" --ready-file "$PF_RUN/ready" &
  BENCH_PID=$!
  BG_PIDS+=("$BENCH_PID")
  until [ -e "$PF_RUN/ready" ] || ! kill -0 "$BENCH_PID" 2>/dev/null; do sleep 1; done
  pool "shared$r" || FAILS=$((FAILS + 1))
  touch "$PF_RUN/stop"
  wait "$BENCH_PID" || FAILS=$((FAILS + 1))
done

echo "== B5, $(date +%T)"
for n in ${CONCURRENCY:-20 40 80}; do
  bench --sessions "$n" --gen 512 --tag "conc$n" --warm-rounds "${WARM_ROUNDS:-3}" \
    || FAILS=$((FAILS + 1))
done
grep -E "KV cache size|Maximum concurrency" "$PF_RUN/vllm.log" | tail -2 || true
echo "done $(date +%T), failures: $FAILS"
[ "$FAILS" = 0 ]
