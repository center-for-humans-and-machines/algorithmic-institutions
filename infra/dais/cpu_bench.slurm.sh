#!/bin/bash -l
# How one real sweep uses CPUs, and whether Slurm job steps cap a process.
#
#   B1a  one slice of a real sweep (part 1 of 32: 8 points x 500 episodes of
#        pf-20261104-1's rule) at 1, 2, 4, 8, 16, 32 threads, each pinned to
#        enough cores for its threads
#   B1b  16 cores filled with k processes of 32/k threads, one slice each
#        (k = 1, 2, 4, 8, 16, 32): throughput per shape
#   B4   srun steps as caps: CPUs seen, CPU use of a busy step, a step that
#        exceeds its memory, and a sim in a step
#
#   cd /u/$USER/algorithmic-institutions-pf
#   sbatch infra/dais/cpu_bench.slurm.sh
#
# Not run yet: the SSH connection to DAIS dropped before it was submitted.
#
#SBATCH -J pf_cpubench
#SBATCH -o /dais/fs/scratch/%u/policy-finder/logs/%x.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=small
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200000
#SBATCH --time=01:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/common.sh"
init_run
hostname
lscpu | grep -E "Model name|Thread\(s\)|Core\(s\) per socket"
echo "CPUs allowed: $(nproc)"
FAILS=0

make_sweep_parts 32 > "$PF_RUN/generate.log" 2>&1 || { tail "$PF_RUN/generate.log"; exit 1; }
tail -2 "$PF_RUN/generate.log"
echo "pairings in part 1: $(grep -c '_vs_ah' "$PF_REPO/configs/simulation/policy_finder/bench_rule_sweep_p1of32.yml")"
SIM_THREADS=3 in_sim_image python -c \
  "import torch; print('torch threads with OMP_NUM_THREADS=3:', torch.get_num_threads())"

echo "== B1a: one slice, threads x"
for t in 1 2 4 8 16 32; do
  cores=$(( (t + 1) / 2 ))
  cfg=$(part_config 1 32 "b1a_t$t")
  SIM_THREADS=$t SIM_CPUS=$(python3 "$PF_HERE/cpu_slots.py" slots 0 "$cores" 1) \
    simulate "$cfg" "b1a_t$t" || FAILS=$((FAILS + 1))
done

echo "== B1b: 16 cores, k processes x 32/k threads, one slice each"
for k in 1 2 4 8 16 32; do
  t=$(( 32 / k ))
  echo "-- k=$k t=$t, $(date +%T)"
  if [ "$t" -ge 2 ]; then
    mapfile -t slots < <(python3 "$PF_HERE/cpu_slots.py" slots 0 $(( t / 2 )) "$k")
  else  # one thread per process: one hyperthread each
    mapfile -t slots < <(python3 "$PF_HERE/cpu_slots.py" slots 0 1 16 | tr , '\n')
  fi
  t0=$SECONDS
  pids=()
  for i in $(seq 1 "$k"); do
    cfg=$(part_config "$i" 32 "b1b_k${k}_$i")
    SIM_THREADS=$t SIM_CPUS=${slots[$((i - 1))]} simulate "$cfg" "b1b_k${k}_$i" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || FAILS=$((FAILS + 1)); done
  echo "B1B k=$k threads=$t episodes=$((k * 4000)) wall=$((SECONDS - t0))"
done

echo "== B4: srun steps as caps"
srun --exact -n1 -c 2 python3 -c \
  "import os; print('step -c 2 may run on CPUs', sorted(os.sched_getaffinity(0)))" \
  || FAILS=$((FAILS + 1))
# 8 busy processes in a 2-core step: CPU seconds per wall second shows the cap
srun --exact -n1 -c 2 bash -c \
  'for i in $(seq 8); do timeout 20 sh -c "while :; do :; done" & done; wait' || true
sleep 2
srun --exact -n1 -c 2 --mem=2G python3 -c \
  "b = bytearray(6 * 2**30); print('allocated 6 GiB in a 2 GiB step')" \
  && echo "memory step was NOT killed" || echo "memory step failed with $?, as intended"
echo "job still alive after the memory step"
cfg=$(part_config 2 32 b4_step)
srun --exact -n1 -c 2 --mem=16G bash -c \
  "source '$PF_HERE/common.sh' >/dev/null; PF_RUN='$PF_RUN'; PF_REPO='$PF_REPO'; \
   SIM_THREADS=2 simulate '$cfg' b4_step" || FAILS=$((FAILS + 1))
sleep 5
sacct -j "$SLURM_JOB_ID" -o JobID%20,JobName%12,State,ExitCode,AllocCPUS,Elapsed,TotalCPU,MaxRSS,ReqMem

echo "failures: $FAILS"
[ "$FAILS" = 0 ]
