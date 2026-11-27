#!/bin/bash -l
# Many sweep slices at once on one allocation's cores (B2): N = 1, 5, 10, 20,
# 40 concurrent slices of a real sweep (part k of 32: 8 points x 500 episodes),
# SIM_THREADS threads each (default 2), pinned one per core versus not pinned.
# Prints node throughput (episodes per second) per round and every slice's
# wall time, CPU time and peak memory.
#
#   cd /u/$USER/algorithmic-institutions-pf
#   sbatch infra/dais/sweep_pack.slurm.sh
#
# Not run yet: the SSH connection to DAIS dropped before it was submitted.
#
#SBATCH -J pf_sweeppack
#SBATCH -o /dais/fs/scratch/%u/policy-finder/logs/%x.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=small
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300000
#SBATCH --time=01:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/common.sh"
init_run
hostname
lscpu | grep -E "Model name"
echo "CPUs allowed: $(nproc)"
export SIM_THREADS="${SIM_THREADS:-2}"
FAILS=0

make_sweep_parts 32 > "$PF_RUN/generate.log" 2>&1 || { tail "$PF_RUN/generate.log"; exit 1; }
for k in $(seq 1 32); do part_config "$k" 32 "part$k" > /dev/null; done

# round <N> <pinned|free>: N slices at once; slice i plays part ((i-1) % 32)+1
round() {
  local n="$1" mode="$2" t0=$SECONDS i cpus pids=() slots=()
  mapfile -t slots < <(python3 "$PF_HERE/cpu_slots.py" slots 0 1 "$n")
  for i in $(seq 1 "$n"); do
    local part=$(( (i - 1) % 32 + 1 )) label="n${n}_${mode}_t${SIM_THREADS}_$i"
    sed "s#/out/part$part\$#/out/$label#" "$PF_RUN/cfg/part$part.yml" > "$PF_RUN/cfg/$label.yml"
    cpus=""
    [ "$mode" = pinned ] && cpus=${slots[$(( (i - 1) % ${#slots[@]} ))]}
    SIM_CPUS=$cpus simulate "$PF_RUN/cfg/$label.yml" "$label" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || FAILS=$((FAILS + 1)); done
  local wall=$(( SECONDS - t0 ))
  echo "B2 N=$n mode=$mode threads=$SIM_THREADS episodes=$((n * 4000)) wall=$wall" \
    "eps_per_s=$(python3 -c "print(round($n * 4000 / $wall, 1))")"
  dmesg 2>/dev/null | grep -i -E "killed process|out of memory" | tail -2 || true
}

for n in 1 5 10 20 40; do
  round "$n" pinned
  round "$n" free
done
echo "== one thread per slice, 40 slices pinned one per core"
SIM_THREADS=1 round 40 pinned

echo "failures: $FAILS"
[ "$FAILS" = 0 ]
