#!/bin/bash -l
# Speed and GPU memory of the new sim image on one H200, on a real sweep
# (generate_sim_config.py's design and split: 256 points x 500 episodes in
# 2 parts of 128 points).
#
#   cd /u/$USER/algorithmic-institutions-gpusim
#   sbatch infra/dais/gpusim_bench.slurm.sh batch 1000 4000 16000
#       one part at each episode_batch_size: wall time, peak GPU memory
#   sbatch infra/dais/gpusim_bench.slurm.sh concurrent POINTS BATCH 1 2 4 8
#       K copies of a POINTS-point slice of the part at once, for each K:
#       wall time, aggregate episodes/s, GPU memory per process and in all
#
#SBATCH -J gpusim_bench
#SBATCH --account=mpib_inst
#SBATCH --partition=gpu1
#SBATCH --qos=g0001
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200000
#SBATCH --time=01:00:00

set -euo pipefail
MODE="${1:?batch or concurrent}"
shift
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/gpusim_common.sh"
use_image new
cd "$PF_REPO"
OUT="$GS_SCRATCH/runs/bench/$SLURM_JOB_ID"
mkdir -p "$OUT"
RULE="${RULE:-configs/managers/rule_based/pf-20261104-12-v7.yml}"
echo "job $SLURM_JOB_ID on $(hostname), $(nproc) cpus, commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
in_sim_image python infra/dais/gpusim_equiv.py sweep-parts "$RULE" "$OUT/parts"
PART="$OUT/parts/sweep_p1of2.yml"

if [ "$MODE" = batch ]; then
  for batch in "$@"; do
    in_sim_image python infra/dais/sim_variant.py config "$PART" "$OUT/b$batch.yml" \
      --batch "$batch" --output-dir "$OUT/b$batch"
    simulate "$OUT/b$batch.yml" "part1_128x500_b$batch"
  done
  exit 0
fi

[ "$MODE" = concurrent ] || { echo "FATAL: mode is batch or concurrent" >&2; exit 2; }
POINTS="${1:?points}" BATCH="${2:?batch}"
shift 2
# a slice of the part: its first POINTS points, unchanged
in_sim_image python - "$PART" "$OUT/slice.yml" "$POINTS" "$BATCH" <<'PY'
import sys, yaml
src, dst, n, batch = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
c = yaml.safe_load(open(src))
c["pairings"] = c["pairings"][:n]
keep = {"ah"} | {p["group_0"] for p in c["pairings"]}
c["managers"] = {k: v for k, v in c["managers"].items() if k in keep}
c["episode_batch_size"] = batch
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
PY
episodes=$(( POINTS * 500 ))
cpus=$(nproc)
for k in "$@"; do
  dir="$OUT/k$k"
  mkdir -p "$dir"
  for i in $(seq 1 "$k"); do
    in_sim_image python infra/dais/sim_variant.py config "$OUT/slice.yml" \
      "$dir/p$i.yml" --output-dir "$dir/p$i"
  done
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
    -lms 1000 > "$dir/apps.csv" &
  apps=$!
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
    -lms 1000 > "$dir/gpu.csv" &
  gpu=$!
  t0=$SECONDS
  pids=()
  for i in $(seq 1 "$k"); do
    # each process gets an equal share of the job's cpus
    APPTAINERENV_OMP_NUM_THREADS=$(( cpus / k > 0 ? cpus / k : 1 )) \
      in_sim_image python -m aimanager simulate "$dir/p$i.yml" > "$dir/p$i.log" 2>&1 &
    pids+=($!)
  done
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  wall=$(( SECONDS - t0 ))
  kill "$apps" "$gpu"
  [ "$rc" = 0 ] || { echo "FATAL: a sim of K=$k failed" >&2; tail -20 "$dir"/p*.log; exit 1; }
  per_proc=$(awk -F', ' '{if ($2 > m[$1]) m[$1] = $2} END {for (p in m) s = s " " m[p]; print s}' "$dir/apps.csv")
  peak=$(sort -t, -k1 -n "$dir/gpu.csv" | tail -1 | cut -d, -f1)
  util=$(awk -F', ' '{s += $2; n++} END {printf "%.0f", s / n}' "$dir/gpu.csv")
  echo "CONC k=$k points=$POINTS batch=$BATCH episodes_each=$episodes wall_s=$wall" \
    "episodes_per_s=$(( k * episodes / (wall > 0 ? wall : 1) ))" \
    "peak_gpu_mib=$peak mean_util=$util per_process_peak_mib=[$per_proc ]"
done
