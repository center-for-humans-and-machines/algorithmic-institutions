#!/bin/bash -l
# Sims on CPUs in the production shape: single-threaded processes, one per
# CPU of the job, each a slice of a config (gpusim_equiv.py slice).
#
#   cd /u/$USER/algorithmic-institutions-gpusim
#   sbatch infra/dais/gpusim_cpu.slurm.sh equiv old CONFIG...
#       every config sliced into one process per point; the points'
#       per-episode scores joined into
#       $GS_SCRATCH/runs/equiv/<image>_cpu1/<config>/episodes.csv
#   sbatch infra/dais/gpusim_cpu.slurm.sh exact old CONFIG
#       the same config sliced per point, in two halves and whole, all
#       single-threaded, into $GS_SCRATCH/runs/exact/<job>/; then a real
#       sweep's first 16 points as 2 parts of 32 and as 1 part of 16
#       (generate_sim_config.py --n-parts); compare with gpusim_equiv.py
#
#SBATCH -J gpusim_cpu
#SBATCH --account=mpib_inst
#SBATCH --partition=small
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200000
#SBATCH --time=01:00:00

set -euo pipefail
MODE="${1:?equiv or exact}"
IMAGE="${2:?old or new}"
shift 2
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/gpusim_common.sh"
use_image "$IMAGE"
export CUDA_VISIBLE_DEVICES=""
# one thread per process (passed into the image through --cleanenv)
export APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1
cd "$PF_REPO"
SLOTS=$(nproc)
echo "job $SLURM_JOB_ID on $(hostname): image $IMAGE, $SLOTS cpus," \
  "commit $(git rev-parse HEAD)"
FAILS=0

# launch <cmd...>: in the background, at most SLOTS at once
launch() {
  while (( $(jobs -rp | wc -l) >= SLOTS )); do
    wait -n || FAILS=$((FAILS + 1))
  done
  "$@" &
}
drain() {
  while (( $(jobs -rp | wc -l) > 0 )); do
    wait -n || FAILS=$((FAILS + 1))
  done
  [ "$FAILS" = 0 ] || { echo "FATAL: $FAILS slices failed" >&2; exit 1; }
}
points() {  # <config>: its design points, one per line
  in_sim_image python -c 'import sys, yaml
print("\n".join(k for k in yaml.safe_load(open(sys.argv[1]))["managers"] if k != "ah"))' "$1"
}
# slice <config> <points,...> <dir>: that slice's sim and per-episode scores
slice() {
  local cfg="$1" pts="$2" dir="$3" t0=$SECONDS
  mkdir -p "$dir"
  in_sim_image python infra/dais/gpusim_equiv.py slice "$cfg" "$dir/config.yml" \
    --points "$pts" --output-dir "$dir/sim"
  in_sim_image python -m aimanager simulate "$dir/config.yml" > "$dir/sim.log" 2>&1 \
    || { echo "FATAL: $dir"; tail -5 "$dir/sim.log"; return 1; }
  in_sim_image python infra/dais/gpusim_equiv.py episodes "$dir/config.yml" \
    "$dir/episodes.csv" > /dev/null
  echo "SLICE $dir seconds=$(( SECONDS - t0 ))"
}
run_sim() {  # <config>: its sim, log beside it
  in_sim_image python -m aimanager simulate "$1" > "${1%.yml}.log" 2>&1 \
    || { echo "FATAL: $1"; tail -5 "${1%.yml}.log"; return 1; }
}
# join <dir>: the slices' per-episode scores in <dir>/*/ as <dir>/episodes.csv
join() {
  local first=1 f
  for f in "$1"/*/episodes.csv; do
    if [ "$first" = 1 ]; then cat "$f"; first=0; else tail -n +2 "$f"; fi
  done > "$1/episodes.csv"
}

t0=$SECONDS
if [ "$MODE" = equiv ]; then
  for cfg in "$@"; do
    stem=$(basename "$cfg" .yml)
    for p in $(points "$cfg"); do
      launch slice "$cfg" "$p" "$GS_SCRATCH/runs/equiv/${IMAGE}_cpu1/$stem/$p"
    done
  done
  drain
  for cfg in "$@"; do join "$GS_SCRATCH/runs/equiv/${IMAGE}_cpu1/$(basename "$cfg" .yml)"; done
  echo "EQUIV OK in $(( SECONDS - t0 )) s"
  exit 0
fi

[ "$MODE" = exact ] || { echo "FATAL: mode is equiv or exact" >&2; exit 2; }
cfg="$1"
OUT="$GS_SCRATCH/runs/exact/$SLURM_JOB_ID"
mapfile -t pts < <(points "$cfg")
half=$(( ${#pts[@]} / 2 ))
for p in "${pts[@]}"; do launch slice "$cfg" "$p" "$OUT/per_point/$p"; done
launch slice "$cfg" "$(IFS=,; echo "${pts[*]:0:$half}")" "$OUT/halves/a"
launch slice "$cfg" "$(IFS=,; echo "${pts[*]:$half}")" "$OUT/halves/b"
launch slice "$cfg" "$(IFS=,; echo "${pts[*]}")" "$OUT/whole/all"
drain
join "$OUT/per_point"
join "$OUT/halves"
join "$OUT/whole"

# a real sweep, split the pipeline's way: its first 16 points as parts 1-2
# of 32 and as part 1 of 16 (seeded by global batch index, so they should
# play the same games)
in_sim_image python infra/dais/gpusim_equiv.py sweep-parts \
  configs/managers/rule_based/pf-20261104-12-v7.yml "$OUT/sweep32" --n-parts 32 > /dev/null
in_sim_image python infra/dais/gpusim_equiv.py sweep-parts \
  configs/managers/rule_based/pf-20261104-12-v7.yml "$OUT/sweep16" --n-parts 16 > /dev/null
for c in "$OUT/sweep32/sweep_p1of32.yml" "$OUT/sweep32/sweep_p2of32.yml" \
  "$OUT/sweep16/sweep_p1of16.yml"; do
  launch run_sim "$c"
done
drain
echo "EXACT OK in $(( SECONDS - t0 )) s: $OUT"
