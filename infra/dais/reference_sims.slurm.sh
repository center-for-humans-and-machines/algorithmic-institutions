#!/bin/bash -l
# Recreate the reference sims' per-round data that the agent definitions
# promise but that is in no checkout (its parquet files are not in git):
# plots/simulation/29_EXP1_reference/ (exp1, policy-finder-exp1.md) and
# plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/ (exp2,
# policy-finder.md). Each config runs unchanged (its size and seed) with
# `python -m aimanager simulate`, in the sim image on the CPUs, in a copy of
# the checkout with the model files of PF_ARTIFACTS; the files it writes that
# git does not track are then copied into the checkout's folder (the tracked
# plots, from the original GPU runs, stay). CPU draws differ from the
# original GPU runs' (same distribution).
#
#   cd <checkout>; sbatch infra/dais/reference_sims.slurm.sh
#
#SBATCH -J pf_refsims
#SBATCH -o /dais/fs/scratch/%u/policy-finder-agents/logs/%x.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=small
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=02:00:00

set -euo pipefail
export PF_SCRATCH="${PF_SCRATCH:-/dais/fs/scratch/$USER/policy-finder-agents}"
CHECKOUT="${SLURM_SUBMIT_DIR:-$PWD}"
export PF_REPO="$CHECKOUT"
source "$CHECKOUT/infra/dais/common.sh"
init_run
ARTIFACTS="${PF_ARTIFACTS:-$PF_SCRATCH/artifacts}"
CONFIGS=(configs/simulation/manager_testing/29_EXP1_reference.yml
         configs/simulation/manager_testing/25_LEVIN_run1_ah_zero_pairings_batched.yml)

tree="$PF_RUN/tree"
mkdir -p "$tree"
(cd "$CHECKOUT" && tar --exclude=.git --exclude=plots -cf - .) | tar -xf - -C "$tree"
cp -r "$ARTIFACTS/." "$tree/artifacts/"
PF_REPO="$tree"
export SIM_THREADS="${SIM_THREADS:-4}"
pids=()
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yml)
  REPO_MODE=rw simulate "$cfg" "$name" & pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
[ "$rc" = 0 ] || { echo "FATAL: a sim failed (logs in $PF_RUN)" >&2; exit 1; }

for cfg in "${CONFIGS[@]}"; do
  out=$(sed -n 's/^output_dir: *//p' "$tree/$cfg")
  mkdir -p "$CHECKOUT/$out"
  for f in "$tree/$out"/*; do
    rel="$out/$(basename "$f")"
    if git -C "$CHECKOUT" ls-files --error-unmatch "$rel" >/dev/null 2>&1; then
      echo "kept tracked $rel (the original run's)"
    else
      cp "$f" "$CHECKOUT/$rel"
      echo "added $rel $(stat -c %s "$f") bytes sha256 $(sha256sum "$f" | cut -c1-16)"
    fi
  done
done
