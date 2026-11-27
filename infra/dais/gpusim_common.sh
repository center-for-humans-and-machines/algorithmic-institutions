# Settings for the GPU sim image checks (infra/dais/gpusim_*.slurm.sh), on top
# of common.sh: their own scratch and checkout, and the two sim images.
# Sourced, not run. See docs/reports/dais_gpu_sim_equivalence.md.

GS_SCRATCH="${GS_SCRATCH:-/dais/fs/scratch/$USER/policy-finder-gpusim}"
# common.sh binds $PF_SCRATCH into the image and keeps its tmp there
export PF_SCRATCH="$GS_SCRATCH"
export PF_REPO="${PF_REPO:-/u/$USER/algorithmic-institutions-gpusim}"
# the old image (torch 1.11+cu113, Raven's algoinst.sif) and the new one
# (torch 2.5.1+cu124, Dockerfile.gpu-sim)
OLD_SIF="${OLD_SIF:-/dais/fs/scratch/$USER/policy-finder/images/algoinst.sif}"
NEW_SIF="${NEW_SIF:-$GS_SCRATCH/images/algoinst_gpusim.sif}"
SIM_SIF="${SIM_SIF:-$NEW_SIF}"

source "$PF_REPO/infra/dais/common.sh"

# use_image old|new: the image in_sim_image and simulate run
use_image() {
  case "$1" in
    old) SIM_SIF="$OLD_SIF" ;;
    new) SIM_SIF="$NEW_SIF" ;;
    *) echo "FATAL: image is old or new, not '$1'" >&2; exit 2 ;;
  esac
  [ -f "$SIM_SIF" ] || { echo "FATAL: no $SIM_SIF" >&2; exit 2; }
}

# run_config <config> <out dir>: the config's sim with its output in
# <out dir>/sim (never in the checkout), then its per-episode scores in
# <out dir>/episodes.csv. Fails if the sim or the scores fail.
run_config() {
  local cfg="$1" out="$2" label
  label="$(basename "$cfg" .yml)"
  mkdir -p "$out"
  in_sim_image python infra/dais/sim_variant.py config "$cfg" "$out/config.yml" \
    --output-dir "$out/sim"
  simulate "$out/config.yml" "$label"
  [ -f "$out/sim/sweep.json" ] || { echo "FATAL: $label wrote no sweep.json" >&2; exit 1; }
  in_sim_image python infra/dais/gpusim_equiv.py episodes "$out/config.yml" \
    "$out/episodes.csv"
}
