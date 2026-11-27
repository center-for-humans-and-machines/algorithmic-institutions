#!/bin/bash -l
# One sim on CPUs only, in the unchanged sim image, with the repo's own entry
# point, its output in the run folder (not in the checkout). Used to compare a
# Raven (A100) result with the same config on DAIS CPUs.
#
#   sbatch infra/dais/sim_cpu.slurm.sh <checkout> <config relative to it>
#
# Job 986869 (2026-11-27, node daisg302, an AMD EPYC RTX node) ran an earlier
# version, which wrote into the checkout's plots/.
#
#SBATCH -J pf_simcpu
#SBATCH -o /dais/fs/scratch/%u/policy-finder/logs/%x.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=small
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=01:00:00

set -euo pipefail
PF_REPO="${1:?checkout}"
CONFIG="${2:?config}"
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/common.sh"
init_run
hostname
nproc
cd "$PF_REPO"
in_sim_image python /opt/dais/sim_variant.py config "$CONFIG" "$PF_RUN/config.yml" \
  --output-dir "$PF_RUN/out"
simulate "$PF_RUN/config.yml" "$(basename "$CONFIG" .yml)"
cat "$PF_RUN/out/sweep.json"
