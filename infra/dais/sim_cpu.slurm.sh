#!/bin/bash -l
# One sim on CPUs only, in the unchanged sim image, with the repo's own entry
# point. Used to compare a Raven (A100) result with the same config on DAIS
# CPUs, where torch 1.11+cu113 has no kernels for H200/B200.
#
#   sbatch infra/dais/sim_cpu.slurm.sh <checkout> <config relative to it>
#
#SBATCH -J pf_simcpu
#SBATCH -o /dais/fs/scratch/levinb/policy-finder/logs/sim_cpu.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=small
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=01:00:00

set -uo pipefail
PF_REPO="${1:?checkout}"
CONFIG="${2:?config}"
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/common.sh"
export CUDA_VISIBLE_DEVICES=""
cd "$PF_REPO"
hostname
nproc
simulate "$CONFIG" "$(basename "$CONFIG" .yml)"
