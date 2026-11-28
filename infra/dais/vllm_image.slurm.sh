#!/bin/bash -l
# Build a vLLM image (the official vllm/vllm-openai Docker image of a release)
# as a SIF on scratch, for a model the group's vllm.sif (v0.25.1) does not
# support, and record its sha256 and vLLM version.
#
#   sbatch --export=VLLM_TAG=v0.34.0 infra/dais/vllm_image.slurm.sh
#
# Writes $PF_SCRATCH/images/vllm-<tag>.sif. Point VLLM_SIF at it (and
# EXPECT_VLLM_TAG=vllm/vllm-openai:<tag>) to serve with it.
#
#SBATCH -J pf_vllmimg
#SBATCH -o /dais/fs/scratch/%u/policy-finder-agents/logs/%x.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=small
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=02:00:00

set -euo pipefail
TAG="${VLLM_TAG:?set VLLM_TAG, e.g. v0.34.0}"
export PF_SCRATCH="${PF_SCRATCH:-/dais/fs/scratch/$USER/policy-finder-agents}"
module load apptainer/1.5.2
sif="$PF_SCRATCH/images/vllm-$TAG.sif"
work="$PF_SCRATCH/tmp/vllm-build.${SLURM_JOB_ID:-$$}"
mkdir -p "$work"
trap 'rm -rf "$work"' EXIT
export APPTAINER_TMPDIR="$work" APPTAINER_CACHEDIR="$work/cache"
apptainer build --force "$sif.partial" "docker://vllm/vllm-openai:$TAG"
mv "$sif.partial" "$sif"
echo "sif $sif sha256 $(sha256sum "$sif" | cut -d' ' -f1)"
ls -la "$sif"
apptainer inspect "$sif" | grep -iE "vllm|version" | head
apptainer exec --containall --cleanenv "$sif" python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
