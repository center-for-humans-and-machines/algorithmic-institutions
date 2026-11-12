#!/bin/bash -l
# vLLM server + annotation command in one job on DAIS (B200 nodes).
# Submit through submit.sh, which overrides partition, time, typed GRES, CPUs,
# memory, account and QoS from config.sh and passes ANNOT_JOB_ENV. The values
# below are only fallbacks. GRES must be typed (gpu:b200:N): --constraint is
# rewritten by the site's job_submit.lua.
#SBATCH --job-name=annot-serve
#SBATCH --partition=gpu1
#SBATCH --account=mpib_inst
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@900

set -euo pipefail
: "${ANNOT_JOB_ENV:?submit through scripts/annotation/cluster/submit.sh}"
# shellcheck source=/dev/null
source "$ANNOT_JOB_ENV"
# shellcheck source=common.sh
source "${CFG_CODE_DIR:-$CFG_PROJECT}/scripts/annotation/cluster/common.sh"

annot_container_flags() {
    # --contain: no host /tmp or home leak into the image. The container gets
    # its own /dev with a private /dev/shm (tmpfs, half the node's RAM), shared
    # by all processes of the server exec, so tensor-parallel workers are fine;
    # a --bind /dev/shm is skipped by apptainer ("already mounted").
    CONTAINER_FLAGS=(--nv --contain --bind /dais)
    # --cleanenv drops Slurm's device selection; forward it when set
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    fi
}

annot_server_flags() {
    SERVER_FLAGS=()
}

annot_gpu_sampler() {
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv,nounits -l "$CFG_SAMPLE_SEC"
}

annot_main
