#!/bin/bash -l
# vLLM server + annotation command in one job on Viper-GPU (2x AMD MI300A).
# Submit through submit.sh, which overrides partition, time, GRES, CPUs and
# memory from config.sh and passes ANNOT_JOB_ENV. The values below are only
# fallbacks. Never load the host `rocm` module outside the GPU sampler: its
# HIP_PATH/ROCM_PATH leak into the container and crash vLLM's AITER.
#SBATCH --job-name=annot-serve
#SBATCH --partition=apu
#SBATCH --constraint=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=220000
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@600

set -euo pipefail
: "${ANNOT_JOB_ENV:?submit through scripts/annotation/cluster/submit.sh}"
# shellcheck source=/dev/null
source "$ANNOT_JOB_ENV"
# shellcheck source=common.sh
source "$CFG_PROJECT/scripts/annotation/cluster/common.sh"

annot_container_flags() {
    # --cleanenv drops Slurm's device selection; forward only that
    if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
        export APPTAINERENV_ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
        export APPTAINERENV_HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$ROCR_VISIBLE_DEVICES}"
    fi
}

annot_server_flags() {
    # writable overlay for AITER's JIT kernels, as in the LLMIS template
    apptainer overlay create --size 1024 "$JOB_TMP/overlay.img" >/dev/null
    SERVER_FLAGS=(--overlay "$JOB_TMP/overlay.img")
}

annot_gpu_sampler() {
    module load rocm/7.0 >/dev/null 2>&1
    while true; do
        echo "### $(date +%s)"
        rocm-smi --showuse --showmemuse --showpower --showmeminfo vram --csv 2>/dev/null
        sleep "$CFG_SAMPLE_SEC"
    done
}

annot_main
