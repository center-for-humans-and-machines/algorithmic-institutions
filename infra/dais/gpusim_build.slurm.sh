#!/bin/bash -l
# The GPU sim image as a SIF: `apptainer build` from the `docker save`
# tarball of Dockerfile.gpu-sim (built elsewhere: DAIS has no Docker), on a
# CPU node, without root.
#
#   cd /u/$USER/algorithmic-institutions-gpusim
#   sbatch infra/dais/gpusim_build.slurm.sh <image.tar.gz>
#
#SBATCH -J gpusim_build
#SBATCH --account=mpib_inst
#SBATCH --partition=small
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=01:00:00

set -euo pipefail
TARBALL="${1:?docker save tarball}"
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/gpusim_common.sh"
tmp="$GS_SCRATCH/tmp/build.$SLURM_JOB_ID"
mkdir -p "$tmp"
export APPTAINER_TMPDIR="$tmp" APPTAINER_CACHEDIR="$tmp/cache"
sha256sum "$TARBALL"
t0=$SECONDS
# docker-archive: reads a plain tar, not a gzipped one
if [[ "$TARBALL" == *.gz ]]; then
  gunzip -c "$TARBALL" > "$tmp/image.tar"
  TARBALL="$tmp/image.tar"
fi
apptainer build --force "$NEW_SIF.partial" "docker-archive://$TARBALL"
mv "$NEW_SIF.partial" "$NEW_SIF"
echo "built in $(( SECONDS - t0 )) s"
ls -la "$NEW_SIF"
sha256sum "$NEW_SIF"
apptainer exec --cleanenv "$NEW_SIF" python -c \
  'import torch, torch_geometric; print(torch.__version__, torch.version.cuda, torch._C._cuda_getArchFlags(), torch_geometric.__version__)'
rm -rf "$tmp"
