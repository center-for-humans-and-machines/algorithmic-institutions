#!/bin/bash
# Download a model's weights into the Hugging Face cache on scratch (HF_HOME),
# pinned to a revision, with the huggingface_hub of the vLLM image. Run on the
# login node (its network is fine; 149 GB took 5 minutes), niced, in the
# background:
#
#   nohup setsid bash infra/dais/download_model.sh <repo> <revision> \
#     > $PF_SCRATCH/logs/download.<name>.log 2>&1 &
#
# No token: the candidates are ungated. Prints DONE <path> at the end; rerun
# to resume an interrupted download.
set -euo pipefail
REPO="${1:?repo}" REV="${2:?revision}"
export HF_HOME="${HF_HOME:-/dais/fs/scratch/$USER/models}"
SIF="${DOWNLOAD_SIF:-/dais/fs/scratch/bsup/container/vllm.sif}"
module load apptainer/1.5.2 >/dev/null 2>&1 || true
echo "$(date +%T) $REPO@$REV into $HF_HOME"
nice -n 10 apptainer exec --containall --cleanenv --bind "$HF_HOME" \
  --env HF_HOME="$HF_HOME" --env HF_HUB_ENABLE_HF_TRANSFER=1 \
  "$SIF" python3 -c "
import sys
from huggingface_hub import snapshot_download
# every file of the revision: a newer huggingface_hub refuses an incomplete
# snapshot offline (IncompleteSnapshotError), and the skipped images and
# PDFs are a few MB
p = snapshot_download(sys.argv[1], revision=sys.argv[2], max_workers=8)
print('DONE', p, flush=True)
" "$REPO" "$REV"
echo "$(date +%T) finished"
