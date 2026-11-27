#!/bin/bash -l
# Convert the OpenCode agent image (algoinst_claude:opencode, built on the
# Hetzner host from infra/claude_proxy/container/CC_env/Dockerfile.opencode)
# from a `docker save | gzip` tarball to a SIF, without root, and check it.
#
#   # on a machine that can reach both (the tarball is streamed, not stored):
#   ssh <docker host> 'docker save algoinst_claude:opencode | gzip -1' \
#     | ssh dais 'cat > $PF_SCRATCH/images/algoinst_claude_opencode.tar.gz'
#   sbatch --export=ALL infra/dais/agent_image.slurm.sh
#
# Writes $PF_SCRATCH/images/algoinst_claude_opencode.sif and prints its
# sha256 (for AGENT_SIF_SHA256 in common.sh) and OpenCode's version. The
# decompressed tarball (Apptainer reads docker-archive only uncompressed) is
# removed afterwards; the .tar.gz is kept.
#
#SBATCH -J pf_agentimg
#SBATCH -o /dais/fs/scratch/%u/policy-finder-agents/logs/%x.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=small
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=01:00:00

set -euo pipefail
export PF_SCRATCH="${PF_SCRATCH:-/dais/fs/scratch/$USER/policy-finder-agents}"
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/common.sh"
init_run

src="$PF_SCRATCH/images/algoinst_claude_opencode.tar.gz"
sif="$PF_SCRATCH/images/algoinst_claude_opencode.sif"
tar="$PF_RUN/tmp/image.tar"
[ -f "$src" ] || { echo "FATAL: no $src" >&2; exit 1; }
echo "tarball sha256 $(sha256sum "$src" | cut -d' ' -f1)"
gzip -dc "$src" > "$tar"
export APPTAINER_TMPDIR="$PF_RUN/tmp" APPTAINER_CACHEDIR="$PF_RUN/tmp/cache"
apptainer build --force "$sif.partial" "docker-archive://$tar"
rm -f "$tar"
mv "$sif.partial" "$sif"
echo "sif sha256 $(sha256sum "$sif" | cut -d' ' -f1)"
ls -la "$sif"
apptainer exec --containall --cleanenv "$sif" opencode --version
apptainer exec --containall --cleanenv "$sif" sh -c 'echo "XDG_CONFIG_HOME=$XDG_CONFIG_HOME"; ls /opt/opencode/config/opencode; command -v python3; python3 --version'
