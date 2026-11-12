#!/bin/bash
# Copy the annotation code (and optionally one run directory) from this
# checkout to the cluster project dir on temporary storage.
#   scripts/annotation/cluster/sync_code.sh <viper|dais> [run_id]
# Code trees are mirrored (stale modules removed). A run directory is only
# added to: no --delete and --update, so outputs a job wrote on the cluster are
# never removed or overwritten by an older local copy.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cluster=${1:?usage: sync_code.sh <viper|dais> [run_id]}
run_id=${2:-}
repo=$(git -C "$here" rev-parse --show-toplevel)
# shellcheck source=config.sh
source "$here/config.sh"
cluster_config "$cluster"
P=$CFG_PROJECT

excludes=(--exclude __pycache__ --exclude '*.pyc' --exclude .pytest_cache
    --exclude .DS_Store --exclude '.ipynb_checkpoints')
ssh -o BatchMode=yes "$cluster" "mkdir -p '$P/src/aimanager' '$P/configs' '$P/scripts' '$P/runs/annotation'"
rsync -a "$repo/src/aimanager/__init__.py" "$cluster:$P/src/aimanager/"
for d in src/aimanager/annotation configs/annotation scripts/annotation; do
    [ -d "$repo/$d" ] || { echo "skip $d (not in checkout)"; continue; }
    rsync -a --delete "${excludes[@]}" "$repo/$d/" "$cluster:$P/$d/"
done
sha=$(git -C "$repo" rev-parse HEAD)
dirty=$(git -C "$repo" status --porcelain -- src/aimanager/annotation configs/annotation scripts/annotation | head -c1)
echo "$sha${dirty:+ dirty} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | ssh -o BatchMode=yes "$cluster" "cat > '$P/CODE_VERSION'"
echo "code -> $cluster:$P ($sha${dirty:+, dirty})"

if [ -n "$run_id" ]; then
    local_run="$repo/runs/annotation/$run_id"
    if [ -d "$local_run" ]; then
        rsync -a --update "${excludes[@]}" "$local_run/" "$cluster:$P/runs/annotation/$run_id/"
        echo "run $run_id -> $cluster:$P/runs/annotation/$run_id"
    else
        echo "no local run dir $local_run; nothing to push"
    fi
fi
