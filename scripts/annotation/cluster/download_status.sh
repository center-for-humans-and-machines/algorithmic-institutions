#!/bin/bash
# Progress of a download started with download_model.sh. Run on the Mac:
#   scripts/annotation/cluster/download_status.sh [cluster] [repo_id] [revision]
# Defaults: dais, the production model and revision from config.sh.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cluster=${1:-dais}
# shellcheck source=config.sh
source "$here/config.sh"
cluster_config "$cluster"
repo=${2:-$CFG_PROD_MODEL}
rev=${3:-$CFG_PROD_REVISION}
hf_home=$CFG_PROD_HF_HOME

ssh -o BatchMode=yes "$cluster" bash -s -- "$repo" "$rev" "$hf_home" <<'EOF'
set -euo pipefail
repo=$1 rev=$2 hf_home=$3
state="$hf_home/annot_downloads/${repo//\//--}/$rev"
model="$hf_home/hub/models--${repo//\//--}"
[ -d "$state" ] || { echo "no download state at $state"; exit 1; }
total=$(python3 -c "import json;print(json.load(open('$state/expected.json'))['total_bytes'])" 2>/dev/null || echo 0)
have=$(du -sb --apparent-size "$model/blobs" 2>/dev/null | cut -f1 || echo 0)
have=${have:-0}
pid=$(cat "$state/pid" 2>/dev/null || echo -)
host=$(cat "$state/host" 2>/dev/null || echo -)
running=no
if [ "$host" = "$(hostname)" ] && kill -0 "$pid" 2>/dev/null; then running=yes
elif [ "$host" != "$(hostname)" ]; then running="unknown (started on $host, this is $(hostname))"; fi
echo "model:    $repo @ $rev"
echo "path:     $model"
if [ "$total" -gt 0 ]; then
  awk -v h="$have" -v t="$total" 'BEGIN{printf "bytes:    %.1f / %.1f GB (%.1f%%)\n", h/1e9, t/1e9, 100*h/t}'
fi
echo "process:  pid $pid, running: $running"
for f in DONE VERIFIED FAILED; do
  [ -f "$state/$f" ] && printf "%-9s %s\n" "$f:" "$(head -c 300 "$state/$f")"
done
echo "--- log tail"
tail -c 2000 "$state/log" 2>/dev/null | tr "\r" "\n" | tail -n 4 || true
EOF
