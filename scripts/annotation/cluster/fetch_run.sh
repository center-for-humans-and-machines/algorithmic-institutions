#!/bin/bash
# Copy a run directory from the cluster to runs/annotation/<run_id>/ here.
#   scripts/annotation/cluster/fetch_run.sh <viper|dais> <run_id> [--no-cache]
# Never deletes local files; skips files that are newer locally (--update).
# --no-cache leaves out the response cache (<run>/cache/).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cluster=${1:?usage: fetch_run.sh <viper|dais> <run_id> [--no-cache]}
run_id=${2:?usage: fetch_run.sh <viper|dais> <run_id> [--no-cache]}
excl=()
[ "${3:-}" = --no-cache ] && excl=(--exclude /cache/)
repo=$(git -C "$here" rev-parse --show-toplevel)
# shellcheck source=config.sh
source "$here/config.sh"
cluster_config "$cluster"
dest="$repo/runs/annotation/$run_id"
mkdir -p "$dest"
rsync -a --update ${excl[@]+"${excl[@]}"} "$cluster:$CFG_RUNS_ROOT/$run_id/" "$dest/"
echo "$cluster:$CFG_RUNS_ROOT/$run_id -> $dest"
