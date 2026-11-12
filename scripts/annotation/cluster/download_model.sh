#!/bin/bash -l
# Download a Hugging Face model snapshot on a cluster LOGIN node (compute nodes
# have no internet), pinned to a revision sha, then verify every file against
# the Hub's size and LFS sha256. Runs detached so it survives the ssh session.
#
#   download_model.sh start <repo_id> <revision_sha> <hf_home> <image>
#   download_model.sh run   <repo_id> <revision_sha> <hf_home> <image>   (foreground)
#
# State lives in <hf_home>/annot_downloads/<repo>/<sha>/ : pid, host, log,
# expected.json, DONE, VERIFIED or FAILED. download_status.sh reads it.
# A token, if needed, comes from HF_TOKEN in the caller's environment (pass it
# through ssh stdin, never on the command line); it is handed to the container
# as APPTAINERENV_HF_TOKEN and never written to disk.
set -euo pipefail

mode=${1:?mode start|run}
repo=${2:?repo_id}
rev=${3:?revision sha}
hf_home=${4:?hf_home}
image=${5:?image}

[[ $rev =~ ^[0-9a-f]{40}$ ]] || { echo "revision must be a 40-char sha: $rev" >&2; exit 2; }
state="$hf_home/annot_downloads/${repo//\//--}/$rev"
mkdir -p "$state"

if [ "$mode" = start ]; then
    if [ -f "$state/pid" ] && kill -0 "$(cat "$state/pid")" 2>/dev/null; then
        echo "already running: pid $(cat "$state/pid") on $(cat "$state/host")"
        exit 0
    fi
    rm -f "$state/FAILED"
    setsid nohup "$0" run "$repo" "$rev" "$hf_home" "$image" \
        >>"$state/log" 2>&1 </dev/null &
    echo $! >"$state/pid"
    hostname >"$state/host"
    echo "started pid $! on $(hostname); log $state/log"
    exit 0
fi

module load apptainer >/dev/null 2>&1 || true
export APPTAINERENV_HF_TOKEN=${HF_TOKEN:-}
unset HF_TOKEN
# shellcheck disable=SC2154
trap 'rc=$?; [ $rc -eq 0 ] || { echo "exit $rc $(date -Is)" >"$state/FAILED"; }' EXIT
echo "### $(date -Is) download $repo@$rev into $hf_home on $(hostname)"

apptainer exec --cleanenv --bind "$hf_home" \
    --env "HF_HOME=$hf_home,HF_HUB_DISABLE_TELEMETRY=1" \
    "$image" python3 - "$repo" "$rev" "$state" <<'EOF'
import hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, snapshot_download

repo, rev, state = sys.argv[1:4]
token = os.environ.get("HF_TOKEN") or None
info = HfApi().model_info(repo, revision=rev, files_metadata=True, token=token)
assert info.sha == rev, (info.sha, rev)
files = {
    s.rfilename: {"size": s.size, "sha256": s.lfs.sha256 if s.lfs else None}
    for s in info.siblings
}
with open(f"{state}/expected.json", "w") as f:
    json.dump({"repo": repo, "revision": rev,
               "total_bytes": sum(v["size"] or 0 for v in files.values()),
               "files": files}, f)
t0 = time.time()
for attempt in range(5):
    try:
        path = snapshot_download(repo, revision=rev, max_workers=16, token=token)
        break
    except Exception as e:  # network blips: retry, resumes from partial files
        print(f"attempt {attempt} failed: {e!r}", flush=True)
        time.sleep(60 * (attempt + 1))
else:
    sys.exit("download failed after 5 attempts")
print(f"downloaded to {path} in {time.time() - t0:.0f} s", flush=True)
open(f"{state}/DONE", "w").write(path + "\n")


def check(item):
    name, meta = item
    p = os.path.join(path, name)
    if not os.path.exists(p):
        return name, "missing"
    if meta["size"] is not None and os.path.getsize(p) != meta["size"]:
        return name, f"size {os.path.getsize(p)} != {meta['size']}"
    if meta["sha256"]:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != meta["sha256"]:
            return name, "sha256 mismatch"
    return name, None


t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    bad = {n: e for n, e in ex.map(check, files.items()) if e}
print(f"verified {len(files)} files in {time.time() - t0:.0f} s, {len(bad)} bad",
      flush=True)
if bad:
    print(json.dumps(bad, indent=1))
    sys.exit(1)
open(f"{state}/VERIFIED", "w").write(f"{len(files)} files ok {time.ctime()}\n")
EOF
echo "### $(date -Is) finished"
