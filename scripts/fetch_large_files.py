"""Restore the data and model files that are no longer in git on this branch.

`docs/large_files.tsv` lists every file that was a Git LFS object on
`yannik/handover` before LFS was removed (2026-11-03): its path, sha256, size
and the key of its copy on S3 (`s3://chm-algorithmic-institutions`). Files
without an S3 key exist only in the LFS history; `--lfs` restores those from
LFS_COMMIT, which needs git-lfs. Every restored file is checked against its
sha256. See `docs/large_files.md`.

    python scripts/fetch_large_files.py --list 'artifacts/artificial_humans/*'
    python scripts/fetch_large_files.py 'experiments/*' 'artifacts/artificial_humans/raven_script_22/*'
    python scripts/fetch_large_files.py --via command-server --lfs   # everything, from the Claude proxy

Patterns are shell globs on the repo path (`*` also crosses `/`); a pattern
without glob characters also matches everything below it as a directory.
No pattern selects every file.
"""

import argparse
import fnmatch
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "docs/large_files.tsv"
BUCKET = "chm-algorithmic-institutions"
# last commit of yannik/handover with the files as LFS pointers (tag lfs-archive/yannik-handover)
LFS_COMMIT = "720a7075767de09330b72d32a049fa7766848a77"
COMMAND_SERVER = "http://cc_command_server:8765"
# the command server only writes under these roots
SERVER_ROOTS = (Path("/opt/algorithmic-institutions"), Path("/var/cluster_transfer"))


def read_table():
    lines = TABLE.read_text().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def matches(path, patterns):
    if not patterns:
        return True
    for p in patterns:
        if fnmatch.fnmatchcase(path, p):
            return True
        if not any(c in p for c in "*?[") and path.startswith(p.rstrip("/") + "/"):
            return True
    return False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_current(dest, row):
    return dest.is_file() and dest.stat().st_size == int(row["bytes"]) and sha256(dest) == row["sha256"]


def via_s3cmd(row, dest):
    subprocess.run(
        ["s3cmd", "get", "--force", "--no-progress", f"s3://{BUCKET}/{row['s3_key']}", str(dest)],
        check=True, stdout=subprocess.DEVNULL,
    )


def _http(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def via_command_server(row, dest):
    if not any(dest.is_relative_to(r) for r in SERVER_ROOTS):
        raise RuntimeError(f"the command server only writes under {', '.join(map(str, SERVER_ROOTS))}")
    run = _http("POST", f"{COMMAND_SERVER}/s3_sync_from",
                {"local_path": str(dest), "key": row["s3_key"]})["run_id"]
    while True:
        time.sleep(2)
        state = _http("GET", f"{COMMAND_SERVER}/runs/{run}")
        if state["status"] == "exited":
            break
    if state["exit_code"] != 0:
        raise RuntimeError(f"download failed, see /var/cc_output/{run}.log")


def via_lfs(row, dest):
    pointer = subprocess.run(["git", "-C", str(ROOT), "show", f"{LFS_COMMIT}:{row['path']}"],
                             check=True, capture_output=True).stdout
    with open(dest, "wb") as out:
        subprocess.run(["git", "-C", str(ROOT), "lfs", "smudge", row["path"]],
                       input=pointer, stdout=out, check=True)


def fetch(row, dest_root, download, use_lfs, force):
    dest = dest_root / row["path"]
    if not force and is_current(dest, row):
        return "present"
    if row["s3_key"]:
        get, source = download, "s3"
    elif use_lfs:
        get, source = via_lfs, "lfs"
    else:
        return "lfs-only"
    dest.parent.mkdir(parents=True, exist_ok=True)
    get(row, dest)
    if not is_current(dest, row):
        raise RuntimeError(f"{row['path']}: sha256 does not match docs/large_files.tsv")
    return source


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="*", help="globs on the repo path; none selects every file")
    ap.add_argument("--via", choices=["s3cmd", "command-server"], default="s3cmd",
                    help="s3cmd with ~/.s3cfg (default), or the Claude proxy's command server")
    ap.add_argument("--lfs", action="store_true",
                    help="restore files without an S3 copy from the LFS history (needs git-lfs)")
    ap.add_argument("--dest", type=Path, default=ROOT, help="write under this root instead of the repo")
    ap.add_argument("--force", action="store_true", help="download even if the file is already current")
    ap.add_argument("--list", action="store_true", help="only list the selected files and their source")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    rows = [r for r in read_table() if matches(r["path"], args.patterns)]
    if not rows:
        sys.exit("no file in docs/large_files.tsv matches")
    if args.list:
        for r in rows:
            print(f"{int(r['bytes']) / 1e6:8.2f} MB  {r['s3_key'] and 's3 ' or 'lfs'}  {r['path']}")
        print(f"{len(rows)} files, {sum(int(r['bytes']) for r in rows) / 1e6:.1f} MB,"
              f" {sum(not r['s3_key'] for r in rows)} only in LFS")
        return
    if args.via == "s3cmd" and not shutil.which("s3cmd"):
        sys.exit("s3cmd not found (or use --via command-server from the Claude proxy)")
    download = via_s3cmd if args.via == "s3cmd" else via_command_server
    dest_root = args.dest.resolve()

    counts, failed = {}, []

    def one(row):
        try:
            return row, fetch(row, dest_root, download, args.lfs, args.force)
        except Exception as e:  # report every failure, keep going
            return row, e

    with ThreadPoolExecutor(args.jobs) as pool:
        for row, result in pool.map(one, rows):
            if isinstance(result, Exception):
                failed.append(row["path"])
                print(f"FAILED {row['path']}: {result}", file=sys.stderr)
                continue
            counts[result] = counts.get(result, 0) + 1
            if result == "lfs-only":
                print(f"skipped (only in LFS, rerun with --lfs): {row['path']}")
    if failed:
        counts["failed"] = len(failed)
    print(", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
