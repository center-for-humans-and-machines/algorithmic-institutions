"""Copy an annotation run directory to and from the project S3 bucket.

    python3 scripts/annotation/s3_sync.py push <run_id> [--include-cache]
    python3 scripts/annotation/s3_sync.py pull <run_id> [--include-cache]
    python3 scripts/annotation/s3_sync.py ls   <run_id>

Local side: runs/annotation/<run_id>/ (or --local-dir). Remote side:
s3://chm-algorithmic-institutions/policy_finder/step_classification/<run_id>/.
A file is copied only when it is missing on the other side or differs in size
or content (MD5, or the multipart ETag computed with the remote part count).
A copy never replaces a newer file: push skips objects modified on S3 after
the local file's mtime, pull skips local files modified after the object
(both reported; --force copies anyway). A pulled file gets the object's
modification time. Nothing is ever deleted, on either side. The response
cache (<run>/cache/) is left out unless --include-cache is given.

Endpoint and keys come from the environment or the worktree .env
(AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY; the MINIO_*
names are a fallback). They are never printed.
"""

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig

REPO = Path(__file__).resolve().parents[2]
BUCKET = "chm-algorithmic-institutions"
PREFIX = "policy_finder/step_classification"
CHUNK = 8 * 1024 * 1024
TRANSFER = TransferConfig(multipart_threshold=CHUNK, multipart_chunksize=CHUNK)
SKIP_NAMES = {".DS_Store", "__pycache__"}


def load_env():
    env = {}
    path = REPO / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")
    env.update(
        {k: v for k, v in os.environ.items() if k.startswith(("AWS_", "MINIO_"))}
    )
    return env


def client():
    env = load_env()
    endpoint = env.get("AWS_ENDPOINT_URL") or env.get("MINIO_URL")
    key = env.get("AWS_ACCESS_KEY_ID") or env.get("MINIO_ACCESS_KEY")
    secret = env.get("AWS_SECRET_ACCESS_KEY") or env.get("MINIO_SECRET_KEY")
    if not (endpoint and key and secret):
        sys.exit("missing S3 endpoint or keys (AWS_* / MINIO_* in env or .env)")
    if not endpoint.startswith("http"):
        endpoint = "https://" + endpoint
    return boto3.client(
        "s3", endpoint_url=endpoint, aws_access_key_id=key, aws_secret_access_key=secret
    )


def etag_of(path, remote_etag):
    """Local ETag in the form of remote_etag: plain MD5, or multipart MD5-of-MD5s."""
    if "-" not in remote_etag:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    parts = int(remote_etag.rsplit("-", 1)[1])
    chunk = _part_size(path.stat().st_size, parts)
    digests = []
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digests.append(hashlib.md5(block).digest())
    return f"{hashlib.md5(b''.join(digests)).hexdigest()}-{len(digests)}"


def _part_size(size, parts):
    """The uploader's part size: ours first, then whole MiB sizes."""
    mib = 1024 * 1024
    for chunk in (CHUNK, *(m * mib for m in range(5, 513))):
        if math.ceil(size / chunk) == parts:
            return chunk
    return math.ceil(size / parts)


def local_files(root, include_cache):
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or SKIP_NAMES & set(rel.parts):
            continue
        if not include_cache and rel.parts[0] == "cache":
            continue
        out[rel.as_posix()] = p
    return out


def remote_files(s3, prefix, include_cache):
    out = {}
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix
    ):
        for o in page.get("Contents", []):
            rel = o["Key"][len(prefix) :]
            if not include_cache and rel.split("/", 1)[0] == "cache":
                continue
            out[rel] = {
                "size": o["Size"],
                "etag": o["ETag"].strip('"'),
                "mtime": o["LastModified"].timestamp(),
            }
    return out


def same(path, meta):
    return (
        path.stat().st_size == meta["size"]
        and etag_of(path, meta["etag"]) == meta["etag"]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["push", "pull", "ls"])
    ap.add_argument("run_id")
    ap.add_argument("--local-dir", type=Path, help="default runs/annotation/<run_id>")
    ap.add_argument("--include-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--force", action="store_true", help="also copy older over newer files"
    )
    a = ap.parse_args()
    if "/" in a.run_id or a.run_id in ("", ".", ".."):
        sys.exit(f"bad run_id {a.run_id!r}")
    root = a.local_dir or REPO / "runs" / "annotation" / a.run_id
    prefix = f"{PREFIX}/{a.run_id}/"
    s3 = client()
    remote = remote_files(s3, prefix, a.include_cache)

    if a.action == "ls":
        for rel, m in sorted(remote.items()):
            print(f"{m['size']:>12}  {rel}")
        total = sum(m["size"] for m in remote.values())
        print(
            f"{len(remote)} objects, {total / 1e6:.1f} MB under s3://{BUCKET}/{prefix}"
        )
        return

    copied = skipped = newer = 0
    if a.action == "push":
        if not root.is_dir():
            sys.exit(f"no local run dir {root}")
        for rel, path in local_files(root, a.include_cache).items():
            if rel in remote and same(path, remote[rel]):
                skipped += 1
                continue
            if rel in remote and remote[rel]["mtime"] > path.stat().st_mtime:
                if not a.force:
                    print(f"skip {rel}: newer on S3 (--force to overwrite)")
                    newer += 1
                    continue
            print(f"push {rel}")
            if not a.dry_run:
                s3.upload_file(str(path), BUCKET, prefix + rel, Config=TRANSFER)
            copied += 1
    else:
        for rel, meta in remote.items():
            path = root / rel
            if path.exists() and same(path, meta):
                skipped += 1
                continue
            if path.exists() and path.stat().st_mtime > meta["mtime"]:
                if not a.force:
                    print(f"skip {rel}: newer locally (--force to overwrite)")
                    newer += 1
                    continue
            print(f"pull {rel}")
            if not a.dry_run:
                path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(BUCKET, prefix + rel, str(path), Config=TRANSFER)
                os.utime(path, (meta["mtime"], meta["mtime"]))
            copied += 1
    verb = "would copy" if a.dry_run else "copied"
    print(
        f"{a.action}: {verb} {copied}, unchanged {skipped}, newer on the target "
        f"{newer} (s3://{BUCKET}/{prefix})"
    )


if __name__ == "__main__":
    main()
