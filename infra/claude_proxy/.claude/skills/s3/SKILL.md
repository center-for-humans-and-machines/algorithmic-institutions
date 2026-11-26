---
name: s3
description: List, upload, download, rename, or delete objects in the algorithmic-institutions S3 bucket via the CC command server. Use whenever the user asks to sync, list, upload, download, rename, or delete objects in s3://chm-algorithmic-institutions. These calls do not touch a cluster.
---

# S3

`s3cmd` runs on the command server with `/root/.s3cfg`. This project may only touch `s3://chm-algorithmic-institutions`. There is no `cluster` field. If `/root/.s3cfg` is missing, stop and tell the user.

- Base URL: `http://cc_command_server:8765/`
- Logs: `/var/cc_output/<run_id>.log` (mounted here; `Read` them directly)

Every endpoint that starts work returns:

```json
{"run_id": "<id>", "output_file": "/var/cc_output/<id>.log"}
```

Poll `GET /runs/{run_id}` (`running` / `exited`, `exit_code`, `tail`). Sleep a few seconds before the first poll.

Pass `key` relative to that bucket, never an `s3://` URI and never another bucket. A leading slash, an empty component, `.` or `..`, and a component that starts with `.` are rejected. `local_path` must be `/opt/algorithmic-institutions` or `/var/cluster_transfer`, or a path under one of them. A symlink that leaves that tree is rejected.

A trailing slash copies directory contents. `delete: true` is `s3cmd --delete-removed`. It is refused for the bucket root on upload and for a mount root on download. `/s3_remove` deletes that key and everything under it, refuses the bucket root, and rejects a trailing slash. `/s3_rename` moves inside the same bucket. `source` and `destination` must differ and must not end with `/`.

`excludes` and `includes` are globs (letters, digits, `._+:@*-/?`), at most 40 each. An entry must not start with `-`.

## Endpoints

### `GET /s3_ls`

Query: `key` (empty lists the bucket root), `recursive` (default false).

### `POST /s3_sync_to` and `POST /s3_sync_from`

Upload is `/s3_sync_to`. Download is `/s3_sync_from`.

| Field | Notes |
|---|---|
| `local_path` | Absolute path under an allowed root. |
| `key` | Relative key. Empty is the bucket root. |
| `excludes` | Optional globs. |
| `includes` | Optional globs, applied with excludes. |
| `delete` | Optional. Refused for the bucket root on upload and a mount root on download. |
| `dry_run` | Optional. |

```bash
curl -s -X POST http://cc_command_server:8765/s3_sync_from \
  -H 'Content-Type: application/json' \
  -d '{"local_path": "/var/cluster_transfer/artifacts/",
       "key": "artifacts/"}'
```

### `POST /s3_remove`

Body: `{"key": "<key>"}`. No trailing slash.

### `POST /s3_rename`

Body: `{"source": "<key>", "destination": "<key>"}`. No trailing slashes.
