---
name: container_spawner
description: Start, watch, and cancel a locked algoinst container through the container spawner. Use when the user asks to spawn a container, run Claude in an isolated container, or call container_spawner. Spawned containers cannot reach this container, the command server, or the spawner.
---

# Container spawner

A sibling container (`container_spawner`) starts allowlisted containers through the host Docker socket. **Do not run `docker` yourself.** This container has no Docker socket.

## Connection

- Base URL: `http://container_spawner:8877/`
- `CC_PROJECT` is `algoinst`, fixed on the server. A request cannot switch it.
- Profile `cpu` runs `algoinst_claude:cpu` (user `1000:1000`). There is no other profile. At startup the server copies `algoinst_claude:latest` onto that tag if it is missing. A rebuild retags `algoinst_claude:cpu` only, not the image of the `claude` service.
- The server reads the setup token and injects it as `CLAUDE_CODE_OAUTH_TOKEN`. Do not send a token, an env block, an image name, a network, or Docker flags. Extra JSON fields are rejected.
- Allowlist and ceilings: `/opt/claude_tools/container_spawner/config/algoinst.yml`. Read it before the first call. If the mount root you need is absent, stop and tell the user. Do not invent a root.

A spawned container is on its own internal network. Its only egress is `api.anthropic.com:443`. It cannot reach `container_spawner`, `cc_command_server`, or this container. You reach it only through this API (create, status, logs, cancel).

## What you can set

| Field | Notes |
|---|---|
| `profile` | `"cpu"`. |
| `command` | Argument list, not a shell string. |
| `workdir` | Optional absolute path inside the spawned container. |
| `mounts` | Subpaths of a named root. See below. |
| `cpus` | Optional. Default 2, ceiling 4. Above the ceiling is HTTP 400. The server sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to that quota (at least 1). Do not wrap `command` in `env` to set them. |
| `memory_mb` | Optional. Default 8192, range 128–16384. |
| `pids_limit` | Optional. Default 256, range 16–512. |
| `timeout_seconds` | Optional. Default 21600 (6 h), ceiling 86400. The server kills the container at the deadline (`status` `expired`). |

At most 8 runs in `starting` or `running`. A further create returns HTTP 409.

`command` whose first element is `claude` or a path ending in `/claude` gets `--dangerously-skip-permissions` inserted after the executable. The stored command is the one that actually runs. There is no stdin after start: a prompt has to be an argument (`claude -p "…"`). Headless flags such as `--agent` and `--strict-mcp-config` belong in `command`.

## Mounts

Each mount is `{"root", "path", "target", "read_only"}`.

- `root` is a key of `mount_roots` in the yaml. `path` is relative to that root (`""` is the root itself). It must already exist. `..`, a trailing slash, and a symlink that leaves the root are rejected.
- `target` is an absolute path inside the spawned container. The server orders mounts so a parent is applied before a child, which is how a read-only tree and a read-write subdirectory are combined.
- `read_only` defaults to true. A root with `writable: false` rejects `read_only: false`.
- Do not target `/`, `/home`, `/home/claude`, or anything under `/home/claude`. The server mounts a fresh home there and keeps it after the container is removed. `home_host_path` in the response is a host path. It is not mounted in this container. Stdout and stderr come from the logs endpoint.

## Endpoints

### `GET /health`

`{"ok": true, "project": "algoinst"}`.

### `POST /runs`

```bash
curl -s -X POST http://container_spawner:8877/runs \
  -H 'Content-Type: application/json' \
  -d '{"profile": "cpu",
       "command": ["claude", "-p", "Design your rule."],
       "workdir": "/opt/algorithmic-institutions",
       "mounts": [
         {"root": "repo", "path": "export/job1", "target": "/opt/algorithmic-institutions", "read_only": true},
         {"root": "repo", "path": "export/job1/notes/policy_finder", "target": "/opt/algorithmic-institutions/notes/policy_finder", "read_only": false}
       ]}'
```

Returns the run: `id`, `status`, `command`, `mounts`, `cpus`, `memory_mb`, `pids_limit`, `timeout_seconds`, `exit_code`, `error`, `started_at`, `deadline_at`, `finished_at`, `home_host_path`. `status` moves through `starting` to `running`, then `succeeded`, `failed`, `cancelled`, `expired`, or `error`.

Poll `GET /runs/{id}`. A run may last hours. Sleep minutes between polls, not a tight loop.

### `GET /runs`

`{"runs": [ ... ]}`, newest first.

### `GET /runs/{id}/logs?tail=200`

`{"id", "stdout", "stderr"}`. `tail` is 1–5000, default 200. The token string is redacted.

### `POST /runs/{id}/cancel`

Kills a `starting` or `running` run. `status` becomes `cancelled`. Cancelling a finished run is HTTP 409.

### `POST /profiles/cpu/rebuild`

No body. Rebuilds `algoinst_claude:cpu` from the Dockerfile fixed in the profile. It does not retag `algoinst_claude:latest`. HTTP 400 if the build context is unset. Do not pass a path.
