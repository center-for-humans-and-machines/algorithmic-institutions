---
name: run_on_cluster
description: Run algorithmic-institutions jobs on MPCDF Raven (`raven`) via the CC external command server. Use whenever the user asks to submit, query, cancel, monitor, sync, or interactively run something on Raven, or to sync, rename, or delete files on the MPCDF archive (slurm, salloc, GPU, Apptainer, archive.mpcdf.mpg.de).
---

# Running on Raven

algorithmic-institutions reaches **one** cluster through this proxy: MPCDF Raven (`raven`). DAIS, Arrhenius, Viper, and MareNostrum are rejected.

A sibling container (`cc_command_server`) SSHes to the Raven login node, runs a whitelisted command, and writes stdout/stderr to a log. **This container has no Raven SSH keys. Never SSH Raven yourself, and never run `scripts/*_cluster.sh` or `scripts/fetch_cluster.sh`.** Those scripts are the manual path on a machine that holds an `ssh raven` ControlMaster.

## Connection

- Base URL: `http://cc_command_server:8765/`
- Logs: `/var/cc_output/<run_id>.log` (mounted here; `Read` them directly)
- `cluster` is always `"raven"` (JSON field on POST, query param on GET)
- Image and binds are fixed. `wandb_mode` (`offline` / `online`) and `checkout` (a snapshot sha, below) are the exceptions.
- Concurrent jobs are capped at `CC_MAX_JOBS` (16). A further submit returns HTTP 409.
- Live checkout on Raven: `/u/$RAVEN_USER/algorithmic-institutions`, inside the image at `/opt/algorithmic-institutions` with `PYTHONPATH=/opt/algorithmic-institutions/src`. A snapshot checkout is `/u/$RAVEN_USER/cc-snapshots/<sha>`, mounted at the same container path.
- Image: `/raven/ptmp/ykeller/cc/containers/apptainer/algoinst.sif` (shared; the same for every user) (Python 3.9, torch 1.11.0+cu113, PyG 2.5). If a job fails because the image is missing, stop and tell the user. Do not build it from here.

`WANDB_ENTITY=chm-ml` and `WANDB_PROJECT=algorithmic-institutions` are set on every container start. The API key is loaded from `/u/$RAVEN_USER/private/cc_secrets`. W&B defaults to offline; pass `"wandb_mode": "online"` to sync.

If SSH fails, the image is missing, or the template is missing at `/u/$RAVEN_USER/slurm_tools/slurm_tools/run_slurm_raven_apptainer_algoinst.sh`, stop and tell the user. Do not work around it.

Every endpoint that starts work returns:

```json
{"run_id": "<id>", "output_file": "/var/cc_output/<id>.log"}
```

Poll `GET /runs/{run_id}` (`running` / `exited`, `exit_code`, `tail`). Sleep a few seconds before the first poll.

## What a job is

`POST /submit_slurm_job` runs one command inside the image. `launcher` is `python`. `program_call` is the arguments after python. The launcher strips a leading `python `, so both of these work:

- `-m aimanager train-ah configs/training/artificial_humans/...yml`
- `-m aimanager train-manager configs/training/rl_manager/...yml`
- `-m aimanager simulate configs/simulation/...yml`

`python -m aimanager train-ah` accepts the orchestrator YAML (keys under `params`). A config with top-level `grid` or `include` still belongs to the SSH orchestrator; the CLI exits and tells you so. Do not submit `src/aimanager/.../run.py`. That calls `sbatch`, and there is no `sbatch` inside the container.

Read resources from the config when it is an artificial-human YAML (`exec.cores`, `exec.memory`). Otherwise use the manual templates' values:

| Job | program_call | time | n_cpu | mem |
|---|---|---|---|---|
| artificial humans | `-m aimanager train-ah <config>` | `10:00:00` | `exec.cores` (else 4) | `<exec.memory>G` (else `16G`) |
| RL manager | `-m aimanager train-manager <config>` | `20:00:00` | 2 | `16G` |
| simulation | `-m aimanager simulate <config>` | `01:00:00` | 2 | `16G` |

`n_gpu` is 1 and `n_nodes` is 1. Raven's wall-time cap is 24 hours. `run_group` and `project` match `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$`. Use `project` `algorithmic-institutions`. Omit `mem` to get the server default `16G`.

Sync before a job. Two syncs exist. The working-tree rsync can change files under a job that is already running, and it will publish uncommitted edits. A snapshot publishes one commit and is never written again.

`POST /sync_to_cluster` with `{"cluster": "raven", "ref": "<sha or branch>"}` resolves `ref` in the mounted repo, checks that commit out with `git worktree add --detach` (so Git LFS files are real files, not pointers), and rsyncs it to `/u/$RAVEN_USER/cc-snapshots/<sha>/`. Unchanged files are hard links to the previous snapshot. The response adds `sha` as soon as the run starts; wait until that run exits 0 before using the sha. If that directory already exists it is left as it is.

Then pass the same sha as `checkout` on `POST /submit_slurm_job` and `POST /run_in_container`. The container bind is the snapshot. Two binds sit on top of it: `/u/$RAVEN_USER/algorithmic-institutions/artifacts` read-only (the shared models), and `/u/$RAVEN_USER/cc-snapshot-outputs/<sha>/{plots,.log}` writable. A snapshot job sees those as `plots/` and `.log/`. It cannot write new artifacts; use the working-tree sync for a job that must save models. `validate-rule` uses `checkout` on `/run_in_container`.

`checkout` on a slurm job needs `run_slurm_raven_apptainer_algoinst.sh` on Raven to contain the `{repo_binds}` placeholder. Until that template is deployed, snapshot jobs fail at submit; login-node `/run_in_container` does not need the template.

The working-tree sync is unchanged: `{"cluster": "raven"}` rsyncs `/opt/algorithmic-institutions/` to `raven:/u/$RAVEN_USER/algorithmic-institutions/`, excluding `.git`, `.venv`, `artifacts`, `plots`, `temp`, `.log`, and `__pycache__`. It does not delete remote files that are excluded, so artifacts already on Raven stay. Data and model files (`*.csv`, `*.parquet`, `*.pt`) are not in git on `main` (no LFS since 2026-11-03): before syncing, make sure the files this job reads exist in the checkout, and fetch missing ones with `python scripts/fetch_large_files.py --via command-server <glob>` (`docs/large_files.md`). On branches that still use LFS, `git lfs pull` them before a snapshot sync, or the server refuses to publish pointer files. Experiments CSV is not excluded.

Results of a live-checkout job are written into that checkout (`artifacts/`, `plots/`, `.log/`). Results of a snapshot job are under `/u/$RAVEN_USER/cc-snapshot-outputs/<sha>/`. To bring a file back, copy it to `/opt/cluster_transfer` from a login-node `/run_in_container` command (same `checkout` for a snapshot job), then `POST /sync_from_cluster`. Locally that tree is `/var/cluster_transfer`. Keep each pull under about 1 GB.

`POST /cleanup_snapshots` with `{"cluster": "raven"}` deletes snapshots older than 14 days (`older_than_days` overrides, or `CC_SNAPSHOT_MAX_AGE_DAYS`) that no queued or running job references. A snapshot whose output is newer than that is kept.

## Endpoints

### `POST /sync_to_cluster`

Working tree: `{"cluster": "raven"}`. Snapshot: `{"cluster": "raven", "ref": "<sha or branch>"}`, which adds `sha` to the response. `ref` and `includes` cannot be combined.

Optional `includes` on the working-tree sync sends specific repo-relative paths the excludes would skip, for example `{"cluster": "raven", "includes": ["artifacts/fig1.pdf"]}`. Literal paths only (no wildcards, no `..`). A directory path includes its contents; other files in an excluded directory stay excluded.

### `POST /submit_slurm_job`

```bash
curl -s -X POST http://cc_command_server:8765/submit_slurm_job \
  -H 'Content-Type: application/json' \
  -d '{"cluster": "raven",
       "program_call": "-m aimanager train-ah configs/training/artificial_humans/contribution/group_switching_contribution_50ep_group_vnode.yml",
       "time": "10:00:00", "n_gpu": 1, "n_nodes": 1, "n_cpu": 4,
       "launcher": "python", "run_group": "group-vnode",
       "project": "algorithmic-institutions", "mem": "8G",
       "wandb_mode": "online",
       "checkout": "<sha from the snapshot sync>"}'
```

Omit `checkout` to run the live tree. The slurm job id is in the run log. `GET /monitor_newest_slurm_run?cluster=raven` tails the newest job's output (`oldness` 0 is newest, `script=true` shows the generated batch script).

### `POST /run_in_container`

Login node, inside the image, no GPU. Body: `{"cluster": "raven", "command": "<shell>", "wandb_mode"?: "offline"|"online", "checkout"?: "<sha>"}`. Use for `python -m pytest ...`, `validate-rule`, and for copying results into `/opt/cluster_transfer`. Command max 4096 bytes, no newlines. `checkout` is the same sha as on the job.

### `POST /request_single_gpu`

One A100 for 2 hours (`salloc ... sleep 7200`). The run stays `running` until you kill it or the sleep ends. The job id is in the log. Pass it to `/run_in_job`.

### `POST /run_in_job`

Body: `{"cluster": "raven", "jobid": "<id>", "command": "<shell>"}`. New container in that allocation via `srun --overlap`.

### `GET /jobs` and `GET /sinfo`

`/jobs?cluster=raven` is `squeue --me`. `/sinfo?cluster=raven` is bare `sinfo`. Both return a `run_id`.

### `POST /cancel_job`

Body: `{"cluster": "raven", "jobid": "<id>"}`.

### `POST /sync_from_cluster`

Body: `{"cluster": "raven"}`. Pulls `/ptmp/$RAVEN_USER/cc/cluster_transfer` into `/var/cluster_transfer`.

### `POST /sync_to_archive`, `POST /sync_from_archive`, `POST /rename_on_archive`, `POST /remove_on_archive`

Move files between Raven and DAIS or Viper through the MPCDF archive (`archive.mpcdf.mpg.de`). This proxy only has Raven, so every call runs there (`cluster` defaults to `raven`). The other project's proxy pushes or pulls the same tree on `dais` or `viper12i`. Do not SSH the archive from a normal container command. The key is mounted only for these endpoints, and the command is a fixed `rsync`, `mv`, or `rm`.

The only allowed tree is `/ghi/r/y/ykeller/algoinst` and paths under it. `local_path` must be a Raven container path: `/opt/algorithmic-institutions`, `/opt/outputs`, `/opt/cluster_transfer`, `/opt/cluster_sync`, `/tmp`, `/home/ykeller`, or `/ptmp/ykeller/cc/container_cache`. A trailing `/` copies directory contents.

Sync body: `local_path`, `archive_path`, and optional `cluster` (`raven` only), `excludes`, `includes`, `delete`, `dry_run`, `rsync_args`, `on_gpu`, `time`. `on_gpu` starts `srun -p gpu --gres=gpu:a100:1 --mem=120GB -J cc-archive` and counts against `CC_MAX_JOBS`. `time` is only valid with `on_gpu` (`HH:MM:SS`, maximum `08:00:00`, default `02:00:00`). Extra `rsync_args` are one token each from `--checksum`, `-v`, `--bwlimit=50M`, `--exclude=*.tar`, `--max-size=1G`, `--info=progress2`.

Rename: `{"source", "destination", "on_gpu"?, "time"?}`. Remove: `{"path", "on_gpu"?, "time"?}` runs `rm -rf`. Both paths must be strictly inside the algoinst root, with no trailing slash. A first read of a tape-only file blocks while the archive recalls it. Killing the SSH run does not `scancel` an `on_gpu` job — cancel the `cc-archive` job with `/cancel_job` on `raven`.

The transfer runs in `/raven/ptmp/ykeller/cc/containers/apptainer/archive.sif` (shared), with this project's data mounts and without `--nv`. Do not SSH the archive yourself.

### `GET /runs/{run_id}` and `POST /runs/{run_id}/kill`

Killing the SSH process does not `scancel` a batch job. Cancel with `/cancel_job`.

## Waiting

For a job you expect to sit in the queue or run for hours, use `wait_for_job.sh` (see the `waiting_for_jobs` skill) instead of polling `/jobs` every few seconds. A training run of many hours should be checked with a long sleep, not a tight loop.
