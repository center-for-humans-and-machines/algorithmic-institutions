# Algorithmic institutions proxy

This folder is the Claude proxy (`infra/claude_proxy` in the algorithmic-institutions repo, mounted here): skills, Dockerfiles, the spawner allowlist and the Raven Apptainer recipe. Setup for a new machine: `docs/handover/handover.md` in the repo. The research code is `/opt/algorithmic-institutions`. Plans, tasks, and reports live there under `docs/`. Read that repo's `CLAUDE.md` for the science, the evaluation suite, and the manual SSH workflow.

## Two stacks

Raven jobs run `algoinst.sif`: Python 3.9, torch 1.11.0+cu113, PyTorch Geometric 2.5. That image is what the command server launches. The recipe is `container/CC_env/algoinst.def`. The image on the cluster is the shared `/raven/ptmp/ykeller/cc/containers/apptainer/algoinst.sif` (`SIF_OWNER` in the command server), the same for every user.

This container is the CPU image (torch 2.5.1 CPU, PyG CPU wheels), started as the `claude` service. On the RTX 4090 machine, start `claude-gpu` instead: `docker compose --profile gpu run --build --rm claude-gpu`. That service uses `container/CC_env/Dockerfile.gpu` (same torch generation, cu124 wheels). Checkpoints from the local GPU image do not load in the Raven image.

## Cluster

Raven only, as `$RAVEN_USER` (set in `.env`, passed into this container). This container has no Raven SSH keys. Every cluster call goes to `http://cc_command_server:8765/` (`CC_PROJECT=algoinst`, at most 16 concurrent jobs). Use the `run_on_cluster` skill. Do not call `ssh raven`, and do not run `scripts/train_cluster.sh`, `scripts/simulate_cluster.sh`, `scripts/remote_test.sh`, or `scripts/fetch_cluster.sh`. Those remain the manual path for a machine that has a ControlMaster session.

The live checkout on Raven is `/u/$RAVEN_USER/algorithmic-institutions`, bind-mounted in the image at `/opt/algorithmic-institutions`. `POST /sync_to_cluster` with `ref` publishes one commit at `/u/$RAVEN_USER/cc-snapshots/<sha>/` and does not write that directory again; pass the sha as `checkout` on the job and on `/run_in_container`. Omitting `ref` still rsyncs the working tree. A job is `POST /submit_slurm_job` with `launcher: python` and a `program_call` of `-m aimanager <train-ah|train-manager|simulate> <config>`. Do not `sbatch` the repo's shell templates.

`WANDB_ENTITY=chm-ml` and `WANDB_PROJECT=algorithmic-institutions` are set on the Apptainer command. `WANDB_API_KEY` comes from `/u/$RAVEN_USER/private/cc_secrets`, which the job loads as an env file when it exists (without it, the job warns and runs without W&B credentials). Pass `"wandb_mode": "online"` when the run should sync.

## Tasks, plans, reports

Write these in the code repo, not in this proxy:

- `docs/tasks/` for the task writeup
- `docs/plans/` for the implementation plan. The first heading is `# [DRAFT|ACTIVE|DONE|PAUSED|ABANDONED] Title`. Completed plans move to `docs/plans/archive/`.
- `docs/reports/` for a finished analysis

For a real feature, write the plan before implementing it. The code repo's `CLAUDE.md` has the rest of that workflow.
