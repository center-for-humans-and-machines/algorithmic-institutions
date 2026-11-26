# [ACTIVE] Portable policy-finder setup for a second user

## Goal

Another person, with their own Raven account, Claude plan and CHM S3 access,
lets their own Claude agent set up the policy-finder pipeline on their machine
from one clone of this repo. Everything that can live in this repo does; what
needs another repo or the Raven account is listed with patches.

## Scope (this branch)

1. The Claude proxy (Dockerfiles, compose, Apptainer recipe, skills, its
   `CLAUDE.md`) moves into `infra/claude_proxy/`. Until now it lived in
   `~/claude_tools/claude_proxies/algoinst`, outside any repo.
2. `docker-compose.yml` takes host paths from a gitignored `.env`
   (`.env.example`), with today's paths as defaults. `RAVEN_USER` is
   required. The repo mounts are relative to the compose file.
3. The `algo-inst-otree` and `slurm_tools` mounts move to an optional
   `docker-compose.extra.yml`.
4. Compose sets `PF_RUNS` in the proxy.
5. The spawner's project allowlist (`algoinst.yml`) lives in
   `infra/claude_proxy/container/spawner_config/` and is mounted from there.
6. The policy finder's base branch defaults to `main` (`PF_BASE` overrides).
7. `.cc_env` is optional. Git identity is a build arg, not a checked-in
   `.gitconfig`.
8. Checks: `infra/claude_proxy/host_check.sh` (host) and
   `scripts/policy_finder/preflight.py` (inside the proxy, including Raven
   through the command server).
9. `infra/claude_proxy/secrets.example/`: the secrets layout, an SSH config
   example for Raven.
10. `infra/raven/raven_setup.sh`: one idempotent script a new user runs on a
    Raven login node (folders, `slurm_tools` and `~/.venv`, `cc_secrets`,
    link to the shared image, live checkout, image smoke test). `--check`
    only reports.
11. `docs/handover/`: `handover.md` (instructions for the other agent),
    `handover_leftover.md` (what could not be done from the proxy) and
    `patches/` for `CC_command_server` and `slurm_tools`.
12. The manual policy-finder pipeline goes (git history keeps it):
    `run_sweep.sh`, `run_final.sh`, the observatory's Raven sync in
    `summarize.py`, and the `../policy-finder-worktrees` default (the worktree
    root is `PF_WORKTREE_ROOT`, else `$PF_RUNS/worktrees`). The AH pipeline's
    manual SSH tooling stays: it is the record of how the AH runs were made.
    Reports, plans and tasks are not changed.

## Out of scope

- A local (non-Raven) sweep backend.
- Changes in `CC_command_server`, `container_spawner` and `slurm_tools`
  themselves: the proxy mounts them read-only or cannot push to them. Patches
  only.

## Done when

- `docker compose config` passes with only `RAVEN_USER` set.
- `preflight.py` passes in this proxy.
- `raven_setup.sh` passes `bash -n` and shellcheck where available.
- The existing policy-finder tests pass.
