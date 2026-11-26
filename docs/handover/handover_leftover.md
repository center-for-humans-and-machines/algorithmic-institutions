# [DRAFT] Handover: status and what is left

What stands between `docs/handover/handover.md` and a working policy finder
for a second person. Updated 2026-11-26, after `handover-setup` was merged
into `main` and this host moved to the in-repo setup. Plan:
`docs/plans/handover-setup.md`.

## Done

| Item | State |
|---|---|
| Proxy setup in the repo (`infra/claude_proxy/`), compose from `.env`, checks (`host_check.sh`, `preflight.py`, `raven_setup.sh`), secrets layout, handover guide | on `main` |
| Manual policy-finder pipeline removed (AH SSH tooling kept) | on `main` |
| `CC_command_server`: Raven account and paths from `env.yml` (`env_config.py`); images from the shared `SIF_OWNER=ykeller` tree; `cc_secrets` optional | GitHub `235b3d3` |
| `slurm_tools`: algoinst template on GitHub, paths from the submitting user (`cluster_layout.py`), `cc_secrets` optional | GitHub `b4b1eb8` |
| Shared `algoinst.sif` readable for other accounts | done (per Yannik) |
| This host on the in-repo compose (`.env`, extra mounts) | done: the proxy runs with `RAVEN_USER` and `PF_RUNS` from compose; `~/claude_tools/claude_proxies/algoinst` is gone |

Follow-up fixes made after that, matching what was built upstream:

- `host_check.sh` checks `CC_command_server/env.yml` (exists, `raven_user`
  equals `RAVEN_USER`) instead of the patch it looked for; compose no longer
  sets the unused `CC_RAVEN_USER`.
- `raven_setup.sh` checks the shared image in place instead of linking it.
- `handover.md`: the `env.yml` step, `raven_setup.sh` fetched on Raven with
  `curl`, `cc_secrets` optional, and `build claude` before the spawner
  starts (see below).
- The proxy's `CLAUDE.md`, `run_on_cluster` skill, `start_command.md` and
  `algoinst.def` name the shared image path.
- `docs/handover/patches/` removed: both repos were changed differently.

## Found while checking: fresh-host start order

The container spawner copies `algoinst_claude:latest` to its agent image
(`algoinst_claude:cpu`) once, when it starts (`runs.py:_ensure_profile_images`),
and fails if the source image does not exist. On a new host the old start
order (spawner up, then `run --build claude`) hits that. `handover.md` and
`start_command.md` now build `claude` first. A lasting fix belongs in
`container_spawner`: seed the image when a run starts, not only at start-up.

## Not verified yet

`preflight.py` passes in full on this host (as `ykeller`, with the new
`env.yml`-based command server), Raven included, and `host_check.sh` passes
on this host (its first run found the Docker-socket check testing for a
regular file; fixed). Not yet tried:

- Anything as a second user: `raven_setup.sh` has not run on Raven, and no
  smoke instance has run with `env.yml` set to someone other than `ykeller`.

## Not blocking, but smoother

- `container_spawner`: the profile's `user: "1000:1000"` is fixed in the
  YAML; a host whose uid is not 1000 must edit
  `infra/claude_proxy/container/spawner_config/algoinst.yml`.
- `container_spawner`: seed the agent image lazily (above).
- `CC_command_server`: the login-node exec binds `raven_repo_root`, the
  snapshot binds use `/u/<user>`; they agree only at the default
  `raven_repo_root: /u/{user}`. `handover.md` says to keep it.
- `CC_command_server`: no endpoint runs `raven_setup.sh`, so step 5 needs
  the human on Raven.
- The `slurm_tools` algoinst template creates only `cluster_sync` and
  `container_cache`; the other folders it binds come from `raven_setup.sh`.
- Online W&B runs need the other person in the `chm-ml` entity.
- Pushing their instance branches needs write access to the GitHub repo.
