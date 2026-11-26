# [DRAFT] Handover: what is left

What the Claude proxy could not do itself for `docs/handover/handover.md`,
and what changes on this host to use the new setup. Written 2026-11-26 on
branch `handover-setup` (off `main`). Plan: `docs/plans/handover-setup.md`.

## Done on `handover-setup` (uncommitted, for review)

| Change | Where |
|---|---|
| The proxy (compose, Dockerfiles, `algoinst.def`, skills, its `CLAUDE.md`) copied into the repo; your user name replaced by `$RAVEN_USER` except the archive tree | `infra/claude_proxy/` |
| Compose: host paths from `infra/claude_proxy/container/CC_env/.env` (defaults = this host's paths), `RAVEN_USER` required, repo mounts relative to the compose file, `PF_RUNS` set, `HOST_UID`/`HOST_GID`, `.cc_env` optional, git identity as build args (no checked-in `.gitconfig` with your email) | `infra/claude_proxy/container/CC_env/` |
| `algo-inst-otree` and `slurm_tools` mounts moved to an optional file | `docker-compose.extra.example.yml` |
| Spawner allowlist mounted from the repo | `infra/claude_proxy/container/spawner_config/algoinst.yml` |
| Host check, secrets layout, SSH config example | `infra/claude_proxy/host_check.sh`, `secrets.example/` |
| One-shot Raven setup, `--check` mode | `infra/raven/raven_setup.sh` |
| In-proxy check including Raven | `scripts/policy_finder/preflight.py` |
| Base branch `yannik/handover` -> `main` | `new_instance.sh`, `check_instance.sh`, `policy-finder` skill, test |
| Patches for the other two repos | `docs/handover/patches/` |
| Manual policy-finder pipeline removed: `run_sweep.sh`, `run_final.sh`, the observatory's Raven sync, the `../policy-finder-worktrees` default. The AH pipeline's SSH tooling, reports, plans and tasks are unchanged | `scripts/policy_finder/`, `observation_app/`, `policy-finder-summarizer` skill, `CLAUDE.md` |

Tested: the 59 policy-finder tests; `preflight.py` against this proxy and
your Raven account (all PASS; `squeue` had a transient Slurm timeout);
`raven_setup.sh` in a fake home here (check mode, clone, venv, secrets);
both patches against their repo's full test suite (`CC_command_server` 706
passed, `slurm_tools` 23 passed). Not tested, because this proxy has no
Docker and no shell on Raven: `docker compose config`, `host_check.sh` on a
host, `raven_setup.sh` on Raven.

Your uncommitted edit of `docs/plans/archive/merge-handover-into-main.md`
and the untracked `djx/` and `33_MULTI_h2h/*_ablation/` are untouched and
not part of this change.

## A. Blocking: needed before the other person starts

1. **Push `slurm_tools`.** GitHub's `slurm_tools` (`a52dce6`, 2026-09-11)
   has no `run_slurm_raven_apptainer_algoinst.sh` and none of the algoinst
   code in `do_slurm_job.py`. Both exist only uncommitted in your local
   checkout (with other templates and tests). `raven_setup.sh` clones from
   GitHub, so a new user gets a `slurm_tools` that cannot run algoinst jobs.
   Commit and push your local work first, then:

   ```bash
   cd ~/github_repos/slurm_tools
   git apply <repo>/docs/handover/patches/slurm_tools.patch
   pytest tests            # 23 passed
   git commit -am "algoinst template: paths from \$USER and \$HOME, create every bound folder" && git push
   ```

   The patch makes the algoinst template and `do_slurm_job.py` use
   `/ptmp/$USER/cc` and `$HOME` instead of `ykeller`, and creates every folder
   the job binds. It changes nothing for you (same paths). Then update your
   own `~/slurm_tools` on Raven, which you have so far updated by hand.

2. **Patch `CC_command_server`.** `RAVEN_USER` becomes
   `os.environ.get("CC_RAVEN_USER") or "ykeller"`; compose passes
   `CC_RAVEN_USER` from `.env`. It applies to GitHub's `HEAD` (`26ccac5`):

   ```bash
   cd ~/claude_tools/CC_command_server       # or wherever your clone lives
   git apply <repo>/docs/handover/patches/CC_command_server.patch
   pytest tests                              # 706 passed
   git commit -am "RAVEN_USER from CC_RAVEN_USER" && git push
   ```

   Without it, `host_check.sh` fails for any user but `ykeller`.

3. **Merge `handover-setup` into `main` and push.** The other agent clones
   `main`: `infra/`, `docs/handover/`, the `main` base default and
   `preflight.py` are only on this branch. The repo is public, so
   `infra/claude_proxy/` becomes public. It holds no secrets: your email is
   gone from the image's `.gitconfig`, and the skills name your archive tree
   `/ghi/r/y/ykeller/algoinst` (the command server's allowlist) and nothing
   else of yours.

4. **Confirm the shared image is reachable.** `raven_setup.sh` links
   `/raven/ptmp/ykeller/cc/containers/apptainer/algoinst.sif` into the new
   user's own path. Their account needs read on the file and execute on
   every parent folder. A link means a rebuilt image in place reaches them
   at once, including mid-campaign; put a new build beside the old one under
   a new name if that matters.

## B. This host: switching to the in-repo setup

The running stack starts from `~/claude_tools/claude_proxies/algoinst`.
After the merge (or with `handover-setup` checked out in
`~/github_repos/algorithmic-institutions`, the checkout compose mounts):

1. Create `infra/claude_proxy/container/CC_env/.env`:

   ```
   RAVEN_USER=ykeller
   GIT_USER_NAME="Claude Code"
   GIT_USER_EMAIL=ykeller@mpib-berlin.mpg.de
   HOST_UID=1000
   HOST_GID=1000
   ```

   (`HOST_UID`/`HOST_GID` from `id -u`/`id -g`.) The path defaults are this
   host's current paths, so nothing moves.
2. To keep `algo-inst-otree` and `slurm_tools` in the proxy: copy
   `docker-compose.extra.example.yml` to `docker-compose.extra.yml` and add
   `-f container/CC_env/docker-compose.extra.yml` to your commands.
3. Run `infra/claude_proxy/host_check.sh`.
4. Stop the old stack and start the new one with the same project name, so
   image and network names stay the same:

   ```bash
   docker compose -p algoinst -f ~/claude_tools/claude_proxies/algoinst/container/CC_env/docker-compose.yml down
   cd ~/github_repos/algorithmic-institutions/infra/claude_proxy
   # then container/CC_env/start_command.md
   ```

   `run --build` rebuilds `algoinst_claude:latest` (git identity is now a
   build arg). The proxy still mounts at `/opt/algoinst_proxy`, so Claude's
   memory for this project stays where it is.
5. Once it works, retire `~/claude_tools/claude_proxies/algoinst`. It is not
   in git; the repo copy is now the source. The spawner reads its allowlist
   from the repo now, so `container_spawner/config/algoinst.yml` in the
   spawner repo is unused (delete it or keep it in step).

## C. Not blocking, but smoother

- `container_spawner`: the profile's `user: "1000:1000"` is fixed in the
  YAML. Reading it from an env var would remove the manual edit for hosts
  whose uid is not 1000.
- `CC_command_server`: no endpoint runs a plain shell command on a Raven
  login node, so `raven_setup.sh` needs a human login. An endpoint that runs
  `raven_setup.sh --check` (or the whole script) would let the other agent do
  step 5 itself after the human opens the SSH connection.
- `CC_command_server`: the archive roots (`/ghi/r/y/ykeller/...`) and the
  DAIS and Viper users are still constants. The policy finder uses none of
  them.
- `~/claude_tools/container_spawner` on this host has no `.git` any more; its
  content matches GitHub (`b4db357`).
- Online W&B runs need the other person in the `chm-ml` entity.
- Pushing their instance branches needs write access to the GitHub repo
  (collaborator or deploy key).
