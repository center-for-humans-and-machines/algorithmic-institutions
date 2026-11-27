# [DRAFT] Setting up the policy-finder pipeline on a new machine

For a Claude agent with full permissions on a Linux host, or a Mac with Docker
Desktop (read "On a Mac" below first), working with its human. Follow the
steps in order. Every step ends with a check; do not go on while it fails.
Steps marked **Human** need the person: a login, a password, a one-time code
(OTP) or a browser. Ask them and wait. For a command they should run in your
session, suggest the `!` prefix.

When you are done, the human can say "run the policy finder" inside the proxy
and get an instance that designs a rule, sweeps it on Raven and commits the
result on its own branch.

## What you are building

```
host (this machine)
├── docker compose project "algoinst"        infra/claude_proxy/container/CC_env/docker-compose.yml
│   ├── claude             the proxy: a Claude session that runs scripts/policy_finder/pipeline.py
│   ├── container_spawner  starts one sealed container per policy-finder agent (Docker socket, Claude token)
│   └── cc_command_server  the only way to Raven (SSH) and S3 (s3cmd)
└── spawned agent containers: no network but api.anthropic.com

MPCDF Raven (your account)
├── algoinst.sif           the shared image in /raven/ptmp/ykeller/... (read access)
├── ~/slurm_tools, ~/.venv the job template and slurm_job
├── ~/algorithmic-institutions/artifacts    the models every sim loads
├── ~/cc-snapshots/<sha>   one published commit per sweep
└── ~/private/cc_secrets   WANDB_API_KEY (optional)
```

Read `infra/claude_proxy/CLAUDE.md` once the repo is cloned; it is the
proxy's own instructions.

## Before you start

Ask the human for, and do not continue without:

- an MPCDF Raven account (login name, password, OTP device),
- a Claude plan for the agents (they run on it, not on API credits),
- CHM S3 credentials (an `.s3cfg` for `s3://chm-algorithmic-institutions`),
- optional: a W&B API key (entity `chm-ml`), for runs with `wandb_mode: online`.

The host needs Docker Engine (Docker Desktop on a Mac) with the Compose
plugin 2.24 or newer, the user in the `docker` group (Linux), `git`, and
Claude Code logged in (`~/.claude` and `~/.claude.json` exist).
`host_check.sh` (step 4) checks all of this.

## 1. Clone

```bash
git clone https://github.com/center-for-humans-and-machines/algorithmic-institutions.git
mkdir -p ~/claude_tools
git clone https://github.com/yannikkellerde/CC_command_server.git ~/claude_tools/CC_command_server
git clone https://github.com/yannikkellerde/container_spawner.git ~/claude_tools/container_spawner
```

Stay on `main`. The rest of this guide calls the clone `<repo>`.

The command server reads the Raven account from `env.yml` beside its code:

```bash
cd ~/claude_tools/CC_command_server
cp env.yml.example env.yml
```

Set `raven_user` and `dais_user` to the human's MPCDF login (DAIS is not
used, but the key must name a user). Leave the paths at their defaults: the
algoinst paths assume `raven_repo_root: /u/{user}`.

## 2. Settings

```bash
cd <repo>/infra/claude_proxy/container/CC_env
cp .env.example .env
```

This is the proxy's settings file. The `.env.example` at the repo root is a
different file, for the manual SSH path; leave it alone. Fill in
`RAVEN_USER` (the human's MPCDF login), `GIT_USER_EMAIL`, and
`HOST_UID`/`HOST_GID` from `id -u` and `id -g`. If those are not 1000, also
change `user: "1000:1000"` in
`infra/claude_proxy/container/spawner_config/algoinst.yml` to the same
values. Leave the other paths at their defaults unless the human wants them
elsewhere; they must be absolute.

## 3. Secrets

Follow `infra/claude_proxy/secrets.example/README.md`. In short, under
`~/secrets` (or `SECRETS_DIR`):

| Path | What to do |
|---|---|
| `claude_setup_token` | **Human:** run `claude setup-token` (browser login to the plan the agents should use). Write the token as the only line, `chmod 600`. |
| `cc_ssh_keys/config` | Copy `infra/claude_proxy/secrets.example/cc_ssh_keys/config.example` (the MPCDF FAQ config) and replace `<raven user>` in both places. The folder must stay writable: the SSH control sockets are created in it. |
| `S3/.s3cfg` | **Human:** provides it. |
| `algoinst_deploy_key/` | Create the folder. It may stay empty: the pipeline does not push. Put a GitHub key there (uid `HOST_UID`) if the proxy should push instance branches. |
| `.cc_env` | Optional on Linux. **On a Mac, required:** `CLAUDE_CODE_OAUTH_TOKEN=<token>` (see "On a Mac"). |

## 4. Host check

```bash
<repo>/infra/claude_proxy/host_check.sh --fix
```

`--fix` creates the folders that hold no secrets (`~/tmp/...`,
`~/pf_runs/exports`). Fix every FAIL. A WARN about missing images is fine
until step 6. It also checks that `env.yml` (step 1) names the same Raven
user as `.env`.

## 5. Raven (**Human** logs in; you can prepare the commands)

The script is idempotent and checks at the end. The human logs in to a
Raven login node the usual way (gateway, password, OTP) and runs:

```bash
curl -fsSLO https://raw.githubusercontent.com/center-for-humans-and-machines/algorithmic-institutions/main/infra/raven/raven_setup.sh
bash raven_setup.sh
```

It creates the `/ptmp/$USER/cc/...` folders every job binds, checks that the
shared `algoinst.sif` is readable and runs it once, clones `slurm_tools` into
`~/slurm_tools` with `~/.venv`, and creates `~/private/cc_secrets`. If it
finds no Python 3.9 or newer for `~/.venv`, load one (`module avail python`)
and rerun. Then:

- optionally put the W&B key in `~/private/cc_secrets` as
  `WANDB_API_KEY=...` (only runs with `wandb_mode: online` need it),
- rerun `bash raven_setup.sh --check`. The expected WARNs are the empty
  `artifacts` folder, which step 7 fills, and a missing W&B key.

## 6. Start the services

From `<repo>/infra/claude_proxy`, as in `container/CC_env/start_command.md`:

```bash
C="docker compose -p algoinst -f container/CC_env/docker-compose.yml"
$C build claude                                                      # first: the spawner copies this image at start-up
$C up -d cc_command_server container_spawner --build
$C exec cc_command_server bash -ic 'ssh -fN raven && sleep 0.5'      # Human: password + OTP
$C run --build --rm claude                                           # the proxy session
```

On an Apple Silicon Mac, add `-f container/CC_env/docker-compose.amd64.yml` to
`C` (see "On a Mac"). Build `claude` before starting the spawner: the spawner
copies `algoinst_claude:latest` to its agent image once, when it starts, and
fails if that image does not exist yet. If you started it too early, `$C
restart container_spawner` after the build.

The `ssh -fN raven` connection carries every Raven call. When it drops
(network change, reboot, Raven maintenance), the command server's calls fail
until the human opens it again with the same command.

The proxy session is a new Claude Code session inside the container; the
remaining steps run there, in `/opt/algorithmic-institutions`.

## 7. Data and the Raven live checkout (inside the proxy)

```bash
cd /opt/algorithmic-institutions
python scripts/policy_finder/preflight.py --local
```

It lists the data files and models that are missing locally, with the
`fetch_large_files.py --via command-server ...` command that gets them from
S3. Run that command, then send the working tree with the models to Raven:

```bash
curl -s -X POST http://cc_command_server:8765/sync_to_cluster \
  -H 'Content-Type: application/json' -d '{"cluster": "raven", "includes": ["artifacts"]}'
```

The response has a `run_id`; follow it with
`curl -s http://cc_command_server:8765/runs/<run_id>` until it exits 0. The
`run_on_cluster` skill in the proxy documents the API.

## 8. Preflight

```bash
python scripts/policy_finder/preflight.py
```

Everything must PASS. A WARN that `squeue` failed with a Slurm timeout is
Raven's scheduler being slow; rerun later. Each FAIL names its fix.

## 9. Smoke run

A small instance through the whole pipeline: the agent, a short sweep on
Raven, the final run. It costs one agent run on the Claude plan (tens of
minutes) and a few small GPU jobs.

```bash
PF_SWEEP_EPISODES=20 PF_FINAL_EPISODES=20 \
  scripts/policy_finder/new_instance.sh pfsmoke-$(date +%Y%m%d) --max-params 2 --sobol-points 16
```

Run it in the background (`run_in_background`) and watch it with
`python scripts/policy_finder/pipeline.py status pfsmoke-<date>` or the log
`$PF_RUNS/worktrees/pfsmoke-<date>.pipeline.log`. It is done when the status
is `scored` (all 14 steps from `export` to `final_fetch`). A `failed` status
names the step; `pipeline.py run pfsmoke-<date>` resumes after a fix. Smoke
branches are not pushed.

## 10. Real runs

In the proxy session: the `policy-finder` skill ("run the policy finder with
up to 5 params"), `scripts/policy_finder/new_instance.sh <name>`, or a
campaign, `python scripts/policy_finder/campaign.py run
scripts/policy_finder/campaigns/<file>.yml`. Instance names are
`pf-<YYYYMMDD>-<n>`.

## On a Mac

Docker Desktop runs the containers in a Linux VM, so most of this guide
applies unchanged. `host_check.sh` checks the points below on macOS.

- **The proxy's Claude login.** Claude Code on macOS keeps its login in the
  Keychain, which the proxy container cannot read, so the proxy would start
  logged out. **Human:** run `claude setup-token` and add
  `CLAUDE_CODE_OAUTH_TOKEN=<token>` to `~/secrets/.cc_env`. The token in
  `claude_setup_token` works; the proxy and the agents then share one plan's
  limits. The agents are not affected either way.
- **Apple Silicon (M1 to M4).** The VM is Linux ARM, and PyG's compiled
  extensions have no Linux ARM wheels: the proxy image fails to build. Add
  `-f container/CC_env/docker-compose.amd64.yml` after the main `-f` in every
  compose command. The proxy and the agents then run as amd64 under
  emulation, slower but working. Do not use the spawner's image rebuild
  endpoint: it builds natively. An Intel Mac needs none of this.
- **uid and gid.** A Mac user is usually `501:20`. Put those in `.env` and in
  `user:` of `infra/claude_proxy/container/spawner_config/algoinst.yml`.
- **File sharing.** Keep the repo, `~/claude_tools`, `~/secrets`, `~/tmp` and
  `~/pf_runs` under `/Users/...`, which Docker Desktop shares by default.
- **Memory.** Up to 4 agents take 4 GB each. Give Docker Desktop at least
  16 GB, or add `PF_AGENT_SLOTS=2` to `~/secrets/.cc_env`.
- **SSH.** Use `config.example` as it is: its connection sockets live inside
  the command server container, because a folder shared from macOS cannot
  hold sockets.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| compose: `RAVEN_USER` required | `infra/claude_proxy/container/CC_env/.env` missing, or `RAVEN_USER` empty in it (step 2). |
| compose: bind source path does not exist | A host path from step 3 or 4 is missing; rerun `host_check.sh`. |
| command server: `Bad owner or permissions` | `sudo chown -R root:root` the `cc_ssh_keys` folder, `chmod 700` it and `600` its files (secrets README). |
| every Raven call fails after working before | The SSH master connection dropped. **Human:** rerun the `ssh -fN raven` line of step 6. |
| agent run fails at once, spawner logs mention the token | `claude_setup_token` is wrong or expired. **Human:** `claude setup-token` again, then `$C restart container_spawner`. |
| agents pause for hours | The Claude plan's 5-hour or weekly limit; the pipeline waits for the reset and resumes. |
| `image_check` fails but local `validate-rule` passed | The rule uses something torch 1.11 lacks. The agent's next version must avoid it. |
| a job fails with a missing `/ptmp/...` folder | Rerun `raven_setup.sh` on Raven. |
| Raven commands run as the wrong user, or fail on `/u/ykeller/...` | `raven_user` in `CC_command_server/env.yml` (step 1) and the `User` in `cc_ssh_keys/config` must both be the human's login. Restart `cc_command_server` after editing `env.yml`. |
| proxy session asks to log in (Mac) | No `CLAUDE_CODE_OAUTH_TOKEN` in `~/secrets/.cc_env` (On a Mac). |
| proxy image build fails on PyG packages (Apple Silicon) | Missing `-f container/CC_env/docker-compose.amd64.yml` (On a Mac). |
| `ssh -fN raven` fails with a `ControlPath` or bind error | The SSH config keeps its sockets in the mounted folder; use `config.example`'s `ControlPath /var/run/cc_ssh/master-%C`. |
| container spawner exits at start: `could not seed algoinst_claude:cpu` | Build `claude` first, then restart the spawner (step 6). |
| a pipeline stopped (crash, restart) | `python scripts/policy_finder/pipeline.py run <name>` resumes where it stopped. |

## If something is missing upstream

Some pieces live in repos only their owner (Yannik Keller) can change. The
checks catch them. If you see one of these, tell the human to ask him, and
do not work around it:

- `raven_setup.sh`: the shared image is not readable by the human's account.
- `raven_setup.sh`: `template ... missing` or `slurm_tools still has ykeller's paths`.
- `host_check.sh`: `CC_command_server is too old (no env.yml support)` after a `git pull`.
