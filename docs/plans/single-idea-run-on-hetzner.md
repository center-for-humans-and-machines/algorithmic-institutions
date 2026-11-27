# [PAUSED] Run the single-idea policy-finder campaign from a Hetzner host

**Paused on 2026-11-27.** The host, Raven and the pipeline work, and the seed version scored. The agent step is blocked: Opus 5.5's safeguards stop the policy-finder agent at its first response (`reasoning_extraction`), as on 2026-11-13. The follow-up is `docs/plans/policy-finder-open-weights.md`, which reuses this host for development.

## Goal

Run the campaign of PR #246 (`scripts/policy_finder/campaigns/single-idea-2026-11.yml`: 5 chains, the seed `punishment = 0` plus 20 one-idea changes each, 105 versions) end to end, with the results back in this repo. The Docker side of the pipeline runs on a rented Hetzner Cloud server, the simulations on Raven under Levin's account, as in `docs/handover/handover.md`.

## Why a Hetzner host

Checked on 2026-11-27:

| Host | Finding |
|---|---|
| Levin's Mac | No Docker, 8 GB RAM, 6 GB free disk, Apple Silicon (amd64 emulation). Too small. |
| MPCDF cloud VM `130.183.217.123` | Docker and network fine, but 7.7 GB RAM with 2 GB available, and it runs 13 database containers of other projects. Not safe to share. |
| Everything on Raven | Needs a redesign (no Docker; the agents' network allowlist; long-running processes on login nodes). Days of work, one part possibly not allowed. Not for this run. |
| Hetzner Cloud | x86 with Docker; the existing setup runs unchanged. `cx53` (0.0563 EUR/h) was the first choice but has no capacity in `nbg1`, `fsn1` or `hel1`; see Decisions. |
| The `mbbf-chm` Kubernetes cluster (chm-platform) | No Docker socket for the spawner (a privileged pod or a spawner rewrite), a platform built for web apps, default project limits of 8 GiB requested memory, and the cluster is being replaced. Not for this run. |

## Decisions

- **Hetzner account:** the project behind `HETZNER_API_TOKEN` in `../life-hub/.env` (it holds the servers `lifeapp` and `behavioral-clones`). The SSH key `Laptop` registered there is this Mac's `~/.ssh/id_ed25519`. The `hetzner` and `hybridhive` entries in `~/.ssh/config` belong to another account and are not used.
- **Server:** name `policy-finder`, type `cpx62` (16 shared vCPUs, 32 GB RAM), image `ubuntu-24.04`, location `hel1`, firewall `policy-finder-fw` with inbound port 22 and ICMP only. About 0.21 EUR/h, so about 15 EUR for 3 days and about 130 EUR per month if it were left running. Approved by Levin on 2026-11-27, after `cx53` turned out to be unavailable in the EU and the fallback `cpx51` no longer orderable there.
- **User on the server:** `pf`, uid and gid 1000, in the `docker` group. The pipeline does not run as root, and uid 1000 means the spawner's `user: "1000:1000"` stays as it is.
- **Code:** the instances are cut from `policy-finder-single-idea` (`PF_BASE`), brought up to date with `main` first. PR #246 is retargeted to `main`. Merging the PR is Levin's decision and not part of this plan.
- **Credentials:** Levin's Claude token and the SSH connection to Raven live on the Hetzner server for the duration of the run (accepted by Levin, 2026-11-27). No secret is committed or written to a log, a report or Slack.
- **Who does what:** Opus agents execute the work packages. Fable (the main session) reviews each package's result before the next one starts and decides whether it passes.

## Work packages

Packages 1 to 3 are independent and run in parallel. Package 4 needs all three and the human inputs below.

### 1. Branch (Opus agent)

1. Merge `origin/main` into `policy-finder-single-idea`; resolve conflicts keeping both the single-idea mode and main's setup changes.
2. Run the local tests: `scripts/tests` and `src/aimanager/tests/test_rule_validate.py` with the project `.venv` and an absolute `PYTHONPATH=<worktree>/src`. `test_overview_charts.py` needs `altair` and is skipped locally.
3. Check the seed with `validate-rule --min-params 0 --max-params 0`.
4. Push the branch, set PR #246's base to `main`, and update its description where the merge changed something.

Done when: the branch contains `origin/main`, the tests pass, the PR targets `main` and is mergeable.

### 2. Host (Opus agent)

1. Create the server in the life-hub Hetzner project (see Decisions). Check first that no server named `policy-finder` exists.
2. Add `Host pf-hetzner` to `~/.ssh/config` (user `pf`, the server's IP, `~/.ssh/id_ed25519`).
3. On the server: updates, Docker Engine with the Compose plugin (2.24 or newer), `git`, the user `pf`, a swap file of 8 GB.
4. Handover steps 1 to 4 as `pf`: the three clones (`algorithmic-institutions` on `main`, `CC_command_server`, `container_spawner`), `env.yml`, the proxy `.env` (`RAVEN_USER=levinb`), the secrets folder with everything that is not a secret (`cc_ssh_keys/config` from `config.example`, an empty `algoinst_deploy_key/`), `host_check.sh --fix`.
5. Build the images: `docker compose ... build claude`, then bring up `cc_command_server` and `container_spawner`. The spawner needs `claude_setup_token`: if it does not start without it, leave it stopped and say so.

Done when: `host_check.sh` shows no FAIL other than the missing human-provided secrets, and the `claude` image is built.

### 3. Raven (Opus agent, over the open SSH master from the Mac)

1. Run `infra/raven/raven_setup.sh` from `main` on a Raven login node, in a login shell so that `module load apptainer` works.
2. Run it again with `--check`.

Done when: the check shows no FAIL; the expected WARN is the missing W&B key. Stop and report if the SSH master is down (only Levin can reopen it) or if a check names one of the upstream problems listed in the handover guide.

### Human inputs (Levin), after package 2

| Input | How |
|---|---|
| Claude token | `claude setup-token` on the Mac, then written to `~/secrets/claude_setup_token` on the server (the main session gives the exact command; the token is not pasted into the chat). |
| Raven login for the command server | On the server: the `ssh -fN raven` line of handover step 6, with password and OTP. |
| S3 credentials (`.s3cfg`) | Only if the data cannot be taken from the Raven checkout or this Mac; decided in package 4. |

### 4. Services, preflight, smoke run (Opus agent)

1. Start the remaining services; check the command server reaches Raven.
2. Handover steps 7 and 8 inside the proxy container: `preflight.py --local`, the data and models, the live checkout on Raven, then the full `preflight.py`. Everything must PASS.
3. Smoke run of the mode itself: a copy of the campaign file with `n_fresh: 1`, `top_k: 1`, `generations: 2`, its own prefix, and `PF_SWEEP_EPISODES=20 PF_FINAL_EPISODES=20`, with `PF_BASE=policy-finder-single-idea`. It exercises what PR #246 could not test: the seeded version 1 through sweep, fit and final with a one-point design, and one agent version on top of it.
4. Fix what the smoke run finds on the PR branch, with tests, and rerun.

Done when: the smoke chain reaches `scored` for the seed and for version 2, and the transcripts of the agent run are kept next to the state file.

### 5. Campaign (main session starts it, after reviewing package 4)

1. Start the campaign on the server, detached, so that it survives the SSH session:
   `PF_RUNS=... PF_BASE=policy-finder-single-idea python scripts/policy_finder/campaign.py run scripts/policy_finder/campaigns/single-idea-2026-11.yml`
2. Watch it at long intervals. The Raven connection of the command server drops on network changes and maintenance; only Levin can reopen it. The pipeline resumes from its state files.

Done when: all 5 chains have 21 scored or failed versions.

### 6. Results and teardown

1. Bring the results back before anything is deleted: the instance branches (`policy-finder/pf-20261126-*`) pushed to GitHub or fetched to the Mac, the state files and the agent transcripts copied off the server.
2. Remove the secrets from the server.
3. Delete the server and its firewall, after Levin confirms.

## Status

| Package | State (2026-11-27) |
|---|---|
| 1. Branch | Done and reviewed: merge commit `176ad254`, no conflicts, 201 tests passed and 3 skipped, PR #246 targets `main` and is mergeable. |
| 2. Host | Done and reviewed: server `policy-finder` (id 177920608, `cpx62`, `hel1`, 5.222.32.225), user `pf`, images built, `host_check.sh` without FAIL. SSH aliases `pf-hetzner` and `pf-hetzner-root` on the Mac. |
| 3. Raven | Done and reviewed: `raven_setup.sh --check` shows no FAIL. Two WARNs: no W&B key, and `module load apptainer` conflicting with the `apptainer/1.2.2` that `~/.bashrc` loads (jobs purge modules first; the image test passed by hand). |
| Human inputs | Done: Claude token, Raven login in the command server. The S3 config was written from the object-store key the sister projects share. |
| 4. Preflight and smoke run | Preflight all PASS. The documented `sync_to_cluster` was not used: it would have overwritten 46 files of the Raven checkout outside `artifacts/`; the 6 missing model files were copied alone. Smoke run `pfsmoke-20261127`: the seed (version 1) ran all 14 steps and scored 55.3 pool (se 9.9, 20 episodes), which covers the parameter-free rule through sweep, fit and final on Raven. Version 2 failed at the agent step, twice: "Opus 5.5's safeguards flagged this message", detail `reasoning_extraction`. Transcripts are kept on the server under `~/pf_runs/worktrees/pfsmoke-20261127-1-v2.transcripts/`. |
| 5. Campaign | Not started: every agent version would fail the same way. |
| 6. Teardown | Open. The server bills about 5 EUR per day. |

## Review (Fable)

After each package the main session checks the agent's report against the "Done when" line by looking at the real state (the branch on GitHub, the server over SSH, Raven's check output), not at the report alone. A package that does not pass goes back to an Opus agent with the finding.

## Risks

| Risk | What we do |
|---|---|
| Hetzner refuses to create the server | Happened for `cx53` (no capacity). The approval covers `cpx62` in `hel1` only; anything else goes back to Levin. |
| The safety classifier blocks the agents' first request (seen on 2026-11-13) | Shows up in the smoke run. Do not reword prompts to get past it; report it. |
| The Claude plan's 5-hour or weekly limit | The pipeline waits and resumes; the campaign then takes longer than the estimated 15 hours. |
| A zero-parameter rule breaks a step on Raven that could not be tested locally (the sim config with empty `params`, the sim's plots) | The smoke run covers it before the campaign. |
| The Raven connection drops during the campaign | Levin reopens it; `campaign.py run` resumes. |
| The server is forgotten and keeps billing | About 130 EUR per month; package 6 deletes it. |
