# [DRAFT] Policy finder through the proxy: a from-scratch campaign, compared with Cem's

## Why

Cem's policy finder ran on his Mac between 2026-10-07 and 2026-10-29. It produced 15
lineages and 38 versions, 37 of them validated. The best is pf-20261008-3 v6 at 73.2 ± 1.1
against `ah`. Levin's rule scores 69.8 and `ah` against `ah` 56.9 (snapshot
https://claude.ai/artifact/QUAxcas7ui7Dv3ihz87NHr, 2026-10-29). That pipeline depends on the
macOS sandbox and an SSH ControlMaster. Neither exists in the proxy.

The goal is to run the policy finder again from scratch with this setup, and to compare the
outcome with Cem's. The setup is one locked container per agent through the container
spawner, and Raven through the command server.

What Cem's results are, precisely:

- **Inputs.** All 38 versions predate #245 (merged 2026-10-27). The 10-29 lineages do too: the
  snapshot's rule code reads `c_prev`, `c_group`, `n_other`, … and never `in_group`. They used
  the per-player summary inputs, not the raw round history the code has now.
- **Score.** Every "final best" is the old validation: the GP peak plus the 5 best measured
  points, rerun at 2000 episodes with seed 1000. The best of those 6 is taken. The current
  `run_final.sh` reruns only the GP peak. Best-of-6 reads higher by construction, because
  the ± 1.1 does not account for choosing among the 6.
- **Iteration.** Cem chose by hand which lineage got `--iterate-on`. Lineages 08-1, 08-3 and
  08-8 got 7–8 versions, 29-1 and 29-2 got 3, and the rest 1.
- **A fresh-seed yardstick.** pf-20261008-3 v6, ported to the #245 inputs, scored 72.7 against
  `ah`, while `ah` against `ah` scored 54.6 (`26_LEVIN_ah_vs_pf_top_batched`, seed 2000,
  1000 episodes; handover 06).

## Decisions

- **One container per agent, no second sandbox (2026-11-03).** The spawned container runs
  `claude --dangerously-skip-permissions`, which the spawner inserts. It has no
  `settings.local.json` permissions, no `policy_finder_guard.py` hook and no Bash sandbox.
  The container is the sandbox. A second layer is an error source.
  - The container sees only its export, and its only egress is `api.anthropic.com:443`. It
    cannot reach the command server, the spawner or the proxy container. This was verified
    with spawner runs on 2026-11-03.
  - `check_instance.sh` stays the acceptance gate.
- **Agents see no other lineage.** That includes Cem's. Isolation is decided by what the
  export contains, not by rules inside it.
- **A full switch from the sandbox (2026-11-03).** Everything of the macOS sandbox setup is
  replaced by the container setup, with no fallback or dual path:
  - the guard hook and its tests;
  - `settings.local.json` permissions and sandbox;
  - the `complement()` deny lists;
  - the host-side `claude -p` launch, and the interactive and `--no-start` modes.

  `new_instance.sh` starts agents only through the spawner. The SSH cluster path
  (`PF_CLUSTER=ssh`) is separate from the sandbox and stays.
- **Rule inputs: the raw round history (#245), the current code (2026-11-03).** Not Cem's
  pre-#245 summaries.
- **Score: GP peak and best-of-6 (2026-11-03).** The final run reruns the GP peak and the top
  5 measured points. The GP peak is the campaign's score. Best-of-6 is reported next to it,
  for the comparison with Cem.

## Settings for the campaign

Decided 2026-11-03.

| | Cem | This campaign |
|---|---|---|
| Rule inputs | per-player summaries (pre-#245) | raw round history (#245, current code) |
| Params, v1 | `--num-params "up to 5"` | up to 5 (`--max-params 5`) |
| Params, later versions | unbounded from 1 | the same |
| Sweep | 256 Sobol points × 500 episodes, seed 42, 2 parts | the same |
| Final score | best of GP peak + top 5, 2000 episodes, seed 1000 | GP peak at 2000 episodes, seed 1000, **plus** the top 5 in the same job, so both numbers are reported |
| Agent | Sonnet relay → Opus subagent (`model: opus`) | the same definitions; record the model ID the alias resolved to |
| Lineages and versions | 15 lineages, 38 versions, chosen by hand | 12 fresh lineages, then 6 generations iterating the top 4: 36 versions |

## Steps

### 0. Environment (proxy side, mostly done)

- **Done (2026-11-03).** The spawner has root `pf_exports` = `${HOME}/pf_runs/exports`,
  mounted at the same path in the spawner and in the claude containers. `max_concurrent` is 8.
- **Done (2026-11-03).** `exports/` exists, and a test run mounted it.
- **To do.** Use a `claude setup-token` token file for `CLAUDE_TOKEN_FILE`. The access token in
  `.credentials.json` lasts hours, and a single-file bind mount can stay on the old file after
  a refresh.
- **Done (2026-11-03).** The spawner sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` from `cpus`.
  With `cpus: 2`, torch uses 2 threads. The inter-op pool stays at 16, which the rule code
  does not use.
- **To do, optional.** Mount `${HOME}/tmp/container_spawner/runs` read-only into the claude
  containers, so the dashboard's activity chart can read the run homes' transcripts.

### 1. Campaign base

- Cut `policy-finder-proxy` from `yannik/handover`. It contains `origin/policy-finder-base`,
  #245 and the final run.
- Instance branches `policy-finder/<name>` are cut from it.
- Nothing in it may tell an agent about earlier results. `yannik/handover` carries
  `configs/managers/rule_based/pf-20261008-3-v6.yml` and its `params/…_best.json`, plus
  `configs/simulation/manager_testing/26_LEVIN_ah_vs_pf_top_batched.yml`. The sparse
  checkout (`instance_paths`) includes `/configs/` and `/.claude/`, so an export of today's
  branch would show Cem's best rule. The export (step 3) drops these paths, and its leak
  check fails the run if any of them gets through.
- Freeze the main checkout for the campaign. `/sync_to_cluster` sends its working tree, so its
  `src/` is the code every sim runs. Record its HEAD and dirty state with each sync.

### 2. Agent definition and instance settings

- In `.claude/agents/policy-finder.md`:
  - remove the `hooks:` block;
  - remove the sentences about a guard and a sandbox, and keep the list of paths it may write;
  - keep `tools`, `disallowedTools` and `omitClaudeMd`, so the agent has Cem's toolset.
- `instance_settings.py` writes only `policy_finder.json`, with `python: python3`.
  Its `settings.local.json` part (permissions, sandbox, hook) goes.
- Delete `.claude/hooks/policy_finder_guard.py` and `scripts/tests/test_policy_finder_guard.py`.
- `new_instance.sh` loses the host-side `claude -p` launch, the interactive `exec claude` and
  `--no-start`. An agent runs only as a spawner run (step 4).
- Update the sandbox and guard references in `CLAUDE.md`, the `policy-finder` skill and
  handover 04.
- `new_instance.sh` and `check_instance.sh` take `PF_PYTHON`, `PF_BASE` and `PF_WORKTREE_ROOT`.
  Here that is `python3`, `policy-finder-proxy` and `${HOME}/pf_runs/worktrees`.
- `check_instance.sh:112` adds to `PYTHONPATH` instead of replacing it.

### 3. The export: what an agent sees

`scripts/policy_finder/export_instance.py <name>` builds `${PF_RUNS}/exports/<stem>/` from the
worktree:

- **Contents.** The sparse paths and the lineage's earlier versions (rules, notes, the scripts
  folder, sweeps with `per_round*.parquet`, `landscapes.json`, `final/`). From `.claude/`,
  only `agents/policy-finder*.md` and `policy_finder.json`.
- **Removed.** `.git`, Cem's rule files listed in step 1, and the policy-finder skills and
  `observation_app/` (they name instances).
- **Leak check.** Grep the export for `pf-2026…` names outside the lineage, and fail if any
  appear.
- **Write directories.** Create `configs/managers/rule_based/`, `notes/policy_finder/` and
  `scripts/policy_finder/<name>/` for the read-write mounts, and write a hash manifest of
  everything else.
- **Copy-back** (`--collect`). Copy the three write paths into the worktree, and list any file
  outside them that differs from the manifest. `check_instance.sh` then fails on those files.

About 165 MB per export, of which 143 MB is `artifacts/`. Plain copies are fine.

### 4. Agent runs through the spawner

`scripts/policy_finder/agent_runner.py`. `POST /runs`:

- **Profile.** `cpu`. Workdir `/opt/algorithmic-institutions`, where the image's `PYTHONPATH`
  already points.
- **Mounts.** `exports/<stem>` read-only at `/opt/algorithmic-institutions`, and the three
  write directories read-write below it.
- **Command.** `claude -p "Design your rule." --agent policy-finder-host --strict-mcp-config`.
- **Limits.** 2 CPUs, 8 GB, 2 h timeout.
- **Outputs.** Poll `/runs/{id}` every minute. The report is `stdout` from `/logs`, saved to
  `${PF_RUNS}/side/<stem>.report.md`. Record `home_host_path` (the transcripts) and the
  model ID.
- **Retry.** A run that is `failed`, `expired` or `error`, or whose report is empty, is
  retried once with a fresh export.

### 5. Cluster adapter

`scripts/policy_finder/cluster.py`, selected by `PF_CLUSTER=ssh|proxy` (default `ssh`). Four
functions, all on the endpoints in the `run_on_cluster` skill:

- **`sync(stage)`** copies the rule and sweep configs into the main checkout, untracked. Their
  names are unique per stem. Then it calls `POST /sync_to_cluster`. That sync is
  `rsync --archive` without `--delete` (`CC_command_server/commands.py:1709`, checked
  2026-11-03), so a sync never removes another instance's staged files on Raven. After a
  version's final run has been fetched, the adapter removes its staged copies locally, and on
  Raven with `/run_in_container`. Otherwise they pile up in the shared checkout.
- **`submit(config) -> jobid`.** `-m aimanager simulate <config>`, 1 GPU, 2 CPUs, 16G,
  01:00:00. On HTTP 409 (the cap of 16 jobs), retry with backoff.
- **`wait(jobids)`** polls `GET /jobs` every 60 s.
- **`fetch(remote_dir, local_dir)`.** `POST /run_in_container` copies into
  `/opt/cluster_transfer/pf/<stem>/`, then `POST /sync_from_cluster`, then copy from
  `/var/cluster_transfer`. Check the size against about 1 GB.

`run_sweep.sh` and `run_final.sh` call these in place of `simulate_cluster.sh`, `ssh` and
`rsync`. `check_instance.sh` also runs `validate-rule` once through `/run_in_container`,
because torch 1.11 rejects code that torch 2.5 accepts.

**Final run.** A new `--top N` option, `generate_sim_config.py --final --top 5`, puts the GP peak and the 5 best
measured points in one job (6 × 2000 episodes, seed 1000). `final/sweep.json` keeps the GP
peak as `best`, which is the campaign's score. A `best_of_6` field holds Cem's metric.

### 6. Rule code on Raven

Rule code runs as `ykeller` in the shared checkout, next to every instance's staged files.
`rule.py`'s name check lets `th.os` through.

- Reject attribute chains on `th` that are not a torch function or constant (an allowlist of
  `th.<name>`), and reject any attribute access on non-tensors.
- Rerun `validate-rule` on `pf-20261008-3-v6.yml`, which must still pass.

### 7. Campaign driver

`scripts/policy_finder/campaign.py <campaign.yml>` runs in the claude container in the
background.

- **State.** It is resumable from `${PF_RUNS}/state/<stem>.json`. Each version moves through
  `EXPORT → AGENT → COLLECT → CHECK+COMMIT → SWEEP → FIT → FINAL → SCORED` (or `FAILED` with
  the step and the log).
- **Parallelism.** At most 8 agents (the spawner's cap) and at most 16 Raven jobs (the
  command server's cap). Final runs go before new sweeps, so versions finish.
- **Generation 1.** 12 fresh lineages, `--max-params 5`, 256 Sobol points.
- **Generations 2–7.** `--iterate-on` the 4 lineages with the highest lineage best, which is
  the highest GP-peak final of any of their versions. A version that fails its check or its
  sweep counts as no improvement, and the slot is not given to another lineage.
- **No information across lineages.** The driver reads every score but passes an agent
  nothing except "another version".
- **Summarizer.** `/policy-finder-summarizer` runs in the claude container after each
  generation, never inside an agent's container.
- **Settings.** `campaign.yml` holds `n_fresh`, `generations`, `top_k`, the param flags, the
  Sobol points, the agent timeout and `PF_RUNS`.

### 8. Tests

- `test_cluster_proxy.py`: the adapter against a stub HTTP server. Covers staging and cleanup,
  409 retry, wait and fetch.
- `test_export_instance.py`: the deny-list, the leak check, the manifest and the copy-back.
- `test_agent_runner.py`: a stub spawner.
- `test_campaign.py`: the state machine on stubs, including resume after a kill, the
  concurrency caps and the selection rule.
- Extend `test_instance_settings.py` (only `policy_finder.json`) and
  `test_policy_finder_check.py` (`PF_PYTHON`).
- Drop the guard tests.

### 9. Smoke

1. One lineage with `--sobol-points 16` and 50 episodes, through the final run.
2. Then one `--iterate-on` version.
3. Checks:
   - the leak check passes;
   - the agent wrote only its three paths;
   - the commit is on `policy-finder/<name>`;
   - `final/sweep.json` has both `best` and `best_of_6`;
   - the Raven staging is cleaned.
4. Time the sweep parts and the final run, to size the campaign's wall time. Each version is
   3 GPU jobs, and at most 16 run at once.

### 10. Campaign

Run `campaign.yml` as in "Settings for the campaign". Republish the observation page as a new
artifact. The old URL is Cem's.

### 11. Comparison: `docs/reports/policy_finder_proxy_vs_cem.md`

- **Scores.**
  - Per generation, the lineage bests as GP-peak final and as best-of-6.
  - Against them, Cem's 37 validated versions (best-of-6, pre-#245 inputs, from the snapshot).
  - Best of campaign, median lineage best, and the share of lineages above Levin (69.8).
- **Head to head on a common seed.** One sim config (`27_LEVIN_pf_proxy_vs_cem_batched`):
  - our top 3 rules and Cem's top (`pf-20261008-3-v6`, already ported) against `ah`, on both
    group sides;
  - `ah` against `ah`, and `zero` against `ah`;
  - 2000 episodes each, on a seed no sweep or final run used.

  This removes both the seed differences and the best-of-6 selection. Porting more of Cem's
  rules (pf-20261008-8 v6, pf-20261008-1 v5) from the snapshot's YAML into the #245 inputs
  is optional.
- **Ideas.** The summarizer's policy categories against Cem's (switch-timed, front-loaded, …),
  and whether our lineages found ideas his did not.
- **Process.** Agent time and tokens per version, failures by step, and Raven job hours.
- **Confounds, stated in the report.**
  - the inputs (#245 holds strictly more information than the old summaries);
  - the iteration (automatic against manual);
  - the model the `opus` alias resolved to;
  - the param bound.

## Open questions

None as of 2026-11-03. Inputs, score, budget and the full switch from the sandbox are decided
(see Decisions and Settings). `/sync_to_cluster` was checked: it does not delete (step 5).
