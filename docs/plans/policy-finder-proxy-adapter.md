# [DONE] Policy finder through the proxy: a from-scratch campaign, compared with Cem's

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
  On the campaign machine (2026-11-04: 6 CPUs, 15 GB RAM) that is
  `PF_RUNS=/home/kappablanca/pf_runs`.
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
- No freeze of the main checkout: every sim runs on a snapshot of its instance's commit
  (step 5), so the code a sim runs is the commit recorded in the version's state.

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
- **Limits.** 2 CPUs, 4 GB, 2 h timeout. At most 4 agents at once (`PF_AGENT_SLOTS`, file
  locks in `${PF_RUNS}/locks`): the machine has 15 GB.
- **Outputs.** Poll `/runs/{id}` every minute. The report is the JSON result's `result`, saved
  to `${PF_RUNS}/worktrees/<stem>.report.md`. The state records cost, tokens and the model IDs
  from the run home's transcripts.
- **Retry.** A run that is `failed`, `expired` or `error`, or whose report is empty, is
  retried once with a fresh export.

### 5. Cluster adapter and pipeline

Built as `scripts/policy_finder/cluster_proxy.py` and `scripts/policy_finder/pipeline.py`
(2026-11-04). The command server's snapshots (`/sync_to_cluster` with `ref`) replaced the
staging into the shared checkout this step first planned:

- **`snapshot(ref)`** publishes the instance branch's commit at
  `/u/ykeller/cc-snapshots/<sha>/`. Jobs and login-node commands pass `checkout: <sha>`.
  Their `plots/` and `.log/` go to `/u/ykeller/cc-snapshot-outputs/<sha>/`, so instances never
  share a tree and nothing is staged or cleaned up in the live checkout. `artifacts/` (the AH
  models) is bound read-only from the live checkout.
- **`submit(config, sha) -> jobid`.** `-m aimanager simulate <config>`, 1 GPU, 2 CPUs, 16G,
  01:00:00. On HTTP 409 (the cap of 16 jobs), retry with backoff.
- **`wait(jobids)`** polls `GET /jobs` (`PF_POLL_SECONDS`, default 120 s). A reply without the
  `squeue` header is an error, never an empty queue.
- **`fetch(sha, remote_dir, local_dir)`.** `/run_in_container` copies into
  `/opt/cluster_transfer/pf/<key>/`, then `/sync_from_cluster`, then a local copy; the
  transfer copies are removed on both ends. Above 1 GB it refuses.
- **`validate_rule`** runs `validate-rule` once in the Raven image on the snapshot, because
  torch 1.11 rejects code that torch 2.5 accepts.

`pipeline.py run <name>` runs one version through its steps (export, agent, collect, check,
sweep config, snapshot, image check, sweep jobs, fetch, fit, final config, snapshot, final
job, fetch) and records each in `${PF_RUNS}/worktrees/<stem>.state.json`, so it resumes where
it stopped. Status `failed` is a verdict on the version; `error` (network, a crash) is
resumable. `run_sweep.sh` and `run_final.sh` stay as the SSH path.

The command server's snapshot script runs `git worktree prune`, and it does not mount
`${PF_RUNS}`: every instance worktree is locked (`git worktree lock`) by `new_instance.sh`.

**Final run.** A new `--top N` option, `generate_sim_config.py --final --top 5`, puts the GP peak and the 5 best
measured points in one job (6 × 2000 episodes, seed 1000). `final/sweep.json` keeps the GP
peak as `best`, which is the campaign's score. A `best_of_6` field holds Cem's metric.

### 6. Rule code on Raven

Rule code runs as `ykeller` on the snapshot (whose `src/` is writable; files there are hard
links shared with older snapshots).
`rule.py`'s name check lets `th.os` through.

- Reject attribute chains on `th` that are not a torch function or constant (an allowlist of
  `th.<name>`), and reject any attribute access on non-tensors.
- Rerun `validate-rule` on `pf-20261008-3-v6.yml`, which must still pass.

### 7. Campaign driver

`scripts/policy_finder/campaign.py run scripts/policy_finder/campaigns/proxy-2026-11.yml` runs
in the claude container in the background (built 2026-11-04).

- **State.** `${PF_RUNS}/campaign/<campaign>.json` holds each generation's members, saved
  when the generation starts. Each version's own progress is its pipeline state (step 5). A
  rerun of the driver skips scored and failed versions and resumes the rest.
- **Parallelism.** All members of a generation run at once (`parallel: 12`). Agents are capped
  by the agent slots (4), Raven jobs by the command server (16, retried on 409).
- **Generation 1.** 12 fresh lineages `pf-20261104-1` … `-12`, `--max-params 5`, 256 Sobol
  points.
- **Generations 2–7.** `--iterate-on` the 4 lineages with the highest lineage best, which is
  the highest GP-peak final of any of their versions, chosen when the generation starts. A
  version that fails its check or its sweep counts as no improvement, and the slot is not
  given to another lineage. The next version of a failed one gets `--after-failure`.
- **Errors.** A version whose pipeline hits an error is resumed up to 3 times, 10 min apart;
  after that the driver stops after the generation's other members, for a human to look.
- **No information across lineages.** The driver reads every score but passes an agent
  nothing except "another version". `campaign.py` and `campaigns/` are not in any export.
- **Summarizer.** `/policy-finder-summarizer` keeps its state on Raven through `ssh raven`,
  which the proxy may not use. The campaign's `summarize` hook stays empty; the driver writes
  `${PF_RUNS}/campaign/<campaign>.summary.md` (per version and per lineage) after each
  generation, and the policy categories for the report are assigned at step 11.

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

None as of 2026-11-04. Inputs, score, budget and the full switch from the sandbox are decided
(see Decisions and Settings).

## Progress

- 2026-11-04: steps 1–8 built (0c3dc45e, da67a6ce, 067d1f5c). Snapshot sync, a snapshot slurm
  job and the `/jobs` format checked against the command server.
- 2026-11-04: smoke (step 9) passed. `pfsmoke-1` (16 points × 50 episodes) scored in 16 min;
  `pfsmoke-1-v2` (`--iterate-on`, full size: 256 × 500 in 2 parts, final 6 × 2000) in 29 min:
  agent 4 min ($0.65), image check 5.5 min, both sweep parts 8 min with the queue, fetch
  2 × 47 MB, fit 17 s, final job 7 min. Leak check, copy-back, commits, `best`/`best_of_n`
  and transfer cleanup all as intended. About 100 MB of sweep output per version on disk.
- 2026-11-04 09:30: campaign (step 10) started, lineages `pf-20261104-1` … `-12`.
- 2026-11-04 12:00: campaign paused after generation 1. 4 versions had failed on the Raven
  login node (12 parallel image checks: threads, container driver), counted as verdicts:
  fixed (44b0de29), reclassified and resumed. 3 (`-4`, `-7`, `-8`) failed the notes check
  without `## Key findings`: the agents' inserts above that heading deleted it. The notes
  instructions now have the file written append-only, the summary sections last, and a
  heading check before finishing. The 3 were archived to `pf_runs/attic/` and rerun fresh.
  The generation 2 chosen from 6 scored lineages was discarded; versions run ahead of
  their selection are adopted when their lineage is selected, else reported as extras.
- 2026-11-04 13:07–17:17: generations 1–7 completed. A Raven outage (about 40 min, in
  generation 3) exposed two fetch faults, fixed in 5eb3a609 and eb79e452. `-7-v7` failed
  at its sweep: a device-less `th.arange` crashes on the GPU and passes every CPU check.
  `validate-rule` now rejects such calls (804a70af). The version stayed failed, as the
  rules say. 36 versions, 35 scored, best GP peak 73.43 (`pf-20261104-12-v7`).
- 2026-11-04 23:00: step 11 done. Categories recorded locally in `summary.json`.
  Head to head on seed 5000 (`27_LEVIN`, job 31911380): `-12-v7` 72.2 and `-9-v3` 72.1
  against Cem's top at 70.3, ± 0.8 each, paired +1.9 ± 1.1. Report:
  `docs/reports/policy_finder_proxy_vs_cem.md`. The observation page was republished as
  the artifact "Policy Finder Proxy Campaign".
- Left open: porting more of Cem's rules, and a second seed for the head to head. Neither
  is needed to read the result.
