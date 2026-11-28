# DAIS scripts

Scripts for the open-weight policy finder on the DAIS cluster (`docs/plans/policy-finder-open-weights.md`). The measurements behind them are in `docs/reports/dais_feasibility.md`; `common.sh` holds the shared settings.

## The pipeline on DAIS

`pipeline.slurm.sh` runs the policy-finder pipeline (`scripts/policy_finder/pipeline.py`, `campaign.py`) inside one CPU job on partition `small`. Its sims do not go to Raven: `PF_CLUSTER_BACKEND=local` makes `cluster_proxy.py` use `scripts/policy_finder/local_cluster.py`, which runs every sim in the unchanged sim image on the job's own cores, one single-threaded process per hyperthread, pinned. A real sweep (256 points x 500 episodes) is split into 32 parts of 4,000 episodes, chosen so that the parts play the same games as one whole sim; `merge_sweeps.py` joins them as for Raven's two parts. The first 2 cores (`PF_RESERVE_CORES`) run the driver.

### Setting up a checkout

```bash
S=/dais/fs/scratch/$USER/policy-finder-pipe          # PF_SCRATCH, the job's default
mkdir -p $S/logs $S/images $S/artifacts
# the code (this branch has no git-lfs files; DAIS's git has no lfs)
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/center-for-humans-and-machines/algorithmic-institutions.git <checkout>
git -C <checkout> config user.name "policy-finder (DAIS)"   # the pipeline commits
git -C <checkout> config user.email "policy-finder@invalid"
# the sim image (a byte copy of Raven's algoinst.sif; its sha256 is checked)
cp <algoinst.sif> $S/images/algoinst.sif
# the model files the sims read: they are not in git (docs/large_files.md;
# scripts/fetch_large_files.py, or a checkout that has them); they are bound
# over each snapshot's artifacts/
cp -r <a checkout with lfs files>/artifacts/. $S/artifacts/
# the driver's python: generate_sim_config.py, fit_surrogate.py and
# check_instance.sh import torch, numpy, scipy, scikit-learn, pandas, yaml
module load python-waterboa/2024.06          # python 3.12, numpy 1.26
python3 -m venv --system-site-packages $S/venv
$S/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
```

The driver runs on the host, not in the sim image: it calls `apptainer` for every sim, and Apptainer does not nest. The sim image's own torch 1.11 checks each rule once more (`image_check`) before its sweep.

### Running

Submit from the checkout, on the branch the instances start from (`PF_BASE`, default the checked-out branch).

**DAIS sets `SBATCH_EXPORT=SRUN_EXPORT_ENV=ALL`, so a job gets none of the submitting shell's variables.** Name each one the job needs, `sbatch --export=VAR=value,...`; a variable set in front of `sbatch` is silently lost (that is how a "smoke" run once ran at the real size). Avoid `--export=ALL`: it also hands the driver the submitting shell's `PYTHONPATH` and loaded modules. Other settings (`PF_RUNS`, `PF_BASE`, `PF_RESERVE_CORES`, `PF_LOCAL_PART_EPISODES`, ...) go into the same list. The account and the log path (`-o`, under the default `PF_SCRATCH`) are fixed in the script; override them on the `sbatch` command line. The job therefore refuses `instance`, `resume` and `campaign` unless `PF_SWEEP_EPISODES` and `PF_FINAL_EPISODES` are each given, as a number or as `default` (the pipeline's per-setting defaults, 500 and 2000 for exp2), and prints the sizes it runs at. A version keeps the sizes it started with.

```bash
cd <checkout>
REAL="--export=PF_SWEEP_EPISODES=default,PF_FINAL_EPISODES=default"
SMOKE="--export=PF_SWEEP_EPISODES=20,PF_FINAL_EPISODES=20"
sbatch $REAL infra/dais/pipeline.slurm.sh instance <name> --mode single_idea --min-params 0 --max-params 0
sbatch $REAL infra/dais/pipeline.slurm.sh resume <name>
sbatch $SMOKE infra/dais/pipeline.slurm.sh campaign <campaign.yml>
# rerun committed final runs unchanged and compare with their recorded results
sbatch infra/dais/pipeline.slurm.sh python infra/dais/rescore.py <out dir> <branch>:<stem> ...
# re-evaluate every version of a campaign on fresh seeds (se about 0.3):
# its table, best-so-far curve and improvement
sbatch infra/dais/pipeline.slurm.sh python infra/dais/reevaluate.py \
    $S/pf_runs/campaign/<campaign>.json <out dir>          # a campaign run here
sbatch infra/dais/pipeline.slurm.sh python infra/dais/reevaluate.py \
    docs/reports/policy_finder_reference_curves/pf-20261104.campaign.json <out dir> \
    --refs origin/policy-finder/                             # a campaign run elsewhere
```

Everything the pipeline keeps is in `PF_RUNS` (default `$S/pf_runs`): worktrees, state files and logs (`worktrees/<stem>.{state.json,pipeline.log}`), campaign state (`campaign/`), snapshots (`local_cluster/snapshots/<sha>/tree`, where the sims also write) and the pool's jobs (`local_cluster/jobs/local-<n>/`: `job.json` with the image, threads and CPUs, `sim.log`, `status.json` with exit code, CPU, wall time and peak memory). A job that ends mid-version leaves the state to resume from with `resume` or `campaign`; sims that were killed with it are rerun on resume without counting as attempts (see the limits below). The job log is `$S/logs/pfpipe.<job id>.out`.

Limits of the local backend:

- **Never run one version (or one campaign) from two jobs at once.** The pipeline holds a POSIX lock per version, which GPFS honours across nodes, so a second pipeline waits for the first; but a second campaign driver on the same `PF_RUNS` would still start its own pipelines for the same versions as soon as the first stops. The agent slots (`agent_runner.slot`) still use `flock`, which may not exclude across nodes.
- A worker that cannot run its sim (a wrong `PF_POOL_CPUS`, an unusable `PF_POOL_LOCKS`, a sim that cannot start) records a failed job with exit code 125, and the version fails as for any failing sim: "wrote no sweep.json in 2 jobs". A job is lost (no status) only when its worker was killed or died before it could record anything: with the Slurm job (the normal case, rerun on resume without counting), or on its own while the pipeline runs (rerun twice, 30 s apart, then the version fails with a pointer to the workers' `worker.log`). A config may lose at most 6 jobs over all resumes.
- A failed version is terminal: `resume` and `campaign` report it as "already failed" (a campaign counts it as no improvement and iterates with `--after-failure`). To run it again after fixing the cause, edit `worktrees/<stem>.state.json`: set `"status"` to `"error"`, remove the jobs step (`sweep_jobs`, or `final_jobs`) from `"done"`, and delete its job lists from `"data"` (`sweep_jobs` and `sweep_jobs_lost`, or `final_jobs` and `final_jobs_lost`); `resume` then submits all of that step's sims afresh.
- A version started on Raven cannot resume here (its snapshots are Raven's; `submit` refuses a snapshot it does not have), nor the other way round.
- A sim killed with its Slurm job (the time limit) is rerun on resume without counting as one of its two attempts; all of a step's missing parts are rerun together.
- Slow but correct splits: a split is only used if its parts play one whole sim's batches. exp2_multi sweeps (512 episodes a point in batches of 1000) have none, so they run in Raven's 2 parts (`parts_aligned: false` in the state). A design with an odd number of distinct points above 16 (a rule with int params merges points) has no such split into 3 or more parts with `generate_sim_config.py`'s edges, so it runs as one or two single-threaded sims: 120 of the 256 possible sizes get parts above 8,000 episodes, minutes instead of about 20 s. The results are the same as one whole sim's.
- The pool's slot locks are in `/dev/shm` (per job, node-local); a queue poll reads only `local_cluster/jobs/active/<host>/`; folders of done or lost jobs are pruned after 7 days (`PF_LOCAL_KEEP_DAYS`), at most once an hour. A snapshot is kept while another node's index lists a job on it, unless `squeue` says that node's Slurm job is gone.

Snapshots are about 165 MB each by `du` plus their sims' output (a real sweep about 100 MB), two per version (sweep and final). When a version is scored, its results are fetched and committed, and the pipeline removes its two snapshots (`cluster_proxy.cleanup`), unless a live pool job still runs on one; a failed version's snapshots stay for debugging.

Measured on 2026-11-27 (`small`, 48 cores, 92 pool slots): the seed of a single-idea instance (one point) through all steps in 80 s at the real size; a 5-parameter rule's real sweep (256 points x 500 episodes, 32 parts) in 20 s and its whole version from `sweep_config` to `scored` in 80 s.

## Agents on DAIS

`campaign.slurm.sh` runs a whole campaign with its agents in ONE GPU job on `gpu1` (2 GPUs of either type, 24 cores): vLLM serves the model on the GPUs (`serve_model` and `wait_ready` of `common.sh`, a free port and a key of the job's own), the agents run OpenCode against it as Apptainer processes on the node (`PF_AGENT_BACKEND=apptainer`, `scripts/policy_finder/agent_apptainer.py`), and the sims run in the CPU pool as in `pipeline.slurm.sh`. Nothing outside the cluster is used.

### The pieces

- **Agent image:** the Hetzner host's `algoinst_claude:opencode` (OpenCode 1.19.6, `infra/claude_proxy/container/CC_env/Dockerfile.opencode`), streamed as `docker save | gzip` into `$PF_SCRATCH/images/` and converted by `agent_image.slurm.sh` to `algoinst_claude_opencode.sif`; its sha256 is `AGENT_SIF_SHA256` in `common.sh`. OpenCode needs nothing from the network at run time: its plugin package is in the image, update checks, model lists, language servers and sharing are off (`agent_check.slurm.sh` starts it under `noinet.py`).
- **Agent backend:** the spawner's `start`, `finish`, `run_home` replaced, the rest of `agent_runner.py` unchanged (slots, command, result parsing, transcripts). The container gets the export read-only at the repo path, the version's write folders writable on top, a home and `/tmp` of its own, `--containall`; CPUs from a group of `PF_AGENT_CPUS` hyperthreads (`taskset`; `PF_AGENT_STEP=1` adds a `srun --overlap` step with a memory cap, `PF_AGENT_MEMORY_MB`: it worked on a `small` node in job 988325, where `taskset` inside the step pinned as asked and a 3 GiB allocation in a 1 GiB step was OOM-killed, but on the GPU nodes the step could not find `taskset` (jobs 988681, 989252, under `sbatch --export=<list>`); the backend now names `taskset` and `apptainer` by absolute path, not yet re-tested there, so the step path is unsupported and the agents' memory is uncapped), and a wall-clock limit (`PF_AGENT_TIMEOUT`, status `expired`; `campaign.slurm.sh` sets 5400 s against `agent_runner.py`'s default of 7200 s, and the runner stops OpenCode 600 s before it). A run whose job ended is `lost` and is not counted as one of the agent's attempts; after more lost runs than attempts the version fails (a run lost for a deterministic reason is not retried without end).
- **Model endpoint:** a campaign with `base_url: "{env:PF_LLM_BASE_URL}"` gets OpenCode provider `local` (`export_instance.opencode_config`), whose URL and key (`PF_LLM_API_KEY`) OpenCode reads from its environment; the job passes the key through `APPTAINERENV_`, never on a command line, in the export, a log or the state, and `opencode_run.py` redacts it from the transcripts. Campaigns without `base_url` stay on OpenRouter. Sampling: the model card's `temperature=1.0, top_p=0.95, top_k=40` for Qwen3-Coder-Next (`export_instance.LOCAL_MODELS`, also its `generation_config.json`). Context: the served one, `MAX_MODEL_LEN` (common.sh; 262144 for Qwen3-Coder-Next, its native length), which `campaign.slurm.sh` passes to the export (`PF_LLM_CONTEXT`) so that a fresh export's OpenCode config has the same; a version that resumes keeps its export's value. OpenCode compacts with a margin (`compaction.reserved` = output + 65536, with `limit.input` set, without which OpenCode ignores it): at 131072, 262144 and 1048576 tokens it compacts at 49152, 180224 and 966656. One step that adds more than about 65k tokens of tool output (several large reads at once) can still pass the server's context; OpenCode then recovers by compacting but exits 1, and the run is a failed attempt, retried on a fresh export, its record naming the overflow (`overflow: {overflows, recovered}` and the reason in `failures`).
- **Preflight:** `preflight.py --dais`, run by the job: both images against their checksums, the weights at the pinned revision, the driver's imports, every setting's agent data in the checkout and the sweep models in `PF_ARTIFACTS`, the run folders writable.

### Network rule

DAIS gives users no network namespace, so the agent's container shares the node's network. What holds instead is **a barrier against ordinary network use by the agent's commands**, plus a transcript audit:

- OpenCode's web tools are off and nothing asks for confirmation (the image's global `opencode.json`);
- the agent's shell is `agent_shell.sh`, set as `$SHELL`, which OpenCode's bash tool runs every command with: each command and everything it starts runs under `noinet.py`'s seccomp filter (no socket but `AF_UNIX`), while OpenCode itself keeps its TCP connection to vLLM. `PF_AGENT_SHELL=none` turns this off;
- the filter is installed before anything a command could have written runs: the wrapper, `noinet.py`, the export with its `.opencode`, the repo's global OpenCode config (bound over the image's) and the image are read-only; the home's config folders (`.opencode`, `.config`, `.claude`, `.agents`) are empty read-only binds; the wrapper names every program by its absolute path, starts Python isolated (`-I -S`: no user site-packages, so no `.pth` file a command wrote in the home runs) and bash without start-up files; no host `/run`, `/tmp` or home is bound, and `--containall` hides the host's user services. What stays writable from a command is the run's home, its `/tmp` and the write folders; OpenCode (unfiltered) also keeps data there (`~/.local/share/opencode`, `~/.cache/opencode`, native libraries its runtime unpacks into `/tmp`), so a process OpenCode starts after a command ran (the runner's `opencode export`) could load something a command left there. That is outside this rule's claim;
- every finished run's transcript is scanned for network use (web tools; in shell commands `curl`, `wget`, `pip install`, `git clone`, URLs, `/dev/tcp`, and network libraries in inline Python and heredocs; network libraries in written scripts): `network_audit` in the run's record in the version's state, and a line in its pipeline log. The audit reads what the agent wrote, not what ran; "audit 0" is not evidence that nothing was tried. The filter is the control, the audit a record.

This is weaker than the Docker sandbox on the Hetzner host, whose container has no route but to one host: here OpenCode's own process could reach anything, an inherited or passed socket would work through the filter, and the host's abstract Unix sockets stay reachable. `agent_check.slurm.sh` checks all of the above inside the container (`WITH_MODEL=1` on `gpudev` adds a session against vLLM).

### Running

```bash
S=/dais/fs/scratch/$USER/policy-finder-agents      # PF_SCRATCH of campaign.slurm.sh
# as for the pipeline (above), plus: $S/env.sh with VLLM_SIF, the agent SIF,
# and the human data files (experiments/*.csv, not in git) in the checkout,
# and the reference sims' per-round data (reference_sims.slurm.sh)
sbatch infra/dais/agent_check.slurm.sh
sbatch --export=PF_SWEEP_EPISODES=20,PF_FINAL_EPISODES=20 \
  infra/dais/campaign.slurm.sh campaign scripts/policy_finder/campaigns/dais-smoke-exp2.yml
# the same command again resumes from the state files in $S/pf_runs
```

The job's cores, in order: `SERVER_CORES` (4) for vLLM, `DRIVER_CORES` (2), `AGENT_CORES` (6, groups of 2 hyperthreads, at most `PF_AGENT_SLOTS`=4 agents at once), the rest (12 cores, 24 slots) the sim pool. `stages <file>` runs several stages behind one server (header of `campaign.slurm.sh`; `stages/node-2026-11-27.txt`: the smoke campaigns, the resume test, then the real campaigns with their re-evaluation). The resume test (`killtest`) kills every process of the job but the server (`kill_job_children.py`) once an agent has run 90 s, and the next stage resumes the campaign from its state files. It stops the server, the driver and every agent and sim of the job at its end, on `scancel` and 4 minutes before the time limit (`--signal`), and exits non-zero unless `campaign_check.py` finds every version `scored` (`PF_EXPECT=finished` accepts failed ones). Agent runs are in `$S/pf_runs/agent_runs/apt-<n>/` (`run.json`, `stdout.log`, `stderr.log`, `status.json`, the home with OpenCode's `opencode/` output), their transcripts also beside the state files; the server's log and its token counters (`vllm_metrics.txt`) in the job's run folder.

### Data on DAIS

- The human data (`experiments/2group_8agent_50ep.csv`, `experiments/pilot_random1_player_round_slim.csv`, not in git) were copied from the Hetzner host's checkout (which has them from S3); they and the 27 model files in `PF_ARTIFACTS` are byte-identical to the Hetzner host's (sha256 of all 29 files compared on 2026-11-27). The sim image is a byte copy of Raven's.
- The reference sims' per-round data the agent definitions promise (`plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet` for exp2, `plots/simulation/29_EXP1_reference/per_round.parquet` for exp1, with `aggregates.csv`) are in no checkout, the Hetzner host's included. `reference_sims.slurm.sh` recreated them on DAIS's CPUs (job 988510) from the unchanged configs at their own size and seed: the same distribution, other draws than the original GPU runs, whose plots (tracked in git) stay beside them. `preflight.py --dais` checks the files, `export_check.sh` that every path the definitions name is in a fresh export and that the definition's `validate-rule` command runs in the agent's container with the instance file's values.
- The agents are told to run Python as the instance file's `python`: `python3`, the agent image's, as for the reference agents (`PF_AGENT_PYTHON`); the driver keeps its own (`PF_PYTHON`).

### Model server failures

`campaign.slurm.sh` watches the server during the stages (process alive, `/health` 200, every 30 s); after 2 minutes down it kills every process of the job, so running versions stop as lost and resume in a new job, and the job exits non-zero. An agent run that failed while the server was unhealthy, or whose session errors are the endpoint's (no connection, a time-out, HTTP 5xx), is not one of the version's attempts: the version stops with an error (resumable; the campaign retries it `error_retries` times, then stops) and the run is kept under `endpoint_runs` in its state. `campaign_check.py` prints each failed version's step and reason, and `--expect finished` refuses a version that failed outside the agent's steps (`agent`, `collect`, `check`, `image_check`) or on endpoint errors. After each stage the job log has vLLM's request counters (errors included), `/health`'s latency and the tail of its log. A stage's expectation `finished/<n>` (the load test) also needs at least n first-generation versions scored and no expired or endpoint run, and `+nolimits` (as in `finished/10+nolimits`) no agent run stopped at the time limit or cut at the output limit, so that a cap that distorts results stops the job before the real campaigns. With `/<n>`, an expired run fails the stage even when its version scored on a later attempt (since 55fd52a5 every run is counted; before, only those of versions lost to a limit); every stages file pairs `/<n>` with `+nolimits`, which refuses such a run anyway. The check reports, per campaign, the runs stopped at either limit, the versions lost with such a run among their attempts (to be rerun with a higher cap, not counted as the model's failures) and the agent run times against the limit (quartiles, maximum, the longest run's share of OpenCode's limit); `limits.json` beside each re-evaluation holds the same. The agent time limit is one number, `PF_AGENT_TIMEOUT` (the campaign job's default 5400 s; the container is killed then), and OpenCode itself is stopped 600 s earlier (4800 s) so that the runner can save the session; both are printed wherever one is, and every run's record carries both (`time_limit_s`, `opencode_time_limit_s`); the check prints the agents' run times (quartiles). A stages file can be resubmitted: a finished campaign returns at once, a finished re-evaluation is not rerun, a kill test that was done (a marker beside the campaign's state) is skipped. But every stage is judged again, so after a change that makes a check stricter (the load gate's refusal of endpoint runs, say) a resubmission uses a stages file without the stages that passed: `stages/node-resume-real.txt` holds only the real campaigns. `stages_selftest.slurm.sh` runs the stages loop with stubbed commands in a small CPU job (two healthy stages, a failing stage, a server dying mid-stage, a done and a partial re-evaluation, a kill test whose campaign ends early, a resubmitted kill test).

### Resuming the full-node job

Which stages file a resubmission of a cut `node-2026-11-27c.txt` job needs depends on the stage it was cut in. The job log (`$PF_SCRATCH/logs/pf_campaign.<job>.out`) prints `== stage: <kind> ...` when a stage starts and `== stage ... failed` or `... stopped` when one ends badly; the campaigns' state files are `$PF_RUNS/campaign/<campaign>.json` (one per started campaign) and the versions' `$PF_RUNS/worktrees/<stem>.state.json`.

| Cut in | Tell by | Resubmit with |
|---|---|---|
| the real campaigns | the last `== stage:` line is the real `run` line (`dais-qwen-exp*`); `campaign/dais-qwen-exp*.json` exist | `stages/node-resume-real.txt`: the real campaigns resume from their state, then their re-evaluation |
| the load stage | the last `== stage:` line is the `finished/10` one; `campaign/dais-load-exp*.json` exist, `dais-qwen-exp*.json` do not | `stages/node-resume-load.txt`: the load campaigns resume and are judged at their end, then the real campaigns. First count the load campaigns' endpoint runs (`campaign_check.py campaign <file> --expect finished` prints `endpoint runs: N`): the gate refuses any, so with N > 0 the load test is rerun with new prefixes (`node-retest-2026-11-28.txt`) |
| a smoke stage or the kill test | the last `== stage:` line is a smoke `run` or the `killtest` | `stages/node-retest-2026-11-28.txt`: the same stages with test campaigns of new prefixes (`dais-*-node2.yml`, `dais-load-exp*-2.yml`, which differ only in the prefix), then the real campaigns |

A job on code older than #254 leaves no `killtest-done` marker, so `node-2026-11-27c.txt` itself cannot be resubmitted on newer code: its kill test would run again on a finished smoke campaign and fail.

### Single-idea campaigns (step 2)

`stages/single-idea.txt` runs single-idea mode for the job's model, both experiments at once (40 agents: `AGENT_CORES=40`, `PF_AGENT_SLOTS=40`): experiment 2 with `.claude/agents/policy-finder-single-idea.md`, experiment 1 with `.claude/agents/policy-finder-exp1-single-idea.md` (built from `policy-finder-exp1.md` and `policy-finder-single-idea.md`), each with its own seed notes.

- Stages: a smoke campaign per experiment (2 chains, 2 changes each, smoke size; `finished/1`: each finishes and at least one change scores, a test of the mechanics), a load test at the real width (20 chains, the seed and one change, smoke size; `finished/10+nolimits`: 10 of the 20 changes scored, no expired, endpoint or limit-stopped run; the gate counts the change, not the seed) and the real campaigns (20 chains x 20 changes, versions after the seed with 0 to 4 parameters). `stages/single-idea-exp2.txt` is experiment 2 alone; after a cut in the real stage, resubmit with `stages/single-idea-real.txt`.
- A note the acceptance check would reject (`change_note.py`: headings, one change, a justification) is a failed agent attempt in single-idea mode, retried on a fresh export within the version's attempts (`note_rejected` in the run's record, counted by `campaign_check.py`); free mode is unchanged.
- The campaign files name the model and the prefix as `{env:PF_LLM_MODEL}` and `{env:PF_MODEL_TAG}` (`campaign.py` expands them; the job sets them from the model profile: `MODEL` and `MODEL_TAG`, the latter by default the profile's name in lower case), so one file serves every model and each model's lineages get branches of their own. The state files are named after the campaign file: give each model's job its own `PF_RUNS`. The preflight fails when a campaign's branches exist without their worktrees in this `PF_RUNS` (a run under another one), and when a reference manager's model file is missing.
- No stage re-evaluates in the GPU job. The real campaigns' re-evaluation (420 versions per experiment, about an hour of pool time each) runs afterwards as a CPU job per experiment, resumable from its finished slices (`--resume`), the improvement labelled as the gain of the best rule over the seed (`--single-idea`), not comparable with the free-mode bars:

```bash
S=/dais/fs/scratch/$USER/policy-finder-agents; R=$S/pf_runs_single_<tag>   # the campaign job's PF_RUNS
for e in exp2:15 exp1:45; do x=${e%%:*}; k=${e##*:}
  sbatch -p small --time=06:00:00 --dependency=afterany:<campaign job> -o $S/logs/%x.%j.out \
    --export=PF_SCRATCH=$S,PF_RUNS=$R infra/dais/pipeline.slurm.sh python infra/dais/reevaluate.py \
    $R/campaign/dais-single-$x.json $R/reevaluate/dais-single-$x --slices $k --seed 100000 \
    --references infra/dais/references/$x.yml --single-idea --resume
done
```

### Model profiles

A model other than the default Qwen3-Coder-Next is one file, `infra/dais/model_profiles/<name>.sh` (selected with `MODEL_PROFILE=<name>`, sourced by `common.sh`), plus its entry in `export_instance.LOCAL_MODELS` (sampling, output limit). A profile sets: `MODEL`, `MODEL_REV` (weights, downloaded with `download_model.sh`), `VLLM_SIF` and `EXPECT_VLLM_TAG` (another vLLM image, built with `vllm_image.slurm.sh`), `TOOL_PARSER`, `REASONING_PARSER`, `MAX_MODEL_LEN`, `MOE_BACKEND` (`auto`, or a backend), `MODEL_VLLM_ARGS` (further vLLM arguments, such as a KV cache format; every server start gets them, before a submission's `VLLM_EXTRA`), `SERVER_GPUS` (tensor parallel), `READY_TIMEOUT` (seconds until the server must answer), `PROMPT_MUST_CONTAIN` (a text the server's rendered chat prompt must contain, checked when it is ready) and `PROBE_MAX_TOKENS` (the endpoint probe's tokens per request; 8192 by default when the profile names a reasoning parser). The campaign job, the serve job and the agent check take all of them; nothing else in the scripts is model-specific. A model with a reasoning effort declares it, and a submission chooses it with `REASONING_EFFORT` (for example `--export=...,MODEL_PROFILE=deepseek-v4.1-flash-1017,REASONING_EFFORT=high`); without it the job runs at the profile's default, so a job that sets nothing runs as before. The profile sets `REASONING_EFFORT_DEFAULT`, `REASONING_EFFORT_LEVELS` (`"name=budget ..."`: the efforts by name and the number the model's prompt encoder renders for each, read in the encoder of the profile's vLLM image), `REASONING_EFFORT_RANGE` (`"min max"` when the model also takes an integer; empty otherwise) and `REASONING_EFFORT_PROMPT` (the rendered prompt's text for a budget, with `{budget}`), and leaves the effort out of `MODEL_VLLM_ARGS` and `PROMPT_MUST_CONTAIN`: `reasoning_effort.sh` (sourced by `common.sh`) adds the server's default (`--default-chat-template-kwargs {"reasoning_effort":...}`) and the prompt check from the effort in force, and an effort the profile does not declare, or any effort for a model without one, stops the job before the server starts. The job log prints the effort and its prompt text, every agent run's record carries it (`reasoning_effort`, beside its time limits), and `campaign_check.py` prints the efforts of a campaign's runs and writes them to `limits.json`, which lies beside each re-evaluation. Only the DeepSeek profile declares an effort on this branch; a GLM, Hy4 or Kimi profile that has one declares it the same way. `opencode_run.py` raises OpenCode's own 32,000-token output cap (`OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX`) to the model's output limit, and a step that ends at the output limit (`finish: length`) is a failed attempt (`length_stops`).
