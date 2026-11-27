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
- **Agent backend:** the spawner's `start`, `finish`, `run_home` replaced, the rest of `agent_runner.py` unchanged (slots, command, result parsing, transcripts). The container gets the export read-only at the repo path, the version's write folders writable on top, a home and `/tmp` of its own, `--containall`; CPUs from a group of `PF_AGENT_CPUS` hyperthreads (`taskset`; `PF_AGENT_STEP=1` adds a `srun --overlap` step with a memory cap, `PF_AGENT_MEMORY_MB`: checked in job 988325, where `taskset` inside the step pinned as asked and a 3 GiB allocation in a 1 GiB step was OOM-killed; not yet used in a campaign, so the agents' memory is uncapped by default), and a wall-clock limit (`PF_AGENT_TIMEOUT`, status `expired`). A run whose job ended is `lost` and is not counted as one of the agent's attempts; after more lost runs than attempts the version fails (a run lost for a deterministic reason is not retried without end).
- **Model endpoint:** a campaign with `base_url: "{env:PF_LLM_BASE_URL}"` gets OpenCode provider `local` (`export_instance.opencode_config`), whose URL and key (`PF_LLM_API_KEY`) OpenCode reads from its environment; the job passes the key through `APPTAINERENV_`, never on a command line, in the export, a log or the state, and `opencode_run.py` redacts it from the transcripts. Campaigns without `base_url` stay on OpenRouter. Sampling: the model card's `temperature=1.0, top_p=0.95, top_k=40` for Qwen3-Coder-Next (`export_instance.LOCAL_MODELS`, also its `generation_config.json`), context 131072 as served.
- **Preflight:** `preflight.py --dais`, run by the job: both images against their checksums, the weights at the pinned revision, the driver's imports, every setting's agent data in the checkout and the sweep models in `PF_ARTIFACTS`, the run folders writable.

### Network rule

DAIS gives users no network namespace, so the agent's container shares the node's network. What holds instead is **a barrier against ordinary network use by the agent's commands**, plus a transcript audit:

- OpenCode's web tools are off and nothing asks for confirmation (the image's global `opencode.json`);
- the agent's shell is `agent_shell.sh`, set as `$SHELL`, which OpenCode's bash tool runs every command with: each command and everything it starts runs under `noinet.py`'s seccomp filter (no socket but `AF_UNIX`), while OpenCode itself keeps its TCP connection to vLLM. `PF_AGENT_SHELL=none` turns this off;
- the filter holds only while nothing the unfiltered side runs or loads is writable from a command: the wrapper, `noinet.py`, the export with its `.opencode`, and the image are read-only, the home's config folders (`.opencode`, `.config`, `.claude`, `.agents`) are empty read-only binds, programs are named by absolute paths, bash reads no start-up files; no host `/run`, `/tmp` or home is bound, and `--containall` hides the host's user services;
- every finished run's transcript is scanned for network use (web tools, `curl`, `wget`, `pip install`, `git clone`, URLs in shell commands, network libraries in written scripts): `network_audit` in the run's record in the version's state, and a line in its pipeline log.

This is weaker than the Docker sandbox on the Hetzner host, whose container has no route but to one host: here OpenCode's own process could reach anything, an inherited or passed socket would work through the filter, and the host's abstract Unix sockets stay reachable. `agent_check.slurm.sh` checks all of the above inside the container (`WITH_MODEL=1` on `gpudev` adds a session against vLLM).

### Running

```bash
S=/dais/fs/scratch/$USER/policy-finder-agents      # PF_SCRATCH of campaign.slurm.sh
# as for the pipeline (above), plus: $S/env.sh with VLLM_SIF, the agent SIF,
# and the human data files (experiments/*.csv, not in git) in the checkout
sbatch infra/dais/agent_check.slurm.sh
sbatch --export=PF_SWEEP_EPISODES=20,PF_FINAL_EPISODES=20 \
  infra/dais/campaign.slurm.sh campaign scripts/policy_finder/campaigns/dais-smoke-exp2.yml
# the same command again resumes from the state files in $S/pf_runs
```

The job's cores, in order: `SERVER_CORES` (4) for vLLM, `DRIVER_CORES` (2), `AGENT_CORES` (6, groups of 2 hyperthreads, at most `PF_AGENT_SLOTS`=4 agents at once), the rest (12 cores, 24 slots) the sim pool. `stages <file>` runs several stages behind one server (header of `campaign.slurm.sh`; `stages/node-2026-11-27.txt`: the smoke campaigns, the resume test, then the real campaigns with their re-evaluation). The resume test (`killtest`) kills every process of the job but the server (`kill_job_children.py`) once an agent has run 90 s, and the next stage resumes the campaign from its state files. It stops the server, the driver and every agent and sim of the job at its end, on `scancel` and 4 minutes before the time limit (`--signal`), and exits non-zero unless `campaign_check.py` finds every version `scored` (`PF_EXPECT=finished` accepts failed ones). Agent runs are in `$S/pf_runs/agent_runs/apt-<n>/` (`run.json`, `stdout.log`, `stderr.log`, `status.json`, the home with OpenCode's `opencode/` output), their transcripts also beside the state files; the server's log and its token counters (`vllm_metrics.txt`) in the job's run folder.
