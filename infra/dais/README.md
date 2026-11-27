# DAIS scripts

Scripts for the open-weight policy finder on the DAIS cluster (`docs/plans/policy-finder-open-weights.md`). The measurements behind them are in `docs/reports/dais_feasibility.md`; `common.sh` holds the shared settings.

## The pipeline on DAIS

`pipeline.slurm.sh` runs the policy-finder pipeline (`scripts/policy_finder/pipeline.py`, `campaign.py`) inside one CPU job on partition `small`. Its sims do not go to Raven: `PF_CLUSTER_BACKEND=local` makes `cluster_proxy.py` use `scripts/policy_finder/local_cluster.py`, which runs every sim in the unchanged sim image on the job's own cores, one single-threaded process per hyperthread, pinned. A real sweep (256 points x 500 episodes) is split into 32 parts of 4,000 episodes, chosen so that the parts play the same games as one whole sim; `merge_sweeps.py` joins them as for Raven's two parts. The first 2 cores (`PF_RESERVE_CORES`) run the driver.

### Setting up a checkout

```bash
S=/dais/fs/scratch/$USER/policy-finder-pipe          # PF_SCRATCH, the job's default
mkdir -p $S/logs $S/images $S/artifacts
# the code: a clone without git-lfs content (DAIS's git has no lfs)
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/center-for-humans-and-machines/algorithmic-institutions.git <checkout>
git -C <checkout> config user.name "policy-finder (DAIS)"   # the pipeline commits
git -C <checkout> config user.email "policy-finder@invalid"
# the sim image (a byte copy of Raven's algoinst.sif; its sha256 is checked)
cp <algoinst.sif> $S/images/algoinst.sif
# the model files the sims read (artifacts/ of a checkout with the lfs files);
# they are bound over each snapshot's artifacts/, which holds lfs pointers
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

**DAIS sets `SBATCH_EXPORT=SRUN_EXPORT_ENV=ALL`, so a job gets none of the submitting shell's variables.** Pass settings with `sbatch --export=ALL,VAR=value,...`; a variable set in front of `sbatch` is silently lost (that is how a "smoke" run once ran at the real size). The job therefore refuses `instance`, `resume` and `campaign` unless `PF_SWEEP_EPISODES` and `PF_FINAL_EPISODES` are each given, as a number or as `default` (the pipeline's per-setting defaults, 500 and 2000 for exp2), and prints the sizes it runs at. A version keeps the sizes it started with.

```bash
cd <checkout>
REAL="--export=ALL,PF_SWEEP_EPISODES=default,PF_FINAL_EPISODES=default"
SMOKE="--export=ALL,PF_SWEEP_EPISODES=20,PF_FINAL_EPISODES=20"
sbatch $REAL infra/dais/pipeline.slurm.sh instance <name> --mode single_idea --min-params 0 --max-params 0
sbatch $REAL infra/dais/pipeline.slurm.sh resume <name>
sbatch $SMOKE infra/dais/pipeline.slurm.sh campaign <campaign.yml>
# rerun committed final runs unchanged and compare with their recorded results
sbatch --export=ALL infra/dais/pipeline.slurm.sh python infra/dais/rescore.py <out dir> <branch>:<stem> ...
# re-evaluate every version of a campaign on fresh seeds (se about 0.3):
# its table, best-so-far curve and improvement
sbatch --export=ALL infra/dais/pipeline.slurm.sh python infra/dais/reevaluate.py \
    $S/pf_runs/campaign/<campaign>.json <out dir>          # a campaign run here
sbatch --export=ALL infra/dais/pipeline.slurm.sh python infra/dais/reevaluate.py \
    docs/reports/policy_finder_reference_curves/pf-20261104.campaign.json <out dir> \
    --refs origin/policy-finder/                             # a campaign run elsewhere
```

Everything the pipeline keeps is in `PF_RUNS` (default `$S/pf_runs`): worktrees, state files and logs (`worktrees/<stem>.{state.json,pipeline.log}`), campaign state (`campaign/`), snapshots (`local_cluster/snapshots/<sha>/tree`, where the sims also write) and the pool's jobs (`local_cluster/jobs/local-<n>/`: `job.json` with the image, threads and CPUs, `sim.log`, `status.json` with exit code, CPU, wall time and peak memory). A job that ends mid-version leaves the state to resume from with `resume` or `campaign`; sims that were killed with it are rerun once, as on Raven. The job log is `$S/logs/pfpipe.<job id>.out`.

Snapshots are about 165 MB each by `du` plus their sims' output (a real sweep about 100 MB), two per version (sweep and final). When a version is scored, its results are fetched and committed, and the pipeline removes its two snapshots (`cluster_proxy.cleanup`), unless a live pool job still runs on one; a failed version's snapshots stay for debugging.

Measured on 2026-11-27 (`small`, 48 cores, 92 pool slots): the seed of a single-idea instance (one point) through all steps in 80 s at the real size; a 5-parameter rule's real sweep (256 points x 500 episodes, 32 parts) in 20 s and its whole version from `sweep_config` to `scored` in 80 s.
