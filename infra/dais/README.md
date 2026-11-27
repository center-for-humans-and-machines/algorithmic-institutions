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

Submit from the checkout, on the branch the instances start from (`PF_BASE`, default the checked-out branch). DAIS sets `SBATCH_EXPORT` so that a job gets none of the submitting shell's variables: pass settings with `--export=ALL,...`.

```bash
cd <checkout>
sbatch --export=ALL infra/dais/pipeline.slurm.sh instance <name> --mode single_idea --min-params 0 --max-params 0
sbatch --export=ALL infra/dais/pipeline.slurm.sh resume <name>
sbatch --export=ALL infra/dais/pipeline.slurm.sh campaign <campaign.yml>
# smoke size
sbatch --export=ALL,PF_SWEEP_EPISODES=20,PF_FINAL_EPISODES=20 infra/dais/pipeline.slurm.sh campaign <campaign.yml>
# rerun committed final runs and compare with their recorded results
sbatch --export=ALL infra/dais/pipeline.slurm.sh python infra/dais/rescore.py <out dir> <branch>:<stem> ...
```

Everything the pipeline keeps is in `PF_RUNS` (default `$S/pf_runs`): worktrees, state files and logs (`worktrees/<stem>.{state.json,pipeline.log}`), campaign state (`campaign/`), snapshots (`local_cluster/snapshots/<sha>/tree`, where the sims also write) and the pool's jobs (`local_cluster/jobs/local-<n>/`: `job.json` with the image, threads and CPUs, `sim.log`, `status.json` with exit code, CPU, wall time and peak memory). A job that ends mid-version leaves the state to resume from with `resume` or `campaign`; sims that were killed with it are rerun once, as on Raven. The job log is `$S/logs/pfpipe.<job id>.out`.

Snapshots are not removed: each is about 165 MB by `du` plus its sims' output (a real sweep about 100 MB), two per version (sweep and final), so a 36-version campaign leaves roughly 15 GB. A scored version's snapshots can be deleted; its results are in its worktree.

Measured on 2026-11-27 (`small`, 48 cores, 92 pool slots): the seed of a single-idea instance (one point) through all steps in 80 s at the real size; a 5-parameter rule's real sweep (256 points x 500 episodes, 32 parts) in 20 s and its whole version from `sweep_config` to `scored` in 80 s.
