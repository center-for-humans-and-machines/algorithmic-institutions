# [DRAFT] Unified CLI entry point

## Goal

Replace the scattered entry points (`artificial_humans/train.py`,
`rl_manager.py`, `simulation/simulate.py`) with a single CLI that
dispatches based on a subcommand. Config validation catches mode/config
mismatches early. Existing code paths are preserved — this is a thin
dispatcher, not a rewrite.

This CLI runs **inside SLURM jobs on the Raven cluster**. The local
workflow remains: skill → shell script in `scripts/` → SSH + sbatch →
SLURM job calls the CLI.

## Current state

| Function | Entry point | Invocation inside jobs |
|---|---|---|
| AH training | `src/aimanager/artificial_humans/train.py` | `python src/aimanager/artificial_humans/train.py <config>` |
| RL manager training | `src/aimanager/rl_manager.py` | `python src/aimanager/rl_manager.py <config>` |
| Simulation | `src/aimanager/simulation/simulate.py` | `python src/aimanager/simulation/simulate.py <config>` |

## Proposed interface

```bash
python -m aimanager train-ah <config>
python -m aimanager train-manager <config>
python -m aimanager simulate <config>
```

Three subcommands, each takes a single positional config path.

## Config validation

Each subcommand checks for required keys before dispatching. Missing
keys → exit with a clear error listing what's missing.

| Subcommand | Required config keys |
|---|---|
| `train-ah` | `data_file`, `model_args`, `train_args`, `optimizer_args` |
| `train-manager` | `artificial_humans`, `manager_args`, `env_args`, `n_update_steps` |
| `simulate` | `artificial_humans`, `managers`, `n_episodes`, `n_episode_steps` |

Cross-mode mismatch detection: if a config contains keys strongly
associated with a *different* mode, warn the user.

| Key present | Likely mode | Warning if used with |
|---|---|---|
| `managers` | `simulate` | `train-ah`, `train-manager` |
| `manager_args` | `train-manager` | `train-ah`, `simulate` |
| `n_update_steps` | `train-manager` | `train-ah`, `simulate` |
| `train_args` | `train-ah` | `train-manager`, `simulate` |

## Plan

| # | File | Change |
|---|------|--------|
| 1 | `src/aimanager/__main__.py` | New file — calls `cli.main()` |
| 2 | `src/aimanager/cli.py` | New file — argparse, config validation, dispatch |
| 3 | existing entry points | Minimal refactor to expose callable `main()` functions |

### 1. `src/aimanager/__main__.py`

```python
from aimanager.cli import main

if __name__ == "__main__":
    main()
```

### 2. `src/aimanager/cli.py`

Single file containing:

- **`main()`**: `argparse` with three subcommands (`train-ah`,
  `train-manager`, `simulate`). Each takes a positional `config` arg.
- **`load_and_validate(config_path, mode)`**: Loads YAML, checks
  required keys for the given mode, warns on cross-mode key presence.
  Returns parsed config dict or exits with error.
- **Dispatch**: imports and calls the existing `main()` from each
  module:
  - `train-ah` → `aimanager.artificial_humans.train.main(config)`
  - `train-manager` → `aimanager.rl_manager.main(config)`
  - `simulate` → `aimanager.simulation.simulate.main(config)`

### 3. Refactoring existing entry points (minimal)

Each entry point needs a callable `main(config_path)` function.

- **`artificial_humans/train.py`**: Already has `main(config)` taking a
  config dict. No change needed.
- **`rl_manager.py`**: Top-level script logic. Extract into
  `main(config_path)` function, keep `if __name__` block.
- **`simulation/simulate.py`**: Has `main(config)` taking a config dict.
  No change needed.

## Logging compaction

### Current state

Each pipeline logs to a different location with inconsistent structure:

| Pipeline | SLURM log location | Job files archived? |
|---|---|---|
| AH training | `temp/training/artificial_humans/{config}/{job_id}/log.log` | Yes: `run.sh`, `job.yml` |
| Manager training | `.log/manager/aimanager_training_%j.out` | No |
| Simulation | `.log/sim/aimanager_simulation_%j.out` | No |

AH training archives both the generated SLURM script (`run.sh`) and the
job config (`job.yml`) alongside the log — making it easy to reproduce
or debug a past run. Manager and simulation only capture SLURM
stdout/stderr with no record of which script or config was used.

### Target state

All pipelines log under `.log/` with a consistent per-job directory
structure. Every job archives its SLURM script and config alongside the
log:

```
.log/
  training/
    ah/{config_path}/{job_id}/
      log.log          # SLURM stdout/stderr
      run.sh           # SLURM job script (archived)
      job.yml          # Job config (archived)
    manager/{config_name}/{job_id}/
      log.log          # SLURM stdout/stderr
      run.sh           # SLURM job script (archived)
      config.yml       # Job config (archived)
  simulation/{config_name}/{job_id}/
    log.log            # SLURM stdout/stderr
    run.sh             # SLURM job script (archived)
    config.yml         # Job config (archived)
```

`{job_id}` is a UUID generated per submission, so repeated runs of
the same config don't overwrite each other.

`{config_name}` is derived from the config file path (e.g.
`configs/training/01_rnn_node.yml` → `01_rnn_node`). If a pipeline
supports grid search (AH), there is an additional `{job_id}` level
beneath the config path.

### Changes per pipeline

#### AH training (`artificial_humans/run.py`)

- Change `temp_dir` in AH training configs from `temp` to `.log`.
  The existing path construction in `run.py` (`os.path.join(temp_dir,
  *exp_dir[1:])`) already produces the right structure:
  `.log/training/artificial_humans/{config_name}/{job_id}/`.
- No code changes needed in `run.py` — just config updates.

#### Manager training — new orchestrator

Manager training currently has no orchestrator. The SLURM script
(`scripts/manager/run_training.sh`) is submitted directly, and
`train_cluster.sh` runs `rl_manager.py` on the login node without
sbatch.

Create `src/aimanager/manager/run.py` following the same pattern as AH's
`run.py` and simulation's `run.py`:

1. Load the config YAML.
2. Derive `config_name` from the config file path.
3. Generate a UUID `{job_id}` for this submission.
4. Create job directory: `.log/training/manager/{config_name}/{job_id}/`.
5. Copy config to `{job_dir}/config.yml`.
6. Generate a job-specific SLURM script from the template
   (`scripts/manager/run_training.sh`) with `--output`/`--error`
   pointing to `{job_dir}/log.log`. Save as `{job_dir}/run.sh`.
7. Submit via `sbatch {job_dir}/run.sh`.

Update `scripts/manager/run_training.sh` to be a template with
`{log_file}` and `{config_path}` placeholders (like AH's template).

Update `train_cluster.sh` to call the new orchestrator:
`python src/aimanager/manager/run.py <config>` instead of
`python src/aimanager/rl_manager.py <config>`.

#### Simulation (`simulation/run.py`)

Enhance the existing orchestrator `simulation/run.py` to create per-job
directories and archive job files:

1. Derive `config_name` from the config file path.
2. Generate a UUID `{job_id}` for this submission.
3. Create job directory: `.log/simulation/{config_name}/{job_id}/`.
4. Copy config to `{job_dir}/config.yml`.
5. Generate a job-specific SLURM script from the template
   (`scripts/run_simulation.sh`) with `--output`/`--error` pointing to
   `{job_dir}/log.log`. Save as `{job_dir}/run.sh`.
6. Submit via `sbatch {job_dir}/run.sh`.

Update `scripts/run_simulation.sh` to be a template with `{log_file}`
and `{config_path}` placeholders.

### SLURM script template pattern

All three pipelines will use the same approach as AH training:
- The SLURM `.sh` file in `scripts/` becomes a **template** with
  placeholders (`{log_file}`, `{config_path}`, etc.)
- The orchestrator fills in the placeholders and writes a
  **job-specific** copy to the `.log/` job directory
- sbatch runs the job-specific copy, so the archived `run.sh` is an
  exact record of what SLURM executed

### Cleanup

- Remove the old `.log/manager/` and `.log/sim/` directories (or let
  them age out)
- AH training configs: change `temp_dir: temp` → `temp_dir: .log`
- Ensure `.gitignore` covers `.log/` (already does)
- Ensure `rsync --exclude='temp/'` in `train_cluster.sh` /
  `simulate_cluster.sh` is no longer needed (or add `.log/` exclude)

## What this does NOT change

- Config file format — unchanged
- Training/simulation logic — unchanged
- SLURM orchestrators (`artificial_humans/run.py`,
  `simulation/run.py`) — unchanged, but jobs they submit should
  switch to calling `python -m aimanager <subcommand> <config>`
- Shell scripts in `scripts/` — updated to call the unified CLI
- Notebook runner (`run.py` at repo root) — separate concern, excluded
- Old `if __name__` blocks — kept for backward compatibility

## Next actions

- [ ] Refactor `rl_manager.py` to expose callable `main()` function
- [ ] Create `src/aimanager/cli.py` with validation and dispatch
- [ ] Create `src/aimanager/__main__.py`
- [ ] Update shell scripts / SLURM orchestrators to use new CLI
- [ ] Logging: update AH training configs `temp_dir: temp` → `.log`
- [ ] Logging: convert `scripts/manager/run_training.sh` to template
- [ ] Logging: create `src/aimanager/manager/run.py` orchestrator
- [ ] Logging: enhance `simulation/run.py` to archive job files
- [ ] Logging: convert `scripts/run_simulation.sh` to template
- [ ] Logging: update `train_cluster.sh` manager branch to use new orchestrator
- [ ] Test all three subcommands with existing configs on Raven
- [ ] Update CLAUDE.md commands section
