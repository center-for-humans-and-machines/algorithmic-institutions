# Algorithmic Institutions

---

## General Remarks

### Purpose

A research project exploring AI-driven group management dynamics. Uses supervised deep learning to create artificial humans that mimic real human contributor behavior, then trains reinforcement learning managers to optimize group outcomes (common good) in a public goods game setting. The pipeline covers data preprocessing, model training, simulation, and evaluation.

### Implementation Remarks

- **Stack**: Python 3.9, uv (package manager), PyTorch + PyTorch Geometric, pandas, seaborn
- **Clusters**: Tardis and Raven GPU clusters via SLURM; `djx` submodule for reproducible experiment management
- **Code style**: Black formatter, flake8 -- enforced via pre-commit hooks on `src/` only. **88-char line limit**. Extend-ignore: `E203, W503`
- **Config-driven**: Experiments defined via YAML configs in `configs/`; notebooks parameterized via Papermill
- **Key pattern**: Artificial humans (supervised learning on pilot data) + RL manager (reinforcement learning to maximize common good) + simulation (testing managers against artificial humans)

---

## Agent Navigation Guide

### Project Structure

```
src/aimanager/                    # Main Python package
  __main__.py                     # Unified CLI entry (python -m aimanager)
  cli.py                          # CLI dispatch, config validation
  rl_manager.py                   # RL manager training logic
  artificial_humans/              # Artificial human models (supervised DL)
    train.py                      # Training logic
    run.py                        # SLURM orchestrator for AH training
    evaluation.py                 # Model evaluation
    grid.py                       # Grid search utilities
  manager/                        # Manager models
    manager.py                    # Base manager logic
    environment.py                # RL environment
    run.py                        # SLURM orchestrator for manager training
    artificial_human_group.py     # Group of artificial humans for training
    api_manager.py                # API-based manager interface
  generic/                        # Generic model components and encoders
  simulation/                     # Simulation framework
    simulate.py                   # Core simulation logic
    run.py                        # SLURM orchestrator for simulations
  utils/                          # Shared utilities
scripts/                          # Executable shell/python scripts
  artificial_humans/              # AH training SLURM templates
  manager/                        # Manager training SLURM templates
  data_creation/                  # Data preprocessing scripts
  plotting/                       # Reusable plotting scripts
  tests/                          # Script-level tests (run locally)
  remote_test.sh                  # Run tests on Raven cluster
  train_cluster.sh                # Submit training on Raven cluster
  simulate_cluster.sh             # Submit simulation on Raven cluster
  fetch_cluster.sh                # Fetch files from Raven cluster
  run_simulation.sh               # Batch GPU simulation SLURM template
notebooks/                        # Jupyter notebooks (excluded from linting)
  artificial_humans/              # AH-related notebooks
  evalutation/                    # Evaluation notebooks
  reports/                        # Report-generating notebooks
  test_manager/                   # Manager testing/simulation notebooks
configs/                          # YAML experiment configurations
  training/artificial_humans/     # AH training configs
  training/rl_manager/            # RL manager training configs
  simulation/                     # Simulation configs
plots/                            # Generated plots and figures
  group_selection/                # AH model evaluation plots
  simulation/                     # Simulation result plots
artifacts/                        # Trained model artifacts
reports/                          # Research documentation and reports
run/                              # DJX experiment run definitions
djx/                              # Git submodule (experiment framework)
doc/plans/                        # Implementation plans (status in title)
  archive/                        # Completed plans (DONE, ABANDONED)
```

### Plan Workflow

Implementation plans live in `doc/plans/` as flat files. Status is tracked in the first heading:

- `# [DRAFT] Title` -- plan is being written
- `# [ACTIVE] Title` -- plan is approved and in progress
- `# [DONE] Title` -- plan is fully implemented
- `# [PAUSED] Title` -- plan is on hold
- `# [ABANDONED] Title` -- plan was dropped, kept for reference

Completed plans (`[DONE]`, `[ABANDONED]`) are moved to `doc/plans/archive/`.

### Issue Labels

Labels control the workflow for GitHub issues:

- `architect-agent-ready` -- Well-specified issue, ready for the architect agent to write a plan
- `human-plan-review` -- Architect has written a plan; awaiting human review
- `engineer-agent-ready` -- Plan approved, ready for the engineer agent to implement
- `research-agent-ready` -- Plan approved, ready for the researcher agent to implement
- `human-review` -- Implementation done (PR open), awaiting human review
- `human-specification-required` -- Issue is unclear or missing detail; needs human clarification before any agent work

### Key Files

- `src/aimanager/cli.py` -- Unified CLI dispatch and config validation
- `src/aimanager/__main__.py` -- Entry point for `python -m aimanager`
- `src/aimanager/artificial_humans/train.py` -- AH model training logic
- `src/aimanager/artificial_humans/run.py` -- SLURM orchestrator for AH training
- `src/aimanager/manager/environment.py` -- RL training environment for managers
- `src/aimanager/manager/manager.py` -- Manager model logic
- `src/aimanager/manager/run.py` -- SLURM orchestrator for manager training
- `src/aimanager/simulation/simulate.py` -- Core simulation (manager vs artificial humans)
- `src/aimanager/simulation/run.py` -- SLURM orchestrator for simulations
- `src/aimanager/rl_manager.py` -- RL manager training logic
- `scripts/plotting/plot_confusion_matrix.py` -- Reusable confusion matrix plot
- `scripts/data_creation/group_switching_preprocess.py` -- Group switching data preprocessing
- `scripts/data_creation/pilot_pseudo_group_matching.py` -- Old 4-player to 8-player data transform
- `run.py` -- Notebook runner using Papermill with YAML parameter files
- `reports/basics.md` -- Game rules and experimental setup reference

### Git Workflow

**IMPORTANT**: All commits must be made using the `/commit` skill. This ensures staged files are reviewed before committing.

### Environment

- PyG/CUDA packages are Linux-only (see `sys_platform` markers in `pyproject.toml`)
- Local macOS has CPU-only `torch==1.11.0` without PyG subpackages
- Full environment (torch + CUDA + PyG) only available on Raven cluster

### Testing

**IMPORTANT**: Tests MUST be run on the Raven HPC cluster, not locally.
The test suite depends on `torch_scatter` and other PyG packages that are only
available on Linux. Even tests that use `device="cpu"` will fail to import locally.

```bash
# Run all tests (syncs code first):
scripts/remote_test.sh

# Sync only (no tests):
scripts/remote_test.sh --sync-only

# Test only (skip sync):
scripts/remote_test.sh --test-only

# Run specific tests:
scripts/remote_test.sh -- -k test_encoder -v
```

**Prerequisites**: SSH ControlMaster must be active (`ssh raven` in a separate terminal, persists 12h).

**Test logs**: `.claude/test-logs/latest.log` (symlink to most recent run)

**Test locations**:
- `src/aimanager/tests/test_encoder.py` - Tensor encoder unit tests
- `src/aimanager/tests/test_environment.py` - RL environment unit tests
- `scripts/tests/test_remote_test.py` - Remote test script tests (runs locally)

### Remote Cluster (Raven)

- Host: raven.mpcdf.mpg.de (via ProxyJump through gate.mpcdf.mpg.de)
- User: certuer
- Project path: ~/algorithmic-institutions
- Remote `.venv` must be pre-configured
- Tests run on login node (no GPU needed)

### Commands

- **Install**: `uv sync`
- **Install djx**: `uv pip install -e djx`
- **Pre-commit install**: `pre-commit install`
- **Pre-commit run**: `pre-commit run --all-files`
- **Format**: `black src/`
- **Lint**: `flake8 src/ --max-line-length=88 --extend-ignore=E203,W503`
- **Run tests**: `scripts/remote_test.sh`
- **Fetch from cluster**: `scripts/fetch_cluster.sh <remote_path>` (path relative to `~/algorithmic-institutions`)
- **Run notebook**: `python run.py run <yaml_config>`
- **Train AH models**: `python -m aimanager train-ah <config>`
- **Train RL manager**: `python -m aimanager train-manager <config>`
- **Run simulation**: `python -m aimanager simulate <config>`
- **Evaluate sim vs human**: `python -m aimanager evaluate <config>` (needs the simulation's `per_round.parquet`)
- **Plot confusion matrix**: `python scripts/plotting/plot_confusion_matrix.py <artifact_dir>`

### Where to Find Things

- Source code: `src/aimanager/`
- Scripts: `scripts/` (cluster orchestration, data creation, plotting)
- Experiment configs: `configs/` (YAML)
- Plots and figures: `plots/`
- DJX run definitions: `run/`
- Notebooks: `notebooks/`
- Trained artifacts: `artifacts/`
- Research docs: `reports/`
- Game rules: `reports/basics.md`
- Cluster setup: `README.md`
