# Algorithmic Institutions

---

## General Remarks

### Purpose

A research project exploring AI-driven group management dynamics. Uses supervised deep learning to create artificial humans that mimic real human contributor behavior, then trains reinforcement learning managers to optimize group outcomes (common good) in a public goods game setting. The pipeline covers data preprocessing, model training, simulation, and evaluation.

### Implementation Remarks

- **Stack**: Python 3.9, uv (package manager), PyTorch + PyTorch Geometric, pandas, seaborn
- **Cluster**: Raven GPU cluster via SLURM
- **Code style**: Black formatter, flake8 -- enforced via pre-commit hooks on `src/` only. **88-char line limit**. Extend-ignore: `E203, W503`. Repo-wide hooks (trailing-whitespace, end-of-file-fixer, check-yaml) also run at commit and can mutate staged non-Python files
- **Config-driven**: Experiments defined via YAML configs in `configs/`
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
  evaluation_suite/               # Sim-vs-human evaluation (metrics + scores + visuals)
    convert.py                    # Canonical agent-round frame; loads human and sim data
    metrics.py                    # Metric row extractions (C/S/P/R families)
    scoring.py                    # Noise-ceiling normalised scores
    evaluate.py                   # Entry point for `python -m aimanager evaluate`
    visuals.py                    # One figure per metric row
  utils/                          # Shared utilities
scripts/                          # Executable shell/python scripts
  artificial_humans/              # AH training SLURM templates
  baselines/                      # Linear baseline training + CV (joblib bundles)
  data_analysis/                  # Analysis scripts (incl. evaluation_sweep.py)
  manager/                        # Manager training SLURM templates
  data_creation/                  # Data preprocessing scripts
  plotting/                       # Reusable plotting scripts
  tests/                          # Script-level tests (run locally)
  remote_test.sh                  # Run tests on Raven cluster
  train_cluster.sh                # Submit training on Raven cluster
  simulate_cluster.sh             # Submit simulation on Raven cluster
  fetch_cluster.sh                # Fetch files from Raven cluster
  run_simulation.sh               # Batch GPU simulation SLURM template
configs/                          # YAML experiment configurations
  training/artificial_humans/     # AH (GNN) training configs
  training/baselines/             # Linear baseline training configs
  training/rl_manager/            # RL manager training configs
  managers/rule_based/            # Rule-based manager rules (YAML) and params (JSON)
  simulation/                     # Simulation configs (stacks: manager_testing/24_*;
                                  #   manager pairings: manager_testing/25_*)
plots/                            # Generated plots and figures
  simulation/                     # Simulation result plots (incl. per-run evaluation/)
  data_analysis/                  # Cross-run analysis outputs (incl. sweep score matrices)
experiments/                      # Human experiment data
artifacts/                        # Trained model artifacts (GNN dirs with .pt files;
                                  #   linear .joblib bundles under baselines/)
reports/                          # Research documentation and reports
notes/                            # Normative definitions (evaluation metrics, scoring schema)
run/                              # Legacy DJX run definitions (djx no longer used)
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
- `src/aimanager/evaluation_suite/evaluate.py` -- Evaluation entry point (metrics, scores, visuals)
- `src/aimanager/evaluation_suite/metrics.py` -- Metric row extractions, in code
- `scripts/data_analysis/evaluation_sweep.py` -- Cross-stack score matrix and sweep figures
- `notes/evaluation_metric_defs.md` -- Normative definitions of every metric row
- `notes/eval_scoring_schema.md` -- The noise-ceiling scoring schema
- `scripts/plotting/plot_confusion_matrix.py` -- Reusable confusion matrix plot
- `scripts/data_creation/group_switching_preprocess.py` -- Group switching data preprocessing
- `reports/basics.md` -- Game rules and experimental setup reference

### Evaluation Suite

Compares finished simulations against the human reference data,
`experiments/2group_8agent_50ep.csv`. Every game appears twice in the file with
group labels mirrored -- the flip augmentation that removes group-label bias.
The GNN models train on this doubled data (some artifact names carry a
`_doubled` suffix); the linear baselines train on the single-copy data, and the
evaluation suite likewise keeps one copy per game.

- The simulation config must set `save_per_round: true` -- `evaluate` reads the
  sim's `per_round.parquet` and hard-fails without it.
- `python -m aimanager evaluate <sim config>` writes to the sim's output dir:
  `evaluation/metrics.csv` (raw discrepancies), `evaluation/scores.csv`
  (normalised scores), and `evaluation/visuals/*.jpg` (one figure per row).
- A score is a multiple of the human-vs-human noise ceiling (500 resampling
  repeats, master seed 42): <= 1 at the ceiling, 1-2 minor, 2-5 clear deviation,
  \> 5 not reproduced. 22 rows; row definitions: `notes/evaluation_metric_defs.md`;
  schema: `notes/eval_scoring_schema.md`. A single seed moves a row by ~0.1-0.3
  (contribution rows most): compare stacks over several seeds.
- Simulation configs reference model artifacts by path and dispatch on
  extension: `.joblib` -> linear-baseline adapter, `.pt` -> GNN; one config may
  mix both (see `simulation/simulate.py`).
- Timeouts: `per_round.parquet` records `contribution_valid`, and `load_sim`
  blanks timed-out contributions as `load_human` does. The sim config's
  `timeout_contribution` sets the contribution recorded for a timed-out player
  (0-20, or `default` = the dataset median, also when absent; training data
  records 0). A punishment aimed at a timed-out player is always charged 0.
- The punisher reads round t's contribution (the manager punishes after that
  round's contributions); see `notes/baseline_feature_defs.md`.
- Reference sim: `plots/simulation/24_FRONTIER_vnode_curpun_self_gnncopar1_contr_gnn_switch`
  (seed replicates and ablations: `configs/simulation/manager_testing/ablation_224/`).
  It is the evaluation-suite reference only; manager sims and RL runs use
  Levin's stack (see Manager Comparison).
- A sim run directory looks like `plots/simulation/<name>/{per_round.parquet,
  evaluation/{metrics.csv, scores.csv, visuals/}}`.
- `scripts/data_analysis/evaluation_sweep.py` aggregates a sweep's `scores.csv`
  files into a score matrix and slot-level figures under
  `plots/data_analysis/evaluation/<name>/`; it parses the sim-dir naming
  convention `..._self_<contr>_contr_<switch>_switch`.

### Manager Comparison

- Experimental stack for manager sims and RL runs: Levin's stack,
  `configs/simulation/manager_testing/24_LEVIN_vnode_skip_timeoutpun_self_gnncopar1_contr_gnn_switch.yml`
  (its `artificial_humans` block names the three models).
- The #226 reference sims on it, `configs/simulation/manager_testing/25_LEVIN_run1_*`:
  the AH manager and the zero punisher, with outputs in
  `plots/simulation/25_LEVIN_run1_*`.
- At 100 episodes per pairing a manager's win rates and margins move with the
  seed by more than most gaps between managers. Compare managers on the
  `*_batched.yml` runs: 1000 episodes per pairing, cheap with
  `episode_batch_size` (#232).
- A sim with `pairings:` puts two managers in the two groups; pairing names are
  `<g0>_vs_<g1>`, which the plotting scripts parse.
- `episode_batch_size: N` in a sim config (1-20000, default 1) plays N episodes
  at once on the GPU, pairings mixed within a batch (#232), several times
  faster than one at a time. Batch i is seeded with `seed + i`, so results
  depend on N and never match a per-episode run bit for bit. Only managers
  with `batched_punish` (`dummy`, `rule_based`, `linear`) run batched; `rl`
  and `human` need `episode_batch_size: 1`.
- A rule-based manager is a config, not code: `type: rule_based` with `rule:`
  (a YAML in `configs/managers/rule_based/` with `params`, optional
  `constraints` and `code`) and `params:` (a JSON of the values), both
  required. `load_rule` in `manager/api_manager.py` documents the schema and
  the load-time checks.
- `scripts/plotting/plot_winrates.py <sim_dir> [<sim_dir> ...]` tables the five
  win definitions of #226. Definitions 1-3 and 5 are head to head. Definition 4
  (Levin's) ranks managers by their group's common pool against `ah`, so it
  compares only managers that played `ah` (others are named in a warning);
  pass the run with `zero_vs_ah` for its reference margins.
- RL training on Levin's stack: `configs/training/rl_manager/04_2g8a_levin.yml`
  (check a setup first with `04_2g8a_levin_smoke.yml`, 50 update steps). Its
  `opponent_manager` may be a `.joblib` linear punisher
  (`manager/linear_opponent.py`, batched) or a `.pt` GNN; set
  `env_args.timeout_contribution` as in the sims.

### Git Workflow

**IMPORTANT**: All commits must be made using the `/commit` skill. This ensures staged files are reviewed before committing.

- `main` is frozen for now: do not commit, push or merge to it (a team
  convention; GitHub does not enforce it).
- Work on the optimized stack (Levin's stack, the rule-based managers, RL
  training on them) goes on `autoresearch-optimized-stack`: branch off it and
  open PRs into it, not into `main`.
- `policy-finder-base` is cut from `autoresearch-optimized-stack` with the
  existing rule-based managers removed (#235). It never merges back: #227 and
  the agent core land on it, and every policy-finder instance branches off it.

### Environment

- PyG/CUDA packages are Linux-only (see `sys_platform` markers in `pyproject.toml`)
- Local macOS has CPU-only `torch==1.11.0` without PyG subpackages
- Full environment (torch + CUDA + PyG) only available on Raven cluster
- `*.csv`, `*.parquet`, `*.pt` are Git LFS-tracked (`.gitattributes`): data and
  artifact files are pointers until `git lfs pull`

### Where Things Run

| Stage | Where |
|---|---|
| `train-ah`, `train-manager`, `simulate` | Raven (SLURM; `train_cluster.sh` / `simulate_cluster.sh`) |
| `evaluate`, plotting/analysis scripts | Local macOS |
| PyG-dependent tests (encoder, environment, edge encoder, linear manager) | Raven (`remote_test.sh`) |
| Evaluation-suite tests, `scripts/tests/` | Local |

Results come back from the cluster via `scripts/fetch_cluster.sh`.

### Testing

PyG-dependent tests MUST be run on the Raven cluster: they import
`torch_scatter` and other PyG packages that are only available on Linux, even
with `device="cpu"`. The evaluation-suite tests (`test_eval_*.py`) and
`scripts/tests/` have no PyG imports and run locally with plain `pytest`.

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
- `src/aimanager/tests/test_encoder.py` - Tensor encoder unit tests (Raven)
- `src/aimanager/tests/test_edge_encoder.py` - Edge encoder unit tests (Raven)
- `src/aimanager/tests/test_environment.py` - RL environment unit tests (Raven)
- `src/aimanager/tests/test_linear_manager.py` - Linear manager unit tests (Raven)
- `src/aimanager/tests/test_eval_convert.py` - Evaluation-suite canonical frame (local)
- `src/aimanager/tests/test_eval_metrics.py` - Evaluation-suite metric rows (local)
- `src/aimanager/tests/test_eval_scoring.py` - Evaluation-suite scoring (local)
- `src/aimanager/tests/test_eval_evaluate.py` - Evaluation-suite end to end (local)
- `src/aimanager/tests/test_eval_visuals.py` - Evaluation-suite figures (local)
- `src/aimanager/tests/test_*copula*.py`, `test_joint_exodus*.py`, `test_group_vnode*.py`,
  `test_conditional_bernoulli.py` - Frontier-stack mechanisms (the copula, joint-exodus
  and group-vnode tests run locally; some substitute PyG stand-ins on macOS)
- `src/aimanager/tests/test_baseline_features.py` - Linear-baseline feature parity (local;
  fixture in `src/aimanager/tests/fixtures/`)
- `src/aimanager/tests/test_linear_opponent.py` - RL linear opponent parity with the
  sim path (local)
- `src/aimanager/tests/test_rule_validate.py` - Rule sweep design and `validate-rule`
  (local; `test_rule_config.py`, the rule-based manager, runs on Raven)
- `scripts/tests/test_remote_test.py` - Remote test script tests (local)

### Remote Cluster (Raven)

- Host: raven.mpcdf.mpg.de (via ProxyJump through gate.mpcdf.mpg.de)
- User: each contributor uses their own account and checkout; record yours in
  `CLAUDE.local.md` (gitignored), which is loaded alongside this file
- Project path: ~/algorithmic-institutions (on the ssh alias `raven`)
- Remote `.venv` must be pre-configured
- Tests run on login node (no GPU needed)

### Commands

- **Install**: `uv sync`
- **Pre-commit install**: `pre-commit install`
- **Pre-commit run**: `pre-commit run --all-files`
- **Format**: `black src/`
- **Lint**: `flake8 src/ --max-line-length=88 --extend-ignore=E203,W503`
- **Run tests**: `scripts/remote_test.sh`
- **Fetch from cluster**: `scripts/fetch_cluster.sh <remote_path>` (path relative to `~/algorithmic-institutions`, no trailing slash)
- **Train AH models**: `python -m aimanager train-ah <config>`
- **Train RL manager**: `python -m aimanager train-manager <config>`
- **Run simulation**: `python -m aimanager simulate <config>` (set `save_per_round: true` if the run will be evaluated; `episode_batch_size` to batch episodes, see Manager Comparison)
- **Validate a rule config**: `python -m aimanager validate-rule <rule.yml> [--max-params N]` (local; the checks and the Sobol design are in `manager/rule.py`)
- **Evaluate sim vs human**: `python -m aimanager evaluate <config>` (needs the simulation's `per_round.parquet`)
- **Plot confusion matrix**: `python scripts/plotting/plot_confusion_matrix.py <artifact_dir>`

### Where to Find Things

- Source code: `src/aimanager/`
- Scripts: `scripts/` (cluster orchestration, data creation, plotting)
- Experiment configs: `configs/` (YAML)
- Plots and figures: `plots/`
- Legacy DJX run definitions: `run/` (djx no longer used)
- Trained artifacts: `artifacts/` (GNN dirs; linear joblib bundles in `artifacts/baselines/`)
- Human reference data: `experiments/2group_8agent_50ep.csv`
- Metric and scoring definitions: `notes/`
- Research docs: `reports/`
- Game rules: `reports/basics.md`
- Cluster setup: `README.md`
