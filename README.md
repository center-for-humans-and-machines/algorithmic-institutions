# Setup

## Clone repository with sub-modules on Tardis Cluster

> You need to run this command on Tardis Cluster.

To clone the repository along with its sub-modules, use the following command:
```bash
git clone --recurse-submodules git@github.com:center-for-humans-and-machines/algorithmic-institutions.git
```

## Install main package

### 0. Installing `uv`
It is advised to use `uv` as the virtual environment manager. Install `uv` if you don't have it already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
check also [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/)

### 1. Create and activate virtual environment

The project uses platform-based environment markers in `pyproject.toml`:
- **Linux**: Installs `torch==1.11.0+cu113` with CUDA support and PyTorch Geometric subpackages (`torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`, `torch-geometric`).
- **Non-Linux (macOS, Windows)**: Installs `torch==1.11.0` (CPU-only) from PyPI without PyTorch Geometric subpackages.

Create and activate a virtual environment using `uv`:
```bash
uv sync
```
this will create a virtual environment in `.venv` folder and install all dependencies listed in `pyproject.toml` file.

You can then activate the virtual environment using:
```bash
source .venv/bin/activate
```

### 2. Install pre-commit for development
For automated checks to keep codebase structure and code format one needs to install pre-commit

```bash
pre-commit install
```

### 3. Secrets (`.env`)

Copy `.env.example` to `.env` and fill in the API keys:

```bash
cp .env.example .env
```

To log training runs to Weights & Biases, set `WANDB_API_KEY`. On Raven, copy the
key from an existing project (e.g. the `collectively-grounded-llms` repo) or grab
it from https://wandb.ai/authorize. `.env` is gitignored; SLURM templates source
it to export `WANDB_*` variables into the job environment.

### Alternative: Manual virtual environment setup
If you prefer to set up the virtual environment manually without `uv`, you can follow these steps:

**On Linux (with CUDA):**
```
python3.9 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install -e ".[dev]"
```

**On macOS/Windows (CPU-only):**
```
python3.9 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install torch==1.11.0
pip install -e ".[dev]"
```

# Development Workflow

See [doc/claude_code_workflow.md](doc/claude_code_workflow.md) for the AI-assisted development workflow using Claude Code agents.

# Training, Simulation, Evaluation

All pipelines are config-driven (YAML in `configs/`) behind one CLI:

```bash
python -m aimanager train-ah <config>       # artificial humans (supervised)
python -m aimanager train-manager <config>  # RL manager
python -m aimanager simulate <config>       # set save_per_round: true if the run will be evaluated
python -m aimanager evaluate <config>       # sim vs human metrics, scores, visuals
```

Training and simulation need the full PyG environment and run on the Raven
cluster; evaluation runs locally against the simulation's `per_round.parquet`.

## From the local machine

With an SSH ControlMaster connection active (`ssh raven` in a separate
terminal, persists 12h), the cluster scripts sync the repo and submit the job
through the SLURM orchestrators:

```bash
scripts/train_cluster.sh ah <config>        # artificial humans
scripts/train_cluster.sh manager <config>   # RL manager
scripts/simulate_cluster.sh <config>        # simulation
scripts/remote_test.sh                      # PyG-dependent tests
scripts/fetch_cluster.sh <remote_path>      # bring results back (no trailing slash)
```

## Skill-based workflow

In Claude Code the same operations are exposed as skills -- `/train`,
`/simulate`, `/test`, `/fetch-cluster`, plus `/commit` and `/pr`. See
[doc/claude_code_workflow.md](doc/claude_code_workflow.md) for the agentic
development workflow around them (issue labels, plans, review).

## Directly on the cluster

From `~/algorithmic-institutions` on Raven with the remote `.venv` activated,
run the SLURM orchestrators yourself:

```bash
python src/aimanager/artificial_humans/run.py <config>  # submits AH training
python src/aimanager/manager/run.py <config>            # submits manager training
python src/aimanager/simulation/run.py <config>         # submits simulation
```

Each submits GPU batch jobs (see `scripts/run_simulation.sh` for the
simulation template); single runs can also be executed in place with
`python -m aimanager <command> <config>` on a GPU node.
