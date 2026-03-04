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

### 3. Installing `djx` sub-module
Install `djx` sub-module in editable mode:
```bash
uv pip install -e djx
```

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
pip install -e djx
```

**On macOS/Windows (CPU-only):**
```
python3.9 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install torch==1.11.0
pip install -e ".[dev]"
pip install -e djx
```

# Up-to-date Docs for Cluster Runs
- The up-to-date version of the pipeline is detailed [here](reports/up_to_date_docs.md).

# Notebooks
Tardis and Raven clusters use different slurm scripts. To run scripts on the GPU infrastructure one needs to modify the script field in the respective config file accordingly.
```yaml
# Tardis
...
exec:
  command: python run.py run {job_file}
  script_name: gpu
  cores: 2
...

# Raven
exec:
  command: python run.py run {job_file}
  script_name: gpu_raven
  cores: 2
...
```

## Retrain Models


### Behavioral Clones

Contribution

```
djx run/behavioral_cloning/21_contribution_model_v4.yml
```

Contribution Is Valid

```
djx run/behavioral_cloning/22_contribution_valid_model_v4.yml
```

Punishments (not used)

```
djx run/behavioral_cloning/23_punishment_autoregressive_v4.yml
```

### RL Manager

```
djx run/manager/07_exp2.yml
```

## Evaluate Models

### Behavioral Clones

Contribution

```
python run.py run notebooks/evalutation/predictive_models_autoreg/21_contribution_model_v4.yml
```

Contribution Is Valid

```
python run.py run notebooks/evalutation/predictive_models_autoreg/22_contribution_valid_model_v4.yml
```

Punishments (not used)

```
python run.py run notebooks/evalutation/predictive_models_autoreg/23_punishment_autoregressive_v4.yml
```

RL Manager

```
python run.py run notebooks/evalutation/rl_models/07_exp2.yml
```

### Run Simulations

Should be run on a GPU node.
```
python run.py run notebooks/test_manager/simulate_mixed/03_all.yml
```


# Simulate existing models

Should be run on a GPU node.
```
python run.py run notebooks/test_manager/simulate_mixed/02_all_artifacts.yml
```


# Reproducing Figures

| Figure | Command                                                          |
| ------ | ---------------------------------------------------------------- |
| 1      | python run_notebook.py evaluation/predictive_models ahc_02_valid |

# List of Runs

| Run Name     | Run Folder        | Run File | Description                                                             |
| ------------ | ----------------- | -------- | ----------------------------------------------------------------------- |
| ahc_02_valid | artificial_humans | 02_valid | Model predicting if human contributors are making a valid contribution. |
