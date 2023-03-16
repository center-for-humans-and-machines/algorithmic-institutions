# Setup

## Install main package

```
python3.9 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install torch -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
pip install -e ".[dev]"
pip install -e djx
```

# Notebooks

## Usage

Run a notebook

```
python run.py evaluation/predictive_models ahc_02_valid
```

Update parameter only

```
python run_notebook.py evaluation/predictive_models ahc_02_valid --prepare_only
```

## Examples

### Train behavioral clones

Clone human contributors

```
python run_notebook.py artificial_humans/graph  ahc_02_valid_node+rnn
```

Clone human manager

```
python run_notebook.py artificial_humans/graph  ahc_02_valid_node+rnn
```

### Train RL-based manager

```
python run_notebook.py artificial_humans/graph  ahc_02_valid_node+rnn
```

### Run simulations with multiple manager

```
python run_notebook.py simulate_rule/simulate_rule_v1_2_1  ahc_02_valid_node+rnn
```

# Train on Slurm Cluster

First ensure that the job_pattern at (`djx/djx/job_pattern`) do fit your local
cluster.

## Run a grid

```
djx run/artificial_humans/02_valid.yml
```

# Reproducing Figures

| Figure | Command                                                          |
| ------ | ---------------------------------------------------------------- |
| 1      | python run_notebook.py evaluation/predictive_models ahc_02_valid |

# List of Runs

| Run Name     | Run Folder        | Run File | Description                                                             |
| ------------ | ----------------- | -------- | ----------------------------------------------------------------------- |
| ahc_02_valid | artificial_humans | 02_valid | Model predicting if human contributors are making a valid contribution. |
