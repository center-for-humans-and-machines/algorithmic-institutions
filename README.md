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

Punishments

```
djx run/behavioral_cloning/23_punishment_autoregressive_v4.yml
```

### RL Manager

```
djx run/manager/06_model.yml
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

Punishments

```
python run.py run notebooks/evalutation/predictive_models_autoreg/23_punishment_autoregressive_v4.yml
```

RL Manager

```
python run.py run notebooks/evalutation/rl_models/06_model.yml
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
