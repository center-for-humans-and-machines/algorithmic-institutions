# Setup

## Install main package

```
python3.9 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install torch -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip install -e ".[dev]"
pip install -e djx
```

# Notebooks

Update parameter from yml

```
papermill -f logreg.yml --prepare-only logreg.ipynb logreg.ipynb
papermill -f neural.yml --prepare-only neural.ipynb neural.ipynb
```

Run notebook from yml.

```
papermill -f logreg.yml logreg.ipynb logreg.ipynb
```

# Cluster

## Setup DJX

Make sure djx submodule is updated.

```
git submodule update --init --remote --recursive
```

Install djx.

```
pip install ./djx
```
