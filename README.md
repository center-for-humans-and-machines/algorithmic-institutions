# Setup

## Install main package

```
python3.9 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install -e ".[dev]"
```

# Notebooks

Update parameter from yml

```
papermill -f logreg.yml --prepare-only logreg.ipynb logreg.ipynb
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
