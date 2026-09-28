#!/bin/bash -l
#
# GPU SLURM job for the herding-copula-v2 simulation, run provenance-isolated.
# Mirrors scripts/run_simulation.sh's resource block and the
# scripts/artificial_humans/*.slurm interpreter pattern: the job is submitted
# from, and --chdir=. runs it in, its own remote dir
# (~/iso_contr_herdcopar1_v2), which carries no venv of its own, so the shared
# checkout supplies only the interpreter while THIS tree supplies the code.
# Prints a full provenance block before simulating.
# Usage: sbatch scripts/simulate_iso.slurm
#
#SBATCH --chdir=.
#SBATCH --output=temp/simulate_iso_%j.log
#SBATCH --error=temp/simulate_iso_%j.log
#SBATCH --job-name=cg_copula_sim
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --time=01:00:00

set -e

module load cuda/11.4

PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG=configs/simulation/manager_testing/23_2g8a_contr_herding_copula_v2_self_gnncopar1_contr_gnn_switch.yml

echo "===== PROVENANCE ====="
echo "cwd        $PWD"
echo "job        ${SLURM_JOB_ID:-none} on $(hostname)"
echo "config     $CONFIG"
md5sum src/aimanager/generic/copula.py \
       src/aimanager/generic/graph.py \
       src/aimanager/manager/environment.py \
       "$CONFIG"

"$PY" - "$CONFIG" <<'PYEOF'
import hashlib
import sys

import torch as th
import yaml

import aimanager
from aimanager.generic.graph import GraphNetwork

print("aimanager  ", aimanager.__file__)
print("torch      ", th.__version__)
print("cuda avail ", th.cuda.is_available())
print("cuda ver   ", th.version.cuda)
if th.cuda.is_available():
    print("gpu        ", th.cuda.get_device_name(0))

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

ah = cfg["artificial_humans"]["group_switching"]
paths = [
    ("contribution_model", ah["contribution_model"]),
    ("valid_model", ah["valid_model"]),
    ("switch_model", ah["switch_model"]),
]
for name, m in cfg["managers"].items():
    paths.append((f"manager:{name}", m["path"]))

print("--- artifact sha256 ---")
for label, p in paths:
    with open(p, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"{digest}  {label}  {p}")

print("--- loaded contribution copula fields ---")
contr = GraphNetwork.load(ah["contribution_model"], device="cpu")
print("y_name             ", contr.y_name)
print("y_levels           ", contr.y_levels)
print("copula_rho         ", repr(contr.copula_rho))
print("copula_phi         ", repr(contr.copula_phi))
print("copula_switch_every", repr(contr.copula_switch_every))
print("seed               ", repr(cfg["seed"]))
print("n_episodes         ", repr(cfg["n_episodes"]))
print("save_per_round     ", repr(cfg["save_per_round"]))
PYEOF

echo "===== SIMULATE ====="
"$PY" -m aimanager simulate "$CONFIG"
