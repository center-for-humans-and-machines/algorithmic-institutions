#!/bin/bash -l
#
# GPU SLURM job for one simulation config, run provenance-isolated.
#
# Same shape as scripts/simulate_iso.slurm, with two differences that the
# rule-manager arms need:
#   * the config is an ARGUMENT, not baked in, so one script serves every
#     seed instead of being edited between submissions;
#   * the provenance block tolerates managers without a `path` (the rule
#     family is defined by its parameters, not by an artifact) and prints
#     those parameters instead, so the log records exactly which rule ran.
#
# --chdir=. runs the job in the directory it was submitted from, which for
# these arms is an isolated remote dir carrying no venv of its own: the
# shared checkout supplies only the interpreter, THIS tree supplies the code.
#
# Usage: sbatch scripts/simulate_config_iso.slurm <config.yml>
#
#SBATCH --chdir=.
#SBATCH --output=temp/simulate_cfg_%j.log
#SBATCH --error=temp/simulate_cfg_%j.log
#SBATCH --job-name=rinv_sim
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

CONFIG="${1:?usage: sbatch scripts/simulate_config_iso.slurm <config.yml>}"

echo "===== PROVENANCE ====="
echo "cwd        $PWD"
echo "job        ${SLURM_JOB_ID:-none} on $(hostname)"
echo "config     $CONFIG"
md5sum src/aimanager/manager/api_manager.py \
       src/aimanager/simulation/simulate.py \
       src/aimanager/manager/environment.py \
       "$CONFIG"

"$PY" - "$CONFIG" <<'PYEOF'
import hashlib
import sys

import torch as th
import yaml

import aimanager

print("aimanager  ", aimanager.__file__)
print("torch      ", th.__version__)
print("cuda avail ", th.cuda.is_available())
if th.cuda.is_available():
    print("gpu        ", th.cuda.get_device_name(0))

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

ah = cfg["artificial_humans"]["group_switching"]
paths = [(f"stack:{k}", v) for k, v in ah.items()]
for name, m in cfg["managers"].items():
    if "path" in m:
        paths.append((f"manager:{name}", m["path"]))

print("--- artifact sha256 ---")
for label, p in paths:
    with open(p, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"{digest}  {label}  {p}")

print("--- managers without an artifact (parameters ARE the manager) ---")
for name, m in cfg["managers"].items():
    if "path" not in m:
        print(f"  {name}: {m}")

print("--- protocol ---")
for k in ("seed", "reseed_per_run", "n_episodes", "n_rounds", "switch_every"):
    print(f"  {k:16s} {cfg.get(k)!r}")
print(f"  {'save_per_round':16s} {cfg.get('save_per_round')!r}")
print(f"  {'pairings':16s} {[p['name'] for p in cfg.get('pairings', [])]}")
PYEOF

echo "===== SIMULATE ====="
"$PY" -m aimanager simulate "$CONFIG"
