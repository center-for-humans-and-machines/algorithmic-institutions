#!/bin/bash -l
#
# CPU-only SLURM job for the RCB step-6 mechanism gate.
# Runs scripts/data_analysis/rcb_teacher_forced.py four times -- the parent
# and the candidate trunk, each teacher-forced over the PARENT's simulated
# trajectories and over the human data -- so the four tables come out of one
# job with one code version. Real compute (four unpickled GNN forward passes
# over 100 episodes), so it goes through the batch system, not a login node.
# Usage: sbatch scripts/data_analysis/rcb_teacher_forced_sim.slurm
#
#SBATCH --chdir=.
#SBATCH --output=temp/rcb_tf_sim_%j.log
#SBATCH --error=temp/rcb_tf_sim_%j.log
#SBATCH --job-name=rcb_tf_sim
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#
#SBATCH --time=00:40:00

set -e

# Provenance-isolated run: this job is submitted from, and --chdir=. runs it
# in, its own remote dir (~/autoresearch/contribution-punish-onehot), which
# carries no venv of its own. Borrow the shared checkout's interpreter but
# keep THIS tree's code ahead of the editable install on sys.path.
PY="$HOME/algorithmic-institutions/.venv/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "cwd        $PWD"
"$PY" -c 'import aimanager; print("aimanager ", aimanager.__file__)'

AH=artifacts/artificial_humans
PARENT="$AH/group_switching_contribution_50ep_group_vnode/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
CANDIDATE="$AH/group_switching_contribution_50ep_vnode_stimulus_skip/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
SIM=plots/simulation/23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch/per_round.parquet

run () {
  echo ""
  echo "############################################################"
  echo "### RUN $1"
  echo "############################################################"
  shift
  "$PY" scripts/data_analysis/rcb_teacher_forced.py "$@"
}

run "1: PARENT trunk x PARENT sim states" \
    --model "$PARENT" --sim-parquet "$SIM"
run "2: CANDIDATE trunk x PARENT sim states" \
    --model "$CANDIDATE" --sim-parquet "$SIM"
run "3: CANDIDATE trunk x HUMAN data" \
    --model "$CANDIDATE"
run "4: PARENT trunk x HUMAN data (step-0 regression check)" \
    --model "$PARENT"
