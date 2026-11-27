#!/bin/bash -l
# One version's "sweep, fit, final" cycle on CPUs, timed step by step, with the
# pipeline's own scripts (as pipeline.py's sweep_config, sweep_jobs, fit,
# final_config, final_jobs and final_fetch steps call them), in a scratch copy
# of the checkout:
#   generate_sim_config.py --n-parts 32       the sweep's 32 part configs
#   32 parts at once, single-threaded, one hyperthread each
#   merge_sweeps.py, fit_surrogate.py         the merged sweep and its GP
#   generate_sim_config.py --final            the final run's config
#   the final run, one single-threaded process (FINAL_THREADS to change)
#   generate_sim_config.py --score-final
#
#   cd /u/$USER/algorithmic-institutions-pf
#   sbatch infra/dais/cycle_bench.slurm.sh
#
#SBATCH -J pf_cycle
#SBATCH -o /dais/fs/scratch/%u/policy-finder/logs/%x.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=small
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120000
#SBATCH --time=01:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/infra/dais/common.sh"
init_run
hostname
lscpu | grep -E "Model name"
export REPO_MODE=rw   # the copy in the run folder; outputs go to its plots/
PF=scripts/policy_finder
OUTDIR=plots/simulation/policy_finder
T0=$SECONDS

step() {  # <label> <command...>: run in the sim image, print its wall time
  local label="$1" t0=$SECONDS
  shift
  in_sim_image "$@" > "$PF_RUN/$label.log" 2>&1 || { tail -20 "$PF_RUN/$label.log"; return 1; }
  echo "STEP $label wall=$(( SECONDS - t0 ))"
}

tree="$PF_RUN/tree"
mkdir -p "$tree"
cp -r "$PF_REPO"/{src,scripts,configs,artifacts} "$tree/"
cp "${SWEEP_RULE:-$PF_SCRATCH/ref-pf-20261104-1/configs/managers/rule_based/pf-20261104-1.yml}" \
  "$tree/configs/managers/rule_based/bench_rule.yml"
PF_REPO="$tree"

step sweep_config python $PF/generate_sim_config.py \
  --config configs/managers/rule_based/bench_rule.yml --max-params none --n-parts 32

mapfile -t CPUS < <(python3 "$PF_HERE/cpu_slots.py" slots 0 1 | tr , '\n')
t0=$SECONDS
pids=()
for k in $(seq 1 32); do
  SIM_CPUS=${CPUS[$(( (k - 1) % ${#CPUS[@]} ))]} \
    simulate "configs/simulation/policy_finder/bench_rule_sweep_p${k}of32.yml" "part$k" &
  pids+=($!)
done
fails=0
for pid in "${pids[@]}"; do wait "$pid" || fails=$((fails + 1)); done
echo "STEP sweep_sims wall=$(( SECONDS - t0 )) parts=32 on ${#CPUS[@]} CPUs failures=$fails"
[ "$fails" = 0 ]

parts=()
for k in $(seq 1 32); do parts+=("$OUTDIR/bench_rule_sweep_p${k}of32"); done
step merge python $PF/merge_sweeps.py "${parts[@]}" --out "$OUTDIR/bench_rule_sweep/sweep.json"
step fit python $PF/fit_surrogate.py "$OUTDIR/bench_rule_sweep"
step final_config python $PF/generate_sim_config.py --final "$OUTDIR/bench_rule_sweep/sweep.json"
grep -E "^seed|^n_episodes|^episode_batch_size" \
  "$tree/configs/simulation/policy_finder/bench_rule_sweep_final.yml"
echo "final run points: $(grep -c '_vs_ah' "$tree/configs/simulation/policy_finder/bench_rule_sweep_final.yml")"
t0=$SECONDS
SIM_THREADS="${FINAL_THREADS:-1}" SIM_CPUS=${CPUS[0]} \
  simulate configs/simulation/policy_finder/bench_rule_sweep_final.yml final
echo "STEP final_sim wall=$(( SECONDS - t0 ))"
step score_final python $PF/generate_sim_config.py --score-final "$OUTDIR/bench_rule_sweep/final/sweep.json"
cat "$PF_RUN/score_final.log"
echo "CYCLE wall=$(( SECONDS - T0 )) (including the copy of the checkout)"
