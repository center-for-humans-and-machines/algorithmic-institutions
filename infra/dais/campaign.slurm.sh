#!/bin/bash -l
# A policy-finder campaign with its agents, entirely on DAIS, in ONE job:
# vLLM serves the model on the job's GPUs, the agents run OpenCode against it
# as Apptainer processes (PF_AGENT_BACKEND=apptainer,
# scripts/policy_finder/agent_apptainer.py) and the sims run in the local
# CPU pool (PF_CLUSTER_BACKEND=local, as in pipeline.slurm.sh).
#
#   cd <checkout>    # see "Agents on DAIS" in infra/dais/README.md
#   sbatch --export=ALL,PF_SWEEP_EPISODES=20,PF_FINAL_EPISODES=20 \
#     infra/dais/campaign.slurm.sh campaign <campaign.yml>
#   sbatch --export=ALL,PF_SWEEP_EPISODES=default,PF_FINAL_EPISODES=default \
#     infra/dais/campaign.slurm.sh resume <instance>
#
# The campaign file names harness opencode, the served model and
# base_url "{env:PF_LLM_BASE_URL}" (scripts/policy_finder/campaigns/dais-*.yml),
# so a resubmitted job, whose server has a new port and key, resumes from the
# state files in PF_RUNS without a new export.
#
# The job's cores, in this order (cpu_slots.py, whole cores with their
# hyperthreads): SERVER_CORES for vLLM, DRIVER_CORES for the pipeline driver,
# AGENT_CORES for the agents (groups of PF_AGENT_CPUS hyperthreads, one per
# agent, PF_AGENT_SLOTS at most at once), the rest the sim pool (one slot per
# hyperthread). Server and pool on separate cores do not slow each other
# (docs/reports/dais_feasibility.md, B3).
#
# It exits non-zero unless the campaign (or instance) reached the expected
# state (campaign_check.py: every version scored; PF_EXPECT=finished accepts
# failed versions). At the end, at the time limit (Slurm's TERM, --signal) and
# on scancel it stops the server, the driver and every agent and sim of the job.
#
#SBATCH -J pf_campaign
#SBATCH -o /dais/fs/scratch/%u/policy-finder-agents/logs/%x.%j.out
#SBATCH --account=mpib_inst
#SBATCH --partition=gpu1
#SBATCH --qos=g0002
#SBATCH --gres=gpu:2
#SBATCH --constraint=hopper|blackwell
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=480000
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@240

set -euo pipefail
MODE="${1:?campaign or resume}"
shift
for var in PF_SWEEP_EPISODES PF_FINAL_EPISODES; do
  value="${!var:-}"
  if [ "$value" = default ]; then
    unset "$var"
  elif [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: set $var to a number or 'default' (sbatch --export=ALL,$var=...)," \
      "not '$value'" >&2
    exit 2
  fi
done
echo "episodes: sweep ${PF_SWEEP_EPISODES:-default}, final ${PF_FINAL_EPISODES:-default}" \
  "(a version that already started keeps its own)"
export PF_SCRATCH="${PF_SCRATCH:-/dais/fs/scratch/$USER/policy-finder-agents}"
export PF_REPO="${PF_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PF_REPO/infra/dais/common.sh"
init_run
cd "$PF_REPO"

export PF_RUNS="${PF_RUNS:-$PF_SCRATCH/pf_runs}"
export PF_CLUSTER_BACKEND=local
export PF_SIM_SIF="$SIM_SIF"
export PF_ARTIFACTS="${PF_ARTIFACTS:-$PF_SCRATCH/artifacts}"
export PF_PYTHON="${PF_PYTHON:-$PF_SCRATCH/venv/bin/python}"
export PF_BASE="${PF_BASE:-$(git rev-parse --abbrev-ref HEAD)}"
export PF_POOL_LOCKS="$PF_RUN/tmp/pool-locks"
export PF_AGENT_BACKEND=apptainer
export PF_AGENT_SIF="${PF_AGENT_SIF:-$AGENT_SIF}"
export PF_AGENT_LOCKS="$PF_RUN/tmp/agent-locks"
export PF_AGENT_CPUS="${PF_AGENT_CPUS:-2}"
export PF_AGENT_SLOTS="${PF_AGENT_SLOTS:-4}"
export PF_AGENT_MEMORY_MB="${PF_AGENT_MEMORY_MB:-16384}"
export PF_AGENT_TIMEOUT="${PF_AGENT_TIMEOUT:-5400}"
# the network rule: the agent's shell commands under noinet.py (no internet
# sockets); PF_AGENT_SHELL=none runs them in plain bash (the audit stays)
if [ "${PF_AGENT_SHELL:-}" = none ]; then
  unset PF_AGENT_SHELL PF_AGENT_NOINET
else
  export PF_AGENT_SHELL="${PF_AGENT_SHELL:-$PF_HERE/agent_shell.sh}"
  export PF_AGENT_NOINET="${PF_AGENT_NOINET:-$PF_HERE/noinet.py}"
fi
mkdir -p "$PF_RUNS"

echo "job ${SLURM_JOB_ID:-local} on $(hostname), commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
  | sed 's/^/GPU /'
export SIM_SIF_SHA256 AGENT_SIF_SHA256 MODEL MODEL_REV
"$PF_PYTHON" scripts/policy_finder/preflight.py --dais || exit 1

# the cores: server, driver, agents, then the pool
SERVER_CORES="${SERVER_CORES:-4}"
DRIVER_CORES="${DRIVER_CORES:-2}"
AGENT_CORES="${AGENT_CORES:-6}"
core_range() {  # core_range <first core> <n cores>: their CPUs, one list
  python3 "$PF_HERE/cpu_slots.py" slots "$1" 1 "$2" | paste -sd, -
}
SERVER_CPUS=$(core_range 0 "$SERVER_CORES")
DRIVER_CPUS=$(core_range "$SERVER_CORES" "$DRIVER_CORES")
first=$(( SERVER_CORES + DRIVER_CORES ))
export PF_AGENT_POOL_CPUS=$(core_range "$first" "$AGENT_CORES")
first=$(( first + AGENT_CORES ))
PF_POOL_CPUS=$( { python3 "$PF_HERE/cpu_slots.py" slots "$first" 1 | cut -d, -f1
                  python3 "$PF_HERE/cpu_slots.py" slots "$first" 1 | cut -s -d, -f2
                } | paste -sd, -)
export PF_POOL_CPUS
for v in SERVER_CPUS DRIVER_CPUS PF_AGENT_POOL_CPUS PF_POOL_CPUS; do
  [ -n "${!v}" ] || { echo "FATAL: no CPUs left for $v" >&2; exit 1; }
  echo "$v ${!v} ($(tr ',' '\n' <<< "${!v}" | wc -l) CPUs)"
done

# stop everything of this job at the end, on scancel and at the time limit:
# the server and the driver, then any agent or sim worker left in the job's
# cgroup (they run in sessions of their own)
stop_all() {
  trap - EXIT TERM INT
  echo "stopping at $(date +%T) after $SECONDS s"
  if [ -n "${SERVER_URL:-}" ]; then
    curl -s -m 10 -H "@$SERVER_AUTH_HEADER" "${SERVER_URL%/v1}/metrics" 2>/dev/null \
      | grep -E '^vllm:(prompt_tokens_total|generation_tokens_total|prefix_cache_(hits|queries)_total|request_success_total)' \
      > "$PF_RUN/vllm_metrics.txt" || true
    sed 's/^/metrics /' "$PF_RUN/vllm_metrics.txt" || true
  fi
  [ -n "${DRIVER_PID:-}" ] && kill -TERM "$DRIVER_PID" 2>/dev/null || true
  cleanup
  local cg procs
  cg=$(sed -n 's/^0:://p' /proc/self/cgroup)
  if [ -n "$cg" ] && [ -r "/sys/fs/cgroup$cg/cgroup.procs" ]; then
    procs=$(grep -vx -e "$$" -e "$BASHPID" "/sys/fs/cgroup$cg/cgroup.procs" || true)
    [ -n "$procs" ] && kill -TERM $procs 2>/dev/null || true
    sleep 10
    procs=$(grep -vx -e "$$" -e "$BASHPID" "/sys/fs/cgroup$cg/cgroup.procs" || true)
    [ -n "$procs" ] && kill -KILL $procs 2>/dev/null || true
  fi
}
trap 'stop_all; exit 143' TERM INT
trap stop_all EXIT

assert_vllm_image
NGPU=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l)
export SERVER_CPUS
# shellcheck disable=SC2086  # VLLM_EXTRA: extra vllm arguments, split
serve_model "${GPU_UTIL:-0.90}" "$NGPU" "$PF_RUN/vllm.log" ${VLLM_EXTRA:-}
wait_ready 1800 || { tail -50 "$PF_RUN/vllm.log"; exit 1; }
export PF_LLM_BASE_URL="$SERVER_URL" PF_LLM_API_KEY_FILE="$SERVER_KEY_FILE"

case "$MODE" in
  campaign) cmd=("$PF_PYTHON" scripts/policy_finder/campaign.py run "$@") ;;
  resume) cmd=("$PF_PYTHON" scripts/policy_finder/pipeline.py run "$@") ;;
  *) echo "FATAL: mode is campaign or resume, not $MODE" >&2; exit 2 ;;
esac
rc=0
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 taskset -c "$DRIVER_CPUS" "${cmd[@]}" &
DRIVER_PID=$!
wait "$DRIVER_PID" || rc=$?
echo "driver exit $rc after $SECONDS s"
"$PF_PYTHON" "$PF_HERE/campaign_check.py" "$MODE" "$@" --expect "${PF_EXPECT:-scored}" || rc=1
exit "$rc"
