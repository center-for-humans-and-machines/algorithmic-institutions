#!/bin/bash -l
# A policy-finder campaign with its agents, entirely on DAIS, in ONE job:
# vLLM serves the model on the job's GPUs, the agents run OpenCode against it
# as Apptainer processes (PF_AGENT_BACKEND=apptainer,
# scripts/policy_finder/agent_apptainer.py) and the sims run in the local
# CPU pool (PF_CLUSTER_BACKEND=local, as in pipeline.slurm.sh).
#
#   cd <checkout>    # see "Agents on DAIS" in infra/dais/README.md
#   sbatch --export=PF_SWEEP_EPISODES=20,PF_FINAL_EPISODES=20 \
#     infra/dais/campaign.slurm.sh campaign <campaign.yml>
#   sbatch --export=PF_SWEEP_EPISODES=default,PF_FINAL_EPISODES=default \
#     infra/dais/campaign.slurm.sh resume <instance>
#   # a full node: several stages behind one server (see "Stages" below)
#   sbatch -p gpu --qos=g0008 --gres=gpu:8 --exclusive --mem=0 -c 96 --time=12:00:00 \
#     --export=PF_RUNS=<dir>,SERVER_GPUS=2,SERVER_CORES=8,DRIVER_CORES=2,AGENT_CORES=24,PF_AGENT_SLOTS=24 \
#     infra/dais/campaign.slurm.sh stages infra/dais/stages/<file>
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
# Stages: a file of lines, run in order behind one server, each only if the
# one before succeeded (# starts a comment):
#   run <expect> <sweep episodes> <final episodes> <campaign.yml>[:<slices>] ...
#       the campaigns at once; each checked with campaign_check.py --expect
#       <expect> (scored or finished) and, with :<slices>, re-evaluated by
#       reevaluate.py into $PF_RUNS/reevaluate/<campaign> (seed 100000)
#   killtest <sweep episodes> <final episodes> <campaign.yml>
#       the resume test: starts the campaign, and once an agent has run for
#       KILL_AFTER seconds (default 90) kills every process of the job but the
#       server, as the end of a job would (kill_job_children.py); a `run` line
#       of the same campaign then resumes it
# Episode sizes are a number or `default`, as PF_SWEEP_EPISODES and
# PF_FINAL_EPISODES. SERVER_GPUS (default: all) are the server's (tensor
# parallel over them); the job's other GPUs stay idle.
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
MODE="${1:?campaign, resume or stages}"
shift
check_episodes() {  # PF_SWEEP_EPISODES, PF_FINAL_EPISODES: a number or default
  local var value
  for var in PF_SWEEP_EPISODES PF_FINAL_EPISODES; do
    value="${!var:-}"
    if [ "$value" = default ]; then
      unset "$var"
    elif [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
      echo "FATAL: set $var to a number or 'default' (sbatch --export=$var=...)," \
        "not '$value'" >&2
      return 2
    fi
  done
  echo "episodes: sweep ${PF_SWEEP_EPISODES:-default}, final ${PF_FINAL_EPISODES:-default}" \
    "(a version that already started keeps its own)"
}
if [ "$MODE" = stages ]; then
  STAGES="${1:?stages file}"
  [ -f "$STAGES" ] || { echo "FATAL: no stages file $STAGES" >&2; exit 2; }
else
  check_episodes || exit 2
fi
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
# the python the agents are told to use (their instance file's `python`):
# the agent image's, as for the reference agents, not the driver's host path
export PF_AGENT_PYTHON="${PF_AGENT_PYTHON:-python3}"
export PF_BASE="${PF_BASE:-$(git rev-parse --abbrev-ref HEAD)}"

export PF_AGENT_BACKEND=apptainer
export PF_AGENT_SIF="${PF_AGENT_SIF:-$AGENT_SIF}"
# slot locks node-local, one folder per job: all pipelines and agents of the
# campaign run in this job (flock on GPFS does not exclude reliably)
SHM_LOCKS="/dev/shm/pf-${SLURM_JOB_ID:-$$}"
export PF_LOCKS="$SHM_LOCKS/locks" PF_AGENT_LOCKS="$SHM_LOCKS/agent-locks"
export PF_POOL_LOCKS="$SHM_LOCKS/pool"
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
PF_AGENT_POOL_CPUS=$(core_range "$first" "$AGENT_CORES")
export PF_AGENT_POOL_CPUS
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
  # the drivers, agents and sims first, the server last: an agent that sees
  # the server go while it still runs would read as the endpoint's failure
  # shellcheck disable=SC2086  # DRIVER_PID: one or more pids
  [ -n "${DRIVER_PID:-}" ] && kill -TERM $DRIVER_PID 2>/dev/null || true
  # every other process of the job's cgroup (not this shell's ancestors), but
  # the server, then the server and the rest
  python3 "$PF_HERE/kill_job_children.py" --term ${SERVER_PID:-} \
    > "$PF_RUN/stop_term.log" 2>&1 || true
  if [ -s "$PF_RUN/stop_term.log" ]; then
    sleep 10
    python3 "$PF_HERE/kill_job_children.py" ${SERVER_PID:-} \
      > "$PF_RUN/stop_kill.log" 2>&1 || true
  fi
  # the server last, with time to shut down cleanly (TERM, up to 30 s)
  cleanup
  local _
  if [ -n "${SERVER_PID:-}" ]; then  # no server started: no wait
    for _ in $(seq 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
  fi
  python3 "$PF_HERE/kill_job_children.py" >> "$PF_RUN/stop_kill.log" 2>&1 || true
  echo "stopped $(grep -c '^kill' "$PF_RUN/stop_term.log" 2>/dev/null || true) processes," \
    "then the server and watch ($(grep -c '^kill' "$PF_RUN/stop_kill.log" 2>/dev/null || true) killed hard)"
  rm -rf "$SHM_LOCKS"
}
trap 'stop_all; exit 143' TERM INT
trap stop_all EXIT

assert_vllm_image
SERVER_GPUS="${SERVER_GPUS:-$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l)}"
SERVER_DEVICES=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | head -n "$SERVER_GPUS" | paste -sd, -)
NGPU=$(tr ',' '\n' <<< "$SERVER_DEVICES" | wc -l)
echo "server on GPUs $SERVER_DEVICES (tensor parallel $NGPU)"
export SERVER_CPUS
# --moe-backend triton: what vLLM picks on H200 by itself; on B200 it would
# pick FlashInfer's TRT-LLM kernels, whose cubins are not in the image and are
# downloaded into it at start (fails: the image is read-only; job 988114)
# shellcheck disable=SC2086  # VLLM_EXTRA: extra vllm arguments, split
CUDA_VISIBLE_DEVICES="$SERVER_DEVICES" \
  serve_model "${GPU_UTIL:-0.90}" "$NGPU" "$PF_RUN/vllm.log" --moe-backend triton ${VLLM_EXTRA:-}
wait_ready 1800 || { tail -50 "$PF_RUN/vllm.log"; exit 1; }
export PF_LLM_BASE_URL="$SERVER_URL" PF_LLM_API_KEY_FILE="$SERVER_KEY_FILE"
# the agents' OpenCode config takes the served context from here
# (export_instance.local_model), so it compacts before the server refuses
export PF_LLM_CONTEXT="$MAX_MODEL_LEN"
echo "served context $MAX_MODEL_LEN tokens"

# server_report: vLLM's request counters (errors included), /health's
# latency and its log's tail. A report: it never fails (a subshell without
# errexit or pipefail, whatever grep finds).
server_report() {
  (
    set +e +o pipefail
    echo "-- model server at $(date +%T): $(kill -0 "$SERVER_PID" 2>/dev/null && echo alive || echo DEAD)"
    curl -s -m 20 -o /dev/null -w '   /health %{http_code} in %{time_total} s\n' \
      -H "@$SERVER_AUTH_HEADER" "${SERVER_URL%/v1}/health"
    curl -s -m 20 -H "@$SERVER_AUTH_HEADER" "${SERVER_URL%/v1}/metrics" 2>/dev/null \
      | grep -E '^vllm:(request_success_total|num_requests_running|num_requests_waiting|generation_tokens_total|prompt_tokens_total)' \
      | sed 's/model_name="[^"]*"//; s/^/   metrics /'
    echo "   vllm.log lines with error/exception: $(grep -ciE 'error|exception|traceback' "$PF_RUN/vllm.log")"
    tail -3 "$PF_RUN/vllm.log" | cut -c1-300 | sed 's/^/   vllm.log | /'
  ) || true
}

# the server watch: every 30 s the process must be alive and /health answer
# 200; after 4 failures in a row (2 minutes) it marks the server down and
# kills every process of the job (kill_job_children.py), so that the running
# versions stop as lost, resumable, rather than burning their agents' attempts
# on a dead endpoint; the stage then fails and the job exits non-zero.
SERVER_DOWN="$PF_RUN/server_down"
WATCH_INTERVAL="${WATCH_INTERVAL:-30}"
WATCH_FAILS="${WATCH_FAILS:-4}"
watch_server() {
  local bad=0
  while sleep "$WATCH_INTERVAL"; do
    if kill -0 "$SERVER_PID" 2>/dev/null && [ "$(curl -s -m 20 -o /dev/null \
        -w '%{http_code}' -H "@$SERVER_AUTH_HEADER" "${SERVER_URL%/v1}/health")" = 200 ]; then
      bad=0
    else
      bad=$(( bad + 1 ))
      echo "[$(date +%T)] model server check failed ($bad/$WATCH_FAILS)"
      if (( bad >= WATCH_FAILS )); then
        echo "[$(date +%T)] FATAL: the model server is down: stopping every process" \
          | tee "$SERVER_DOWN"
        # first the kill (the versions stop as lost), then the report
        python3 "$PF_HERE/kill_job_children.py" || true
        server_report
        return 0
      fi
    fi
  done
}
watch_server &
WATCH_PID=$!
BG_PIDS+=("$WATCH_PID")

driver() {  # driver <command...>: on the driver's CPUs
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 taskset -c "$DRIVER_CPUS" "$@"
}

# campaign_stage <expect> <campaign.yml>[:<slices>]: run, check, re-evaluate.
# A resubmitted job can use the same stages file: campaign.py resumes an
# unfinished campaign from its state files and returns at once for a finished
# one, a finished re-evaluation is not rerun, and a kill test that was done
# (its marker beside the campaign's state) is skipped. But a stage is judged
# again on rerun: when a check became stricter, use a stages file without the
# stages that passed (stages/node-resume-real.txt).
campaign_stage() {
  local expect="$1" file="${2%%:*}" slices="" name rc=0 t0=$SECONDS
  [[ "$2" == *:* ]] && slices="${2##*:}"
  name=$(basename "$file" .yml)
  echo "[$(date +%T)] campaign $name: start"
  driver "$PF_PYTHON" scripts/policy_finder/campaign.py run "$file" \
    >> "$PF_RUN/$name.campaign.log" 2>&1 || rc=$?
  echo "[$(date +%T)] campaign $name: driver exit $rc after $(( SECONDS - t0 )) s"
  [ -e "$SERVER_DOWN" ] && { echo "campaign $name: stopped, the model server is down"; return 3; }
  # <expect>[/<n>]: with /n, at least n first-generation versions scored and
  # no expired agent run (the load test)
  local check=(--expect "${expect%%/*}")
  [[ "$expect" == */* ]] && check+=(--min-first-scored "${expect##*/}" --no-expired)
  "$PF_PYTHON" "$PF_HERE/campaign_check.py" campaign "$file" "${check[@]}" || rc=1
  local out="$PF_RUNS/reevaluate/$name"
  if [ "$rc" = 0 ] && [ -n "$slices" ] && [ -f "$out/summary.json" ]; then
    echo "[$(date +%T)] campaign $name: re-evaluation already done ($out)"
  elif [ "$rc" = 0 ] && [ -n "$slices" ]; then
    # an earlier, partial re-evaluation is moved aside; this one starts anew
    # at the expected path
    if [ -e "$out" ]; then
      mv "$out" "$out.partial.$(date +%Y%m%d-%H%M%S)"
    fi
    echo "[$(date +%T)] campaign $name: re-evaluation in $slices slices into $out"
    driver "$PF_PYTHON" "$PF_HERE/reevaluate.py" "$PF_RUNS/campaign/$name.json" \
      "$out" --slices "$slices" --seed 100000 \
      >> "$PF_RUN/$name.reevaluate.log" 2>&1 || rc=$?
    echo "[$(date +%T)] campaign $name: re-evaluation exit $rc"
    cat "$out/summary.json" 2>/dev/null || true
  fi
  return "$rc"
}

# killtest_stage <campaign.yml>: start it, kill the job's processes (not the
# server) once an agent has run KILL_AFTER seconds
killtest_stage() {
  local file="$1" t0=$SECONDS waited=0
  echo "[$(date +%T)] killtest $(basename "$file" .yml): start"
  local done_marker
  done_marker="$PF_RUNS/campaign/$(basename "$file" .yml).killtest-done"
  if [ -e "$done_marker" ]; then
    # a resubmitted job: the kill was done before; the run stage resumes it
    echo "[$(date +%T)] killtest: done in an earlier job ($(cat "$done_marker")), skipped"
    return 0
  fi
  driver "$PF_PYTHON" scripts/policy_finder/campaign.py run "$file" < /dev/null \
    >> "$PF_RUN/$(basename "$file" .yml).campaign.log" 2>&1 &
  local campaign_pid=$!
  local since
  since=$(date +%s)
  # a run of this stage (started after it began) running for KILL_AFTER s; a
  # lost run of an earlier job keeps `running` in its status for ever
  until "$PF_PYTHON" - "$PF_RUNS/agent_runs" "${KILL_AFTER:-90}" "$since" <<'EOF'
import json, sys, time
from pathlib import Path
for s in Path(sys.argv[1]).glob("apt-*/status.json"):
    try:
        st = json.loads(s.read_text())
        started = time.mktime(time.strptime(st["started_at"], "%Y-%m-%dT%H:%M:%S"))
    except (OSError, ValueError, KeyError):
        continue
    if (st.get("status") == "running" and started >= float(sys.argv[3])
            and time.time() - started > float(sys.argv[2])):
        print(f"killing while {s.parent.name} runs", flush=True)
        sys.exit(0)
sys.exit(1)
EOF
  do
    sleep 10; waited=$(( SECONDS - t0 ))
    if [ -e "$SERVER_DOWN" ]; then echo "killtest: the model server is down" >&2; return 3; fi
    if ! kill -0 "$campaign_pid" 2>/dev/null; then
      echo "FATAL: killtest: the campaign ended before an agent had run ${KILL_AFTER:-90} s" >&2
      return 1
    fi
    if (( waited > 3600 )); then echo "FATAL: no agent ran in an hour" >&2; return 1; fi
  done
  echo "[$(date +%T)] killtest: killing every process but the server after $(( SECONDS - t0 )) s"
  python3 "$PF_HERE/kill_job_children.py" "$SERVER_PID" "$WATCH_PID"
  echo "job ${SLURM_JOB_ID:-local} $(date +%Y-%m-%dT%H:%M:%S)" > "$done_marker"
  sleep 5
  rm -rf "$PF_LOCKS" "$PF_AGENT_LOCKS" "$PF_POOL_LOCKS"  # as a new job would find them
}

rc=0
case "$MODE" in
  campaign)
    driver "$PF_PYTHON" scripts/policy_finder/campaign.py run "$@" &
    DRIVER_PID=$!
    wait "$DRIVER_PID" || rc=$?
    echo "driver exit $rc after $SECONDS s"
    "$PF_PYTHON" "$PF_HERE/campaign_check.py" campaign "$@" --expect "${PF_EXPECT:-scored}" || rc=1 ;;
  resume)
    driver "$PF_PYTHON" scripts/policy_finder/pipeline.py run "$@" &
    DRIVER_PID=$!
    wait "$DRIVER_PID" || rc=$?
    echo "driver exit $rc after $SECONDS s"
    "$PF_PYTHON" "$PF_HERE/campaign_check.py" resume "$@" --expect "${PF_EXPECT:-scored}" || rc=1 ;;
  stages)
    while read -r -u 3 kind rest || [ -n "${kind:-}" ]; do
      [[ -z "$kind" || "$kind" == \#* ]] && continue
      read -r -a args <<< "$rest"
      echo "== stage: $kind ${args[*]}"
      case "$kind" in
        run)
          export PF_SWEEP_EPISODES="${args[1]}" PF_FINAL_EPISODES="${args[2]}"
          check_episodes || { rc=2; break; }
          pids=()
          for c in "${args[@]:3}"; do
            campaign_stage "${args[0]}" "$c" < /dev/null & pids+=($!)
          done
          DRIVER_PID="${pids[*]}"
          for pid in "${pids[@]}"; do wait "$pid" || rc=1; done ;;
        killtest)
          export PF_SWEEP_EPISODES="${args[0]}" PF_FINAL_EPISODES="${args[1]}"
          check_episodes || { rc=2; break; }
          killtest_stage "${args[2]}" || rc=1 ;;
        *) echo "FATAL: unknown stage $kind" >&2; rc=2 ;;
      esac
      server_report
      if [ -e "$SERVER_DOWN" ]; then
        echo "== stage $kind stopped: the model server went down"; rc=3; break
      fi
      [ "$rc" = 0 ] || { echo "== stage $kind failed: stopping"; break; }
    done 3< "$STAGES" ;;
  *) echo "FATAL: mode is campaign, resume or stages, not $MODE" >&2; rc=2 ;;
esac
echo "exit $rc after $SECONDS s"
exit "$rc"
