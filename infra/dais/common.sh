# Shared settings and helpers for the DAIS job scripts in infra/dais/.
# Sourced, not run. See docs/reports/dais_feasibility.md for the measurements
# these defaults come from.
#
# Every job writes into its own run folder, $PF_RUN (init_run), never into the
# checkout: the checkout holds committed results (sweep.json files from Raven)
# that a failed sim must not be mistaken for.
#
# Prerequisites (see docs/reports/dais_feasibility.md, "Prerequisites"):
# the logs folder (Slurm opens the -o file before the script runs), the sim
# image, the weights, and $PF_SCRATCH/env.sh, not in git, which sets VLLM_SIF
# to the vLLM image's path on this cluster.

PF_SCRATCH="${PF_SCRATCH:-/dais/fs/scratch/$USER/policy-finder}"
PF_REPO="${PF_REPO:-/u/$USER/algorithmic-institutions-pf}"
PF_LOGS="$PF_SCRATCH/logs"
PF_HERE="${SLURM_SUBMIT_DIR:-$PF_REPO}/infra/dais"

# Site settings kept out of git (VLLM_SIF: the vLLM image's path).
[ -f "$PF_SCRATCH/env.sh" ] && source "$PF_SCRATCH/env.sh"
# vLLM image: its version is asserted, not assumed.
VLLM_SIF="${VLLM_SIF:-}"
EXPECT_VLLM_TAG="vllm/vllm-openai:v0.25.1"

# Sim image: a byte copy of Raven's shared algoinst.sif (torch 1.11.0+cu113).
# It has no kernels for H200/B200, so sims run on the CPUs.
SIM_SIF="${SIM_SIF:-$PF_SCRATCH/images/algoinst.sif}"
SIM_SIF_SHA256="fe358c025561e9fac4831abd1f03afcdf2931ed7bb683d6a86f19a39591550ff"

# Weights: Hugging Face cache layout under HF_HOME, pinned revision.
export HF_HOME="${HF_HOME:-/dais/fs/scratch/$USER/models}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-Next}"
MODEL_REV="${MODEL_REV:-a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

module load apptainer/1.5.2

# init_run: the job's own folder under scratch ($PF_RUN), and a cleanup trap
# that stops every background process registered in BG_PIDS.
init_run() {
  PF_RUN="$PF_SCRATCH/runs/${SLURM_JOB_NAME:-local}.${SLURM_JOB_ID:-$$}"
  mkdir -p "$PF_RUN/tmp" "$PF_LOGS"
  BG_PIDS=()
  trap cleanup EXIT
  echo "run folder: $PF_RUN"
}

cleanup() {
  local pid
  for pid in "${BG_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}

assert_vllm_image() {
  local tag
  if [ -z "$VLLM_SIF" ]; then
    echo "FATAL: VLLM_SIF is not set (put it in $PF_SCRATCH/env.sh)" >&2
    return 1
  fi
  tag=$(apptainer inspect "$VLLM_SIF" | sed -n 's/^ai\.vllm\.image\.tag: *//p' | head -1)
  if [ "$tag" != "$EXPECT_VLLM_TAG" ]; then
    echo "FATAL: $VLLM_SIF is '$tag', expected '$EXPECT_VLLM_TAG'" >&2
    return 1
  fi
  echo "vLLM image: $tag"
}

# free_port: a TCP port nothing on this node listens on right now.
free_port() {
  python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

# serve_model and wait_ready follow serve() and wait_ready() of the sister
# project ../understanding (scripts/slurm/lib/evo_common.sh, used by
# evo_dais.slurm.sh): the same image, `apptainer exec --nv`, a home that is not
# the real one (they bind the cwd as $HOME), HF_HOME on scratch, explicit
# --enable-prefix-caching, and readiness gated on /v1/models listing the model.
# Added here: offline mode, a pinned revision, tool calling, a port and an API
# key of the job's own (the partition is shared with other users' jobs).
#
# serve_model <gpu-memory-utilization> <tp> <log> [extra vllm args...]
# Starts vLLM in the background on the GPUs in CUDA_VISIBLE_DEVICES, pinned to
# SERVER_CPUS if set. Sets SERVER_PID, SERVER_PORT, SERVER_URL,
# SERVER_KEY_FILE and SERVER_AUTH_HEADER (both mode 600; the key is never
# printed and never on a command line: vLLM reads it from its environment,
# curl from the header file).
#
# The container: --containall, the weights read-only, a home of the job's own
# ($PF_RUN/home) and one cache folder shared by all jobs ($PF_SCRATCH/cache:
# torch.compile, Triton and FlashInfer caches are keyed by content).
serve_model() {
  local util="$1" tp="$2" log="$3"
  shift 3
  SERVER_PORT=$(free_port)
  SERVER_URL="http://127.0.0.1:$SERVER_PORT/v1"
  SERVER_KEY_FILE="$PF_RUN/api_key"
  SERVER_AUTH_HEADER="$PF_RUN/auth_header"
  (umask 077
   python3 -c 'import secrets; print(secrets.token_hex(24))' > "$SERVER_KEY_FILE"
   echo "Authorization: Bearer $(cat "$SERVER_KEY_FILE")" > "$SERVER_AUTH_HEADER")
  mkdir -p "$PF_RUN/home" "$PF_SCRATCH/cache"
  local pin=()
  [ -n "${SERVER_CPUS:-}" ] && pin=(taskset -c "$SERVER_CPUS")
  APPTAINERENV_VLLM_API_KEY="$(cat "$SERVER_KEY_FILE")" \
    "${pin[@]}" apptainer exec --nv --containall --cleanenv \
    --home "$PF_RUN/home" \
    --bind "$HF_HOME:$HF_HOME:ro" \
    --bind "$PF_SCRATCH/cache" \
    --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 \
    --env XDG_CACHE_HOME="$PF_SCRATCH/cache" \
    --env VLLM_CACHE_ROOT="$PF_SCRATCH/cache/vllm" \
    --env TRITON_CACHE_DIR="$PF_SCRATCH/cache/triton" \
    --env FLASHINFER_WORKSPACE_BASE="$PF_SCRATCH/cache" \
    --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$VLLM_SIF" \
    vllm serve "$MODEL" --revision "$MODEL_REV" \
    --host 127.0.0.1 --port "$SERVER_PORT" \
    --tensor-parallel-size "$tp" \
    --gpu-memory-utilization "$util" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-prefix-caching --enable-prompt-tokens-details \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
    "$@" > "$log" 2>&1 &
  SERVER_PID=$!
  BG_PIDS+=("$SERVER_PID")
}

# wait_ready <timeout s>: until OUR server answers: /v1/models lists $MODEL
# with our key and refuses a request without it (401), so another job's
# server on the same port cannot pass. Prints the load time.
wait_ready() {
  local timeout="$1" t0=$SECONDS code
  while (( SECONDS - t0 < timeout )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "FATAL: vLLM exited during start-up" >&2
      return 1
    fi
    if curl -sf -m 10 -H "@$SERVER_AUTH_HEADER" "$SERVER_URL/models" 2>/dev/null \
        | grep -qF "\"$MODEL\""; then
      code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$SERVER_URL/models")
      if [ "$code" != 401 ]; then
        echo "FATAL: the server on $SERVER_PORT answers without a key ($code)" >&2
        return 1
      fi
      echo "ready after $(( SECONDS - t0 )) s on port $SERVER_PORT"
      return 0
    fi
    sleep 5
  done
  echo "FATAL: not ready after $timeout s" >&2
  return 1
}

# in_vllm_image <command...>: a client (probe, benchmark) in the vLLM image,
# with the run folder and this folder's scripts visible.
in_vllm_image() {
  apptainer exec --containall --cleanenv \
    --bind "$PF_RUN" --bind "$PF_HERE:/opt/dais:ro" \
    "$VLLM_SIF" "$@"
}

# in_sim_image <command...>: run a command in the sim image, set up as
# Raven's job template does (slurm_tools run_slurm_raven_apptainer_algoinst.sh:
# repo at /opt/algorithmic-institutions, PATH and PYTHONPATH as there), with
# no GPU (SIM_GPU=<index> exposes one, for sim_check's GPU test only). The
# checkout is read-only (REPO_MODE=rw to write configs into it); only the run
# folder is writable; this folder's scripts are at /opt/dais. SIM_THREADS
# (default 1, the cheapest shape per CPU in B1) sets the OMP/MKL threads, which
# torch 1.11 honours (the sim itself does not set them).
in_sim_image() {
  sim_argv
  "${SIM_ARGV[@]}" "$@"
}

# sim_argv: the apptainer command line of in_sim_image, in SIM_ARGV.
sim_argv() {
  SIM_ARGV=(apptainer exec --containall --cleanenv --writable-tmpfs
    --no-mount cwd,tmp,bind-paths
    --bind "$PF_RUN/tmp:/tmp"
    --bind "$PF_REPO:/opt/algorithmic-institutions:${REPO_MODE:-ro}"
    --bind "$PF_RUN"
    --bind "$PF_HERE:/opt/dais:ro"
    --env PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin
    --env PYTHONPATH=/opt/algorithmic-institutions/src
    --env PYTHONDONTWRITEBYTECODE=1)
  if [ -n "${SIM_GPU:-}" ]; then  # only for sim_check's GPU test
    SIM_ARGV+=(--nv --env "CUDA_VISIBLE_DEVICES=$SIM_GPU")
  else
    SIM_ARGV+=(--env CUDA_VISIBLE_DEVICES=)
  fi
  SIM_ARGV+=(--env "OMP_NUM_THREADS=${SIM_THREADS:-1}" --env "MKL_NUM_THREADS=${SIM_THREADS:-1}")
  SIM_ARGV+=(--pwd /opt/algorithmic-institutions "$SIM_SIF")
}

# simulate <config> <label>: `python -m aimanager simulate <config>` on the
# CPUs, pinned to SIM_CPUS if set; prints one SIM line (exit code, wall, user
# and system CPU seconds, peak RSS) and returns the sim's exit code.
simulate() {
  local cfg="$1" label="$2" timing rc=0
  local pin=() argv
  [ -n "${SIM_CPUS:-}" ] && pin=(taskset -c "$SIM_CPUS")
  sim_argv
  argv=("${SIM_ARGV[@]}")
  timing=$(mktemp -p "$PF_RUN/tmp")
  /usr/bin/time -o "$timing" -f "wall=%e user=%U sys=%S maxrss_kb=%M" \
    "${pin[@]}" "${argv[@]}" python -m aimanager simulate "$cfg" \
    > "$PF_RUN/$label.log" 2>&1 || rc=$?
  echo "SIM $label rc=$rc threads=${SIM_THREADS:-1} cpus=${SIM_CPUS:-all} $(tail -1 "$timing")"
  rm -f "$timing"
  return "$rc"
}

# make_sweep_parts <n parts>: a real sweep, split by the repo's own
# mechanism. Copies the checkout's code, configs and model files to
# $PF_RUN/tree, adds the rule SWEEP_RULE (default: pf-20261104-1's, a real
# 5-parameter rule from the Opus reference run) and runs
# `generate_sim_config.py --n-parts N` there (256 Sobol points x 500 episodes,
# the defaults). Sets PF_REPO to the copy, for the sims that follow.
make_sweep_parts() {
  local n="$1" rule="${SWEEP_RULE:-$PF_SCRATCH/ref-pf-20261104-1/configs/managers/rule_based/pf-20261104-1.yml}"
  local tree="$PF_RUN/tree"
  mkdir -p "$tree" "$PF_RUN/cfg" "$PF_RUN/out"
  cp -r "$PF_REPO"/{src,scripts,configs,artifacts} "$tree/"
  cp "$rule" "$tree/configs/managers/rule_based/bench_rule.yml"
  PF_REPO="$tree"
  REPO_MODE=rw in_sim_image python scripts/policy_finder/generate_sim_config.py \
    --config configs/managers/rule_based/bench_rule.yml --max-params none --n-parts "$n"
}

# part_config <k> <n parts> <label>: part k's config with its output in
# $PF_RUN/out/<label>; prints the config's path.
part_config() {
  local k="$1" n="$2" label="$3" src
  src="configs/simulation/policy_finder/bench_rule_sweep_p${k}of${n}.yml"
  in_sim_image python /opt/dais/sim_variant.py config "$src" \
    "$PF_RUN/cfg/$label.yml" --output-dir "$PF_RUN/out/$label" >&2
  echo "$PF_RUN/cfg/$label.yml"
}
