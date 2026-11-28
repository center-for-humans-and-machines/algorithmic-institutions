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
# MODEL_PROFILE=<name>: a model other than the default Qwen3-Coder-Next, its
# settings (weights, revision, vLLM image, parsers, context, MoE backend,
# server GPUs) in infra/dais/model_profiles/<name>.sh; the defaults below fill the rest.
if [ -n "${MODEL_PROFILE:-}" ]; then
  _profile="${PF_HERE:-$(dirname "${BASH_SOURCE[0]}")}/model_profiles/$MODEL_PROFILE.sh"
  [ -f "$_profile" ] || { echo "FATAL: no model profile $_profile" >&2; exit 2; }
  # shellcheck source=/dev/null
  source "$_profile"
fi
# vLLM image: its version is asserted, not assumed.
VLLM_SIF="${VLLM_SIF:-}"
EXPECT_VLLM_TAG="${EXPECT_VLLM_TAG:-vllm/vllm-openai:v0.25.1}"

# Sim image: a byte copy of Raven's shared algoinst.sif (torch 1.11.0+cu113).
# It has no kernels for H200/B200, so sims run on the CPUs.
SIM_SIF="${SIM_SIF:-$PF_SCRATCH/images/algoinst.sif}"
SIM_SIF_SHA256="fe358c025561e9fac4831abd1f03afcdf2931ed7bb683d6a86f19a39591550ff"

# Agent image: the Hetzner host's algoinst_claude:opencode (Docker image
# sha256:9ad094d1..., OpenCode 1.19.6, built from
# infra/claude_proxy/container/CC_env/Dockerfile.opencode), converted by
# agent_image.slurm.sh. Used by campaign.slurm.sh.
AGENT_SIF="${AGENT_SIF:-$PF_SCRATCH/images/algoinst_claude_opencode.sif}"
AGENT_SIF_SHA256="4d3ba8cc170616c804184177ec969227ff9d014f084bcce1ba057ab485c328d4"

# Weights: Hugging Face cache layout under HF_HOME, pinned revision.
export HF_HOME="${HF_HOME:-/dais/fs/scratch/$USER/models}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-Next}"
MODEL_REV="${MODEL_REV:-a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"
# a text the server's rendered chat prompt must contain (check_prompt; empty:
# no check), such as a reasoning effort a profile sets as the server's default
PROMPT_MUST_CONTAIN="${PROMPT_MUST_CONTAIN:-}"
# a reasoning model's parser (empty: none, as for Qwen3-Coder-Next)
REASONING_PARSER="${REASONING_PARSER:-}"
# the MoE kernels: triton (Qwen on H200 and B200, see campaign.slurm.sh), or
# auto for vLLM's own choice
MOE_BACKEND="${MOE_BACKEND:-triton}"
# a profile's own vLLM arguments (a KV cache format, an attention backend),
# split on spaces; every server start gets them (serve_model), before the
# caller's (VLLM_EXTRA, a per-submission override, comes after)
MODEL_VLLM_ARGS="${MODEL_VLLM_ARGS:-}"
# a profile's environment for vLLM in the container, VAR=value words split on
# spaces (such as a longer VLLM_ENGINE_READY_TIMEOUT_S for a slow load)
MODEL_VLLM_ENV="${MODEL_VLLM_ENV:-}"
# the nodes the model server spans (a profile's SERVER_NODES; serve_model):
# 1, the server on this node alone, or more: pipeline parallel over the job's
# first SERVER_NODES nodes, tensor parallel over each one's SERVER_GPUS, the
# head with the API (and the agents and sims) on the first node. Across
# nodes vLLM's own traffic runs over SERVER_NET_IF (IP over InfiniBand on
# DAIS) and NCCL's data over InfiniBand.
SERVER_NODES="${SERVER_NODES:-1}"
SERVER_NET_IF="${SERVER_NET_IF:-ib0}"
# the endpoint probe's tokens per request (probe_endpoint.py): a reasoning
# model needs room to reason before it answers
if [ -n "$REASONING_PARSER" ]; then
  PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-8192}"
fi
# the served context; campaign.slurm.sh passes it to the agents' OpenCode
# config (PF_LLM_CONTEXT). Qwen3-Coder-Next: 262,144 natively. Its KV cache
# costs about 25 KB per token over both GPUs at TP 2 (B5), so a full-length
# session holds about 6.6 GB; 2 B200 at 0.90 leave about 170 GB of KV cache
# (about 6.8M tokens), 2 H200 about 4M (measured), and vLLM preempts rather
# than fails when sessions together need more.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"

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
  if [ -z "$tag" ]; then  # an official image converted here: no such label
    tag="vllm/vllm-openai:v$(apptainer exec --containall --cleanenv "$VLLM_SIF" \
      python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tail -1)"
  fi
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
# SERVER_CPUS if set. Sets SERVER_PID, SERVER_PIDS (the server's processes on
# this node: SERVER_PID and, across nodes, each worker's srun), SERVER_PORT,
# SERVER_URL, SERVER_KEY_FILE and SERVER_AUTH_HEADER (both mode 600; the key
# is never printed and never on a command line: vLLM reads it from its
# environment, curl from the header file).
#
# The container: --containall, the weights read-only, a home of the job's own
# ($PF_RUN/home) and one cache folder shared by all jobs ($PF_SCRATCH/cache:
# torch.compile, Triton and FlashInfer caches are keyed by content).
#
# Across nodes (SERVER_NODES > 1, a profile): tensor parallel over <tp> GPUs
# on each node, pipeline parallel over the nodes, vLLM's own multi-node mode
# (--nnodes, --node-rank, --master-addr, --master-port; the other nodes
# --headless; no Ray). The head is this node, the job's first, and serves the
# API; each further node of the job runs a worker in the same container
# through `srun --overlap` (server_worker), its log beside the head's
# (<log without .log>.node<k>.log). vLLM's own traffic and NCCL's and Gloo's
# bootstrap use SERVER_NET_IF (each node's IPv4 address there is its
# VLLM_HOST_IP); NCCL's data goes over InfiniBand, /dev/infiniband bound in.
# With SERVE_DRY_RUN=1 it prints what it would start on which node, and
# starts nothing.
serve_model() {
  local util="$1" tp="$2" log="$3"
  shift 3
  local multi=() net=() nodes=() k
  if (( SERVER_NODES > 1 )); then
    server_nodes || return 1
    nodes=("${SERVER_NODE_LIST[@]}")
    net_args || return 1
    net=("${NET_ARGV[@]}" --env VLLM_HOST_IP="$SERVER_HEAD_IP")
    SERVER_MASTER_PORT=$(free_port)
    multi=(--pipeline-parallel-size "$SERVER_NODES" --nnodes "$SERVER_NODES"
           --master-addr "$SERVER_HEAD_IP" --master-port "$SERVER_MASTER_PORT"
           --distributed-executor-backend mp)
  fi
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
  vllm_container "$PF_RUN/home"
  local head=("${pin[@]}" "${CONTAINER_ARGV[@]}"
    --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${net[@]}" "$VLLM_SIF")
  vllm_argv "$util" "$tp" head
  head+=("${VLLM_ARGV[@]}" "${multi[@]}" ${multi[0]:+--node-rank 0} "$@")
  if [ "${SERVE_DRY_RUN:-0}" = 1 ]; then
    echo "head on ${nodes[0]:-$(hostname -s)}, log $log:"
    echo "  APPTAINERENV_VLLM_API_KEY=<key> $(printf '%q ' "${head[@]}")"
  else
    APPTAINERENV_VLLM_API_KEY="$(cat "$SERVER_KEY_FILE")" \
      "${head[@]}" > "$log" 2>&1 &
    SERVER_PID=$!
    BG_PIDS+=("$SERVER_PID")
  fi
  SERVER_PIDS=(${SERVER_PID:-})
  for (( k = 1; k < ${#nodes[@]} && k < SERVER_NODES; k++ )); do
    server_worker "$k" "${nodes[$k]}" "${log%.log}.node$k.log" "$util" "$tp" \
      "${multi[@]}" "$@" || return 1
  done
}

# vllm_container <home>: CONTAINER_ARGV, the apptainer command up to the
# GPUs, the network and the image; with the profile's MODEL_VLLM_ENV.
vllm_container() {
  local var
  CONTAINER_ARGV=(apptainer exec --nv --containall --cleanenv
    --home "$1"
    --bind "$HF_HOME:$HF_HOME:ro"
    --bind "$PF_SCRATCH/cache"
    --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1
    --env XDG_CACHE_HOME="$PF_SCRATCH/cache"
    --env VLLM_CACHE_ROOT="$PF_SCRATCH/cache/vllm"
    --env TRITON_CACHE_DIR="$PF_SCRATCH/cache/triton"
    --env FLASHINFER_WORKSPACE_BASE="$PF_SCRATCH/cache")
  for var in ${MODEL_VLLM_ENV:-}; do
    CONTAINER_ARGV+=(--env "$var")
  done
}

# vllm_argv <util> <tp> head|worker: VLLM_ARGV, the `vllm serve` command
# without the caller's arguments; the head's has the API's address and port.
vllm_argv() {
  VLLM_ARGV=(vllm serve "$MODEL" --revision "$MODEL_REV")
  [ "$3" = head ] && VLLM_ARGV+=(--host 127.0.0.1 --port "$SERVER_PORT")
  # shellcheck disable=SC2206  # MODEL_VLLM_ARGS: split on spaces
  VLLM_ARGV+=(--tensor-parallel-size "$2"
    --gpu-memory-utilization "$1"
    --max-model-len "$MAX_MODEL_LEN"
    --enable-prefix-caching --enable-prompt-tokens-details
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER"
    ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"}
    $([ "$MOE_BACKEND" = auto ] || echo --moe-backend "$MOE_BACKEND")
    ${MODEL_VLLM_ARGS:-})
}

# server_nodes: SERVER_NODE_LIST, the nodes of a server across SERVER_NODES
# nodes, this one (the head) first: the job's nodes, or SERVER_NODES_OVERRIDE
# (space-separated; the connectivity probe puts head and worker on one node).
# Fails when the job has fewer nodes than the profile asks for.
server_nodes() {
  if [ -n "${SERVER_NODES_OVERRIDE:-}" ]; then
    read -r -a SERVER_NODE_LIST <<< "$SERVER_NODES_OVERRIDE"
  else
    mapfile -t SERVER_NODE_LIST < <(scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>/dev/null)
  fi
  if (( ${#SERVER_NODE_LIST[@]} < SERVER_NODES )); then
    echo "FATAL: the model server spans $SERVER_NODES nodes (SERVER_NODES, profile" \
      "${MODEL_PROFILE:-none}), but this job has ${#SERVER_NODE_LIST[@]}" \
      "(${SERVER_NODE_LIST[*]:-none}): submit with -N $SERVER_NODES" \
      "--ntasks=$SERVER_NODES --ntasks-per-node=1 (the scripts' --ntasks=1 alone" \
      "makes Slurm shrink -N to one node)" >&2
    return 1
  fi
  if [ "${SERVE_DRY_RUN:-0}" != 1 ] && [ "${SERVER_NODE_LIST[0]}" != "$(hostname -s)" ]; then
    echo "FATAL: the head must run on the job's first node ${SERVER_NODE_LIST[0]}," \
      "not on $(hostname -s)" >&2
    return 1
  fi
}

# net_args: NET_ARGV, the container's network settings across nodes, and
# SERVER_HEAD_IP, this node's IPv4 address on SERVER_NET_IF.
net_args() {
  SERVER_HEAD_IP="${SERVER_HEAD_IP:-$(iface_ip "$SERVER_NET_IF")}"
  if [ -z "$SERVER_HEAD_IP" ]; then
    echo "FATAL: no IPv4 address on $SERVER_NET_IF on $(hostname -s) (SERVER_NET_IF)" >&2
    return 1
  fi
  if [ "${SERVE_DRY_RUN:-0}" != 1 ] && [ ! -d /dev/infiniband ]; then
    echo "FATAL: no /dev/infiniband on $(hostname -s)" >&2
    return 1
  fi
  NET_ARGV=(--bind /dev/infiniband
    --env GLOO_SOCKET_IFNAME="$SERVER_NET_IF" --env NCCL_SOCKET_IFNAME="$SERVER_NET_IF"
    ${NCCL_DEBUG:+--env NCCL_DEBUG="$NCCL_DEBUG"})
}

# iface_ip <interface>: its first IPv4 address on this node (empty: none)
iface_ip() {
  ip -4 -o addr show dev "$1" 2>/dev/null | awk '{split($4, a, "/"); print a[1]; exit}'
}

# server_worker <k> <node> <log> <util> <tp> [vllm args...]: worker k of the
# server, on <node>, in the background through srun (one task, the node's
# GPUs: SERVER_WORKER_GPUS, default <tp>; SERVER_WORKER_CPUS cores' worth of
# CPUs, default 32). On the node, WORKER_PRELUDE finds its address on
# SERVER_NET_IF and its GPUs (SERVER_WORKER_DEVICES, or the step's) and passes
# them in through APPTAINERENV_ (--cleanenv drops the rest). Adds the srun's
# pid to SERVER_PIDS and BG_PIDS: stopping it ends the worker (srun forwards
# the signal), and its end shows in server_alive.
server_worker() {
  local k="$1" node="$2" log="$3" util="$4" tp="$5"
  shift 5
  mkdir -p "$PF_RUN/home.node$k"
  vllm_container "$PF_RUN/home.node$k"
  # by absolute path: a step on the GPU nodes may not have the batch's PATH
  # (infra/dais/README.md, the agents' taskset)
  CONTAINER_ARGV[0]=$(command -v apptainer || echo apptainer)
  vllm_argv "$util" "$tp" worker
  local argv=(srun --nodes=1 --ntasks=1 --nodelist="$node" --overlap
    --kill-on-bad-exit=1 --gres="gpu:${SERVER_WORKER_GPUS:-$tp}"
    --cpus-per-task="${SERVER_WORKER_CPUS:-32}" --mem=0
    /bin/bash -c "$WORKER_PRELUDE" worker "$SERVER_NET_IF" "${SERVER_WORKER_DEVICES:-}"
    "${CONTAINER_ARGV[@]}" "${NET_ARGV[@]}" "$VLLM_SIF"
    "${VLLM_ARGV[@]}" "$@" --node-rank "$k" --headless)
  if [ "${SERVE_DRY_RUN:-0}" = 1 ]; then
    echo "worker $k on $node, log $log:"
    echo "  $(printf '%q ' "${argv[@]}")"
    return 0
  fi
  "${argv[@]}" > "$log" 2>&1 &
  SERVER_PIDS+=($!)
  BG_PIDS+=($!)
  echo "worker $k on $node: srun pid $!, log $log"
}

# what a worker runs on its node before the container: bash -c "$WORKER_PRELUDE"
# worker <interface> <devices or empty> <command...>
WORKER_PRELUDE='if_=$1 devices=$2; shift 2
export PATH=/usr/sbin:/usr/bin:/bin:$PATH
ip=$(ip -4 -o addr show dev "$if_" 2>/dev/null | awk '"'"'{split($4, a, "/"); print a[1]; exit}'"'"')
[ -n "$ip" ] || { echo "FATAL: no IPv4 address on $if_ on $(hostname -s)" >&2; exit 1; }
[ -d /dev/infiniband ] || { echo "FATAL: no /dev/infiniband on $(hostname -s)" >&2; exit 1; }
export APPTAINERENV_VLLM_HOST_IP=$ip
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${devices:-${CUDA_VISIBLE_DEVICES:-}}
echo "worker on $(hostname -s): $if_ $ip, GPUs $APPTAINERENV_CUDA_VISIBLE_DEVICES"
exec "$@"'

# server_alive: every process of the server on this node runs (SERVER_PIDS:
# the head, and each worker's srun, which ends with its worker)
server_alive() {
  local pid
  for pid in "${SERVER_PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null || return 1
  done
}

# wait_ready <timeout s>: until OUR server answers: /v1/models lists $MODEL
# with our key and refuses a request without it (401), so another job's
# server on the same port cannot pass. Prints the load time. Across nodes the
# head answers only once every worker has loaded its part.
wait_ready() {
  local timeout="$1" t0=$SECONDS code
  while (( SECONDS - t0 < timeout )); do
    if ! server_alive; then
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
      check_prompt || return 1
      return 0
    fi
    sleep 5
  done
  echo "FATAL: not ready after $timeout s" >&2
  return 1
}

# check_prompt: with PROMPT_MUST_CONTAIN set (a profile), the prompt the
# server renders for a plain chat request, defaults included (such as a
# reasoning effort set by --default-chat-template-kwargs), must contain it:
# /tokenize with the chat, then /detokenize. Prints the rendered prompt's
# start either way.
check_prompt() {
  [ -n "${PROMPT_MUST_CONTAIN:-}" ] || return 0
  python3 - "$SERVER_URL" "$SERVER_KEY_FILE" "$MODEL" "$PROMPT_MUST_CONTAIN" <<'EOF'
import json, sys, urllib.request
base, key_file, model, want = sys.argv[1:5]
key = open(key_file).read().strip()

def post(path, body):
    req = urllib.request.Request(base.rsplit("/v1", 1)[0] + path, json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {key}",
                                  "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())

tokens = post("/tokenize", {"model": model, "add_generation_prompt": True,
                            "messages": [{"role": "user", "content": "Say ok."}]})["tokens"]
prompt = post("/detokenize", {"model": model, "tokens": tokens})["prompt"]
print("rendered prompt:", repr(prompt[:300]), flush=True)
if want not in prompt:
    print(f"FATAL: the server's prompt lacks {want!r}", file=sys.stderr)
    sys.exit(1)
print(f"the server's prompt has {want!r}")
EOF
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
