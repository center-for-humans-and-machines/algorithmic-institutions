# Shared settings and helpers for the DAIS job scripts in infra/dais/.
# Sourced, not run. See docs/reports/dais_feasibility.md for the measurements
# these defaults come from.

PF_SCRATCH="${PF_SCRATCH:-/dais/fs/scratch/$USER/policy-finder}"
PF_REPO="${PF_REPO:-/u/$USER/algorithmic-institutions-pf}"
PF_LOGS="$PF_SCRATCH/logs"

# vLLM image: group-readable, in another user's scratch; asserted, not assumed.
VLLM_SIF="${VLLM_SIF:-/dais/fs/scratch/bsup/container/vllm.sif}"
EXPECT_VLLM_TAG="vllm/vllm-openai:v0.25.1"

# Sim image: a byte copy of Raven's shared algoinst.sif (torch 1.11.0+cu113).
SIM_SIF="${SIM_SIF:-$PF_SCRATCH/images/algoinst.sif}"
SIM_SIF_SHA256="fe358c025561e9fac4831abd1f03afcdf2931ed7bb683d6a86f19a39591550ff"

# Weights: Hugging Face cache layout under HF_HOME, pinned revision.
export HF_HOME="${HF_HOME:-/dais/fs/scratch/$USER/models}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-Next}"
MODEL_REV="${MODEL_REV:-a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

module load apptainer/1.5.2

assert_vllm_image() {
  local tag
  tag=$(apptainer inspect "$VLLM_SIF" | sed -n 's/^ai\.vllm\.image\.tag: *//p' | head -1)
  if [ "$tag" != "$EXPECT_VLLM_TAG" ]; then
    echo "FATAL: $VLLM_SIF is '$tag', expected '$EXPECT_VLLM_TAG'" >&2
    exit 1
  fi
  echo "vLLM image: $tag"
}

# serve_model and wait_ready follow serve() and wait_ready() of the sister
# project ../understanding (scripts/slurm/lib/evo_common.sh, used by
# evo_dais.slurm.sh): the same image, `apptainer exec --nv`, a home that is not
# the real one (they bind the cwd as $HOME), HF_HOME on scratch, explicit
# --enable-prefix-caching, and readiness gated on /v1/models listing the model
# rather than on /health alone. Added here: offline mode, a pinned revision
# and tool calling.
#
# serve_model <port> <gpu-memory-utilization> <tp> <log> [extra vllm args...]
# Starts vLLM in the background on the GPUs in CUDA_VISIBLE_DEVICES, offline
# (the weights must already be in HF_HOME). Sets SERVER_PID.
serve_model() {
  local port="$1" util="$2" tp="$3" log="$4"
  shift 4
  # a home on scratch: vLLM, torch.compile and FlashInfer's JIT write their
  # caches under ~/.cache, which would otherwise be the real home
  mkdir -p "$PF_SCRATCH/home"
  apptainer exec --nv --cleanenv \
    --home "$PF_SCRATCH/home" \
    --bind /dais/fs/scratch \
    --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 \
    --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$VLLM_SIF" \
    vllm serve "$MODEL" --revision "$MODEL_REV" \
    --host 127.0.0.1 --port "$port" \
    --tensor-parallel-size "$tp" \
    --gpu-memory-utilization "$util" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-prefix-caching --enable-prompt-tokens-details \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
    "$@" > "$log" 2>&1 &
  SERVER_PID=$!
}

# wait_ready <port> <timeout s>: until /v1/models lists $MODEL. Prints the
# load time in seconds.
wait_ready() {
  local port="$1" timeout="$2" t0=$SECONDS
  while (( SECONDS - t0 < timeout )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "FATAL: vLLM exited during start-up" >&2
      return 1
    fi
    if curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | grep -qF "\"$MODEL\""; then
      echo "ready after $(( SECONDS - t0 )) s"
      return 0
    fi
    sleep 5
  done
  echo "FATAL: not ready after $timeout s" >&2
  return 1
}

# gpu_mem_log <file> &: per-GPU memory used, every 2 s, until killed.
gpu_mem_log() {
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits -l 2 > "$1"
}

# in_sim_image <command...>: run a command in the sim image, set up as
# Raven's job template does (slurm_tools run_slurm_raven_apptainer_algoinst.sh:
# repo at /opt/algorithmic-institutions, PATH and PYTHONPATH as there), on the
# GPUs in CUDA_VISIBLE_DEVICES.
in_sim_image() {
  mkdir -p "$PF_SCRATCH/tmp"
  apptainer exec --nv --containall --cleanenv --writable-tmpfs \
    --no-mount cwd,tmp,bind-paths \
    --bind "$PF_SCRATCH/tmp:/tmp" \
    --bind "$PF_REPO:/opt/algorithmic-institutions" \
    --bind "$PF_SCRATCH:$PF_SCRATCH" \
    --env PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    --env PYTHONPATH=/opt/algorithmic-institutions/src \
    --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    --pwd /opt/algorithmic-institutions \
    "$SIM_SIF" "$@"
}

# simulate <config> <label>: `python -m aimanager simulate <config>` on the one
# GPU in CUDA_VISIBLE_DEVICES; prints a SIM line with the wall time and the
# peak rise of that GPU's used memory while it ran (the sim's whole footprint,
# CUDA context included, as long as nothing else on the GPU grows meanwhile).
simulate() {
  local cfg="$1" label="$2" gpu="$CUDA_VISIBLE_DEVICES" samples base t0 rc sampler
  if [ -z "$gpu" ]; then  # on the CPU: time only
    t0=$SECONDS
    in_sim_image python -m aimanager simulate "$cfg"
    echo "SIM $label rc=$? seconds=$(( SECONDS - t0 )) device=cpu"
    return
  fi
  samples=$(mktemp)
  base=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  (while true; do
    nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits
    sleep 1
  done) > "$samples" &
  sampler=$!
  t0=$SECONDS
  in_sim_image python -m aimanager simulate "$cfg"
  rc=$?
  kill "$sampler"
  echo "SIM $label rc=$rc seconds=$(( SECONDS - t0 )) gpu_mem_base_mib=$base" \
    "peak_rise_mib=$(( $(sort -n "$samples" | tail -1) - base ))"
  rm -f "$samples"
}
