# Job-side helpers for serve_and_run.<cluster>.sbatch: start a vLLM server in
# an Apptainer image, wait until it serves the model, sample its metrics and
# the GPUs, run the pipeline command in the same image, then shut down.
#
# The sbatch file sources the job env file (ANNOT_JOB_ENV, written by
# submit.sh: all CFG_* values plus ANNOT_RUN_ID and ANNOT_CMD), then this file,
# then defines three cluster hooks and calls annot_main:
#   annot_container_flags   fills the array CONTAINER_FLAGS (device flags,
#                           extra binds) used for every exec
#   annot_server_flags      fills SERVER_FLAGS (flags only the server exec gets)
#   annot_gpu_sampler       runs forever, appending GPU samples to stdout
#
# Contract with the pipeline (src/aimanager/annotation): the command runs with
# cwd = project dir, PYTHONPATH=<project>/src and
#   ANNOT_BASE_URL=http://127.0.0.1:<port>/v1  ANNOT_MODEL=<served model id>
#   ANNOT_RUNS_ROOT=<project>/runs/annotation  ANNOT_RUN_ID  ANNOT_JOB_LOG_DIR
# and the job exits with the command's status.
# shellcheck shell=bash

mark() { echo "$(date +%s) $(date -Is) $*" | tee -a "$LOG_DIR/timeline.txt"; }

annot_init() {
    : "${ANNOT_RUN_ID:?}" "${ANNOT_CMD:?}" "${CFG_PROJECT:?}" "${CFG_MODEL:?}"
    RUN_DIR="$CFG_RUNS_ROOT/$ANNOT_RUN_ID"
    LOG_DIR="$RUN_DIR/logs/job-${SLURM_JOB_ID}"
    JOB_TMP="${JOB_SHMTMPDIR:-${TMPDIR:-/tmp}}/annot_${SLURM_JOB_ID}"
    mkdir -p "$LOG_DIR" "$JOB_TMP"/{home,shim,tmp}
    cp "$ANNOT_JOB_ENV" "$LOG_DIR/job.env"
    # per-job port window so co-scheduled jobs on one node never collide
    PORT=$((20000 + (SLURM_JOB_ID % 2000) * 10))
    while (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; do PORT=$((PORT + 1)); done
    BASE_URL="http://127.0.0.1:$PORT"
    SERVER_PID="" METRICS_PID="" GPU_PID="" CMD_PID="" STOP_REASON=""
    # `python` -> the image's python3 when the image has no `python` (CUDA image)
    # shellcheck disable=SC2016
    SHIM_PRELUDE='command -v python >/dev/null || { ln -sfn "$(command -v python3)" "$ANNOT_SHIM/python"; export PATH="$ANNOT_SHIM:$PATH"; };'

    mark "job ${SLURM_JOB_ID} on $(hostname) cluster=$CFG_CLUSTER run=$ANNOT_RUN_ID"
    mark "command: $ANNOT_CMD"
    mark "model=$CFG_MODEL@$CFG_REVISION image=$CFG_IMAGE tp=$CFG_TP gpus=$CFG_GPUS" \
        "max_model_len=$CFG_MAX_MODEL_LEN max_num_seqs=$CFG_MAX_NUM_SEQS" \
        "gpu_mem_util=$CFG_GPU_MEM_UTIL kv_dtype=$CFG_KV_DTYPE port=$PORT"

    module purge >/dev/null 2>&1 || true
    module load "$CFG_APPTAINER_MODULE"

    # Environment for every exec. APPTAINERENV_* passes through --cleanenv,
    # keeps host modules out and allows commas in values.
    export APPTAINERENV_HF_HOME="$CFG_HF_HOME"
    export APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_TRANSFORMERS_OFFLINE=1
    export APPTAINERENV_HF_HUB_DISABLE_TELEMETRY=1 APPTAINERENV_VLLM_NO_USAGE_STATS=1
    export APPTAINERENV_PYTHONNOUSERSITE=1 APPTAINERENV_PYTHONUNBUFFERED=1
    export APPTAINERENV_TMPDIR="$JOB_TMP/tmp"
    annot_compile_cache

    COMMON_FLAGS=(--cleanenv --home "$JOB_TMP/home" --pwd "$CFG_PROJECT"
        --bind "$JOB_TMP" --bind "$CFG_PROJECT")
    local src
    for src in $CFG_HF_SOURCES; do
        if [ -d "$src" ]; then COMMON_FLAGS+=(--bind "$src:$src:ro"); fi
    done
    CONTAINER_FLAGS=()
    annot_container_flags
}

# Compile caches (vLLM, Triton, Inductor, FlashInfer). Per job by default: a
# shared cache was once corrupted by concurrent jobs. With
# CFG_PERSIST_COMPILE_CACHE=1 (models whose first boot autotunes for a long
# time) one directory per image/model/TP is reused, held under an exclusive
# flock for the whole job; a second concurrent job falls back to per job.
annot_compile_cache() {
    local dir="$JOB_TMP/cc"
    if [ "${CFG_PERSIST_COMPILE_CACHE:-0}" = 1 ]; then
        local key
        key="$(basename "$CFG_IMAGE" .sif)__${CFG_MODEL//\//--}__tp${CFG_TP}"
        mkdir -p "$CFG_PROJECT/compile_cache/$key"
        exec 9>"$CFG_PROJECT/compile_cache/$key.lock"
        if flock -n 9; then
            dir="$CFG_PROJECT/compile_cache/$key"
        else
            mark "compile cache $key in use by another job; using a per-job cache"
        fi
    fi
    mkdir -p "$dir"/{vllm,triton,inductor,xdg,flashinfer}
    mark "compile cache: $dir"
    export APPTAINERENV_VLLM_CACHE_ROOT="$dir/vllm"
    export APPTAINERENV_TRITON_CACHE_DIR="$dir/triton"
    export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$dir/inductor"
    export APPTAINERENV_XDG_CACHE_HOME="$dir/xdg"
    export APPTAINERENV_FLASHINFER_WORKSPACE_BASE="$dir/flashinfer"
}

cexec() { apptainer exec "${COMMON_FLAGS[@]}" "${CONTAINER_FLAGS[@]}" "$CFG_IMAGE" "$@"; }

# Writable HF_HOME whose model dir is a symlink into a read-only cache, so no
# copy of the weights is made. Fails before any GPU time if the pinned
# snapshot is missing or has dangling links (purged blobs).
annot_link_model() {
    local name="models--${CFG_MODEL//\//--}" src found=""
    mkdir -p "$CFG_HF_HOME/hub"
    if [ ! -e "$CFG_HF_HOME/hub/$name" ] || [ -L "$CFG_HF_HOME/hub/$name" ]; then
        for src in $CFG_HF_SOURCES; do
            if [ -d "$src/$name/snapshots/$CFG_REVISION" ]; then found="$src/$name"; break; fi
        done
        [ -n "$found" ] || { mark "FAIL no snapshot $CFG_REVISION of $CFG_MODEL in: $CFG_HF_SOURCES"; return 1; }
        ln -sfn "$found" "$CFG_HF_HOME/hub/$name"
    fi
    local snap="$CFG_HF_HOME/hub/$name/snapshots/$CFG_REVISION"
    [ -f "$snap/config.json" ] || { mark "FAIL $snap has no config.json"; return 1; }
    if find -L "$snap" -type l | grep -q .; then
        mark "FAIL dangling links in $snap (blobs purged or download incomplete)"
        return 1
    fi
    mark "model snapshot ok: $(readlink -f "$CFG_HF_HOME/hub/$name") ($(du -shL "$snap" | cut -f1))"
}

annot_preflight() {
    [ -r "$CFG_IMAGE" ] || { mark "FAIL image not readable: $CFG_IMAGE"; return 1; }
    local out
    out=$(cexec python3 -c 'import vllm, pydantic, yaml, openai, httpx; print(vllm.__version__)' 2>"$LOG_DIR/preflight.err" | tail -1) \
        || { mark "FAIL image imports"; tail -5 "$LOG_DIR/preflight.err"; return 1; }
    [ "$out" = "$CFG_VLLM_VERSION" ] || { mark "FAIL vllm $out != pinned $CFG_VLLM_VERSION"; return 1; }
    mark "preflight ok: vllm $out, pydantic/yaml/openai/httpx importable"
    annot_link_model
}

annot_start_server() {
    local args=(vllm serve "$CFG_MODEL" --revision "$CFG_REVISION"
        --served-model-name "$CFG_MODEL"
        --host 127.0.0.1 --port "$PORT"
        --tensor-parallel-size "$CFG_TP"
        --max-model-len "$CFG_MAX_MODEL_LEN"
        --max-num-seqs "$CFG_MAX_NUM_SEQS"
        --max-num-batched-tokens "$CFG_MAX_BATCHED_TOKENS"
        --gpu-memory-utilization "$CFG_GPU_MEM_UTIL"
        --kv-cache-dtype "$CFG_KV_DTYPE"
        --enable-prefix-caching
        --structured-outputs-config.backend "$CFG_STRUCTURED_BACKEND"
        --uvicorn-log-level warning)
    if [ -n "$CFG_REASONING_PARSER" ]; then args+=(--reasoning-parser "$CFG_REASONING_PARSER"); fi
    args+=("${CFG_MODEL_ARGS[@]}")
    # shellcheck disable=SC2206
    if [ -n "$CFG_EXTRA_ARGS" ]; then args+=($CFG_EXTRA_ARGS); fi
    local kv
    for kv in "${CFG_SERVE_ENV[@]}"; do export "APPTAINERENV_$kv"; done
    SERVER_FLAGS=()
    annot_server_flags
    printf '%q ' "${args[@]}" >"$LOG_DIR/serve_cmd.txt"
    mark "starting server"
    setsid apptainer exec "${COMMON_FLAGS[@]}" "${CONTAINER_FLAGS[@]}" "${SERVER_FLAGS[@]}" \
        "$CFG_IMAGE" "${args[@]}" >"$LOG_DIR/vllm.log" 2>&1 &
    SERVER_PID=$!
    for kv in "${CFG_SERVE_ENV[@]}"; do unset "APPTAINERENV_${kv%%=*}"; done
}

annot_wait_ready() {
    local deadline=$(($(date +%s) + CFG_WAIT_MIN * 60))
    until curl -sf "$BASE_URL/v1/models" 2>/dev/null | grep -q "\"$CFG_MODEL\""; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            mark "FAIL server died before becoming ready"
            tail -40 "$LOG_DIR/vllm.log"
            return 1
        fi
        if [ "$(date +%s)" -gt "$deadline" ]; then
            mark "FAIL server not ready after $CFG_WAIT_MIN min"
            tail -40 "$LOG_DIR/vllm.log"
            return 1
        fi
        sleep 5
    done
    mark "server ready"
    curl -s "$BASE_URL/version" >"$LOG_DIR/version.json" || true
    curl -s "$BASE_URL/metrics" >"$LOG_DIR/metrics_start.prom" || true
    grep -E "KV cache size|Maximum concurrency|Available KV cache|Model loading took|Loading weights took|torch.compile takes|Graph capturing finished|init engine" \
        "$LOG_DIR/vllm.log" | cut -c1-300 | tee "$LOG_DIR/startup_lines.txt" || true
}

annot_start_samplers() {
    (
        while true; do
            echo "### $(date +%s)"
            curl -s -m 5 "$BASE_URL/metrics" | grep -E "^vllm:(kv_cache_usage_perc|gpu_cache_usage_perc|num_requests_(running|waiting)|prompt_tokens_total|generation_tokens_total|prefix_cache_(queries|hits)_total|num_preemptions_total|request_success_total)" || true
            sleep "$CFG_SAMPLE_SEC"
        done
    ) >"$LOG_DIR/metrics_samples.txt" 2>&1 &
    METRICS_PID=$!
    annot_gpu_sampler >"$LOG_DIR/gpu_samples.txt" 2>&1 &
    GPU_PID=$!
}

# Seconds left before the internal wall-time limit (job end minus buffer).
annot_seconds_left() {
    local end
    end=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null || true)
    if [ -z "$end" ] || ! end=$(date -d "$end" +%s 2>/dev/null); then
        echo 86400
        return
    fi
    echo $((end - $(date +%s) - CFG_WALL_BUFFER_MIN * 60))
}

annot_run_command() {
    local left
    left=$(annot_seconds_left)
    [ "$left" -gt 60 ] || { mark "FAIL only ${left}s left before the wall-time buffer"; return 1; }
    mark "START command (internal limit ${left}s)"
    setsid timeout --signal=TERM --kill-after=60 "$left" \
        apptainer exec "${COMMON_FLAGS[@]}" "${CONTAINER_FLAGS[@]}" "$CFG_IMAGE" \
        env PYTHONPATH="$CFG_PROJECT/src" \
        ANNOT_BASE_URL="$BASE_URL/v1" ANNOT_MODEL="$CFG_MODEL" \
        ANNOT_RUNS_ROOT="$CFG_RUNS_ROOT" ANNOT_RUN_ID="$ANNOT_RUN_ID" \
        ANNOT_JOB_LOG_DIR="$LOG_DIR" ANNOT_SHIM="$JOB_TMP/shim" \
        bash -c "$SHIM_PRELUDE $ANNOT_CMD" >"$LOG_DIR/command.log" 2>&1 &
    CMD_PID=$!
    local rc=0
    wait "$CMD_PID" || rc=$?
    # a trapped signal interrupts wait; wait again for the real status
    while kill -0 "$CMD_PID" 2>/dev/null; do wait "$CMD_PID" || rc=$?; done
    CMD_PID=""
    [ "$rc" -eq 124 ] && STOP_REASON=${STOP_REASON:-internal_walltime}
    mark "END command rc=$rc ${STOP_REASON}"
    tail -20 "$LOG_DIR/command.log"
    return "$rc"
}

annot_on_signal() {
    STOP_REASON=$1
    mark "signal $1"
    if [ -n "$CMD_PID" ]; then kill -TERM -- "-$CMD_PID" 2>/dev/null || true; fi
    return 0
}

annot_cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        curl -s -m 10 "$BASE_URL/metrics" >"$LOG_DIR/metrics_end.prom" || true
        curl -s -m 10 "$BASE_URL/version" >"$LOG_DIR/version.json" || true
    fi
    local pid
    for pid in "$METRICS_PID" "$GPU_PID"; do
        if [ -n "$pid" ]; then kill "$pid" 2>/dev/null || true; fi
    done
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null
        for _ in $(seq 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    mark "server stopped"
    cexec python3 "$CFG_PROJECT/scripts/annotation/cluster/metrics_summary.py" "$LOG_DIR" \
        >"$LOG_DIR/summary.json" 2>"$LOG_DIR/summary.err" && cat "$LOG_DIR/summary.json" || true
    rm -rf "$JOB_TMP"
}

annot_main() {
    annot_init
    trap 'annot_on_signal USR1_walltime' USR1
    trap 'annot_on_signal TERM' TERM
    trap annot_cleanup EXIT
    annot_preflight
    annot_start_server
    annot_wait_ready
    annot_start_samplers
    local rc=0
    annot_run_command || rc=$?
    mark "job exit rc=$rc"
    exit "$rc"
}
