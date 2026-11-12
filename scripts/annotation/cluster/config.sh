# Per-cluster and per-model defaults for the annotation inference jobs. The one
# place where model, image, parallelism, context length, sequence cap and time
# limit are set. Sourced by the Mac helpers (submit.sh, sync_code.sh, ...) and,
# through the job env file that submit.sh writes, by the jobs themselves.
#
# Every value is a default: a CFG_* variable already set in the environment
# (or given as KEY=VALUE to submit.sh) wins. Call cluster_config <cluster>
# first, then model_config, which reads CFG_MODEL.
# shellcheck shell=bash disable=SC2034

cluster_config() {
    case "$1" in
    viper)
        # Development: 2x AMD MI300A (one node), ROCm vLLM image of another
        # user (readable), weights from the shared LLMIS cache on /ptmp.
        : "${CFG_PROJECT:=/ptmp/levinb/algorithmic-institutions-annot}"
        : "${CFG_IMAGE:=/ptmp/lumi/.llmis/vllm_vllm-openai-rocm_v0.30.0.sif}"
        : "${CFG_VLLM_VERSION:=0.30.0}"
        : "${CFG_APPTAINER_MODULE:=apptainer/1.5.2}"
        # read-only caches whose model dirs are symlinked into CFG_HF_HOME
        : "${CFG_HF_SOURCES:=/ptmp/lumi/.llmis/huggingface/hub /ptmp/levinb/hf_cache/hub}"
        : "${CFG_MODEL:=Qwen/Qwen3-30B-A3B-Instruct-2507}"
        : "${CFG_TP:=2}"
        : "${CFG_GPUS:=2}"
        : "${CFG_GPU_MEM_UTIL:=0.90}"
        : "${CFG_PARTITION:=apu}"
        : "${CFG_TIME:=02:00:00}"
        : "${CFG_CPUS:=48}"
        : "${CFG_MEM:=220000}"
        : "${CFG_SBATCH_EXTRA:=--constraint=apu}"
        : "${CFG_WALL_BUFFER_MIN:=10}"
        # the KV cache holds 1.45M tokens at TP 2, so a long context is cheap
        # and gives proposal prompts room (the pipeline cuts them to fit)
        : "${CFG_MAX_MODEL_LEN:=65536}"
        ;;
    dais)
        # Production: B200 nodes (8x 183 GB). gpu1 takes up to 4 GPUs per job
        # with at most 12 CPUs and 240G per GPU; gpu is whole-node exclusive.
        : "${CFG_PROJECT:=/dais/fs/scratch/levinb/algorithmic-institutions-annot}"
        : "${CFG_IMAGE:=/dais/fs/scratch/levinb/containers/vllm-openai_v0.32.0.sif}"
        : "${CFG_VLLM_VERSION:=0.32.0}"
        : "${CFG_APPTAINER_MODULE:=apptainer/1.4.3}"
        : "${CFG_HF_SOURCES:=/dais/fs/scratch/levinb/hf_cache/hub /u/levinb/.cache/huggingface/hub}"
        : "${CFG_MODEL:=deepseek-ai/DeepSeek-V4.1-Flash-1017}"
        : "${CFG_TP:=2}"
        : "${CFG_GPUS:=2}"
        : "${CFG_GPU_MEM_UTIL:=0.92}"
        : "${CFG_PARTITION:=gpu1}"
        : "${CFG_TIME:=06:00:00}"
        : "${CFG_CPUS:=$((12 * CFG_GPUS))}"
        : "${CFG_MEM:=$((240 * CFG_GPUS))G}"
        : "${CFG_SBATCH_EXTRA:=--account=mpib_inst --qos=normal}"
        : "${CFG_GRES_TYPE:=b200}"
        : "${CFG_WALL_BUFFER_MIN:=15}"
        # where download_model.sh puts the production weights
        : "${CFG_PROD_MODEL:=deepseek-ai/DeepSeek-V4.1-Flash-1017}"
        : "${CFG_PROD_REVISION:=08c8657f02291e11b6f6b63e0e16014f8e748518}"
        : "${CFG_PROD_HF_HOME:=/dais/fs/scratch/levinb/hf_cache}"
        ;;
    *)
        echo "config.sh: unknown cluster '$1' (viper|dais)" >&2
        return 2
        ;;
    esac
    : "${CFG_CLUSTER:=$1}"
    : "${CFG_HF_HOME:=$CFG_PROJECT/hf}"
    : "${CFG_RUNS_ROOT:=$CFG_PROJECT/runs/annotation}"
    # longest prompt is ~12k tokens incl. the codebook; prompt + max_tokens
    # must stay below this with a margin
    : "${CFG_MAX_MODEL_LEN:=32768}"
    : "${CFG_MAX_NUM_SEQS:=128}"
    : "${CFG_MAX_BATCHED_TOKENS:=8192}"
    : "${CFG_STRUCTURED_BACKEND:=xgrammar}"
    : "${CFG_SAMPLE_SEC:=15}"
    # run the pipeline preflight (one request per schema) before the command
    : "${CFG_PIPELINE_PREFLIGHT:=1}"
    : "${CFG_EXTRA_ARGS:=}"
}

model_config() {
    # Revision pins, reasoning parser, KV dtype and model-specific serve flags.
    # CFG_MODEL_ARGS and CFG_SERVE_ENV are arrays (JSON values keep their commas).
    # A CFG_MODEL_ARGS override replaces the model's flags; CFG_SERVE_ENV
    # entries are appended to the defaults (later entries win). Call once.
    local serve_env=() user_env=()
    if declare -p CFG_SERVE_ENV >/dev/null 2>&1; then user_env=("${CFG_SERVE_ENV[@]}"); fi
    case "$CFG_MODEL" in
    Qwen/Qwen3-30B-A3B-Instruct-2507)
        : "${CFG_REVISION:=0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe}"
        : "${CFG_REASONING_PARSER:=}" # instruct-only model, never thinks
        : "${CFG_KV_DTYPE:=auto}"
        : "${CFG_WAIT_MIN:=30}"
        ;;
    Qwen/Qwen3-0.6B)
        : "${CFG_REVISION:=c1899de289a04d12100db370d81485cdf75e47ca}"
        : "${CFG_REASONING_PARSER:=qwen3}" # hybrid model: thinks unless told not to
        : "${CFG_KV_DTYPE:=auto}"
        : "${CFG_WAIT_MIN:=15}"
        ;;
    deepseek-ai/DeepSeek-V4.1-Flash-1017)
        # recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash, hardware=b200,
        # strategy=single_node_tp, features=reasoning,text_only (vLLM >= 0.31).
        : "${CFG_REVISION:=08c8657f02291e11b6f6b63e0e16014f8e748518}"
        : "${CFG_REASONING_PARSER:=deepseek_v41}"
        : "${CFG_KV_DTYPE:=fp8}"
        # first boot: 510 GB load plus a FlashInfer autotune of 75-90 min
        # (recipe), so the wait and the engine timeout sit well above it
        : "${CFG_WAIT_MIN:=200}"
        # FlashInfer autotune on first boot can take over an hour (recipe);
        # keep its cache across jobs
        : "${CFG_PERSIST_COMPILE_CACHE:=1}"
        if ! declare -p CFG_MODEL_ARGS >/dev/null 2>&1; then
            CFG_MODEL_ARGS=(
                --tokenizer-mode deepseek_v41
                --attention-config
                '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'
                --engram-config '{"cpu_offload":true}'
                --language-model-only
            )
        fi
        # first load can be long (engine timeout above CFG_WAIT_MIN); the Rust
        # frontend of the recipe is left off until response_format and
        # chat_template_kwargs are proven on it
        serve_env=(VLLM_ENGINE_READY_TIMEOUT_S=14400)
        ;;
    *)
        if [ -z "${CFG_REVISION:-}" ]; then
            echo "config.sh: no profile for model $CFG_MODEL; set CFG_REVISION" >&2
            return 2
        fi
        ;;
    esac
    : "${CFG_REASONING_PARSER:=}"
    : "${CFG_KV_DTYPE:=auto}"
    : "${CFG_WAIT_MIN:=30}"
    if ! declare -p CFG_MODEL_ARGS >/dev/null 2>&1; then CFG_MODEL_ARGS=(); fi
    if [ "$CFG_CLUSTER" = viper ]; then serve_env=(VLLM_ROCM_USE_AITER=1 "${serve_env[@]}"); fi
    CFG_SERVE_ENV=("${serve_env[@]}" "${user_env[@]}")
}
