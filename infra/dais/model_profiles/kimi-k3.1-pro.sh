# Model profile (common.sh MODEL_PROFILE): Kimi K3.1 Pro, the official
# checkpoint, on TWO nodes of 8 B200. Kimi K3.1 Pro licence (modified MIT;
# research use allowed), ungated, pinned revision. See
# docs/reports/dais_kimi_k31_pro.md.
#
# 1,561,293,160,568 bytes (1,454 GiB): routed experts MXFP4, the rest BF16
# (KimiK31ForConditionalGeneration: 93 layers, 69 KDA + 24 gated DSA, 896
# experts with top-16). One node does not hold it: at TP 8 about 191 GiB of
# weights per GPU against 179 GiB. On two nodes, TP 8 on each and pipeline
# parallel over both (as NVIDIA's Dynamo recipe and InferenceX run it on 16
# B200), about 96 GiB per GPU.
#
# vLLM 0.34.0 has the model (served as KimiK3ForConditionalGeneration, which
# supports pipeline parallelism), the kimi_k3 tool and reasoning parsers and
# its own multi-node mode (--nnodes, --node-rank, --master-addr, --headless);
# the recipe names vLLM >= 0.35 and the image vllm/vllm-openai:kimi-k31.
# Card: temperature 1.0, top_p 0.95 (export_instance.LOCAL_MODELS; its
# generation_config.json sets none), reasoning effort max for its results.
MODEL=moonshotai/Kimi-K3.1-Pro
MODEL_REV=80aff651e9b2d2f5b9f70f92a0a7a77054418e16
VLLM_SIF="$PF_SCRATCH/images/vllm-v0.34.0.sif"
EXPECT_VLLM_TAG=vllm/vllm-openai:v0.34.0
TOOL_PARSER=kimi_k3
REASONING_PARSER=kimi_k3
# 1M natively. KV cache: FP8 (as NVIDIA's evaluations), 16,992 bytes per token
# and GPU at TP 8 over the 24 DSA layers (vLLM's blog), half that per GPU
# with each pipeline stage holding 12; with about 45 GiB per GPU left at 0.90,
# about 5M tokens
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MOE_BACKEND=auto
# the maximum reasoning effort as the server's default for every chat request
# (vLLM's kimi_k3 renderer passes it to the chat encoder as thinking_effort,
# which writes it into the prompt), checked on the running server
# (PROMPT_MUST_CONTAIN); --trust-remote-code for the repository's tokenizer;
# text only (no vision encoder); the KDA state in float32 as in the recipe
MODEL_VLLM_ARGS='--trust-remote-code --language-model-only --kv-cache-dtype fp8 --mamba-ssm-cache-dtype float32 --max-num-seqs 32 --default-chat-template-kwargs {"reasoning_effort":"max"}'
PROMPT_MUST_CONTAIN="thinking_effort=max"
# two nodes: TP 8 on each, pipeline parallel over the two; vLLM's own traffic
# and NCCL's bootstrap over IP on InfiniBand, NCCL's data over InfiniBand
SERVER_NODES="${SERVER_NODES:-2}"
SERVER_GPUS="${SERVER_GPUS:-8}"
SERVER_NET_IF="${SERVER_NET_IF:-ib0}"
# loading about 780 GB per node from GPFS (DeepSeek's 476 GB took 526 s on
# one node), then compilation and CUDA graphs: the engine's own start-up
# limit (600 s by default) and the job's both an hour and a half
MODEL_VLLM_ENV="VLLM_ENGINE_READY_TIMEOUT_S=5400"
READY_TIMEOUT="${READY_TIMEOUT:-5400}"
