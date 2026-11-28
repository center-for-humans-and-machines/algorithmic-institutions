# Model profile (common.sh MODEL_PROFILE): Kimi K3.1 Pro on one node of 8 B200,
# as nota-ai's expert-pruned build Kimi-K3.1-Pro-Global-Pruned-37 (37.5% of
# the routed experts removed, non-uniformly per layer; Kimi K3.1 Pro licence,
# ungated), pinned revision. See docs/reports/dais_kimi_k31_pro.md.
#
# The official checkpoint (moonshotai/Kimi-K3.1-Pro, 1,454 GiB, MXFP4 routed
# experts and BF16 rest) does not fit one node: about 191 GiB of weights per
# GPU at TP 8 (the router and the latent MoE projections are replicated)
# against 179 GiB of HBM. The pruned build is 948 GiB, about 128 GiB per GPU.
#
# vLLM 0.34.0 serves KimiK31ForConditionalGeneration with the kimi_k3 parsers,
# but builds every MoE layer with 896 experts: VLLM_PATCH binds two patched
# files that take the layer's count from num_experts_per_layer
# (vllm_patches/kimi_k31_nonuniform_experts.py). The pruned repo carries an
# older chat encoder (encoding_k3.py, before the official fix of 2026-11-25),
# so the tokenizer comes from the official repo at its revision (the small
# files only: download_model.sh with DOWNLOAD_ALLOW).
MODEL=nota-ai/Kimi-K3.1-Pro-Global-Pruned-37
MODEL_REV=72f0f57d5eb3fc2df513271598e8d2185df145fd
KIMI_TOKENIZER=moonshotai/Kimi-K3.1-Pro
KIMI_TOKENIZER_REV=80aff651e9b2d2f5b9f70f92a0a7a77054418e16
VLLM_SIF="$PF_SCRATCH/images/vllm-v0.34.0.sif"
EXPECT_VLLM_TAG=vllm/vllm-openai:v0.34.0
VLLM_PATCH=kimi_k31_nonuniform_experts
TOOL_PARSER=kimi_k3
REASONING_PARSER=kimi_k3
# 1M natively; 262,144 keeps the KV cache for 12-24 sessions
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MOE_BACKEND=auto
SERVER_GPUS="${SERVER_GPUS:-8}"
# about 42 GiB per GPU beside the weights at 0.95, 33 GiB at 0.90
GPU_UTIL="${GPU_UTIL:-0.95}"
# about 1 TB to load from GPFS (DeepSeek's 476 GB took 526 s), then warm-up
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
# KV cache: NVFP4 MLA latent (352 instead of 576 bytes per token and layer;
# KV_CACHE_DTYPE=fp8 for FP8); text only (no vision encoder); the recurrent
# KDA state in float32 as in the recipe; at most 32 sequences at once
MODEL_ARGS=(
  --trust-remote-code
  --tokenizer "$KIMI_TOKENIZER" --tokenizer-revision "$KIMI_TOKENIZER_REV"
  --language-model-only
  --kv-cache-dtype "${KV_CACHE_DTYPE:-nvfp4_ds_mla}"
  --mamba-ssm-cache-dtype float32
  --max-num-seqs "${MAX_NUM_SEQS:-32}"
)
# serve.slurm.sh's probe: room to think, and little thinking
PROBE_ARGS=(--max-tokens 8192 --extra-body '{"reasoning_effort": "low"}')
