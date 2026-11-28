# Model profile (common.sh MODEL_PROFILE): GLM-5.3 (zai-org), licence
# "GLM-5.3 License" (MIT-style; a safety review only for Model-as-a-Service
# businesses above USD 10 billion revenue), ungated. 753B parameters, native
# FP8 (e4m3, 128x128 blocks): 755 GB of weights in 141 shards
# (GlmMoeDsaForCausalLM, DeepSeek-V3.2-style sparse attention, MLA, 1 MTP layer).
# vLLM serves it from v0.29 (recipes.vllm.ai/zai-org/GLM-5.3); the group's
# image is v0.25.1, so it is served from the official v0.34.0 image, whose
# registry maps the architecture to models/deepseek_v32.py.
# Parsers glm47 (tool calls as <tool_call>name<arg_key>..</arg_key>
# <arg_value>..</arg_value></tool_call>; reasoning in <think>, always on).
# Sampling: generation_config.json temperature 1.0, top_p 0.95
# (export_instance.LOCAL_MODELS). Native context 1,048,576 tokens.
MODEL=zai-org/GLM-5.3
MODEL_REV=ba32f8c19c8738e2c57c71798c601d477a7bf844
VLLM_SIF="$PF_SCRATCH/images/vllm-v0.34.0.sif"
EXPECT_VLLM_TAG=vllm/vllm-openai:v0.34.0
TOOL_PARSER=glm47
REASONING_PARSER=glm47
# The KV cache stays at the model's dtype (BF16): the MLA latent cache is
# replicated on every GPU at TP 8, an estimated 100 KB per token (78 layers),
# so the about 68 GiB left per B200 at 0.90 hold roughly 0.7M tokens in all
# (8 H200: about 0.35M); 262,144 per session. Estimates, not measured. (The
# vLLM recipe adds --kv-cache-dtype fp8_e4m3, which a profile cannot pass;
# VLLM_EXTRA can.)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MOE_BACKEND=auto
# 755 GB (704 GiB, 88 GiB per GPU at TP 8): 8 B200 or 8 H200. Not fewer:
# 4 B200 hold 4 x 179 GiB, less than the weights, and TP 6 does not divide
# the 64 attention heads
SERVER_GPUS="${SERVER_GPUS:-8}"
# 755 GB from GPFS (DeepSeek's 510 GB loaded in 526 s at TP 4, job 993167),
# then DeepGEMM warm-up and CUDA graphs: up to an hour
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
