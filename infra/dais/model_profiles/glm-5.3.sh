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
# the campaign prefixes' model tag ({env:PF_MODEL_TAG}, where a branch uses it)
MODEL_TAG=glm53
VLLM_SIF="$PF_SCRATCH/images/vllm-v0.34.0.sif"
EXPECT_VLLM_TAG=vllm/vllm-openai:v0.34.0
TOOL_PARSER=glm47
REASONING_PARSER=glm47
# The KV cache: measured on 8 B200 at 0.90 (job 993477), vLLM chose an FP8
# cache by itself ("standard fp8 KV cache format"): 1,256,064 tokens in
# 55.8 GiB per GPU (the MLA cache is replicated, 47.7 KB per token), so the
# recipe's --kv-cache-dtype fp8_e4m3 is not needed. A session holds up to
# OpenCode's compaction point, the context less the output limit (32,768,
# LOCAL_MODELS) and the margin (65,536): at 147,456 that is 49,152 tokens,
# room for 25 sessions at their compaction point at once (24 agents run).
MODEL_VLLM_ARGS=""
MAX_MODEL_LEN="${MAX_MODEL_LEN:-147456}"
MOE_BACKEND=auto
# the reasoning effort (reasoning_effort.sh): a submission's REASONING_EFFORT,
# by default max, the model's own (its card: "It defaults to max if not
# passed (or if set to any other value)"; its benchmarks ran at max; the
# template renders "Max" for a request without one). The campaigns run at
# high (Levin, 2026-11-28: high for every model). The
# chat template (chat_template.jinja at this revision) takes low, high or
# max and renders "Reasoning Effort: Low|High|Max" as the first system line;
# it has no integer efforts and no way to turn thinking off.
REASONING_EFFORT_DEFAULT=max
REASONING_EFFORT_LEVELS="low=Low high=High max=Max"
REASONING_EFFORT_RANGE=""
REASONING_EFFORT_PROMPT="Reasoning Effort: {budget}"
# 755 GB (704 GiB; measured 89.5 GiB per GPU at TP 8): 8 B200 or 8 H200.
# Not fewer: 4 B200 hold 4 x 179 GiB, less than the weights, and TP 6 does
# not divide the 64 attention heads
SERVER_GPUS="${SERVER_GPUS:-8}"
# job 993477: weights loaded in 1,166 s from GPFS, ready after 1,771 s
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
