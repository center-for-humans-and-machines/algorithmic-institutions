# Model profile (common.sh MODEL_PROFILE): DeepSeek-V4.1-Flash-1017, MIT,
# ungated, 510 GB of weights (DeepseekV41ForCausalLM), pinned revision.
# vLLM supports it from v0.32; the group's image is v0.25.1, so it is served
# from the official v0.34.0 image converted by vllm_image.slurm.sh.
# Its dated card (-1017, at this revision): temperature 1.0, top_p 1.0,
# max_tokens >= 256K, contexts up to 1M tokens (export_instance.LOCAL_MODELS);
# a KV cache of about 890 bytes per token (FP4 on B200, FP8 on H200);
# reasoning effort low/high/max or 1-100 (reasoning_effort.sh below).
MODEL=deepseek-ai/DeepSeek-V4.1-Flash-1017
MODEL_REV=08c8657f02291e11b6f6b63e0e16014f8e748518
VLLM_SIF="$PF_SCRATCH/images/vllm-v0.34.0.sif"
EXPECT_VLLM_TAG=vllm/vllm-openai:v0.34.0
TOOL_PARSER=deepseek_v41
REASONING_PARSER=deepseek_v41
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MOE_BACKEND=auto
# the reasoning effort (reasoning_effort.sh): a submission's REASONING_EFFORT,
# by default max, the server's default for every chat request, checked on the
# running server before any agent starts. The card's names and the budgets
# vLLM v0.34.0's encoder (tokenizers/deepseek_v41_encoding.py) renders: low
# 50, high 75 (the model's own default), max 100, or an integer 1 to 100
# (vLLM's further alias xhigh is 75 as well, and none turns thinking off:
# neither is offered). The card's sampling and max_tokens are the same at
# every effort (LOCAL_MODELS).
MODEL_VLLM_ARGS=""
REASONING_EFFORT_DEFAULT=max
REASONING_EFFORT_LEVELS="low=50 high=75 max=100"
REASONING_EFFORT_RANGE="1 100"
REASONING_EFFORT_PROMPT="Reasoning Effort: {budget} (range 1-100"
# 510 GB: 4 B200 hold it (about 128 GB per GPU); the campaign takes 8
SERVER_GPUS="${SERVER_GPUS:-8}"
# loading took 526 s at TP 4 from GPFS (job 993167), then warm-up and CUDA
# graphs: give the server up to an hour to answer
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
