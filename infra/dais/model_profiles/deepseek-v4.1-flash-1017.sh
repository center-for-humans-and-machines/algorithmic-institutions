# Model profile (common.sh MODEL_PROFILE): DeepSeek-V4.1-Flash-1017, MIT,
# ungated, 510 GB of weights (DeepseekV41ForCausalLM), pinned revision.
# vLLM supports it from v0.32; the group's image is v0.25.1, so it is served
# from the official v0.34.0 image converted by vllm_image.slurm.sh.
# Its dated card (-1017, at this revision): temperature 1.0, top_p 1.0,
# max_tokens >= 256K, contexts up to 1M tokens (export_instance.LOCAL_MODELS);
# a KV cache of about 890 bytes per token (FP4 on B200, FP8 on H200);
# reasoning effort low/high/xhigh/max or 1-100, its agentic results at max.
MODEL=deepseek-ai/DeepSeek-V4.1-Flash-1017
MODEL_REV=08c8657f02291e11b6f6b63e0e16014f8e748518
MODEL_TAG=dsv41
VLLM_SIF="$PF_SCRATCH/images/vllm-v0.34.0.sif"
EXPECT_VLLM_TAG=vllm/vllm-openai:v0.34.0
TOOL_PARSER=deepseek_v41
REASONING_PARSER=deepseek_v41
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MOE_BACKEND=auto
# the maximum reasoning effort as the server's default for every chat request
# (the prompt encoder renders it as "Reasoning Effort: 100"), checked on the
# running server before any agent starts (PROMPT_MUST_CONTAIN, common.sh)
MODEL_VLLM_ARGS='--default-chat-template-kwargs {"reasoning_effort":"max"}'
PROMPT_MUST_CONTAIN="Reasoning Effort: 100"
# 510 GB: 4 B200 hold it (about 128 GB per GPU); the campaign takes 8
SERVER_GPUS="${SERVER_GPUS:-8}"
# loading took 526 s at TP 4 from GPFS (job 993167), then warm-up and CUDA
# graphs: give the server up to an hour to answer
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
