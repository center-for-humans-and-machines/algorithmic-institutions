# Model profile (common.sh MODEL_PROFILE): DeepSeek-V4.1-Flash-1017, MIT,
# ungated, 510 GB of weights (DeepseekV41ForCausalLM), pinned revision.
# vLLM supports it from v0.32; the group's image is v0.25.1, so it is served
# from the official v0.34.0 image converted by vllm_image.slurm.sh.
# Model card: temperature 1.0, top_p 0.95 (export_instance.LOCAL_MODELS),
# contexts up to 1M tokens with a KV cache of about 890 bytes per token
# (FP4 on B200, FP8 on H200), reasoning effort low/high/xhigh/max or 1-100.
MODEL=deepseek-ai/DeepSeek-V4.1-Flash-1017
MODEL_REV=08c8657f02291e11b6f6b63e0e16014f8e748518
VLLM_SIF="$PF_SCRATCH/images/vllm-v0.34.0.sif"
EXPECT_VLLM_TAG=vllm/vllm-openai:v0.34.0
TOOL_PARSER=deepseek_v41
REASONING_PARSER=deepseek_v41
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
MOE_BACKEND=auto
# 510 GB: 4 B200 hold it (about 128 GB per GPU); the campaign takes 8
SERVER_GPUS="${SERVER_GPUS:-8}"
