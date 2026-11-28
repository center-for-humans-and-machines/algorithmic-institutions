# Model profile (common.sh MODEL_PROFILE): Tencent Hy4, FP8 checkpoint
# (tencent/Hy4-FP8), Tencent Hy License 1.0 (Apache 2.0 with a separate
# licence only for MaaS businesses above US$50M revenue or 100M MAU; internal
# use exempt), ungated, 786 GB of block-FP8 weights (about 1.57 TB of GPFS
# quota), pinned revision. 770B MoE, 49B active, gated sparse MLA (DSA with
# IndexCache), plus a 1.23B vision encoder we do not use (it loads anyway,
# about 2.5 GB per GPU) and the built-in DFly2 drafter (not enabled here).
#
# vLLM: no release serves it. The checkpoint is HYV4VLForConditionalGeneration,
# which v0.34.0 does not register (vLLM PR #68296, 2026-11-19); the recipe
# (recipes/models/tencent/Hy4.yaml) names 0.35.0 as the first release. The
# model card's vllm/vllm-openai:hy4 image (2026-11-20 03:08) predates the
# streaming fix of the hy_v4 reasoning parser (#68689, 2026-11-20 14:09), and
# OpenCode streams. So: the nightly at commit 70b67cf8 (2026-11-28), built by
# vllm_image.slurm.sh (VLLM_TAG=nightly-<sha>). Move to v0.35.0 once released.
#
# Model card: temperature 0.9, top_p 1.0 (export_instance.LOCAL_MODELS);
# reasoning_effort high (default) or no_think; 1,048,576 positions natively,
# served at 262,144. B200 only (8 x 183 GB; the recipe needs 944 GB): about
# 100 GB of weights per GPU at TP 8. H200 is untested and had a FlashMLA fix
# for SM90 after the hy4 image.
MODEL=tencent/Hy4-FP8
MODEL_REV=78037597c0ed338334957bd94f881f6853cf5dc5
VLLM_SIF="$PF_SCRATCH/images/vllm-nightly-70b67cf80585b671b57ea748232ff6c68e7ba4d7.sif"
# the image's own tag label (assert_vllm_image reads it first); vLLM reports
# itself as 0.34.1rc1.dev319+g70b67cf80 (job 993433), SIF sha256
# 1a78e3218e2881111a948da63196bdeec0dcd43feb2f3048fa07c59bbc0db6c1
EXPECT_VLLM_TAG=vllm/vllm-openai:nightly-70b67cf80585b671b57ea748232ff6c68e7ba4d7
TOOL_PARSER=hy_v4
REASONING_PARSER=hy_v4
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MOE_BACKEND=auto
# 786 GB: only all 8 GPUs of a node hold it with room for the KV cache
SERVER_GPUS="${SERVER_GPUS:-8}"
# 786 GB from GPFS (DeepSeek's 510 GB took 526 s at TP 4, job 993167), then
# warm-up and CUDA graphs: give the server up to an hour to answer
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
