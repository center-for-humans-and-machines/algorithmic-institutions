# Annotation inference on viper and dais

How the step-classification pipeline (`docs/plans/step-classification-pipeline.md`) gets its LLM: one Slurm job starts a vLLM server, waits until it serves the model, runs a pipeline command against it inside the same container image, records server and GPU metrics, and stops. viper (2x AMD MI300A) is for development, dais (B200) for the production model. Scripts: `scripts/annotation/cluster/`.

## Files

| file | what |
|---|---|
| `config.sh` | the one place for per-cluster and per-model defaults: project dir, image and its pinned vLLM version, model and revision, tensor-parallel size, GPU count, `--max-model-len`, `--max-num-seqs`, GPU memory share, KV dtype, reasoning parser, model flags, partition, time, CPUs, memory |
| `common.sh` | job side: preflight, model link, server start, readiness wait, samplers, command, shutdown |
| `serve_and_run.viper.sbatch`, `serve_and_run.dais.sbatch` | thin per-cluster job scripts (device flags, GPU sampler) |
| `sync_code.sh <cluster> [run_id]` | rsync the annotation code (and a run dir) to the cluster |
| `submit.sh <cluster> <run_id> "<command>" [CFG_KEY=VALUE ...]` | sync, resolve the config on the login node, `sbatch` |
| `fetch_run.sh <cluster> <run_id> [--no-cache]` | rsync the run dir back to `runs/annotation/<run_id>/` |
| `probe.py` | preflight check of an endpoint (chat, JSON schema, thinking off, 12k prompt, throughput) |
| `metrics_summary.py` | turns a job's log dir into `summary.json` |
| `download_model.sh`, `download_status.sh` | pinned weight download on a login node, and its progress |
| `../s3_sync.py push\|pull\|ls <run_id>` | run dir to and from S3 |

## Contract between job and pipeline

The job runs the command given to `submit.sh` with `bash -c` inside the serving image, with the working directory at the project dir and:

- `PYTHONPATH=<code>/src`, where `<code>` is the per-job snapshot `<run>/jobs/<stamp>.code/` that `submit.sh` copies from the synced project dir (package, `configs/annotation`, `scripts/annotation`, `CODE_VERSION`), so a later sync never changes a queued or running job; `ANNOT_CODE_DIR=<code>`
- `ANNOT_MAX_MODEL_LEN=<--max-model-len>` (the pipeline cuts proposal prompts to fit it)
- `ANNOT_BASE_URL=http://127.0.0.1:<port>/v1` (port unique per job)
- `ANNOT_MODEL=<served model id>` (the Hugging Face id, e.g. `Qwen/Qwen3-30B-A3B-Instruct-2507`)
- `ANNOT_RUNS_ROOT=<project>/runs/annotation`, `ANNOT_RUN_ID`, `ANNOT_JOB_LOG_DIR`

Unless `CFG_PIPELINE_PREFLIGHT=0`, the job runs `python -m aimanager.annotation preflight --run <run_id>` first and the command only if every schema passes, so a submitted command need not include it. `python` resolves to the image's `python3` (the CUDA image has no `python`). All three images in use have `pydantic` 2, `yaml`, `openai` and `httpx` (checked at every job start). The job exits with the command's status; judge a stage by the rows it wrote, not by `COMPLETED`.

## Workflow

```bash
# 0. on the Mac, in this checkout: corpus etc. under runs/annotation/<run_id>/
# 1. sync code + run dir and submit (sync is part of submit.sh)
scripts/annotation/cluster/submit.sh viper r1 "python -m aimanager.annotation stage A --run r1"
#    overrides: any CFG_* from config.sh, plus --dry-run (sbatch --test-only, no sync), --dependency=afterok:<id>
scripts/annotation/cluster/submit.sh viper r1 "python -m aimanager.annotation stage A --run r1" \
    CFG_TIME=04:00:00 CFG_MAX_NUM_SEQS=64
# 2. monitor
ssh viper squeue -u levinb
ssh viper tail -f /ptmp/levinb/algorithmic-institutions-annot/runs/annotation/r1/logs/slurm-<jobid>.out
# 3. fetch (never deletes local files; --no-cache skips the response cache)
scripts/annotation/cluster/fetch_run.sh viper r1
# 4. archive to s3://chm-algorithmic-institutions/policy_finder/step_classification/r1/
python3 scripts/annotation/s3_sync.py push r1            # cache/ left out unless --include-cache
python3 scripts/annotation/s3_sync.py ls r1
python3 scripts/annotation/s3_sync.py pull r1            # on another machine
```

Preflight before a real stage: `submit.sh <cluster> <run_id> "python scripts/annotation/cluster/probe.py"` with the production settings. On viper the `apudev` partition (15 min, `CFG_PARTITION=apudev CFG_TIME=00:15:00 CFG_WALL_BUFFER_MIN=2`) fits even the 30B model; on dais `gpudev` (15 min, at most 4 GPUs, one job per user) fits a small model, not the production one.

Every job leaves in `<run>/logs/job-<jobid>/`: `CODE_VERSION` (commit synced, `dirty` if uncommitted changes), `timeline.txt` (job start, server start, ready, command start/end), `job.env` (resolved config), `serve_cmd.txt`, `vllm.log`, `startup_lines.txt` (weight load, KV cache size, compile, graph capture), `metrics_start.prom`, `metrics_end.prom`, `metrics_samples.txt` (every 15 s: KV cache use, running and waiting requests incl. grammar-deferred, prompt and generation token counters, prefix-cache hits, preemptions), `gpu_samples.txt` (rocm-smi or nvidia-smi), `version.json`, `command.log`, `summary.json`. Slurm's own log is `<run>/logs/slurm-<jobid>.out`.

## What the job does

1. Sources the job env file written by `submit.sh` (`<run>/jobs/<stamp>.env`: every resolved `CFG_*`, the run id and the command; values with commas or quotes survive, and the resolved command is logged first).
2. Preflight before loading anything: image readable, `vllm.__version__` equals the pinned `CFG_VLLM_VERSION`, pipeline imports work, the pinned snapshot exists and has no dangling links (purged blobs are caught here).
3. Writable `HF_HOME` in the project dir whose model dir is a symlink into a read-only cache (`CFG_HF_SOURCES`), so weights are never copied; `HF_HUB_OFFLINE=1`, `--revision <sha>` always (no reliance on `refs/main`).
4. `apptainer exec --cleanenv` (dais adds `--nv --contain`), environment passed as `APPTAINERENV_*`, per-job compile caches (vLLM, Triton, Inductor) in node-local temp, removed at the end. On viper only `ROCR/HIP_VISIBLE_DEVICES` are forwarded and the host `rocm` module is loaded only in the GPU sampler subshell.
5. `vllm serve` with explicit `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization`, `--kv-cache-dtype`, `--enable-prefix-caching`, `--structured-outputs-config.backend xgrammar`, `--served-model-name`, the model's reasoning parser and flags.
6. Waits until `/v1/models` lists the exact model id, failing fast if the server dies or after `CFG_WAIT_MIN`.
7. Runs the command under `timeout` set to the job's end minus `CFG_WALL_BUFFER_MIN` (10 min viper, 15 min dais), and `--signal=B:USR1` at the same point, so the shutdown and the metric snapshots always happen.
8. Snapshots `/metrics` and `/version`, stops the server, writes `summary.json`, exits with the command's status.

## Measured (smoke tests, 2026-11-12)

All probe checks passed in both viper jobs: plain chat, top-level `response_format` json_schema (parsed and schema-valid, also 0 violations in 96 concurrent schema requests), thinking disabled (no `<think>` markers and an empty reasoning field; for the hybrid Qwen3-0.6B the qwen3 reasoning parser was on, so this is a real test), a 17k-token prompt within `--max-model-len 32768`.

| | viper, Qwen3-0.6B | viper, Qwen3-30B-A3B-Instruct-2507 | dais, Qwen3-30B-A3B-Instruct-2507 |
|---|---|---|---|
| job | 13075390, `apudev` | 13075553, `apudev` | 878226, `gpudev` (queued) |
| GPUs, TP | 1 MI300A, TP 1 | 2 MI300A, TP 2 | 1 B200, TP 1 |
| image | ROCm vLLM 0.30.0 | ROCm vLLM 0.30.0 | vLLM 0.32.0 (CUDA) |
| job start to server ready | 186 s | 512 s | pending |
| weight loading | 4 s | 272 s (28.5 GiB per GPU from `/ptmp/lumi`, about 0.2 GB/s) | pending |
| engine init (profile, KV, warmup) | 66 s (compile 18 s) | 110 s (compile 28 s, graph capture 14-18 s) | pending |
| KV cache | 96.5 GiB, 903,440 tokens | 66.5 GiB per GPU, 1,451,824 tokens (44x at 32k) | pending |
| 96 schema requests, 48 in flight, shared 3k prefix | 2.4 s, 40 req/s | 4.9 s, 19.8 req/s, prefix-cache hit 93% | pending |
| decode, 128 x 384 tokens in flight | not run | 6.2 s, 7,952 output tok/s | pending |
| GPU memory in use | 99 GiB | 103 GiB per APU | pending |
| whole job | 3 min 41 s | 9 min 26 s | pending |

The dais smoke (job 878226: `serve_and_run.dais.sbatch`, vLLM 0.32.0 image, Qwen3-30B-A3B-Instruct-2507 read through a symlink into `~/.cache/huggingface` on one B200, the same probe) was submitted at 16:22 with an estimated start of 17:25. Its results land in `/dais/fs/scratch/levinb/algorithmic-institutions-annot/runs/annotation/smoke-dais/logs/job-878226/` (`probe.json`, `summary.json`); fetch them with `scripts/annotation/cluster/fetch_run.sh dais smoke-dais`. The submission itself passed `sbatch --test-only`, as did the production request (2x B200 on `gpu1`, 24 CPUs, 480G, estimated start the next morning).

What this means for the pipeline: on viper the 30B model spends 8.5 min loading, 5 min of it reading weights, so a stage should be one job against one server. The KV cache is far larger than the workload needs (110 documents per pass, prompts up to about 12k tokens): at 128 sequences of 14k tokens it would be 1.8M tokens, above the 1.45M of TP 2, and the pipeline keeps at most 128 requests in flight (`llm.concurrency` in `configs/annotation/default.yml`, = `--max-num-seqs`; at 64 the server never queued a request in viper1/viper2, peak KV use 24%). It submits the longest prompts of a pass first, so the slowest calls do not start last.

## Storage and retention

| what | where | retention |
|---|---|---|
| viper project dir (code, runs, HF_HOME links) | `/ptmp/levinb/algorithmic-institutions-annot` | `/ptmp` (`/viper/ptmp2`): files not accessed for 12 weeks are removed automatically, no backup. This already happened: the blobs of every model in `/ptmp/levinb/hf_cache` were gone (dangling snapshot links); only `Qwen/Qwen3-0.6B` was re-downloaded there |
| viper weights | `/ptmp/lumi/.llmis/huggingface` (another user's cache, read-only) | same 12-week rule, outside our control |
| viper image | `/ptmp/lumi/.llmis/vllm_vllm-openai-rocm_v0.30.0.sif` (another user's) | same; a copy of our own is advisable before production use on viper |
| dais project dir | `/dais/fs/scratch/levinb/algorithmic-institutions-annot` | dais scratch: no backup, quota 8 TB and 8M files per user; the DAIS user guide (docs.mpcdf.mpg.de/doc/computing/dais-user-guide.html) names no automatic purge for it, but says the filesystem details "are not yet settled and are thus subject to change" |
| dais weights | `/dais/fs/scratch/levinb/hf_cache` | as above |
| dais images | `/dais/fs/scratch/levinb/containers` (`MANIFEST.tsv` lists name, sha256, size, versions) | as above |
| dais home | `/u/levinb` (1 TB quota, 637 GB used) | nothing large goes here; the dais smoke test only reads the Qwen weights that already were there |
| results | `s3://chm-algorithmic-institutions/policy_finder/step_classification/<run_id>/` | durable copy; push after every stage |

## Production model on dais

Model `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `08c8657f02291e11b6f6b63e0e16014f8e748518` (MIT licence, not gated). The repo is 510.3 GB (476 GiB) in 84 files: routed and DSpark experts in MXFP4 (259.5 GiB), the two Engram n-gram tables in FP8 (183.1 GiB), attention and dense parts in FP8 (6.9 GiB), embeddings and LM head in BF16, UE8M0 block scales (21.9 GiB). 40 layers, 64 attention heads, 384 routed experts (6 active), vision tower included.

**Weights.** Downloaded on the dais login node into `/dais/fs/scratch/levinb/hf_cache/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017` on 2026-11-12 with `download_model.sh` (pinned revision, `snapshot_download` through `hf_xet` inside `glm_52_vllm.sif`): 167 s for the download, 140 s to check every file's size and LFS sha256 against the Hub, 84 of 84 files verified. Check with `scripts/annotation/cluster/download_status.sh` (state files under `<hf_cache>/annot_downloads/<repo>/<sha>/`). Scratch use went from 1.95 TB to 2.42 TB of 8 TB (`du` shows twice the apparent size on this GPFS; the quota counts it once).

**vLLM version and image.** The architecture (`DeepseekV41ForCausalLM`) arrived in vLLM 0.30.0 and the vLLM recipe requires 0.31.0 or later (KV-offload fix vllm#57145, Engram host-memory gate vllm#57491). The installed `glm_52_vllm.sif` is vLLM 0.23.0 and cannot serve it. Pulled: `vllm/vllm-openai:v0.32.0-x86_64` by digest `sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9` (released 2026-10-18) into `/dais/fs/scratch/levinb/containers/vllm-openai_v0.32.0.sif` with `pull_vllm_v0.32.0.sh` in the same folder (log `pull_vllm_v0.32.0.log`, writes `vllm-openai_v0.32.0.sif.sha256` when done). The pull was at "Creating SIF file" when cluster access dropped; its `MANIFEST.tsv` line (name, path, sha256, bytes, mtime, source `docker://vllm/vllm-openai@sha256:6ee1...`, torch/transformers/vllm versions) is still to be added. v0.33.0 (2026-11-02) exists but is newer than two weeks.

**Serve flags** (in `config.sh`, profile `deepseek-ai/DeepSeek-V4.1-Flash-1017`; source: the vLLM recipe at recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash for `hardware=b200, strategy=single_node_tp, features=reasoning,text_only`, which treats the 1017 checkpoint as the default):

- `--tokenizer-mode deepseek_v41`: the prompt encoding (the repo now also ships a `chat_template.jinja` ported from `encoding/encoding.py`, but the recipe uses the tokenizer mode)
- `--reasoning-parser deepseek_v41`
- `--attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'`
- `--kv-cache-dtype fp8`, `--max-num-seqs 128` (recipe value, also our default)
- `--engram-config '{"cpu_offload":true}'`: the Engram tables (183 GiB) live in pinned host memory and are read through UVA, output unchanged
- `--language-model-only`: skips the vision tower, frees its memory for KV cache
- no `--trust-remote-code`: the architecture is native in vLLM
- env `VLLM_ENGINE_READY_TIMEOUT_S` (recipe 3600; set to 7200). The recipe also sets `VLLM_USE_RUST_FRONTEND=1`; it is left off here until `response_format` and `chat_template_kwargs` are proven on it (`CFG_SERVE_ENV` turns it on)
- plus the common flags: `--max-model-len 32768`, `--enable-prefix-caching`, `--structured-outputs-config.backend xgrammar`, `--gpu-memory-utilization 0.92`
- weights are already quantised (MXFP4 experts, MXFP8 rest); no `--quantization` or `--dtype` flag
- thinking off per request: `chat_template_kwargs: {"thinking": false}` (or `enable_thinking: false`; if both are sent they must agree). With both unset, thinking is ON at effort 50, so a small `max_tokens` returns empty content with `finish_reason=length`. The pipeline config already sends `thinking: false` for DeepSeek. Recommended sampling is temperature 1.0, top_p 1.0 (model card); the pipeline uses temperature 0.

**GPU count: 2 B200 (TP 2), on `gpu1` with 24 CPUs and 480G.** Supported TP sizes in the recipe: 2, 4, 8 (64 heads, powers of two). TP 1 cannot fit: resident weights without the Engram tables are about 476 - 183 = 293 GiB, more than one B200 (183 GB = about 179 GiB). At TP 2 each rank holds about 145 GiB (the recipe measures "roughly 145 GiB per rank" at TP 2 with offload on MI300X). At `--gpu-memory-utilization 0.92` a rank may use 0.92 x 179 = 165 GiB, which leaves about 20 GiB for activations, CUDA graphs and KV cache, so on the order of 10-15 GiB of KV per rank. The model's global KV is about 890 bytes per token plus a fixed 128-token window per layer (recipe), so even 128 concurrent 14k-token sequences (12k prompt + 2k output, 1.8M tokens) need under 2 GB of global KV. The recipe's Blackwell default is TP 2 with Engram offload for this reason. Host memory: 183 GiB of pinned Engram tables plus loader buffers; `gpu1` allows 240G per GPU, so 480G (Slurm's G is GiB: 480 GiB) for TP 2, against about 430 GiB in use on a DGX Station that offloads 249 GiB. If the first load fails on host memory or KV, the fallback is TP 4 (`CFG_TP=4 CFG_GPUS=4`, 960G, still `gpu1`), which the recipe recommends only for latency.

**First load time is unknown.** 510 GB from scratch (this GPFS read the model at about 3.6 GB/s for the checksum) plus Engram pinning plus a FlashInfer autotune that the recipe reports at 75-90 min on a first boot of a new flag set (GB300). So `CFG_WAIT_MIN=120` and the compile caches for this model persist across jobs in `<project>/compile_cache/<image>__<model>__tp<N>/` under a flock (`CFG_PERSIST_COMPILE_CACHE=1`).

**Remaining before the production run:**

1. One `gpu1` job with the probe on the real model: `submit.sh dais prod-preflight "python scripts/annotation/cluster/probe.py" CFG_TIME=03:00:00`. It records load time, KV size, whether TP 2 fits, and whether `response_format` works with `--tokenizer-mode deepseek_v41` plus the reasoning parser (not tested anywhere yet). The first boot fills the persistent compile cache.
2. If that works, stage A as one job (`CFG_TIME` sized from the measured load time plus the work), then stage B.
3. Optional: try `VLLM_USE_RUST_FRONTEND=1` in a second probe and keep it if all checks pass.
4. Second judge `zai-org/GLM-5.3`: not downloaded, no profile in `config.sh` yet.
