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
4. `apptainer exec --cleanenv` (dais adds `--nv --contain` and forwards `CUDA_VISIBLE_DEVICES`), environment passed as `APPTAINERENV_*`, per-job compile caches (vLLM, Triton, Inductor) in node-local temp, removed at the end. The job temp dir (container home, `TMPDIR`, per-job caches) is `CFG_JOB_TMP_BASE`, else `JOB_SHMTMPDIR`; dais sets `/tmp` (node-local xfs, 32G per job), because under `--contain` the container mounts its own `/dev` and a `--home` in the host `/dev/shm` cannot be mounted (the first dais smoke, job 878226, died on exactly this as `FAIL image imports`). The private `/dev/shm` of a `--contain` exec is a tmpfs of half the node's RAM shared by all processes of that exec, so tensor-parallel workers need no bind (`--bind /dev/shm` is skipped by apptainer as "already mounted"). On viper only `ROCR/HIP_VISIBLE_DEVICES` are forwarded and the host `rocm` module is loaded only in the GPU sampler subshell.
5. `vllm serve` with explicit `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization`, `--kv-cache-dtype`, `--enable-prefix-caching`, `--structured-outputs-config.backend xgrammar`, `--served-model-name`, the model's reasoning parser and flags.
6. Waits until `/v1/models` lists the exact model id, failing fast if the server dies or after `CFG_WAIT_MIN`.
7. Runs the command under `timeout` set to the job's end minus `CFG_WALL_BUFFER_MIN` (10 min viper, 15 min dais), and `--signal=B:USR1` at the same point, so the shutdown and the metric snapshots always happen.
8. Snapshots `/metrics` and `/version`, stops the server, writes `summary.json`, exits with the command's status.

## Measured (smoke tests, 2026-11-12)

All probe checks passed in both viper jobs: plain chat, top-level `response_format` json_schema (parsed and schema-valid, also 0 violations in 96 concurrent schema requests), thinking disabled (no `<think>` markers and an empty reasoning field; for the hybrid Qwen3-0.6B the qwen3 reasoning parser was on, so this is a real test), a 17k-token prompt within `--max-model-len 32768`.

| | viper, Qwen3-0.6B | viper, Qwen3-30B-A3B-Instruct-2507 | dais, Qwen3-30B-A3B-Instruct-2507 |
|---|---|---|---|
| job | 13075390, `apudev` | 13075553, `apudev` | 878637, `gpudev` |
| GPUs, TP | 1 MI300A, TP 1 | 2 MI300A, TP 2 | 1 B200, TP 1 |
| image | ROCm vLLM 0.30.0 | ROCm vLLM 0.30.0 | vLLM 0.32.0 (CUDA) |
| job start to server ready | 186 s | 512 s | 209 s |
| weight loading | 4 s | 272 s (28.5 GiB per GPU from `/ptmp/lumi`, about 0.2 GB/s) | 50 s (56.9 GiB from home) |
| engine init (profile, KV, warmup) | 66 s (compile 18 s) | 110 s (compile 28 s, graph capture 14-18 s) | 67 s (compile 30 s) |
| KV cache | 96.5 GiB, 903,440 tokens | 66.5 GiB per GPU, 1,451,824 tokens (44x at 32k) | 105.3 GiB, 1,150,064 tokens (35x at 32k) |
| 96 schema requests, 48 in flight, shared 3k prefix | 2.4 s, 40 req/s | 4.9 s, 19.8 req/s, prefix-cache hit 93% | 2.8 s, 34.2 req/s, prefix-cache hit 93% |
| decode, 128 x 384 tokens in flight | not run | 6.2 s, 7,952 output tok/s | 3.9 s, 12,645 output tok/s |
| GPU memory in use | 99 GiB | 103 GiB per APU | 165 GiB |
| whole job | 3 min 41 s | 9 min 26 s | 3 min 47 s |

The first dais smoke (job 878226) failed 2 s in: the job temp dir, and with it the container home, was in the host `/dev/shm`, which `--contain` hides (fixed with `CFG_JOB_TMP_BASE=/tmp`, see above). Job 878637 then passed every probe check in 3 min 47 s. A second smoke at TP 2 on `gpudev` (job 878812) was cancelled unrun because `gpudev` had no free B200 pair for hours; the DeepSeek first boot (below) is the TP 2 test under the production container options. Results of both: `runs/annotation/smoke-dais/` (also on S3).

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

**Weights.** Downloaded on the dais login node into `/dais/fs/scratch/levinb/hf_cache/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017` on 2026-11-12 with `download_model.sh` (pinned revision, `snapshot_download` through `hf_xet` inside `glm_52_vllm.sif`): 167 s for the download, 140 s to check every file's size and LFS sha256 against the Hub, 84 of 84 files verified. Check with `scripts/annotation/cluster/download_status.sh` (state files under `<hf_cache>/annot_downloads/<repo>/<sha>/`). The verification result is `<hf_cache>/annot_downloads/deepseek-ai--DeepSeek-V4.1-Flash-1017/<sha>/VERIFIED` ("84 files ok", four levels below `hf_cache`), the file `submit.sh` requires. Scratch use went from 1.95 TB to 2.42 TB of 8 TB. `du -sh` shows 951G against an apparent 476 GiB (510.3 GB, `du --apparent-size`): every blob has twice its size in allocated blocks (a 101.5 GB shard occupies 203 GB), with 84 blobs for 84 files, no duplicates and no hard links, which is GPFS keeping two data replicas; the quota counts the data once.

**vLLM version and image.** The architecture (`DeepseekV41ForCausalLM`) arrived in vLLM 0.30.0 and the vLLM recipe requires 0.31.0 or later (KV-offload fix vllm#57145, Engram host-memory gate vllm#57491). The installed `glm_52_vllm.sif` is vLLM 0.23.0 and cannot serve it. Pulled: `vllm/vllm-openai:v0.32.0-x86_64` by digest `sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9` (released 2026-10-18) into `/dais/fs/scratch/levinb/containers/vllm-openai_v0.32.0.sif` with `pull_vllm_v0.32.0.sh` in the same folder (log `pull_vllm_v0.32.0.log`, writes `vllm-openai_v0.32.0.sif.sha256` when done). The pull finished at 16:30; the image (8,467,496,960 bytes, sha256 `ef45566c...1848c`, rechecked against the sidecar, now read-only) is listed in `MANIFEST.tsv` with torch 2.13.0+cu130, transformers 5.19.0, vLLM 0.32.0. v0.33.0 (2026-11-02) exists but is newer than two weeks.

**Serve flags** (in `config.sh`, profile `deepseek-ai/DeepSeek-V4.1-Flash-1017`; source: the vLLM recipe at recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash for `hardware=b200, strategy=single_node_tp, features=reasoning,text_only`, which treats the 1017 checkpoint as the default):

- `--tokenizer-mode deepseek_v41`: the prompt encoding (the repo now also ships a `chat_template.jinja` ported from `encoding/encoding.py`, but the recipe uses the tokenizer mode)
- `--reasoning-parser deepseek_v41`
- `--attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'`
- `--kv-cache-dtype fp8`, `--max-num-seqs 128` (recipe value, also our default)
- `--engram-config '{"cpu_offload":true}'`: the Engram tables (183 GiB) live in pinned host memory and are read through UVA, output unchanged
- `--language-model-only`: skips the vision tower, frees its memory for KV cache
- no `--trust-remote-code`: the architecture is native in vLLM
- env `VLLM_ENGINE_READY_TIMEOUT_S` (recipe 3600; set to 14400, above `CFG_WAIT_MIN=200`). The recipe also sets `VLLM_USE_RUST_FRONTEND=1`; it is left off here until `response_format` and `chat_template_kwargs` are proven on it (`CFG_SERVE_ENV` turns it on)
- plus the common flags: `--max-model-len 32768`, `--enable-prefix-caching`, `--structured-outputs-config.backend xgrammar`, `--gpu-memory-utilization 0.92`
- weights are already quantised (MXFP4 experts, MXFP8 rest); no `--quantization` or `--dtype` flag
- thinking off per request: `chat_template_kwargs: {"thinking": false}` (or `enable_thinking: false`; if both are sent they must agree). With both unset, thinking is ON at effort 50, so a small `max_tokens` returns empty content with `finish_reason=length`. The pipeline config already sends `thinking: false` for DeepSeek. Recommended sampling is temperature 1.0, top_p 1.0 (model card); the pipeline uses temperature 0.

**GPU count: 2 B200 (TP 2), on `gpu1` with 24 CPUs and 480G.** Supported TP sizes in the recipe: 2, 4, 8 (64 heads, powers of two). TP 1 cannot fit: resident weights without the Engram tables are about 476 - 183 = 293 GiB, more than one B200 (183 GB = about 179 GiB). At TP 2 each rank holds about 145 GiB (the recipe measures "roughly 145 GiB per rank" at TP 2 with offload on MI300X). At `--gpu-memory-utilization 0.92` a rank may use 0.92 x 179 = 165 GiB, which leaves about 20 GiB for activations, CUDA graphs and KV cache, so on the order of 10-15 GiB of KV per rank. The model's global KV is about 890 bytes per token plus a fixed 128-token window per layer (recipe), so even 128 concurrent 14k-token sequences (12k prompt + 2k output, 1.8M tokens) need under 2 GB of global KV. The recipe's Blackwell default is TP 2 with Engram offload for this reason. Host memory: 183 GiB of pinned Engram tables plus loader buffers; `gpu1` allows 240G per GPU, so 480G (Slurm's G is GiB: 480 GiB) for TP 2, against about 430 GiB in use on a DGX Station that offloads 249 GiB. If the first load fails on host memory or KV, the fallback is TP 4 (`CFG_TP=4 CFG_GPUS=4`, 960G, still `gpu1`), which the recipe recommends only for latency.

**First load (job 878977, 2026-11-12).** The compile caches for this model persist across jobs in `<project>/compile_cache/<image>__<model>__tp<N>/` under a flock (`CFG_PERSIST_COMPILE_CACHE=1`); the cache gets a `COMPLETE` marker once the server is ready. The first boot was planned with `CFG_WAIT_MIN=200` and a 5 h limit, from a recipe autotune of 75-90 min on GB300; it took 22 min.

## Measured: DeepSeek-V4.1-Flash-1017 on 2 B200 (job 878977)

`gpu1`, 2 B200 at TP 2, 24 CPUs, 480G, vLLM 0.32.0 image, the serve flags above, `--contain`. Command: pipeline preflight, `probe.py`, `stage A --run dais0 --limit 6`. Queued 18:23, started 19:05:55 (`gpu1` had about 900 jobs pending; the estimate swung between 20:51 the same day and 11:00 the next), whole job 23 min 16 s, rc 0. Logs and outputs: `runs/annotation/dais0/` (also on S3).

Boot timeline (from the server start at 19:06:16):

- +0:50 Engram configuration resolved (`cpu_offload=True`); +1:41 to +2:00 the two Engram tables pinned in host memory, 47.21 GiB per table per rank (188.8 GiB in all)
- +11:00 weights loaded: 536.6 s for 510 GB (0.95 GB/s from scratch); model loading 586 s, 140.28 GiB per GPU
- +12:44 profiling and the first graph capture (70 s) done; KV cache 18.95 GiB per GPU, 1,538,376 tokens, 47x at 32k
- +13:32 to +21:12 FlashInfer autotune (21 token buckets up to 8192), 147 configs saved into the persistent cache; some `mxfp8_gemm` shapes have no tuned config and fall back to the CuteDSL runner (warnings only)
- +21:35 server ready (engine init 625 s); 21.6 min in all, 22 min from job start

Memory: GPU 166-169 GiB of 179 GiB in use per GPU, on the two GPUs Slurm assigned (`SLURM_JOB_GPUS=2,3`, inside the job CUDA 0,1; `nvidia-smi` in the job shows the two TP workers on exactly those bus ids). Host: the job cgroup sat at its 480 GiB limit (1.04M reclaim events, no OOM), but only about 278 GiB of it was non-reclaimable (shmem 264 GiB, which holds the pinned Engram tables, anon 11 GiB, kernel 2.4 GiB); the rest was page cache from the weight read. Slurm's `MaxRSS` (503 GB) counts that cache and overstates the need. 480G leaves about 200 GiB of headroom.

Answers from the probe and the trial:

- top-level `response_format` json_schema works with `--tokenizer-mode deepseek_v41` plus `--reasoning-parser deepseek_v41`: parsed and schema-valid; 96 concurrent schema requests with 0 violations and 0 `length` finishes
- `chat_template_kwargs {"thinking": false}` alone (what the pipeline sends, `llm.extra_body` matched on `DeepSeek`) suppresses reasoning: the pipeline preflight passed all 5 schemas with no reasoning leak, and the probe found 0 reasoning characters with thinking off against 287 in its thinking-on control, so the parser works and the switch is real
- stage A on 6 versions: 45 calls, all `stop`, 0 failed, 0 `<think>` strips; open-rules 27 free codes, what-codebook 5 codes, coverage 1 candidate, why-codebook 5 codes; longest open-notes call 5.2k prompt and 2.6k completion tokens in 19.6 s
- throughput: 96 schema requests at 48 in flight with a shared 3k prefix in 9.3 s (10.3 req/s, about 3.6x slower than Qwen3-30B on one B200); decode 128 x 384 tokens in 6.1 s, 8,070 output tok/s; generation over the whole command peaked at 3,489 tok/s in a 15 s window; prefix-cache hit rate 86%; 0 preemptions; KV cache use never above 1.1%
- a 14.6k-token prompt fits (`--max-model-len 32768`), 2.2 s

**Second boot (job 879378, probe only, started 19:39 on other GPUs of another node) was not faster: 21.5 min to ready** (weights 542.6 s, engine init 618 s against 625 s). The 147 FlashInfer autotune configs were loaded on both ranks ("Loaded 147 configs"), yet the autotune ran in full again (19:52:34 to 20:00:40, 8.1 min) and rewrote the same 147 entries, so the runtime lookups miss the stored keys (the "No tuned config covers mxfp8_gemm" warnings point the same way); the TileLang JIT warmup (48 s) also recompiles each boot because its cache sits in the per-job container home. What the persistent cache did help: the DeepGEMM kernels JIT-compiled during the first job's requests (`vllm/deep_gemm/cache`, written at 19:29) were reused, and the probe's 96 schema requests ran at 66 req/s (1.45 s) against 10.3 req/s in the first job, whose first requests paid those compiles (the JIT monitor flagged Triton compiles during inference there); decode 8,094 tok/s, prefix-cache hit rate 93%, probe all ok. Untested ways to cut the 8 min: `--kernel-config '{"enable_flashinfer_autotune": false}'` (heuristic kernels, may cost throughput) or `VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS=mxfp8_gemm`; with a 22 min boot against a stage of well under an hour, it is not worth a GPU job before the production run.

**Recommended production settings** (stage A, then stage B, each one job): 2 B200 at TP 2 on `gpu1` with 24 CPUs and 480G (the `config.sh` defaults; TP 4 is not needed); `CFG_WAIT_MIN=60` suffices (every boot takes about 22 min; 200 only wastes time if a boot hangs); `CFG_TIME=03:00:00` per stage (22 min boot, a full stage A is 110 versions at about 10 schema requests per second, and the 15 min wall buffer); pipeline concurrency 64 (`llm.concurrency`), `--max-num-seqs 128`; `--max-model-len 32768` (the KV cache holds 1.54M tokens, so 65536 as on viper would also fit 64 requests of 16k if proposal prompts need the room). Queue time on `gpu1` is the real cost: submit early.

**Remaining before the production run:**

1. Full stage A as one job once the prompts are settled on viper, then the codebook review, then stage B.
2. Optional: try `VLLM_USE_RUST_FRONTEND=1` in a second probe and keep it if all checks pass.
3. Second judge `zai-org/GLM-5.3`: not downloaded, no profile in `config.sh` yet.
