# DAIS feasibility for the open-weight policy finder

Phase 0 (DAIS part) and the first half of phase 2 of `docs/plans/policy-finder-open-weights.md`, measured on 2026-11-27. Scripts: `infra/dais/`. All jobs ran under account `mpib_inst`, user `levinb`.

## Summary

- **The simulations should run on the CPUs, not the GPUs.** The sim image `algoinst.sif` (byte-identical to Raven's) has no kernels for H200 (`no kernel image is available for execution on the device`: torch 1.11+cu113 ships sm_37 to sm_86 and no PTX; H200 is sm_90, B200 sm_100). On the CPUs the same unchanged image runs the repo's `python -m aimanager simulate` fast: a sweep-shaped sim (16 points x 500 episodes, batches of 1000) takes 62 to 69 s on 32 hardware threads, and a 2000-episode, 6-point final run of `pf-20261104-1` reproduces Raven's A100 result within noise (chi-squared 7.3 on 6 degrees of freedom).
- **Qwen3-Coder-Next serves on 2 H200 with the group's `vllm.sif` (v0.25.1), unchanged.** Ready in 281 to 386 s, tool calls well formed (plain, streamed, after an 83k-token context), 5 concurrent sessions at 82 to 133 tokens/s each.
- **So decision 1 becomes simpler:** vLLM gets the GPUs, the sims get the CPUs of the same node, and no GPU memory has to be held back. A sim next to 5 generating sessions was 1.6x slower (108 s against 62 to 69 s); generation was not slowed at all.
- **The agent's network cannot be cut with Apptainer on DAIS** (network namespaces are disabled for users), and there is no Docker-compatible runtime, so the spawner cannot run unchanged. Decision 4's fallback applies. A Unix-socket path to the model server works, and file isolation (binds, read-only image, fake home) works.

## 1. Cluster facts

| Item | Finding | Evidence |
|---|---|---|
| Partitions | `gpu` is `OverSubscribe=EXCLUSIVE`: any job there gets a whole node, whatever it asks for. `gpu1` is the shared partition (docs: for jobs with fewer than 4 GPUs; more than half a node triggers a full-node allocation). `gpudev`: shared, 15 min, at most 4 GPUs, one job per user. `small`: CPU only, on the RTX nodes, 6 h, half a node. `gpux`: 10 days, 2 H200 nodes. | `scontrol show partition`, DAIS user guide |
| GPU type | `--gres=gpu:h200:N` selects H200 (every job here landed on H200). Without a type, Slurm picks any node. | jobs 986457, 986513, 986771, 986879 |
| QoS | `g0001` to `g0008` carry no TRES limit in `sacctmgr`; the GPU count comes from `--gres`. A full node is `--partition=gpu --gres=gpu:h200:8 --qos=g0008` (not tested: no full-node jobs in this phase). By the exclusive setting, the sister project's 2-GPU jobs on `gpu` would have held whole nodes (not checked in their accounting). | `sacctmgr show qos`, `show assoc` |
| Nodes | H200 nodes: 8 x H200 (143,771 MiB each), 192 threads, 2 TB RAM; driver 580.196.07. | `nvidia-smi` in jobs |
| Network | Login node and compute nodes reach `huggingface.co`, `openrouter.ai`, `pypi.org`, `github.com` (HTTP 200). | job 986457, 986771 |
| Network cut | Refused: `apptainer exec --net --network none` fails with `Failed to create network namespace: maximum number of network namespaces exceeded` (`/proc/sys/user/max_net_namespaces` = 0) on the login node and in jobs. `unshare -rn` fails too. | login node, job 986771 |
| Scratch | Quota 8 TB and 8M files (2.65 TB used, mostly older data); no backup. From 2027-02-01, files not accessed for 12 weeks are deleted automatically (login message). GPFS: `du` shows about twice the apparent size (297 GB for 149 GB of weights); whether the quota counts the doubled figure is unclear. | login message, `mmlsquota`, `du` |
| Home | 1 TB and 500k files. | user guide |
| Weight downloads | No site rule found against downloads on the login node (login nodes have per-user CPU/memory limits); earlier downloads in this scratch ran there too. The 149 GB download took 5 minutes with 4 workers, niced. | `download_qwen3_coder_next.log` |

### Apptainer as the agent sandbox (all as an unprivileged user, apptainer 1.5.2)

| Test | Result |
|---|---|
| `--containall --net --network none` | Refused (see above). |
| Without a network namespace | The container shares the host network: services on the host's `127.0.0.1` are reachable from inside (HTTP 200 from a test server). |
| Unix socket across the boundary | Works: an HTTP server on a Unix socket in a bound directory answers `curl --unix-socket` from inside `--containall`. |
| Relay to a base URL | Works: a 15-line Python relay inside the container (no `socat` in the image) serves `http://127.0.0.1:18766` from the Unix socket. Without a network namespace this only matters if the server itself listens on a socket instead of TCP. |
| Binds | `--bind dir:/work:ro` is read-only inside; a second `--bind` of a subfolder is writable; with `--containall` neither `/dais/fs/scratch` nor the real home is visible, `$HOME` is an empty tmpfs, the SIF root is read-only, `/tmp` writable. |
| `--cpus`, `--memory` | Refused on the login node (`cpu.max: no such file`: the cpu and memory controllers are not delegated) and in a job (`XDG_RUNTIME_DIR /run/user/51692 is not writable`). |
| `--pids-limit` | Works on the login node (forks fail above the limit); refused in a job for the same `XDG_RUNTIME_DIR` reason. |
| Slurm instead | A job step (`srun --cpus-per-task=N --mem=M`) inside an allocation gets its own cgroup: the way to cap an agent's CPU and memory. |
| Docker image to SIF | Works without root: `apptainer build x.sif docker://alpine:3.20` (registry pull) and `apptainer build x.sif docker-archive://alpine.tar` from a `docker save` tarball. |
| Docker-compatible runtime | None: no `podman`, `docker`, `nerdctl`, `buildah` on the path or as modules, no subuid range for the user, no `slirp4netns`/`pasta`. The spawner, which drives Docker through its socket, cannot run unchanged. |

MPCDF maintains an Apptainer sandbox for coding agents (`gitlab.mpcdf.mpg.de/mpcdf/ai-cli-agents-container`, Bits and Bytes 222) that ships `opencode` and confines an agent to the working directory plus a fake home. It has the same network limits as above, but its launcher and image are a candidate for reuse in phase 3 instead of writing our own.

## 2. The sim image on DAIS

- The image streamed from Raven (`/raven/ptmp/ykeller/cc/containers/apptainer/algoinst.sif`, read-only, through the Hetzner command server) has the same SHA-256 on DAIS: `fe358c02...91550ff`. It sits at `/dais/fs/scratch/levinb/policy-finder/images/algoinst.sif` (6.5 GB).
- **H200: does not work.** `torch.cuda.is_available()` is `True`, then the first kernel fails: `RuntimeError: CUDA error: no kernel image is available for execution on the device`, with torch's warning `NVIDIA H200 with CUDA capability sm_90 is not compatible with the current PyTorch installation. The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86.` (job 986771). B200 (sm_100) was not tried: the same arch list rules it out.
- **CPU: works, unchanged.** With `CUDA_VISIBLE_DEVICES=""` `aimanager` uses the CPU. Timings with the repo's entry point, seed 42 (job 986771, 32 threads):

| Sim | Episodes | Wall time |
|---|---|---|
| smoke sweep (1 point x 20) | 20 | 29 s |
| 2 points x 50, batch 100 | 100 | 30 s |
| 2 points x 500, batch 1000 | 1000 | 34 s |
| 8 points x 500, batch 4000 | 4000 | 41 s |
| 16 points x 500, batch 1000 (a real sweep's shape) | 8000 | 66 s |
| `pf-20261104-1` final run, 6 points x 2000 (job 986869) | 12000 | 167 s |

About 25 s of each is start-up (imports, model loading). Python warns that the linear baseline was pickled with scikit-learn 1.6.1 and loaded with 1.5.2; the same image does the same on Raven.

**Against Raven.** The smoke reference sims (20 episodes) differ from Raven because the CPU and CUDA random streams differ, so identical numbers are not expected: sweep `s000` 70.23 +- 10.97 against Raven's 65.37 +- 10.06; final `gp` 82.36 +- 11.31 against 55.31 +- 9.89 (1.8 combined standard errors), `s000` 53.91 +- 7.39 against 55.31 +- 6.78. At 20 episodes these say little. The decisive check is the 2000-episode final run of `pf-20261104-1` (code at that branch, same model files: the joblib's git blob and the `.pt` checksums match), seed 1000:

| Point | Raven (A100) | DAIS (CPU) | z |
|---|---|---|---|
| gp | 66.00 +- 1.07 | 64.28 +- 1.06 | -1.14 |
| s183 | 67.58 +- 1.07 | 66.52 +- 1.06 | -0.70 |
| s063 | 64.99 +- 0.99 | 66.35 +- 1.03 | +0.96 |
| s043 | 63.33 +- 0.98 | 64.93 +- 1.00 | +1.14 |
| s255 | 63.63 +- 1.03 | 65.18 +- 1.01 | +1.07 |
| s103 | 62.38 +- 0.98 | 64.43 +- 1.00 | +1.46 |

Chi-squared 7.3 on 6 degrees of freedom (p about 0.3), mean z +0.47: no systematic difference at this resolution. The best point changes (gp on Raven, s183 on DAIS), as expected when the top points lie within one standard error. Judged as **within noise, not identical**. A caveat: points of one sim share games, so the six z values are not fully independent.

If GPU sims were wanted on DAIS after all, the repo already has the recipe: `infra/claude_proxy/container/CC_env/Dockerfile.gpu` (torch 2.5.1+cu124, PyG wheels for it) covers sm_90 (H200) but not sm_100 (B200, which needs cu128 or later). It would be built with Docker elsewhere, saved, and converted with `apptainer build ... docker-archive://` (shown to work above), then validated against the reference sims, because a newer torch changes the random streams. Not done: the CPU path makes it unnecessary for now.

## 3. Serving Qwen3-Coder-Next

- Weights: `Qwen/Qwen3-Coder-Next` revision `a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb` (Apache 2.0, not gated), Hugging Face cache layout with `HF_HOME=/dais/fs/scratch/levinb/models`, served offline (`HF_HUB_OFFLINE=1`). 148.4 GiB of safetensors.
- Image: `/dais/fs/scratch/bsup/container/vllm.sif`, label `ai.vllm.image.tag: vllm/vllm-openai:v0.25.1` (commit 752a3a5), asserted at start as the sister project does. The model card asks for vLLM >= 0.15.
- Serve arguments (`infra/dais/common.sh`): `--tensor-parallel-size 2 --gpu-memory-utilization 0.88 --max-model-len 131072 --enable-prefix-caching --enable-prompt-tokens-details --enable-auto-tool-choice --tool-call-parser qwen3_xml`, in a home on scratch (vLLM, torch.compile and FlashInfer write caches under `~/.cache`).
- **Parsers:** in v0.25.1 `qwen3_coder` and `qwen3_xml` are the same class (`vllm/tool_parsers/__init__.py`: both map to `qwen3_engine_tool_parser.Qwen3EngineToolParser`), so the comparison asked for in the HF discussion (#17: the old `qwen3_coder` looped with `!!!` after long inputs with tool calls) is moot for this version. All checks passed (jobs 986513, 986879): a chat completion; a tool call with `finish_reason: tool_calls` and JSON arguments with all required keys; the same streamed (34 chunks, arguments joined to valid JSON); a tool call after an 83,132-token context (2.8 s).
- **Usage:** `prompt_tokens`, `completion_tokens`, `total_tokens` always; `prompt_tokens_details.cached_tokens` only with `--enable-prompt-tokens-details` (now set), and streamed usage with `stream_options.include_usage`.
- **Load time:** 386 s cold (weights 145 s from GPFS, torch.compile 50 s, engine init with profiling and CUDA graphs 96 s), 281 s on the second start.
- **Memory per GPU** at utilisation 0.88: weights 74.3 GiB, KV cache 44.3 GiB (3.78M tokens in all: the model's linear-attention layers keep the KV small; 28.8 concurrent requests at 131k), CUDA graphs 1.1 GiB; `nvidia-smi` 129,371 of 143,771 MiB used, 14 GB free.

## 4. Sharing a node

Because the sim image cannot use these GPUs, the sharing measured is GPU for vLLM and CPUs for the sims, in one 2-GPU, 64-thread allocation (job 986879). Five concurrent sessions, 1024 generated tokens each (`ignore_eos`), cold prompt and then the next turn with the prefix cached:

| Prompt per session | Round | Alone: decode tok/s per session, aggregate, TTFT | With a sim on the CPUs |
|---|---|---|---|
| 29k tokens | cold | 109, 426, 2.6 s | 120, 495, 1.8 s |
| 29k tokens | warm | 133, 643, 0.23 s | 131, 637, 0.19 s |
| 94k tokens | cold | 82, 265, 6.7 s | 82, 266, 6.6 s |
| 94k tokens | warm | 119, 552, 0.53 s | 118, 554, 0.42 s |

- Generation is not slowed by a CPU sim (the differences are run-to-run noise; the cold 29k round was the first after start-up).
- The sim is: 108 s while sessions generated against 69 s before the server started and 62 s next to an idle server, 1.6x. torch uses every thread of the allocation by default and competes with vLLM's API server and scheduler; pinning the sims (`taskset` or `OMP_NUM_THREADS`, or one Slurm job step per sim with its own CPUs) should remove most of this. Not measured yet.
- GPU memory of a sim (question 4a) could not be measured on DAIS. From the code (`MAX_EPISODE_BATCH_SIZE` comment in `simulation/simulate.py`): about 9 GB on an A100 at batch 20000; a batch of 1000 is far below that. Memory is driven by `episode_batch_size` (env and AH model state per episode in the batch) plus the per-batch records, which are kept on the device until the end of the sim (`run_batched`).
- Prefix caching works for the agent pattern: the next turn reuses 28.8k of 29.2k and 93.0k of 93.5k prompt tokens, and its time to first token drops from 2.6 to 0.2 s and from 6.7 to 0.5 s.

## 5. The frontier candidates on a full 8 x H200 node

With the sims on the CPUs no slice must stay free, so vLLM can take `--gpu-memory-utilization 0.92` (129.2 GiB of the 140.4 GiB per GPU). From the Qwen run, about 5 GiB per GPU goes to activations, CUDA graphs and buffers inside that budget. Weight sizes are the published files (Hugging Face API), divided evenly over the tensor-parallel GPUs.

| Model | Weights | Layout | Weights per GPU | Left for KV per GPU (0.92) | Verdict |
|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 765 GB (712 GiB), FP8 + int8 | TP 8 | 89 GiB | about 35 GiB | Fits with the sims on the CPUs; whole node. |
| GLM-5.3 | 755 GB (703 GiB), FP8 | TP 8 | 88 GiB | about 36 GiB | Fits; whole node. |
| GLM-5.3-Flash | 328 GB (305 GiB), FP8 | TP 4 | 76 GiB | about 48 GiB | Fits on half a node; a second replica on the other 4 GPUs is possible. TP 2 does not fit (153 GiB per GPU). |
| MiniMax-M3.1-Flash | 224 GB (209 GiB), mostly packed 8-bit (`U8`, likely FP4) | TP 2 or TP 4 | 104 or 52 GiB | about 20 or 72 GiB | TP 2 fits but leaves a small KV cache; TP 4 is comfortable. |

If GPU sims were ever needed beside the model (a rebuilt image), a 10 GiB slice per GPU (utilisation about 0.85) would still leave the two 760 GB models about 25 GiB of KV per GPU. Uncertain: vLLM v0.25.1's support for each model's architecture and quantisation on Hopper (int8 experts in DeepSeek-V4.1-Flash, the packed `U8` weights of MiniMax-M3.1-Flash), and each model's KV size per token, which decides how much context 35 GiB holds. Storage: the two large models need about 1.5 TB each on scratch by `du`; the quota has 5.3 TB left. None was downloaded or served.

## What can be reused

| Part | Reuse |
|---|---|
| `algoinst.sif` | Unchanged, on the CPUs. Not on the H200/B200 GPUs. |
| `vllm.sif` (group image) | Unchanged; version asserted at start. |
| Sister project's serve conventions (`../understanding/scripts/slurm/lib/evo_common.sh`) | Followed: `apptainer exec --nv`, a home that is not the real home, `HF_HOME` on scratch, explicit prefix caching, readiness on `/v1/models`. |
| Repo entry points | `python -m aimanager simulate <config>` on the policy finder's generated configs, unchanged; the Raven job template's container setup (repo at `/opt/algorithmic-institutions`, `PATH`, `PYTHONPATH`) copied into `in_sim_image`. |
| Agent spawner (Docker socket) | Cannot run: no Docker-compatible runtime. Needs an Apptainer backend (phase 3), possibly MPCDF's agents container. |
| Network rule of the Docker sandbox | Cannot be reproduced: no network namespaces. File isolation can. |
| Raven `cluster_proxy` functions | A thin adapter: on DAIS a sim is a CPU process in the sim image, so the "queue" is a local process pool or Slurm job steps, as phase 3 planned. |

## Scripts

- `infra/dais/common.sh`: settings (images, weights, revision), `serve_model`, `wait_ready`, `in_sim_image`, `simulate`.
- `infra/dais/serve.slurm.sh`: serve on 2 H200; `MODE=probe` checks the endpoint, `MODE=serve` keeps it up.
- `infra/dais/probe_endpoint.py`: chat, tool calls (plain, streamed, long context), usage.
- `infra/dais/bench_load.py`: N concurrent long-context sessions, cold and warm.
- `infra/dais/sim_check.slurm.sh`: network and Apptainer checks, the sim image on a GPU (CPU fallback), reference sims, timing per batch size.
- `infra/dais/sim_cpu.slurm.sh`: one sim on the CPUs of the `small` partition.
- `infra/dais/share_bench.slurm.sh`: sim alone, server, probe, sessions alone and beside sims.
- `infra/dais/sim_variant.py`: the GPU check and sweep-shaped config variants.

The scripts expect a checkout at `/u/$USER/algorithmic-institutions-pf` with the model files and `infra/dais/`, and are submitted from it.

## Changes this suggests for the plan

- Decision 1: run the sims on the node's CPUs in the existing image, not in a GPU slice; give vLLM the GPUs. Pin the sims to their own cores.
- Phase 3's cluster backend: a local process pool of CPU sims (or `srun` job steps for CPU and memory caps); no `CUDA_VISIBLE_DEVICES` bookkeeping.
- Decision 4: the network cannot be cut, so the fallback (restricted tools, audited transcripts) applies; consider MPCDF's agents container for the file isolation.
- Full-node jobs go to `gpu` (exclusive) or `gpux` (10 days, 2 nodes); `gpu1` is for the shared, smaller jobs.
- Copy the weights and any results off scratch before 2027-02-01's 12-week cleanup starts.
