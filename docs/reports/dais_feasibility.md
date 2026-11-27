# DAIS feasibility for the open-weight policy finder

Phase 0 (DAIS part) and the first half of phase 2 of `docs/plans/policy-finder-open-weights.md`, measured on 2026-11-27 under account `mpib_inst`. Scripts: `infra/dais/`. Revised after a review of the first version, with the measurements that decide the node layout (B1 to B5) added; section "Not yet measured" lists what is still open.

## Where the numbers come from

| Job | Partition, QoS | Node | What |
|---|---|---|---|
| 986457 | gpudev, debug | daisg103 (H200) | network test |
| 986513 | gpudev, debug | daisg103 (H200), 2 GPUs | serve and probe Qwen3-Coder-Next |
| 986771 | gpudev, debug | daisg103 (H200), 1 GPU, 32 threads | sim image on the GPU, CPU fallback, smoke-rule timings |
| 986879 | gpudev, debug | daisg103 (H200), 2 GPUs, 64 threads | serve, probe, 5-session benchmark, smoke-rule sims beside it |
| 986869 | small, cpu | daisg302 (AMD EPYC 9555, an RTX node), 32 threads | `pf-20261104-1` final run on the CPUs |
| 987428 | small, cpu | daisg301 (AMD), 48 cores | B1 threads and processes, B4 Slurm steps (`cpu_bench.slurm.sh`) |
| 987622 | small, cpu | daisg302 (AMD), 48 cores | B2 many slices at once (`sweep_pack.slurm.sh`) |
| 987623 | gpudev | daisg103 (H200 node's Intel CPUs), 1 GPU unused, 12 cores | B1 on the H200 nodes' CPU (`cpu_bench.slurm.sh`, short shapes) |
| 987430 | gpudev | daisg103 (H200), 2 GPUs, 24 cores | B3 sims beside the server, B5 concurrency, revised probe (`serve_bench.slurm.sh`) |

The first five jobs ran earlier versions of the scripts. All GPU jobs ran on `gpudev` (15 minutes; one job per user, shared nodes). The scripts' default `gpu1`/`g0002` path was submitted once (job 987431) and cancelled before it started, because the `gpudev` run (987430) answered B3 and B5; `serve.slurm.sh`'s `MODE=serve` has not been run. The `small` nodes are shared with other users' jobs, which adds noise. No Raven baseline could be run for this report (no Raven access); Raven numbers are the committed results of earlier runs.

## Summary

- **The sim image cannot use the H200 or B200 GPUs, but runs unchanged on the CPUs.** `algoinst.sif` (byte-identical to Raven's) has no kernels for H200 (`no kernel image is available for execution on the device`; torch 1.11+cu113 ships sm_37 to sm_86, no PTX). On the CPUs the repo's `python -m aimanager simulate` runs, and the right shape is many single-threaded processes: extra torch threads only cost CPU. On 12 cores of an H200 node, 24 single-threaded processes (one per hyperthread) played 96,000 episodes of a real rule in 38 s (about 2,500 episodes per second, 211 per core); on the AMD `small` nodes 96 processes on 48 cores reached 11,600 episodes per second. A real sweep (256 points x 500 episodes = 128,000 episodes, split by `--n-parts 32`) takes about 40 s on 16 cores.
- **CPU and A100 results: no difference detected at this power.** A 2000-episode, 6-point final run gives no detectable mean shift (each difference has a standard error of about 1.45 pool points, so shifts below about 4 points would go unseen). The ranking of points may differ: Raven's points are heterogeneous, DAIS's are not (see section 2).
- **Qwen3-Coder-Next serves on 2 H200 with the group's `vllm.sif` (v0.25.1), unchanged.** Ready in 276 to 386 s. Tool calls were well formed in every trial, including after a 70k-token history of earlier tool calls. Warm 30k-token sessions decode at 135 tokens/s each with 5 at once and 43 each with 80 at once; the aggregate grows from 630 to 2,770 tokens/s and is still growing at 80.
- **Server and sims on separate cores do not slow each other** (job 987430): 16 pinned sim slices took 34.0 to 36.4 s beside steady generation and 33.4 to 35.8 s alone; generation ran at 698 tokens/s beside them and 678 alone.
- **The agent's network can be cut with a seccomp filter, not with a namespace.** Users have no network namespaces (`max_net_namespaces` = 0) and there is no Docker-compatible runtime, so the spawner cannot run unchanged. But an unprivileged `apptainer exec --security seccomp:<profile>` that denies `AF_INET`/`AF_INET6` sockets (and `io_uring_setup`) blocks all IP traffic while a Unix-socket path to the model server still works (login node). Slurm job steps cap CPUs and memory (an over-limit step is OOM-killed, the job survives); Apptainer's own `--cpus`/`--memory` are refused.

## 1. Cluster facts

| Item | Finding | Evidence |
|---|---|---|
| Partitions | `gpu` is `OverSubscribe=EXCLUSIVE`: any job there holds a whole node. `gpu1` is the shared partition; per GPU at most 12 cores (24 threads) and 250 GB (an `sbatch --test-only` for 2 GPUs refused more than 24 cores and 500,000 MB). `gpudev`: shared, 15 min, at most 4 GPUs, one job per user. `small`: CPU only, on the two AMD RTX nodes, at most half a node (64 cores, 750,000 MB). `gpux`: 10 days, 2 H200 nodes. | `scontrol show partition`, `sbatch --test-only`, user guide |
| GPU type | `--gres=gpu:h200:N` selects H200. | jobs above |
| QoS | `g0001` to `g0008` carry no TRES limit in `sacctmgr`. A full node would be `-p gpu --gres=gpu:h200:8 --qos=g0008` (not tested). | `sacctmgr` |
| Nodes | H200 nodes: 8 x H200 (143,771 MiB), 2 x Xeon Platinum 8568Y+ (96 cores, 192 threads), 2 TB RAM. The `small` partition's nodes are AMD EPYC 9555, a different CPU. | `lscpu`, `nvidia-smi` |
| Network | The login node and compute nodes reach `huggingface.co`, `openrouter.ai`, `pypi.org`, `github.com`. | 986457, 986771 |
| Scratch | Quota 8 TB and 8M files; no backup; from 2027-02-01 files not accessed for 12 weeks are deleted (login message). GPFS `du` shows about twice the apparent size (297 GB for 149 GB of weights); whether the quota counts the doubled figure is unclear. | login message, `mmlsquota`, `du` |
| Weights download | 149 GB in 5 minutes on the login node, niced, 4 workers; no site rule against it was found. | download log |

### Apptainer as the agent sandbox

What was shown, as an unprivileged user with apptainer 1.5.2. Login-node rows: `infra/dais/iso_check.sh`, output in `docs/reports/dais_iso_check_login.txt`. In-job rows: job 986771's log and job 987428 (B4).

| Test | Result | Evidence |
|---|---|---|
| `--containall --net --network none` | Refused: `Failed to create network namespace: maximum number of network namespaces exceeded` (`/proc/sys/user/max_net_namespaces` = 0). `unshare -rn` fails as well. | login node; job 986771 |
| Without a network namespace | The container shares the host network; a test server on the host's `127.0.0.1` answered from inside. | login node |
| HTTP over a Unix socket in a bound directory | Works with `--containall`. | login node |
| A relay inside the container to a base URL | Works: a 15-line Python relay (no `socat` in the image) served the Unix-socket server at `http://127.0.0.1:18766`. | login node |
| Binds | A `:ro` bind is read-only, a second bind of a subfolder is writable; with `--containall` neither scratch nor the real home is visible and `$HOME` is empty; the image root is read-only; `/tmp` is writable. | login node |
| `--cpus`, `--memory` | Refused on the login node (`cpu.max: no such file or directory`) and in a job (`cannot use cgroups - XDG_RUNTIME_DIR ... is not writable`). | login node; job 986771 |
| `--pids-limit` | Works on the login node: with `--pids-limit 64` a shell starting 200 background processes stops with `Cannot fork`; without the limit all 200 start. Refused in a job (same `XDG_RUNTIME_DIR` error). | `iso_check.sh`; job 986771 |
| A seccomp filter denying `AF_INET`/`AF_INET6` sockets and `io_uring_setup` | Works unprivileged (`--security seccomp:<profile>`): `socket(AF_INET)` and `socket(AF_INET6)` fail with `Operation not permitted`, `curl` fails by name and by IP address, `io_uring_setup` is denied; `AF_UNIX` sockets work and the Unix-socket model server answers. Without the filter the same image opens `AF_INET` sockets. Tested on the login node only, not yet in a job, and not yet with OpenCode inside. | `iso_check.sh` |
| Slurm job steps as caps | Work. `srun --exact -n1 -c 2` gives the step two CPUs (affinity `[12, 24]`); 8 busy loops for 20 s in such a step used 40 CPU-seconds, so the step is held to 2 CPUs. A step with `--mem=2G` that allocates 6 GiB is OOM-killed (`OUT_OF_MEMORY`, exit 0:125) and the job carries on. A sim in a `-c 2 --mem=16G` step ran normally (MaxRSS 2.3 GB by `sacct`). | job 987428 |
| Docker image to SIF | Works without root, from `docker://alpine:3.20` and from a `docker save` tarball (`docker-archive://`). | login node |
| Docker-compatible runtime | None: no `podman`, `docker`, `nerdctl`, `buildah`, no subuid range for the user, no `slirp4netns`/`pasta`. | login node |

So the agent's network cannot be cut by a namespace, but a seccomp filter blocks IP sockets, which leaves a Unix socket (with a relay inside the container if OpenCode needs a base URL) as the agent's only way to the model. MPCDF's Apptainer sandbox for coding agents (`gitlab.mpcdf.mpg.de/mpcdf/ai-cli-agents-container`, Bits and Bytes 222) ships `opencode` and confines an agent to its working directory and a fake home; it has the same network situation.

## 2. The sim image on DAIS

- The image was streamed read-only from Raven's shared copy; its SHA-256 on DAIS is `fe358c02...91550ff`, the same.
- **H200: does not work** (job 986771): `torch.cuda.is_available()` is `True`, then `RuntimeError: CUDA error: no kernel image is available for execution on the device`, after torch's warning that H200 (sm_90) is not among sm_37 ... sm_86. B200 (sm_100) was not tried; the arch list rules it out.
- **CPU: works, unchanged**, with `CUDA_VISIBLE_DEVICES=""`.

**First timings (superseded by B1 and B2 below).** The smoke rule (`punishment = 0`, job 986771, 32 threads) is cheap and not representative; its 66 s "sweep" (16 points x 500 episodes) is 16 times smaller than a real one, and 62 and 69 s in the 64-thread job 986879 against 66 s in the 32-thread job show that more threads gained nothing. The real rule of `pf-20261104-1` in one 32-thread process (job 986869, daisg302, AMD) took 167 s for 12,000 episodes. B1 shows why that was slow: torch's intra-op threads cost far more than they save at this tensor size.

**Against Raven.** The 20-episode smoke sims are not identical to Raven's (CPU and CUDA random streams differ) and at 20 episodes say little: sweep `s000` 70.23 +- 10.97 against Raven's 65.37 +- 10.06; final `gp` 82.36 +- 11.31 against 55.31 +- 9.89, `s000` 53.91 +- 7.39 against 55.31 +- 6.78.

The stronger check is the 2000-episode final run of `pf-20261104-1` (that branch's code; the model files' checksums match), seed 1000:

| Point | Raven (A100) | DAIS (CPU, daisg302) | Difference | z |
|---|---|---|---|---|
| gp | 66.00 +- 1.07 | 64.28 +- 1.06 | -1.72 +- 1.51 | -1.14 |
| s183 | 67.58 +- 1.07 | 66.52 +- 1.06 | -1.06 +- 1.51 | -0.70 |
| s063 | 64.99 +- 0.99 | 66.35 +- 1.03 | +1.37 +- 1.42 | +0.96 |
| s043 | 63.33 +- 0.98 | 64.93 +- 1.00 | +1.60 +- 1.40 | +1.14 |
| s255 | 63.63 +- 1.03 | 65.18 +- 1.01 | +1.55 +- 1.44 | +1.07 |
| s103 | 62.38 +- 0.98 | 64.43 +- 1.00 | +2.05 +- 1.40 | +1.46 |

- No difference is detected at this power: chi-squared 7.3 on 6 degrees of freedom. But each difference has a standard error of about 1.45, so shifts below about 4 pool points would go undetected, while the policy finder ranks points that differ by 1 to 5 points.
- A difference in the ranking of points is not excluded. On Raven the six points are heterogeneous (chi-squared 17.2 on 5 df around their weighted mean); on DAIS they are not (4.2). The lead of the top two points over the other four is +3.2 +- 0.9 on Raven and +0.2 +- 0.9 on DAIS. This comparison was chosen after seeing the data, so it is a hint, not a finding.
- The points do not share games in this config: `run_batched` seeds batch i with `seed + i` and each point fills its own batches, so the six differences are independent.
- A stronger test needs the same config at many more episodes on both systems, and Raven is not reachable from here.

**The two identical Raven smoke points.** In Raven's smoke final run, `gp` and `s000` (the same rule) have the same pool mean to 14 digits but different standard errors and member counts. The rule punishes nothing, so a point's pool mean is 1.6 x (its group-0 contributions) / 480 (20 episodes x 24 rounds). The two runs' group-0 contribution totals are both 16,594, while their per-episode values, punishments of the other group and group-0 row counts (2,468 and 2,072 agent-rounds) all differ (`per_round.parquet`). No shared data was found, so as far as the code and data show, the equal integer totals are a coincidence; this is not fully explained. The DAIS run of the same config does not repeat it.

If GPU sims are wanted on DAIS after all, the repo has a recipe: `infra/claude_proxy/container/CC_env/Dockerfile.gpu` (torch 2.5.1+cu124) covers H200 (sm_90) but not B200 (sm_100, which needs cu128 or later). It would be built with Docker elsewhere, converted with `apptainer build ... docker-archive://`, and validated against the reference sims (a newer torch changes the random streams). Not done.

## 3. Serving Qwen3-Coder-Next

- Weights: `Qwen/Qwen3-Coder-Next` revision `a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb` (Apache 2.0, not gated), Hugging Face cache layout under `HF_HOME=/dais/fs/scratch/$USER/models`, served offline. 148.4 GiB of safetensors.
- Image: the group's `vllm.sif`, label `vllm/vllm-openai:v0.25.1`, asserted at start. The model card asks for vLLM >= 0.15.
- Serve arguments as run: `--tensor-parallel-size 2 --gpu-memory-utilization 0.88 --max-model-len 131072 --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_xml` (986879 added `--enable-prompt-tokens-details`).
- **Parsers:** in v0.25.1 `qwen3_coder` and `qwen3_xml` are the same class (`vllm/tool_parsers/__init__.py` maps both to `Qwen3EngineToolParser`). The looping reported in the model's discussion #17 with the old `qwen3_coder` was not reproduced, but it was tested only once.
- **Checks** (986513, 986879: one trial each at temperature 1.0; 987430: the revised `probe_endpoint.py`, 2 trials per check at temperature 0.2), all passed. A pass needs `finish_reason: tool_calls`, the expected tool, JSON arguments with every required key and no degenerate repetition. The checks were: a chat completion; a tool call; the same streamed; a tool call after one 71k-token user message; a tool call after a 70.7k-token history of three earlier `run_shell` calls with long tool results, plain and streamed (0.6 to 2.7 s each). Two trials per check is still few; the agent bake-off will exercise this far more.
- **Usage:** `prompt_tokens`, `completion_tokens`, `total_tokens`; `prompt_tokens_details.cached_tokens` only with `--enable-prompt-tokens-details`; streamed usage with `stream_options.include_usage`.
- **Load time:** 386 s cold (weights 145 s from GPFS, torch.compile 50 s, engine start-up with profiling and CUDA graphs 96 s), 276 to 281 s later.
- **Memory per GPU** at utilisation 0.88: weights 74.3 GiB, KV cache 44.3 GiB, CUDA graphs 1.1 GiB; 129,371 of 143,771 MiB used. At 0.90 (987430): 132,247 MiB used, 4.02M tokens of KV cache.
- **KV cache:** 44.3 GiB per GPU held 3.78M tokens, about 25 KB per token across both GPUs. That is unusually small, because most of the model's layers are linear attention; other models need far more per token.

## 4. Sims on the CPUs, generation on the GPUs

### B1: threads and processes for one real sweep

A slice is one part of a real sweep split by the repo's own `generate_sim_config.py --n-parts 32` (`pf-20261104-1`'s 5-parameter rule, 256 Sobol points x 500 episodes): 8 points x 500 episodes = 4,000 episodes, 4 batches of 1,000. Every process ran `python -m aimanager simulate` in the unchanged sim image, `OMP_NUM_THREADS`/`MKL_NUM_THREADS` set to the thread count (torch 1.11 honours them; the sim sets none itself), pinned with `taskset`. Times are wall and user+system CPU seconds by GNU `time`, per process, including start-up.

One slice alone, by threads:

| Threads | AMD (987428) wall, CPU s | Intel H200 node (987623) wall, CPU s |
|---|---|---|
| 1 | 15.4, 15.7 | 29.6, 30.5 |
| 2 | 23.8, 35.9 | 26.8, 39.1 |
| 4 | 25.5, 62.3 | 33.7, 79.1 |
| 8 | 26.8, 124.2 | |
| 16 | 33.8, 326.8 | |
| 32 | 76.5, 1933.8 | |

Cores filled with k processes of t threads, one slice each:

| Node, cores | k x t | Episodes | Slowest process | Episodes per second | Per core |
|---|---|---|---|---|---|
| AMD, 16 | 1 x 32 | 4,000 | 80.5 s | 50 | 3 |
| AMD, 16 | 4 x 8 | 16,000 | 37.9 s | 420 | 26 |
| AMD, 16 | 8 x 4 | 32,000 | 50.7 s | 630 | 39 |
| AMD, 16 | 16 x 2 | 64,000 | 25.5 s | 2,510 | 157 |
| AMD, 16 | 32 x 1 | 128,000 | 27.8 s | 4,600 | 288 |
| Intel, 12 | 6 x 4 | 24,000 | 33.9 s | 710 | 59 |
| Intel, 12 | 12 x 1 (one per core) | 48,000 | 31.3 s | 1,530 | 128 |
| Intel, 12 | 12 x 2 | 48,000 | 28.7 s | 1,670 | 139 |
| Intel, 12 | 24 x 1 (one per hyperthread) | 96,000 | 38.0 s | 2,530 | 211 |

- **Best shape: one single-threaded process per hyperthread.** Every extra torch thread adds CPU time without saving wall time; system time grows with the threads (thread synchronisation).
- **CPU cost:** about 9 CPU-seconds per 1,000 episodes per process on the Intel H200 nodes with every hyperthread busy (24 x 1: 36 s per 4,000 episodes), and about 7 on the AMD nodes (32 x 1).
- **A whole real sweep is 128,000 episodes**, the 32 x 1 row: 28 s on 16 AMD cores, so about 40 s on 16 Intel cores.
- **Caveats.** The k x t rows above count only the processes' own times. In job 987428 the round times (B1B lines) also include the per-slice config preparation, which the revised script takes out of the timed part. The `gpudev` and `small` nodes are shared with other jobs.

### B2: many slices at once (job 987622, AMD, 48 cores, single-threaded slices)

| Slices at once | Pinned: slowest, episodes/s | Not pinned: slowest, episodes/s |
|---|---|---|
| 1 | 19.6 s, 210 | 15.7 s, 250 |
| 5 | 20.0 s, 1,000 | 16.9 s, 1,180 |
| 10 | 19.1 s, 2,110 | 17.9 s, 2,220 |
| 20 | 19.1 s, 4,210 | 19.3 s, 4,000 |
| 40 | 21.1 s, 7,620 | 26.5 s, 5,930 |
| 96 (every hyperthread) | 32.4 s, 11,640 | 35.4 s, 10,970 |

- Episodes per second counts the round's whole wall time.
- Pinning (one per hyperthread, one per core first) helps once the slices compete for cores (+29 percent at 40, +6 percent at 96). At small numbers the unpinned slices were faster: they can move to idle CPUs, and these nodes are shared with other users' jobs.
- All 344 slices exited 0 and wrote their `sweep.json`; nothing was killed.
- Peak RSS per slice was 1.2 to 1.4 GB, so 96 slices need about 130 GB. Each slice's output is about 3 MB, roughly 95 MB per sweep.

### B3: sims beside the server, each on its own cores (job 987430, Intel, 2 H200, 24 cores)

- **Layout:** vLLM (TP 2, utilisation 0.90) and the benchmark clients were pinned to 4 cores. 16 slices of the real sweep (2 threads each, one core each) ran on 16 of the other cores. Generation was steady: 5 warm 27k-token sessions repeating 1,024-token requests from 17:41:00 to 17:41:51, while the slices ran from 17:40:52 to 17:41:45.
- **Sims:** 34.0 to 36.4 s each beside generation, 33.4 to 35.8 s alone after the server was stopped; the pools are identical (the CPU sims are deterministic).
- **Generation:** 698 tokens/s in all and 143 per session beside the sims; 678 and 139 alone; median time to first token 0.18 s and 0.16 s.
- **Conclusion:** with separate cores, no interference was measurable. The earlier unpinned attempt (job 986879) only showed that an unpinned sim nearly stalls while the server takes the CPUs: generation overlapped about 47 s of its 108 s, and in that window it did roughly 1 to 9 s of work.

### B5: generation against concurrency (job 987430, Qwen3-Coder-Next on 2 H200)

Sessions with 27k to 29k-token prompts, 512 generated tokens each, every session once cold and then once warm (prefix cached, 99 percent cached tokens):

| Sessions | Warm: decode tok/s per session (median) | Warm: aggregate tok/s | Warm: time to first token (median, max) | Cold: aggregate tok/s | Cold: time to first token (median, max) |
|---|---|---|---|---|---|
| 5 | 135 | 631 | 0.25 s, 0.36 s | 348 | 2.9 s, 3.7 s |
| 10 | 102 | 934 | 0.41 s, 0.68 s | 516 | 3.0 s, 5.2 s |
| 20 | 86 | 1,329 | 1.78 s, 1.79 s | 566 | 6.6 s, 12.2 s |
| 40 | 62 | 2,159 | 0.82 s, 1.72 s | 714 | 10.4 s, 21.1 s |
| 80 | 43 | 2,769 | 2.06 s, 4.13 s | 793 | 20.8 s, 42.1 s |

- The aggregate is still growing at 80 sessions (+28 percent from 40), but per-session speed falls by two thirds from 5 to 80.
- Cold prompts are limited by prefill, which ran at about 54,000 tokens/s (vLLM's log).
- 80 sessions of 29k tokens fill about 2.3M of the 4.0M tokens of KV cache.
- This ran on `gpudev`, not on the `gpu1`/`g0002` default; the hardware is the same.

The sim's GPU memory could not be measured on DAIS (no kernels for H200); on the CPUs a slice's resident memory is 1.2 to 1.4 GB.

## 5. The frontier candidates on a full 8 x H200 node

Expected to fit or not, unverified. Weight sizes are the summed safetensors files from the Hugging Face API on 2026-11-27, spread evenly over the tensor-parallel GPUs. With the sims on the CPUs, vLLM can take utilisation 0.92 (129.2 GiB of 140.4 GiB per GPU). About 5 GiB of that goes to activations, CUDA graphs and buffers; this figure comes from one model at TP 2 (Qwen) and may differ for others.

| Model | Weights | Layout | Weights per GPU | Left for KV per GPU | Assessment |
|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 510 GB (475 GiB) | TP 8 | 59 GiB | about 65 GiB | Expected to fit; TP 4 would leave about 5 GiB, too little. |
| GLM-5.3 | 756 GB (704 GiB), FP8 | TP 8 | 88 GiB | about 36 GiB | Expected to fit; whole node. |
| GLM-5.3-Flash | 328 GB (306 GiB), FP8 | TP 4 | 76 GiB | about 48 GiB | Expected to fit on half a node; TP 2 does not fit (153 GiB per GPU). |
| MiniMax-M3.1-Flash | 252 GB (235 GiB) | TP 4 | 59 GiB | about 65 GiB | Expected to fit on half a node; TP 2 would leave about 7 GiB, too little to be useful. |

- **Unverified:** whether vLLM v0.25.1 supports each architecture and its quantisation on Hopper (DeepSeek-V4.1-Flash's int8 tensors, MiniMax's packed `U8` weights).
- **Unknown:** each model's KV size per token, which decides how much context the remainder holds. Qwen's 25 KB per token is no guide.
- **Storage:** about twice the file size by `du` on scratch.
- None of these was downloaded or served.

## A node layout for 20 chains

For one full H200 node (96 cores, 192 hyperthreads, 8 GPUs, 2 TB), with the sims on the CPUs:

| Use | Cores | How |
|---|---|---|
| vLLM and its clients | 8 | the server pinned to them (`SERVER_CPUS`); 4 cores served TP 2 without interference in B3, TP 8 has more worker processes, so twice that (not measured) |
| 20 agents (OpenCode, their shell tools, `validate-rule`) | 16 | one Slurm job step or `taskset` set per agent; `OMP_NUM_THREADS=1` in the agents' environment, since an unbounded `validate-rule` used about 7 cores for 13 s on the login node (not measured under agent load) |
| Sweeps | 72 (144 hyperthreads) | a pool of 144 single-threaded slice processes (`OMP_NUM_THREADS=1`), pinned one per hyperthread, fed with slices from `generate_sim_config.py --n-parts 32` (4,000 episodes each) |

Expected sweep times under that layout, from B1 and B2 (Intel, every hyperthread busy: about 36 s per slice, 211 episodes per second per core):

- **One sweep on an idle pool:** its 32 slices run at once, about 30 to 40 s, plus merging (`merge_sweeps.py`, not timed).
- **All 20 chains' sweeps at once (the worst case):** 640 slices in 144 slots, about 5 waves of about 36 s, so about 3 minutes for all 20. The pool's capacity is about 15,000 episodes per second, about 9 s per real sweep.
- **Memory:** 144 slices x 1.4 GB, about 200 GB.

So the sims are not the bottleneck: a round of 20 sweeps costs minutes, against an agent version that takes many model calls. On the model side, B5 for Qwen3-Coder-Next on 2 H200 gives 86 tokens/s per session with 20 warm sessions (1,330 in all) and still more aggregate at 40 and 80. For a frontier model on 8 GPUs the numbers are unmeasured.

## Not yet measured

- The layout above as a whole: a full node, a TP 8 server, 20 real agents and the sim pool together.
- Throughput of the frontier models; their vLLM support on Hopper.
- Agents' CPU use (OpenCode, tools, `validate-rule`) under load.
- The seccomp filter inside a job and with OpenCode in the container.
- Per-process start-up cost separately from simulation time; `merge_sweeps.py` and the final run's timing.
- The `gpu1`/`g0002` default path and `serve.slurm.sh MODE=serve` (B5 ran on `gpudev`, the same hardware).
- A Raven baseline for any of these: not available from here.

## What can be reused

| Part | Reuse |
|---|---|
| `algoinst.sif` | Unchanged, on the CPUs. Not on H200/B200 GPUs. |
| The group's `vllm.sif` | Unchanged; version asserted at start. |
| Sister project's serve conventions (`../understanding/scripts/slurm/lib/evo_common.sh`) | Followed: `apptainer exec --nv`, a home that is not the real one, `HF_HOME` on scratch, explicit prefix caching, readiness on `/v1/models`. Added: a port and API key per job, readiness that refuses a server answering without the key. |
| Repo entry points | `python -m aimanager simulate` on configs from `generate_sim_config.py` (including `--n-parts`), unchanged; the Raven job template's container layout copied into `in_sim_image`. |
| Agent spawner (Docker socket) | Cannot run: no Docker-compatible runtime. Needs an Apptainer backend, possibly MPCDF's agents container. |
| Network rule of the Docker sandbox | Not with namespaces; a seccomp filter that denies IP sockets works (login node). File isolation is reproducible. |
| Raven `cluster_proxy` functions | A thin adapter: a sim becomes a CPU process (or a Slurm job step) in the sim image. |

## Scripts

`infra/dais/common.sh` holds the settings and helpers. Every job writes into its own run folder under scratch (`$PF_RUN`), never into the checkout; the checkout is bound read-only into the sim container, and failures make the job exit non-zero.

| Script | Purpose | Run? |
|---|---|---|
| `serve.slurm.sh` | serve and probe, or keep serving | earlier version, on gpudev |
| `probe_endpoint.py` | tool-call checks, several trials | yes (987430) |
| `bench_load.py` | concurrent sessions, rounds or steady load | yes (987430) |
| `sim_check.slurm.sh` | network and Apptainer checks, GPU test, reference sims on the CPUs | earlier version (986771) |
| `sim_cpu.slurm.sh` | one sim on the CPUs of the `small` partition | earlier version (986869) |
| `cpu_bench.slurm.sh` | B1, B4 | yes (987428; 987623 with short shapes); the B1 round times of 987428 still included config preparation |
| `sweep_pack.slurm.sh` | B2 | yes (987622) |
| `serve_bench.slurm.sh` | B3, B5 | yes, on gpudev (987430) |
| `iso_check.sh` | sandbox checks | yes, login node (`dais_iso_check_login.txt`) |
| `cpu_slots.py` | the job's CPUs grouped by core, for pinning | yes |
| `sim_variant.py` | GPU check; a config copy with another output folder or shape | yes |

The scripts expect a checkout at `/u/$USER/algorithmic-institutions-pf` with the model files, submitted from it, and `/dais/fs/scratch/$USER/policy-finder/logs` to exist.

## Changes this suggests for the plan

- **Decision 1:** run the sims on the node's CPUs in the existing image, single-threaded and one per hyperthread, and give vLLM the GPUs and a few pinned cores; the layout above.
- **Phase 3's cluster backend:** a pool of single-threaded CPU sim processes fed with `--n-parts 32` slices; `srun --exact` steps cap CPU and memory where a hard cap is wanted (for the agents).
- **Decision 4:** the network can be cut with a seccomp filter (no IP sockets, a Unix socket to the model server); to be confirmed in a job with OpenCode before relying on it.
- **Full-node jobs:** they go to `gpu` (exclusive) or `gpux`; `gpu1` is for jobs of at most half a node.
- **Scratch cleanup:** copy weights and results off scratch before the 12-week cleanup starts on 2027-02-01.
