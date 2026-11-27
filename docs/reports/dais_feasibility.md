# DAIS feasibility for the open-weight policy finder

Phase 0 (DAIS part) and the first half of phase 2 of `docs/plans/policy-finder-open-weights.md`, measured on 2026-11-27 under account `mpib_inst`. Scripts: `infra/dais/`. Revised after a review of the first version; section "Not yet measured" lists what the revision planned but could not run (the SSH connection to DAIS dropped before the new jobs were submitted).

## Where the numbers come from

| Job | Partition, QoS | Node | What |
|---|---|---|---|
| 986457 | gpudev, debug | daisg103 (H200) | network test |
| 986513 | gpudev, debug | daisg103 (H200), 2 GPUs | serve and probe Qwen3-Coder-Next |
| 986771 | gpudev, debug | daisg103 (H200), 1 GPU, 32 threads | sim image on the GPU, CPU fallback, smoke-rule timings |
| 986879 | gpudev, debug | daisg103 (H200), 2 GPUs, 64 threads | serve, probe, 5-session benchmark, smoke-rule sims beside it |
| 986869 | small, cpu | daisg302 (AMD EPYC 9555, an RTX node), 32 threads | `pf-20261104-1` final run on the CPUs |

All GPU jobs ran on `gpudev` (15 minutes) with an earlier version of the scripts. The scripts' default path (`gpu1`, QoS `g0002`) and `serve.slurm.sh`'s `MODE=serve` have not been run. No Raven baseline could be run for this report (no Raven access); Raven numbers are the committed results of earlier runs.

## Summary

- **The sim image cannot use the H200 or B200 GPUs, but runs unchanged on the CPUs.** `algoinst.sif` (byte-identical to Raven's) has no kernels for H200 (`no kernel image is available for execution on the device`; torch 1.11+cu113 ships sm_37 to sm_86, no PTX). On the CPUs the repo's `python -m aimanager simulate` runs. A real rule costs about 12 to 14 s of wall time per 1,000 episodes in one 32-thread process (job 986869), so a real sweep (256 points x 500 episodes = 128,000 episodes) would take about 25 to 30 minutes in one such process; how that improves with more processes is not yet measured.
- **CPU and A100 results: no difference detected at this power.** A 2000-episode, 6-point final run gives no detectable mean shift (each difference has a standard error of about 1.45 pool points, so shifts below about 4 points would go unseen). The ranking of points may differ: Raven's points are heterogeneous, DAIS's are not (see section 2).
- **Qwen3-Coder-Next serves on 2 H200 with the group's `vllm.sif` (v0.25.1), unchanged.** Ready in 281 to 386 s; tool calls were well formed in one trial each of four checks; 5 concurrent sessions decode at 82 to 133 tokens/s each.
- **Sharing a node between vLLM and CPU sims is not yet measured properly.** The one attempt (job 986879) overlapped generation with only about 47 s of a 108 s smoke-rule sim, during which the unpinned sim made little progress; it supports no conclusion about contention either way.
- **No network namespace for users on DAIS**, so `apptainer --net --network none` is refused, and there is no Docker-compatible runtime: the spawner cannot run unchanged. A Unix-socket path to a server works; file isolation works. A seccomp filter against IP sockets is written but untested.

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

What was shown, as an unprivileged user with apptainer 1.5.2. The login-node tests ran from an ad-hoc script whose output was not saved; `infra/dais/iso_check.sh` is that script, extended, and has not been run yet. The in-job rows come from job 986771's saved log.

| Test | Result | Evidence |
|---|---|---|
| `--containall --net --network none` | Refused: `Failed to create network namespace: maximum number of network namespaces exceeded` (`/proc/sys/user/max_net_namespaces` = 0). `unshare -rn` fails as well. | login node; job 986771 |
| Without a network namespace | The container shares the host network; a test server on the host's `127.0.0.1` answered from inside. | login node |
| HTTP over a Unix socket in a bound directory | Works with `--containall`. | login node |
| A relay inside the container to a base URL | Works: a 15-line Python relay (no `socat` in the image) served the Unix-socket server at `http://127.0.0.1:18766`. | login node |
| Binds | A `:ro` bind is read-only, a second bind of a subfolder is writable; with `--containall` neither scratch nor the real home is visible and `$HOME` is empty; the image root is read-only; `/tmp` is writable. | login node |
| `--cpus`, `--memory` | Refused on the login node (`cpu.max: no such file or directory`) and in a job (`cannot use cgroups - XDG_RUNTIME_DIR ... is not writable`). | login node; job 986771 |
| `--pids-limit` | Refused in a job (same `XDG_RUNTIME_DIR` error). On the login node it was accepted, but the test did not show that forks fail above the limit; `iso_check.sh` now counts them. | job 986771 |
| A seccomp filter denying `AF_INET`/`AF_INET6` sockets | Not tested yet (`iso_check.sh` section 2b). | none |
| Slurm job steps as CPU and memory caps | Not tested yet (`cpu_bench.slurm.sh`, B4). | none |
| Docker image to SIF | Works without root, from `docker://alpine:3.20` and from a `docker save` tarball (`docker-archive://`). | login node |
| Docker-compatible runtime | None: no `podman`, `docker`, `nerdctl`, `buildah`, no subuid range for the user, no `slirp4netns`/`pasta`. | login node |

So the agent's network can be neither cut by a namespace nor, so far, filtered. MPCDF's Apptainer sandbox for coding agents (`gitlab.mpcdf.mpg.de/mpcdf/ai-cli-agents-container`, Bits and Bytes 222) ships `opencode` and confines an agent to its working directory and a fake home; it has the same network situation.

## 2. The sim image on DAIS

- The image was streamed read-only from Raven's shared copy; its SHA-256 on DAIS is `fe358c02...91550ff`, the same.
- **H200: does not work** (job 986771): `torch.cuda.is_available()` is `True`, then `RuntimeError: CUDA error: no kernel image is available for execution on the device`, after torch's warning that H200 (sm_90) is not among sm_37 ... sm_86. B200 (sm_100) was not tried; the arch list rules it out.
- **CPU: works, unchanged**, with `CUDA_VISIBLE_DEVICES=""`.

**Timings.** The smoke rule (`punishment = 0`, job 986771, 32 threads on daisg103) is cheap and not representative: 34 s for one batch of 1,000 episodes and 66 s for 16 points x 500 episodes (8,000 episodes), each with about 25 s of start-up. That "sweep" is 16 times smaller than a real one. The real rule of `pf-20261104-1` (5 parameters, job 986869, 32 threads on daisg302, an AMD node) took 167 s for 12,000 episodes, about 12 to 14 s per 1,000 episodes depending on how much of the start-up is counted. A real sweep, `generate_sim_config.py`'s default of 256 Sobol points x 500 episodes = 128,000 episodes, would take about 25 to 30 minutes in one such process, by extrapolation. Thread count: the smoke-rule sweep took 62 and 69 s in the 64-thread job 986879 and 66 s in the 32-thread job 986771, so doubling the threads gained nothing at that size. How time scales with threads and with splitting into processes is not measured yet (B1).

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
- **Checks run** (986513, 986879), one trial each at temperature 1.0, all passed: a chat completion; a tool call with `finish_reason: tool_calls` and JSON arguments with every required key; the same streamed; a tool call after one 83,132-token user message.
- **Not yet run:** the revised `probe_endpoint.py`, which adds a multi-turn history of earlier tool calls and long tool results, several trials per check at temperature 0.2, and checks of `finish_reason`, the tool name and degenerate repetition.
- **Usage:** `prompt_tokens`, `completion_tokens`, `total_tokens`; `prompt_tokens_details.cached_tokens` only with `--enable-prompt-tokens-details`; streamed usage with `stream_options.include_usage`.
- **Load time:** 386 s cold (weights 145 s from GPFS, torch.compile 50 s, engine start-up with profiling and CUDA graphs 96 s), 281 s the second time.
- **Memory per GPU** at utilisation 0.88: weights 74.3 GiB, KV cache 44.3 GiB, CUDA graphs 1.1 GiB; 129,371 of 143,771 MiB used.
- **KV cache:** 44.3 GiB per GPU held 3.78M tokens, about 25 KB per token across both GPUs. That is unusually small, because most of the model's layers are linear attention; other models need far more per token.

## 4. Sharing a node: what was and was not measured

Job 986879 measured five concurrent sessions alone (1024 generated tokens each, `ignore_eos`):

| Prompt per session | Round | Decode tok/s per session (median) | Aggregate tok/s | Time to first token (median) |
|---|---|---|---|---|
| 29k | cold | 109 | 426 | 2.6 s |
| 29k | warm (prefix cached) | 133 | 643 | 0.23 s |
| 94k | cold | 82 | 265 | 6.7 s |
| 94k | warm | 119 | 552 | 0.53 s |

- The cached fraction of the warm prompts was 98 to 99 percent.
- The same job then ran smoke-rule sims, unpinned, on the job's CPUs while repeating the benchmark. The file timestamps show that generation overlapped only about 47 s of the 108 s sim, and in those 47 s the sim did roughly 1 to 9 s of work: an unpinned sim nearly stalls while the server generates.
- The benchmark rounds during that window resemble the solo rounds, but they ran against a sim that was losing the CPU, so neither "generation is not slowed" nor a sim slowdown factor follows.
- `serve_bench.slurm.sh` (B3) repeats this properly: the server on its own cores, sims pinned to the others, and steady generation for the whole sim.

The sim's GPU memory could not be measured on DAIS. The code comment on `MAX_EPISODE_BATCH_SIZE` says about 9 GB on an A100 at batch 20,000. Memory is driven by `episode_batch_size`, plus per-batch records that stay on the device until the sim ends (`run_batched`).

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

## Not yet measured

Planned for this revision and written as scripts, but not run (the SSH connection to DAIS dropped):

- **B1** (`cpu_bench.slurm.sh`): one slice of a real sweep (`generate_sim_config.py --n-parts 32`: 8 points x 500 episodes of `pf-20261104-1`'s rule) at 1 to 32 threads, then 16 cores filled with k processes of 32/k threads: CPU seconds per 1,000 episodes and the best process/thread shape. torch 1.11 in the image does honour `OMP_NUM_THREADS` (checked on the login node: 3 threads when set to 3, otherwise all cores); the sim does not set it itself, while the Raven pipeline sets `OMP_NUM_THREADS=2` only for `validate-rule`.
- **B2** (`sweep_pack.slurm.sh`): 1, 5, 10, 20, 40 concurrent slices, pinned one per core versus not pinned; node throughput, peak memory per process, kills.
- **B3** (`serve_bench.slurm.sh`): the server on 4 reserved cores, 16 slices pinned to the other cores, steady generation for the whole duration; both against their solo values.
- **B4** (`cpu_bench.slurm.sh`): `srun --exact -c 2 --mem=...` steps: CPUs seen, CPU use of a busy step, a step over its memory, a sim in a step.
- **B5** (`serve_bench.slurm.sh`): 5, 10, 20, 40, 80 concurrent warm sessions with 30k-token prompts on the default `gpu1`/`g0002` path.
- The isolation script `iso_check.sh` (pids limit with fork counts, seccomp filter) and the revised probe.
- A Raven baseline for any of these: not available from here.

The node layout for about 20 concurrent chains depends on B1 to B3 and is not proposed yet.

## What can be reused

| Part | Reuse |
|---|---|
| `algoinst.sif` | Unchanged, on the CPUs. Not on H200/B200 GPUs. |
| The group's `vllm.sif` | Unchanged; version asserted at start. |
| Sister project's serve conventions (`../understanding/scripts/slurm/lib/evo_common.sh`) | Followed: `apptainer exec --nv`, a home that is not the real one, `HF_HOME` on scratch, explicit prefix caching, readiness on `/v1/models`. Added: a port and API key per job, readiness that refuses a server answering without the key. |
| Repo entry points | `python -m aimanager simulate` on configs from `generate_sim_config.py` (including `--n-parts`), unchanged; the Raven job template's container layout copied into `in_sim_image`. |
| Agent spawner (Docker socket) | Cannot run: no Docker-compatible runtime. Needs an Apptainer backend, possibly MPCDF's agents container. |
| Network rule of the Docker sandbox | Not reproducible with namespaces; a seccomp filter is untested. File isolation is reproducible. |
| Raven `cluster_proxy` functions | A thin adapter: a sim becomes a CPU process (or a Slurm job step) in the sim image. |

## Scripts

`infra/dais/common.sh` holds the settings and helpers. Every job writes into its own run folder under scratch (`$PF_RUN`), never into the checkout; the checkout is bound read-only into the sim container, and failures make the job exit non-zero.

| Script | Purpose | Run? |
|---|---|---|
| `serve.slurm.sh` | serve and probe, or keep serving | earlier version, on gpudev |
| `probe_endpoint.py` | tool-call checks, several trials | earlier version |
| `bench_load.py` | concurrent sessions, rounds or steady load | earlier version |
| `sim_check.slurm.sh` | network and Apptainer checks, GPU test, reference sims on the CPUs | earlier version (986771) |
| `sim_cpu.slurm.sh` | one sim on the CPUs of the `small` partition | earlier version (986869) |
| `cpu_bench.slurm.sh` | B1, B4 | no |
| `sweep_pack.slurm.sh` | B2 | no |
| `serve_bench.slurm.sh` | B3, B5 | no |
| `iso_check.sh` | sandbox checks | earlier ad-hoc version, output not saved |
| `cpu_slots.py` | the job's CPUs grouped by core, for pinning | no |
| `sim_variant.py` | GPU check; a config copy with another output folder or shape | yes |

The scripts expect a checkout at `/u/$USER/algorithmic-institutions-pf` with the model files, submitted from it, and `/dais/fs/scratch/$USER/policy-finder/logs` to exist.

## Changes this suggests for the plan

- **Decision 1:** run the sims on the node's CPUs in the existing image and give vLLM the GPUs. The core split between server, agents and sims waits for B1 to B3.
- **Phase 3's cluster backend:** a pool of CPU sim processes, or `srun` job steps if B4 shows they cap CPU and memory.
- **Decision 4:** the network cannot be cut by a namespace; the fallback (restricted tools, audited transcripts) applies unless a seccomp filter works.
- **Full-node jobs:** they go to `gpu` (exclusive) or `gpux`; `gpu1` is for jobs of at most half a node.
- **Scratch cleanup:** copy weights and results off scratch before the 12-week cleanup starts on 2027-02-01.
