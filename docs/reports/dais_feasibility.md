# DAIS feasibility for the open-weight policy finder

Phase 0 (DAIS part) and the first half of phase 2 of `docs/plans/policy-finder-open-weights.md`, measured on 2026-11-27 under account `mpib_inst`. Scripts: `infra/dais/`. Revised twice after independent reviews.

## Where the numbers come from

| Job | Partition | Node | What |
|---|---|---|---|
| 986457 | gpudev | daisg103 (H200) | network test |
| 986513 | gpudev | daisg103 (H200), 2 GPUs | serve and probe Qwen3-Coder-Next |
| 986771 | gpudev | daisg103 (H200), 1 GPU | sim image on the GPU, CPU fallback |
| 986879 | gpudev | daisg103 (H200), 2 GPUs | serve, probe, 5-session benchmark, unpinned smoke-rule sims beside it |
| 986869 | small | daisg302 (AMD EPYC 9555) | `pf-20261104-1` final run on the CPUs (32 threads) |
| 987428 | small | daisg301 (AMD), 48 cores | B1, B4, earlier version of `cpu_bench.slurm.sh` |
| 987622 | small | daisg302 (AMD), 48 cores | B2 (`sweep_pack.slurm.sh`) |
| 987623 | gpudev | daisg103 (Intel CPUs of an H200 node), 12 cores | B1 on Intel, short shapes (`cpu_bench.slurm.sh`) |
| 987430 | gpudev | daisg103 (H200), 2 GPUs, 24 cores | B3 and B5, earlier version of `serve_bench.slurm.sh` |
| 987881, 988058 | small | daisg301 (AMD) | `iso_check.sh` inside a job (988058 with the committed scripts) |
| 987890 | gpudev | daisg111 (H200), 2 GPUs, 24 cores | the layout at 20 sessions with repeats, B5 repeats (`serve_bench.slurm.sh`, cut at the 15-minute limit) |
| 987891 | small | daisg302 (AMD), 24 cores | a whole "sweep, fit, final" cycle (`cycle_bench.slurm.sh`) |
| 987892 | small | daisg301 (AMD), 48 cores | B1 and B4 with the committed `cpu_bench.slurm.sh` |
| 987893 | small | daisg301 (AMD) | committed `sim_cpu.slurm.sh` |
| 987950, 987951 | gpudev | see section 3 and "Scripts" | committed `serve.slurm.sh` (probe) and `sim_check.slurm.sh` |

All GPU jobs ran on `gpudev` (15 minutes, one job per user, nodes shared with other jobs); a `gpu1` job of 2 GPUs and 24 cores was estimated to start a day later, so none ran there. The `small` nodes are shared with other users' jobs as well. No Raven baseline could be run (no Raven access); Raven numbers are committed results of earlier runs.

## Summary

- **The sim image cannot use the H200 or B200 GPUs, but runs unchanged on the CPUs**, and the cheapest shape is single-threaded processes, one per hyperthread. With 36 such processes on 18 Intel cores of an H200 node, 144,000 episodes (a little more than one real sweep, 128,000) took 32 to 34 s per round, about 4,300 episodes per second, unchanged while vLLM served 20 sessions on its own cores (job 987890, two repeats).
- **A whole "sweep, fit, final" cycle took 78 s** on 24 AMD cores (job 987891), of which the final run, one process of 12,000 episodes, took 33 s; on the Intel nodes and under load it should take about 2 to 3 minutes (an estimate, not measured there).
- **CPU and A100 results: no difference detected at this power** (2000-episode final run of `pf-20261104-1`); shifts below about 4 pool points, or a different ranking of close points, are not excluded.
- **Qwen3-Coder-Next serves on 2 H200 with the vLLM image v0.25.1, unchanged.** Tool calls were well formed in every trial, including after a 70k-token history of tool calls. Warm aggregate generation: 1,320 to 1,570 tokens/s at 20 sessions, about 2,040 at 40, 2,770 at 80 (one round).
- **Network isolation: a seccomp allow-list works, as a barrier against ordinary network use.** It refuses every socket but `AF_UNIX`, including a 64-bit-argument bypass that defeated the first, deny-list version, and is applied either to a whole container (Apptainer profile) or only to the commands an agent harness runs (`noinet.py`, tested with a stand-in harness that keeps its own TCP connection). It does not cover sockets a process already holds or receives, the host's abstract Unix sockets, path Unix sockets of services that act for the caller (the user's systemd/D-Bus socket, a tmux or screen server), or files that the unfiltered harness later runs or reads; so the agent runs inside `--containall` and the wrapper and the harness configuration must be read-only to it. Users get no network namespace and no Docker-compatible runtime on DAIS.

## 1. Cluster facts

| Item | Finding | Evidence |
|---|---|---|
| Partitions | `gpu` is `OverSubscribe=EXCLUSIVE`: any job there holds a whole node. `gpu1` is the shared partition; per GPU at most 12 cores (24 threads) and 250 GB. `gpudev`: shared, 15 min, at most 4 GPUs, one job per user. `small`: CPU only, on the two AMD RTX nodes, at most 64 cores and 750,000 MB. `gpux`: 10 days, 2 H200 nodes. | `scontrol show partition`, `sbatch --test-only`, user guide |
| GPU type | `--gres=gpu:h200:N` selects H200. | jobs above |
| QoS | `g0001` to `g0008` carry no TRES limit in `sacctmgr`. A full node would be `-p gpu --gres=gpu:h200:8 --qos=g0008` (not tested). | `sacctmgr` |
| Nodes | H200 nodes: 8 x H200 (143,771 MiB), 2 x Xeon Platinum 8568Y+ (96 cores, 192 threads), 2 TB RAM. `small`: AMD EPYC 9555. | `lscpu`, `nvidia-smi` |
| Slurm CPUs | `-c N` in a job gives N cores (2N hyperthreads); a `srun --exact -c 2` step got 2 CPUs on two different cores but is accounted 4 (`CR_ONE_TASK_PER_CORE`), so exclusive steps use a core each at least. | jobs 987428, 987892 |
| Network | Login and compute nodes reach `huggingface.co`, `openrouter.ai`, `pypi.org`, `github.com`. | 986457, 986771 |
| Scratch | Quota 8 TB and 8M files; no backup; from 2027-02-01 files not accessed for 12 weeks are deleted. GPFS `du` shows about twice the apparent size. | login message, `mmlsquota` |
| Weights download | 149 GB in 5 minutes on the login node, niced, 4 workers. | download log |

## 2. Network isolation and the agent sandbox

Tests as an unprivileged user with apptainer 1.5.2, by `infra/dais/iso_check.sh` with `net_probe.py`, `noinet.py` and `harness_standin.py`. Outputs: `docs/reports/dais_iso_check_login.txt` (login node) and `docs/reports/dais_iso_check_job.txt` (job 988058 on daisg301, an AMD compute node; not yet repeated on an H200 node), both from the committed scripts.

| Test | Result |
|---|---|
| `--net --network none` | Refused: `maximum number of network namespaces exceeded` (`max_net_namespaces` = 0), login node and job. |
| Without a network namespace | The container shares the host network: the host's `127.0.0.1` services and its abstract Unix sockets are reachable. |
| Unix socket in a bound folder | Works with `--containall`. |
| First seccomp profile (deny-list: `socket` with family `== AF_INET` or `== AF_INET6`) | **Bypassable.** `syscall(SYS_socket, 0x100000002, SOCK_STREAM, 0)` created a socket (the comparison sees all 64 bits, the kernel uses the low 32), and `AF_NETLINK` sockets were allowed. |
| Allow-list profile `infra/dais/no_inet.seccomp.json` (`socket`/`socketpair` refused unless the family is exactly `AF_UNIX`, all 64 bits compared; `io_uring_setup` refused; architectures `SCMP_ARCH_X86_64`) | `AF_INET`, `AF_INET6`, `AF_NETLINK`, `AF_PACKET` and the 64-bit bypass refused with EPERM; `AF_UNIX` sockets and `socketpair` work and the Unix-socket server answers; an x32 or i386 (`int 0x80`) socket call kills the calling process with SIGSYS. Login node and job alike. |
| `noinet.py` (the same allow-list installed by a process on itself, with `PR_SET_NO_NEW_PRIVS`, then `exec`) | The same refusals, foreign-ABI calls return EPERM instead of killing; `curl http://127.0.0.1:<port>` fails. |
| io_uring | Not testable here: the kernel disables it for everyone (`io_uring_disabled` = 2). Both filters refuse `io_uring_setup` anyway. |
| What the filters do not cover | An inherited TCP socket stays usable (shown: the parent opens a fresh socket for each run, the filtered child sends its own request over it and reads the answer); sockets received over a Unix socket (`SCM_RIGHTS`) would too (follows, not tested); the host's abstract Unix sockets stay reachable (shown). By design also: path Unix sockets of services that act for the caller, and files the unfiltered harness later runs or reads (next paragraph). |
| A relay inside the container to give a client a TCP base URL | Works without a filter; **cannot work with the filter**, which refuses the relay's own `AF_INET` socket. Seccomp cannot restrict an address to loopback. |
| Binds | A `:ro` bind is read-only, a second bind of a subfolder is writable; with `--containall` neither scratch nor the real home is visible; the image root is read-only. |
| `--cpus`, `--memory` | Refused (login node: controllers not delegated; job: `XDG_RUNTIME_DIR ... is not writable`). |
| `--pids-limit` | Works on the login node (with 64, a shell stops at `Cannot fork` before starting 200 processes; without, all 200 start); refused in a job. |
| Slurm steps as caps | Work (jobs 987428, 987892): a `-c 2` step is held to 2 CPUs (8 busy loops for 20 s used 40 CPU-s); a step over its `--mem` is OOM-killed and the job carries on. With `--overlap`, 8 steps of `-c 1 --mem=1G` ran at once on 2 cores, and an overlapping step over its memory was still killed. |
| Docker | No Docker-compatible runtime (no podman, docker, nerdctl, subuid range, slirp4netns); `apptainer build` from `docker://` and from `docker save` tarballs works. |

**The harness and its tools.** The stand-in harness (`harness_standin.py`: a parent that talks HTTP to a local server and runs its "shell tool") showed the design that fits an agent harness such as OpenCode, which needs TCP to the model server:

- the harness keeps the network and talks TCP to vLLM on `127.0.0.1` with the job's API key;
- its shell runs through `noinet.py` (`noinet.py /bin/bash ...`), so every command the model runs, and everything those commands start, cannot open IP, netlink or packet sockets;
- the parent reached its server before and after its filtered child failed to.

What this does not cover. The filtered commands run as the same user, on the same filesystem, as the unfiltered harness, so:

- **The harness's own actions:** its built-in network tools (for example OpenCode's web fetch) must be switched off in its configuration.
- **Sockets it passes on:** file descriptors given to the shell must not include open sockets; an inherited socket works through the filter.
- **The host's abstract Unix sockets:** reachable, because the network namespace is shared.
- **Services that act for the caller, over path Unix sockets.** On the login node the user's systemd/D-Bus socket `/run/user/<uid>/bus` exists, and `systemd-run`, `busctl`, `tmux` and `screen` are installed. A filtered command could ask the systemd user manager, or a tmux server started outside, to run an unfiltered process (not tried). In a job on daisg301 the socket was absent. Inside `--containall` it is not visible, on the login node and in the job (`iso_check.sh` section 5b).
- **Files the unfiltered harness later runs or reads:** a filtered command that can write `noinet.py`, the harness configuration or shell start-up files removes the filter for later commands.

Consequences for use: the agent runs inside `apptainer exec --containall`, and the wrapper and the harness configuration are read-only to it (bound `:ro`, outside the agent's writable folders). Even then the whole claim is "a barrier against ordinary network use by the agent's commands"; it is not a sandbox against a hostile process. Not tested: OpenCode itself, and the filter on an H200 node. The `noinet.py` refusals, including the x32 and i386 ones, hold in the container and on the host's own Python 3.6 (`iso_check.sh` section 5).

## 3. The sim image on DAIS

- The image was streamed read-only from Raven's shared copy; its SHA-256 on DAIS is the same (`fe358c02...91550ff`, asserted by `sim_check.slurm.sh`).
- **H200: does not work** (job 986771): `RuntimeError: CUDA error: no kernel image is available for execution on the device`; torch 1.11+cu113 ships sm_37 to sm_86 and no PTX. B200 (sm_100) is ruled out by the same list.
- **CPU: works, unchanged**, with `CUDA_VISIBLE_DEVICES=""`. The results do not depend on the thread count: the `pf-20261104-1` final run gave the same six pools with 32 threads (986869) and with 1 thread (987893).
- **First timings (superseded).** The smoke rule (`punishment = 0`) is cheap and not representative, its 66 s "sweep" was 16 times smaller than a real one, and the 32-thread final run (986869, 167 s) was slow because of the threads (B1).

**Against Raven.** The 20-episode smoke sims say little (CPU and CUDA random streams differ). The 2000-episode final run of `pf-20261104-1` (that branch's code; the model files' checksums match), seed 1000:

| Point | Raven (A100) | DAIS (CPU) | Difference | z |
|---|---|---|---|---|
| gp | 66.00 +- 1.07 | 64.28 +- 1.06 | -1.72 +- 1.51 | -1.14 |
| s183 | 67.58 +- 1.07 | 66.52 +- 1.06 | -1.06 +- 1.51 | -0.70 |
| s063 | 64.99 +- 0.99 | 66.35 +- 1.03 | +1.37 +- 1.42 | +0.96 |
| s043 | 63.33 +- 0.98 | 64.93 +- 1.00 | +1.60 +- 1.40 | +1.14 |
| s255 | 63.63 +- 1.03 | 65.18 +- 1.01 | +1.55 +- 1.44 | +1.07 |
| s103 | 62.38 +- 0.98 | 64.43 +- 1.00 | +2.05 +- 1.40 | +1.46 |

- **No difference detected at this power** (chi-squared 7.3 on 6 df). Shifts below about 4 pool points would go undetected, while the policy finder ranks points 1 to 5 points apart.
- **A ranking difference is not excluded.** The six points are heterogeneous on Raven (chi-squared 17.2 on 5 df) but not on DAIS (4.2). The top two lead the other four by +3.2 +- 0.9 on Raven and +0.2 +- 0.9 on DAIS. This comparison was chosen after seeing the data, so it is a hint only.
- **The six differences are independent:** `run_batched` seeds batch i with `seed + i` and each point fills its own batches.

**The two identical Raven smoke points.** In Raven's smoke final run, `gp` and `s000` (the same rule, which punishes nothing) have the same pool mean to 14 digits but different standard errors. With zero punishment the mean is 1.6 x group-0 contributions / 480. Both runs' group-0 totals are 16,594, while their per-episode values and row counts (2,468 and 2,072 agent-rounds) differ. No shared data was found: as far as code and data show, a coincidence of integer totals; not fully explained.

GPU sims would need a rebuilt image; the repo's `Dockerfile.gpu` (torch 2.5.1+cu124) covers H200 but not B200, and its results would need validating against the reference sims. Not done here.

## 4. Serving Qwen3-Coder-Next

- Weights: `Qwen/Qwen3-Coder-Next` revision `a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb` (Apache 2.0, not gated), served offline from a read-only `HF_HOME`; 148.4 GiB.
- Image: the vLLM image named by `VLLM_SIF`, label `vllm/vllm-openai:v0.25.1`, asserted at start (the model needs >= 0.15).
- Serve arguments: `--tensor-parallel-size 2 --gpu-memory-utilization 0.90 --max-model-len 131072 --enable-prefix-caching --enable-prompt-tokens-details --enable-auto-tool-choice --tool-call-parser qwen3_xml`, in `--containall` with a per-job home, a shared compile cache, a free port and a per-job API key (refused without it, which `wait_ready` checks).
- **Parsers:** in v0.25.1 `qwen3_coder` and `qwen3_xml` are the same class (`Qwen3EngineToolParser`).
- **Tool calls:** 986513 and 986879, one trial each at temperature 1.0; the revised `probe_endpoint.py` at temperature 0.2, 2 trials per check in 987430 and 3 in 987950 (the committed `serve.slurm.sh`). All passed. A pass needs `finish_reason: tool_calls`, the expected tool, JSON arguments with every required key and no degenerate repetition. The checks: chat; a tool call, plain and streamed; after one 71k-token message; after a 70.7k-token history of three `run_shell` calls with long results, plain and streamed. Few trials; the bake-off will test far more.
- **Load time:** 231 to 412 s (the longest on a node whose page cache did not hold the weights and with an empty compile cache).
- **Memory per GPU** at 0.90: 132,247 MiB used; KV cache 4.02M tokens, about 25 KB per token, unusually small (most layers are linear attention).

## 5. Sims on the CPUs, generation on the GPUs

A slice is one part of a real sweep split by `generate_sim_config.py --n-parts 32` (`pf-20261104-1`'s 5-parameter rule, 256 Sobol points x 500 episodes): 8 points x 500 episodes = 4,000 episodes. Each process runs `python -m aimanager simulate` in the unchanged image with `OMP_NUM_THREADS`/`MKL_NUM_THREADS` set (torch 1.11 honours them; the sim sets none), pinned with `taskset`. Times are GNU `time` per process, start-up included.

### B1: threads and processes

One slice alone:

| Threads | AMD, committed script (987892): wall, CPU s | Intel H200 node (987623): wall, CPU s |
|---|---|---|
| 1 | 15.3, 15.6 | 29.6, 30.5 |
| 2 | 23.3, 34.8 | 26.8, 39.1 |
| 4 | 27.1, 67.5 | 33.7, 79.1 |
| 8 | 28.6, 136.6 | |
| 16 | 33.9, 337.8 | |
| 32 | 68.5, 1702.8 | |

16 AMD cores (987892) or 12 Intel cores (987623) filled with k processes of t threads, one slice each; configs prepared before the timed part (987892) and the round's wall time:

| Cores | k x t | Episodes | Round wall | Process wall, median (slowest) | Episodes/s |
|---|---|---|---|---|---|
| AMD 16 | 1 x 32 | 4,000 | 71 s | 70.5 s | 56 |
| AMD 16 | 2 x 16 | 8,000 | 38 s | 33.7 (37.5) s | 210 |
| AMD 16 | 4 x 8 | 16,000 | 44 s | 27.0 (43.8) s | 360 |
| AMD 16 | 8 x 4 | 32,000 | 37 s | 24.0 (36.1) s | 860 |
| AMD 16 | 16 x 2 | 64,000 | 25 s | 25.4 (25.7) s | 2,560 |
| AMD 16 | 32 x 1 | 128,000 | 29 s | 28.4 (29.0) s | 4,410 |
| Intel 12 | 6 x 4 | 24,000 | 34 s | 32.9 (33.9) s | 710 |
| Intel 12 | 12 x 1, one per core | 48,000 | 31 s | 30.3 (31.3) s | 1,530 |
| Intel 12 | 12 x 2 | 48,000 | 28 s | 28.2 (28.7) s | 1,670 |
| Intel 12 | 24 x 1, one per hyperthread | 96,000 | 38 s | 34.6 (38.0) s | 2,530 |

- **Cheapest per CPU: single-threaded, one per hyperthread.** A second thread can shorten one slice's wall time when the CPUs would otherwise idle (Intel: 26.8 s against 29.6 s), but it costs about 30 percent more CPU on Intel and more than twice as much on AMD, so a full pool does more work single-threaded.
- **The 4 x 8 and 8 x 4 AMD rows are set by one or two slow processes each.** Their medians are close to the 16 x 2 row; the `small` nodes are shared with other users' jobs.
- **Two Intel oddities.**
  - One single-threaded slice alone costs 30.5 CPU-s on Intel against 15.6 on AMD.
  - Filling each core's second hyperthread gives 1.65x on Intel (24 x 1 against 12 x 1) but about 1.25x on AMD (B2: 190 episodes/s per core at one slice per core, 242 at two).
  - Both suggest the lone Intel process ran slower than it could (clock, or a busy sibling hyperthread on the shared node), so single-process Intel numbers are less reliable than the full-pool ones.

### B2: many slices at once (987622, AMD, 48 cores, single-threaded)

| Slices at once | Pinned: slowest, episodes/s | Not pinned: slowest, episodes/s |
|---|---|---|
| 1 | 19.6 s, 210 | 15.7 s, 250 |
| 5 | 20.0 s, 1,000 | 16.9 s, 1,180 |
| 10 | 19.1 s, 2,110 | 17.9 s, 2,220 |
| 20 | 19.1 s, 4,210 | 19.3 s, 4,000 |
| 40 | 21.1 s, 7,620 | 26.5 s, 5,930 |
| 96 (every hyperthread) | 32.4 s, 11,640 | 35.4 s, 10,970 |

- Pinning helps once slices compete for cores (+29 percent at 40, +6 percent at 96).
- At small numbers the pinned rounds were slower. Two confounds: each pinned round ran first, and a few CPUs at the head of the list were slower than the rest throughout, likely from other users' load on the shared node.
- All 344 slices exited 0; peak RSS was 1.1 to 1.4 GB per slice; output about 3 MB per slice.

### B3: the layout on one allocation (987890, Intel, 2 H200, 24 cores)

- **Layout:** vLLM (TP 2) on 4 cores; the benchmark client on 2 other cores; a pool of 36 single-threaded slices on the remaining 18 cores (one per hyperthread). The slices were launched together, with their configs prepared beforehand.
- **Order of each repeat:** (a) the pool alone with the server loaded and idle; (b) 20 warm 28k-token sessions alone, steady, for as long as (a) took; (c) the pool started once the same 20-session generation was steady.

| Repeat | Pool alone: round, slice median | Pool beside generation: round, slice median | Generation alone: tok/s, per session | Generation beside the pool |
|---|---|---|---|---|
| 1 | 33 s, 33.0 s | 34 s, 33.0 s | 1,571, 80.9 | 1,564, 81.0 |
| 2 | 32 s, 31.8 s | 34 s, 32.6 s | 1,574, 81.2 | 1,565, 80.1 |

- Each pool round played 144,000 episodes (36 slices), about 4,300 episodes per second, or 240 per core.
- With separate cores, sims and generation did not measurably slow each other (slice medians at most 3 percent slower beside generation, rounds 1 to 2 s longer; generation under 1 percent slower).
- Two repeats at 20 sessions, one node; the earlier run 987430 (5 sessions, 2-thread slices started one by one, client on the server's cores) agreed.

### B5: generation against concurrency (Qwen3-Coder-Next on 2 H200)

Warm sessions (99 percent of the 27k to 29k-token prompt cached), 512 generated tokens each:

| Sessions | Run | Per session, median | Aggregate | Time to first token, median (max) |
|---|---|---|---|---|
| 5 | 987430 | 135 tok/s | 631 tok/s | 0.25 (0.36) s |
| 10 | 987430 | 102 | 934 | 0.41 (0.68) s |
| 20 | 987430 | 86 | 1,329 | 1.78 (1.79) s |
| 20 | 987890, rounds 1 and 2 | 73, 83 | 1,325, 1,461 | 0.67, 0.61 s |
| 40 | 987430 | 62 | 2,159 | 0.82 (1.72) s |
| 40 | 987890, rounds 1 and 2 | 61, 62 | 2,039, 2,048 | 1.14, 1.24 s |
| 80 | 987430 | 43 | 2,769 | 2.06 (4.13) s |

- Aggregate throughput was still growing at 80 sessions in the one round measured there; per-session speed falls to about half from 5 to 40 sessions.
- In 987430 the client shared the server's 4 cores; in 987890 it had 2 cores of its own. At 20 and 40 sessions the two agree within run-to-run variation, so the client was not the limit there. The 80-session row was not repeated (987890 hit its time limit) and may be client-limited.
- Time to first token varies between rounds; the dip at 40 in 987430 is noise.
- Cold prompts are limited by prefill (about 54,000 tokens/s).

### B4: Slurm steps

See the sandbox table in section 2. Exclusive steps take at least one core each, so 20 agents cannot be 20 exclusive steps on 16 cores; `--overlap` steps on a shared CPU set can, with a memory cap each (tested with 8 steps on 2 cores).

### The "sweep, fit, final" cycle (987891, AMD daisg302, 24 cores)

With the pipeline's own scripts in a scratch copy of the checkout, single-threaded throughout:

| Step | Wall |
|---|---|
| `generate_sim_config.py --n-parts 32` | 2 s |
| 32 parts at once (128,000 episodes) | 30 s |
| `merge_sweeps.py` | 2 s |
| `fit_surrogate.py` | 7 s |
| `generate_sim_config.py --final` | 2 s |
| final run: 6 points x 2,000 episodes, one process | 33 s |
| `--score-final` | 1 s |
| whole cycle, with copying the checkout | 78 s |

- The final run cannot be split with the repo's mechanism: `--final` writes one config (`--out` refuses `--n-parts`). Splitting it would need code, and would change its seeds unless done the way `--n-parts` offsets them.
- It is the longest step and runs as one process. Alone it took 33 s on AMD. Estimate for an H200 node, from the single-slice times (29.6 s on Intel against 15.3 s on AMD): about 33 x 29.6 / 15.3, about 64 s; under a full pool, up to twice that. Not measured on Intel.

## 6. A node layout for 20 chains

For one full H200 node (96 cores, 192 hyperthreads, 8 GPUs, 2 TB):

| Use | Cores | How |
|---|---|---|
| vLLM | 8 | pinned with `taskset`; 4 cores served TP 2 at 20 sessions without interference, TP 8 has more worker processes (not measured) |
| benchmark or monitoring clients, the campaign driver | 2 | pinned |
| 20 agents | 16 | one shared CPU set (`taskset`) with one `srun --overlap -c 1 --mem=<cap>` step per agent for a memory cap. Not exclusive steps: those take a core each (`CR_ONE_TASK_PER_CORE`), so 20 would need 20 cores. `OMP_NUM_THREADS=1` in the agents' environment (an unbounded `validate-rule` used about 7 cores for 13 s on the login node). |
| Sweeps | 70 (140 hyperthreads) | a pool of 140 single-threaded slice processes, one per hyperthread, fed with `--n-parts 32` slices; final runs take a pool slot each |

Expected times, from B1 to B3 and the cycle, **each within about a factor of two**:

- **Pool capacity:** about 240 episodes per second per core (987890), so about 17,000 episodes per second for 70 cores, or about 8 s of pool time per real sweep.
- **One cycle on an idle pool:** about 2 minutes (estimate). Its 32 slices take about 35 s, merge and fit about 15 s, and the single-process final run about 64 s on Intel (estimate).
- **All 20 chains' sweeps at once (the worst case):** 2.56M episodes, about 2.5 minutes of pool time, then 20 final runs in parallel of about 1 to 2 minutes each (estimate), so about 4 to 5 minutes for the round.
- **Memory:** about 1.4 GB per slice, about 200 GB for the pool.

Which inputs were measured on other hardware or at another scale than the target:

| Input | Measured on |
|---|---|
| Pool throughput | 18 Intel cores of a shared H200 node (987890); 12 Intel cores (987623); the full node is extrapolated about 4x |
| Many slices at once, memory per slice, nothing killed | AMD `small` nodes (987622) |
| Server cores | TP 2 with 5 and 20 sessions; not TP 8, not a frontier model |
| Step caps, overlapping steps | AMD `small` node (987892) |
| Agents' CPU use | one `validate-rule` on the login node; no agent measured |
| Cycle steps | AMD (987891); the Intel final-run time is estimated from single-thread speed |

So the sims are not the bottleneck: a round of 20 cycles costs minutes on the CPUs, against agent versions that take many model calls. The model side for a frontier model on 8 GPUs is unmeasured.

## 7. The frontier candidates on a full 8 x H200 node

Expected to fit or not, unverified. Weight sizes are the summed safetensors files from the Hugging Face API on 2026-11-27. With the sims on the CPUs, vLLM can take utilisation 0.92 (129.2 GiB of 140.4 GiB per GPU); about 5 GiB of that went to activations, CUDA graphs and buffers for Qwen at TP 2, and may differ for others.

| Model | Weights | Layout | Weights per GPU | Left for KV per GPU | Assessment |
|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 510 GB (475 GiB) | TP 8 | 59 GiB | about 65 GiB | expected to fit; TP 4 leaves about 5 GiB |
| GLM-5.3 | 756 GB (704 GiB), FP8 | TP 8 | 88 GiB | about 36 GiB | expected to fit; whole node |
| GLM-5.3-Flash | 328 GB (306 GiB), FP8 | TP 4 | 76 GiB | about 48 GiB | expected to fit on half a node; TP 2 does not fit |
| MiniMax-M3.1-Flash | 252 GB (235 GiB) | TP 4 | 59 GiB | about 65 GiB | expected to fit on half a node; TP 2 leaves about 7 GiB |

- **Unverified:** vLLM v0.25.1 support for each architecture and quantisation on Hopper.
- **Unknown:** each model's KV size per token; Qwen's 25 KB per token is no guide.
- None of these was downloaded or served.

### GLM-5.3 (profile `glm-5.3`, 2026-11-28)

`MODEL_PROFILE=glm-5.3` (`infra/dais/model_profiles/glm-5.3.sh`) serves `zai-org/GLM-5.3` at revision `ba32f8c1` (the Hub's `main` on 2026-11-28), from the official vLLM v0.34.0 image, with tensor parallel 8; `export_instance.LOCAL_MODELS` holds its OpenCode settings.

| Fact | Value | Source |
|---|---|---|
| Licence | "GLM-5.3 License": MIT-style; a Z.AI safety review only for Model-as-a-Service businesses above USD 10 billion revenue. Ungated, no token needed | `LICENSE` in the snapshot; Hub API `gated: false` |
| Weights | 753.3B parameters, native FP8 (e4m3, 128x128 blocks; 2.1B in BF16): 755 GB (704 GiB) in 141 shards, 156 files in all | `config.json`, Hub API `safetensors` |
| Architecture | `GlmMoeDsaForCausalLM`: 78 layers, 256 experts (8 active), MLA (`kv_lora_rank` 512) with DeepSeek-V3.2-style sparse attention (top 2048), one MTP layer | `config.json` |
| vLLM | from v0.29 per the recipe; v0.34.0 maps the architecture to `models/deepseek_v32.py`, transformers 5.22.2 (5.15 needed) | recipes.vllm.ai/zai-org/GLM-5.3; the image's registry |
| Parsers | `--tool-call-parser glm47 --reasoning-parser glm47` | recipe; both registered in v0.34.0 |
| Context | 1,048,576 natively; served 262,144 | `max_position_embeddings`; profile |
| Sampling | temperature 1.0, top_p 0.95 | `generation_config.json`; the card's HLE and SWE-Marathon runs |
| Reasoning | always on; `reasoning_effort` low, high or max (default max, any other value is max), a chat template variable passed as `chat_template_kwargs` | model card "Note"; `chat_template.jinja` |

Chat template quirks (rendered by vLLM's HF renderer on the Python frontend; `--tokenizer-mode glm5_next` is the same there, the recipe's Rust frontend is not used): the reasoning effort becomes a first `<|system|>Reasoning Effort: Max` line, ahead of the tools block and the client's own system prompt; tool calls are XML (`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`), and the template needs the arguments as a mapping, which vLLM's `_postprocess_messages` makes from the OpenAI JSON string; the generation prompt ends in `<think>`, so the model's output starts inside its reasoning; `clear_thinking` defaults to false, so earlier turns keep their reasoning, if the client sends it back as `reasoning_content` (otherwise they get an empty `<think></think>`).

Memory at TP 8 (estimates, not measured): 88 GiB of weights per GPU. On 8 B200 (179 GiB each) at utilisation 0.90 about 68 GiB per GPU remain for the KV cache after about 5 GiB of overhead; on 8 H200 about 33 GiB. The MLA latent cache is replicated on every GPU, about 100 KB per token in BF16 (576 values per layer plus the indexer's keys, 78 layers), so about 0.7M tokens in all on B200 and 0.35M on H200; an FP8 cache (`--kv-cache-dtype fp8_e4m3`, the recipe's) would hold about 1.6 times as many. It does not fit on fewer GPUs: 4 B200 (716 GiB) hold less than the weights, and TP must divide the 64 heads.

What ran (job IDs on DAIS):

- 993473 (`small`, CPU): in the v0.34.0 image, the snapshot's tokenizer loads (`glm5_next`, vocabulary 154,856), the template renders tools, a tool-call history and both effort levels; the `glm47` tool parser turns a GLM tool call into `tool_calls` (`bash`, `{"command": "ls -la /tmp", "timeout": 30}`) with the text before it as content, and the `glm47` reasoning parser splits `reasoning</think>answer`. PASS.
- `gpudev` cannot prove serving: its QOS allows 4 GPUs per user, fewer than the weights need.
- The full-node proof (load time, memory, `/health`, chat, tool call unstreamed and streamed, one OpenCode session) is prepared and checked with `sbatch --test-only`, not yet run.

Known problems and open points:

- `probe_endpoint.py` gives a chat 200 tokens and a tool call 600; a model that reasons at maximum effort spends those on reasoning, so `serve.slurm.sh MODE=probe` can fail for it although the server is fine.
- `agent_check.slurm.sh WITH_MODEL=1` writes OpenCode's config for `Qwen/Qwen3-Coder-Next` and waits 720 s for the server, whatever the profile.
- The KV cache at BF16 is the profile's only option; FP8 needs `VLLM_EXTRA=--kv-cache-dtype fp8_e4m3` on each submission. MTP speculative decoding (the recipe's `--speculative-config.method mtp`) is off for the same reason.
- Whether OpenCode 1.19.6 sends `reasoning_content` back in its history is unverified.

## Prerequisites

| What | Where | How to create it |
|---|---|---|
| A checkout with the model files | `/u/$USER/algorithmic-institutions-pf` | `git clone` the repo; copy `artifacts/artificial_humans/**/*.pt` (not in git) from a machine that has them |
| The logs folder | `/dais/fs/scratch/$USER/policy-finder/logs` | `mkdir -p` |
| Site settings | `/dais/fs/scratch/$USER/policy-finder/env.sh` (not in git) | one line, `VLLM_SIF=<path of the group's vLLM image on DAIS>` |
| The sim image | `/dais/fs/scratch/$USER/policy-finder/images/algoinst.sif` | a byte copy of Raven's shared `algoinst.sif` (SHA-256 asserted by `sim_check.slurm.sh`), or built from `infra/claude_proxy/container/CC_env/algoinst.def` |
| The weights | `HF_HOME=/dais/fs/scratch/$USER/models` | `huggingface_hub.snapshot_download("Qwen/Qwen3-Coder-Next", revision="a7fbcb5c...")` with that `HF_HOME`, inside the vLLM image |
| The reference rule for the benchmarks | `/dais/fs/scratch/$USER/policy-finder/ref-pf-20261104-1/configs/managers/rule_based/pf-20261104-1.yml` | `git worktree add` of the branch `policy-finder/pf-20261104-1` there, or set `SWEEP_RULE` to any rule file |

## Not yet measured

- The whole layout on a full node: a TP 8 server, 20 real agents and the pool together.
- The frontier models' throughput and their vLLM support on Hopper.
- Agents' CPU use under load; OpenCode with `noinet.py` as its shell; the seccomp filter on an H200 node.
- The cycle on Intel CPUs; per-process start-up separately from simulation time.
- B5 at 80 sessions with repeats and with the client on its own cores.
- The `gpu1`/`g0002` default path and `serve.slurm.sh MODE=serve`.
- A Raven baseline for any of these.

## What can be reused

| Part | Reuse |
|---|---|
| `algoinst.sif` | Unchanged, on the CPUs. Not on H200/B200 GPUs. |
| The vLLM image | Unchanged; version asserted at start. |
| Sister project's serve conventions (`../understanding/scripts/slurm/lib/evo_common.sh`) | Followed (`apptainer exec --nv`, a home that is not the real one, `HF_HOME` on scratch, prefix caching, readiness on `/v1/models`); added a port and API key per job and `--containall`. |
| Repo entry points | `python -m aimanager simulate`, `generate_sim_config.py` (`--n-parts`, `--final`, `--score-final`), `merge_sweeps.py`, `fit_surrogate.py`, unchanged. |
| Agent spawner (Docker socket) | Cannot run: no Docker-compatible runtime. Needs an Apptainer backend, possibly MPCDF's agents container (`gitlab.mpcdf.mpg.de/mpcdf/ai-cli-agents-container`, which ships `opencode`). |
| Network rule of the Docker sandbox | Not with namespaces. A seccomp allow-list on the agent's commands (`noinet.py`) or on a whole container without a TCP client (`no_inet.seccomp.json`). |
| Raven `cluster_proxy` functions | A thin adapter: a sim becomes a CPU process in a pool. |

## Scripts

`infra/dais/common.sh` holds the settings and helpers. Every job writes into its own run folder under scratch (`$PF_RUN`), never into the checkout; the checkout is bound read-only into the sim container; failures make the job exit non-zero; the API key is never on a command line.

| Script | Purpose | Last run of the committed form |
|---|---|---|
| `serve.slurm.sh` | serve and probe, or keep serving | 987950 (`MODE=probe`, gpudev); `MODE=serve` not run |
| `probe_endpoint.py`, `bench_load.py` | tool-call checks; concurrent sessions | 987890, 987950; `bench_load.py` was reformatted by `black` afterwards (18:45), with no other change intended. The version that ran was not kept, so the diff cannot be shown, and it has not been rerun since. |
| `serve_bench.slurm.sh` | the layout (B3), B5 | 987890 (gpudev, cut at the time limit during B5) |
| `cpu_bench.slurm.sh` | B1, B4 | 987892; since then its memory checks read the step's `sacct` state (`OUT_OF_MEMORY`) instead of any non-zero exit (not rerun) |
| `sweep_pack.slurm.sh` | B2 | 987622; since then only a check of the output folder was added and the slot list no longer repeats a CPU |
| `cycle_bench.slurm.sh` | the "sweep, fit, final" cycle | 987891 |
| `sim_check.slurm.sh` | network and Apptainer checks, GPU test, reference sims | 987951 (the same smoke pools as before: 70.23; 82.36 and 53.91) |
| `sim_cpu.slurm.sh` | one sim on the CPUs | 987893 |
| `iso_check.sh`, `net_probe.py`, `noinet.py`, `harness_standin.py`, `no_inet.seccomp.json` | the sandbox checks | 988058 and the login node, after the last changes. Those changes: `black` formatting (18:45), the fixed inherited-socket probe, Python 3.6 support in `net_probe.py`, the section 5b checks, and the wider docstring of `noinet.py` (its filter is unchanged). |
| `cpu_slots.py`, `sim_variant.py` | pinning slots; config copies | in the jobs above |

## Changes this suggests for the plan

- **Decision 1:** run the sims on the node's CPUs in the existing image, single-threaded and one per hyperthread, with vLLM on the GPUs and a few pinned cores; the layout in section 6.
- **Phase 3's cluster backend:** a pool of single-threaded sim processes fed with `--n-parts 32` slices; agents in `--overlap` steps with memory caps on a shared CPU set.
- **Decision 4:** run the agent harness inside `--containall`, with `noinet.py` as its shell. Disable the harness's own network tools, and keep the wrapper and its configuration read-only to the agent. The harness keeps its TCP connection to the model. The claim is a barrier against ordinary network use by the agent's commands, nothing more; to be tested with OpenCode.
- **Full-node jobs:** they go to `gpu` (exclusive) or `gpux`.
- **Scratch cleanup:** copy weights and results off scratch before the 12-week cleanup starts on 2027-02-01.
