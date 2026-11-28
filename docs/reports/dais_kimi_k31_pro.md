# Kimi K3.1 Pro on DAIS: the model server across two nodes

`moonshotai/Kimi-K3.1-Pro` (weights released 2026-11-04) for the policy-finder campaigns on DAIS. Levin's decision (2026-11-28): the full official model on two B200 nodes, one run of both experiments after the production runs of the other models. This report covers what was prepared and proven without two full nodes; the model itself has not been downloaded or served yet. Profile: `infra/dais/model_profiles/kimi-k3.1-pro.sh`.

Marked throughout: **proven** (run or measured here), **verified** (read from the files, the API or vLLM's code), **read** (from a web page, not checked), **open** (only the real two-node run shows it).

## The model

| | Value | |
|---|---|---|
| Licence | "Kimi K3.1-Pro License", modified MIT: use, copying and modification free; conditions only for a Model-as-a-Service business over USD 20M revenue or a product over 100M monthly users or USD 20M a month; internal use exempt. Research use at a public institute is allowed. | verified, `LICENSE` |
| Gated | no | verified, HF API |
| Revision | `80aff651e9b2d2f5b9f70f92a0a7a77054418e16` (2026-11-25) | verified |
| Size | 1,561,293,160,568 B of weights in 96 files (1,454.0 GiB), 1,561,355,967,331 B with every file: routed experts MXFP4 1,446,456,066,048 B, the rest BF16 113,866,593,280 B, vision 894,717,952 B | verified, all safetensors headers |
| Architecture | `KimiK31ForConditionalGeneration`: 93 layers (69 KDA, 24 gated DSA, layer 0 dense), 896 routed experts, top-16, context 1,048,576 | verified, `config.json` |
| Sampling | temperature 1.0, top_p 0.95, reasoning effort max (the card's evaluation settings); `generation_config.json` sets none | verified |

One node does not hold it: at TP 8 the router, the latent MoE projections, MLA's `kv_a` and the DSA indexer's `wq_index` are replicated on every GPU (vLLM 0.34's `ReplicatedLinear`, verified), about 191 GiB of weights per GPU against 179.06 GiB (8 × 183,359 MiB per node). `nvidia/Kimi-K3.1-Pro-NVFP4` is larger (1,605,125,645,632 B, verified). An expert-pruned build that fits one node was assessed and prepared but not chosen; that work is on branch `policy-finder-dais-kimi-pruned`, unmerged.

On two nodes, TP 8 on each and pipeline parallel over the two, as NVIDIA's Dynamo recipe and InferenceX (#4797) run it on 16 B200 (read): about 96 GiB of weights per GPU (verified from the headers: (1,347.1 + 95.3) / 16 + 10.7 / 2 GiB). With FP8 KV, 16,992 bytes per token and GPU at TP 8 over the 24 DSA layers (vLLM's blog, read), half that per GPU with each stage holding 12 layers, and roughly 45 GiB per GPU left at utilisation 0.90 after an assumed 20 GiB of runtime memory: about 5M tokens of KV cache, room for the full 1M context and 12 to 24 sessions (estimate; open).

## vLLM 0.34.0 (our image)

- Has the model: `KimiK31ForConditionalGeneration` maps to `vllm.models.kimi_k3.KimiK3ForConditionalGeneration` (verified, registry).
- That class supports pipeline parallelism: `SupportsPP`, `make_layers`, `PPMissingLayer`, `get_pp_group()` (verified, `models/kimi_k3/nvidia/model.py`). PP > 1 only turns off sequence parallelism, an expert-parallel feature not used here; only the opt-in `latency_specialized` decode path refuses PP (verified). The recipe names vLLM ≥ 0.35 and `vllm/vllm-openai:kimi-k31` (read); whether 0.34.0 serves this model at all, with PP, is open.
- Multi-node without Ray: `--nnodes`/`-n`, `--node-rank`/`-r`, `--master-addr`, `--master-port` (`engine/arg_utils.py`), `--headless`, which `entrypoints/cli/serve.py` documents as "Run headless workers (for multi-node PP/TP)", and `--distributed-executor-backend mp`, whose docs say multi-node `mp` needs `nnodes` (verified). Each node's `VLLM_HOST_IP` is its address for vLLM's message queues (verified, `utils/network_utils.py`).
- `kimi_k3` tool and reasoning parsers and renderer (verified). The renderer passes `reasoning_effort` to the chat encoder as `thinking_effort`, which writes "Now the system is invoked with \`thinking_effort=max\`." into the prompt (verified, `renderers/kimi_k3.py` and the official `encoding_k3.py`).
- `VLLM_ENGINE_READY_TIMEOUT_S` defaults to 600 s: the API server's wait for the engine, model loading included (verified, `envs.py`). The recipe sets 3600; the profile sets 5400.
- The image has `libibverbs` and the mlx5 provider, NCCL 2.30.7 (torch) and 2.31.2 (pip) (verified). Under `--containall` its `/dev` is minimal, so `/dev/infiniband` has to be bound in (verified).

## DAIS

- Only partition `gpu` takes more than one node per job: `gpudev`, `gpu1`, `small`, `rvs` and `gpux` have MaxNodes=1 (verified, `scontrol show partition`, and `sbatch --test-only` refusing `-N 2`).
- The job scripts' `#SBATCH --ntasks=1` makes Slurm shrink `-N 2` to one node ("can't run 1 processes on 2 nodes, setting nnodes to 1", verified with `--test-only`); a two-node job needs `-N 2 --ntasks=2 --ntasks-per-node=1`.
- A B200 node (daisg209, job 993684): `eth0` 130.183.176.78/24 (the address the node name resolves to) and `ib0` 10.179.39.78/21 (IP over InfiniBand), no `bond0`; `/dev/infiniband` present; 12 InfiniBand ports active (mlx5_0 to 5 and 10, 11 at 200 Gb/s NDR, mlx5_6 to 9 at 100 Gb/s HDR), two Ethernet ports (mlx5_12 25 Gb/s, mlx5_13 down); every GPU has HCAs on its PCIe switch (`nvidia-smi topo -m`) (proven). MPCDF states InfiniBand 8 × 200 Gbit/s per node (read). The fabric between two nodes is open.
- A step on the GPU nodes does not have the batch's `PATH`: `srun ... bash` failed with "execve(): bash: No such file or directory" (job 993684, proven), as the agents' `taskset` did before (README). The worker therefore runs `/bin/bash` and `apptainer` by absolute path.

## The setup across nodes

One setup: the same `campaign.slurm.sh` and `serve.slurm.sh`, the profile saying how many nodes the server spans.

- `SERVER_NODES` (profile; default 1). With 1 the server's command line is byte for byte the old one: `scripts/tests/test_serve_nodes.py` compares it, for the default and the DeepSeek profile, with the line captured from the frontier branch before the change (proven, on DAIS).
- With 2: the head on the job's first node, with the API, the driver, the agents and the sims as before; a worker on the second, started by `serve_model` (`server_worker`) through `srun --nodes=1 --ntasks=1 --nodelist=<node> --overlap --gres=gpu:<SERVER_GPUS> --mem=0`, with the same container and `vllm serve` arguments plus `--node-rank 1 --headless`. Both get `--pipeline-parallel-size 2 --nnodes 2 --master-addr <head's ib0 address> --master-port <free port> --distributed-executor-backend mp`; the head `--node-rank 0`.
- Start order and readiness: the head first, then the worker; `wait_ready` waits for the head's `/v1/models` (it answers only once every stage has loaded) and fails as soon as the head or a worker's `srun` ends (`server_alive`).
- Network: `SERVER_NET_IF` (default `ib0`) for vLLM's own traffic and NCCL's and Gloo's bootstrap (`GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`); each node's `VLLM_HOST_IP` is its address there, found on that node by a small prelude before the container starts; NCCL's data over InfiniBand, `/dev/infiniband` bound in. A node without an address on the interface or without `/dev/infiniband` fails with a message.
- Logs: the worker's is `vllm.node1.log` beside the head's `vllm.log` in the run folder; `server_report` and `serve.slurm.sh` show both.
- The server watch, the kill test and stop-everything use `SERVER_PIDS` (the head and each worker's `srun` on the head node): the kill test spares them, the watch counts a worker's end as the server down, the stop sends TERM to the head and to the `srun`s (which pass it on to their node) and waits for all of them, then kills what is left.
- Fewer nodes than the profile asks for: `campaign.slurm.sh` fails before the preflight, `serve_model` at once, with "the model server spans 2 nodes (SERVER_NODES, profile kimi-k3.1-pro), but this job has 1 (...): submit with -N 2 --ntasks=2 --ntasks-per-node=1 ..." (proven, dry run and test).
- `SERVE_DRY_RUN=1` prints what would start on which node, with the full command lines, and starts nothing (`serve.slurm.sh`; proven on the login node).
- `MODEL_VLLM_ENV` (profile; default empty): environment variables for vLLM in the container.

What the frontier already had and this uses: `MODEL_VLLM_ARGS`, `PROMPT_MUST_CONTAIN`, `PROBE_MAX_TOKENS`, `download_model.sh` fetching every file, the agent check taking `$MODEL` and `READY_TIMEOUT`. Added here: `SERVER_NODES`, `SERVER_NET_IF`, `MODEL_VLLM_ENV`, `SERVE_DRY_RUN`, `SERVER_PIDS` and `server_alive`, the worker launch, the node-count check, a `/health` check in `serve.slurm.sh`'s probe mode, `nodes_probe.slurm.sh`.

Not adapted: `agent_check.slurm.sh` and `serve_bench.slurm.sh` still start a one-node server; with a two-node profile they would need the same `-N 2` and are not part of the plan.

## The connectivity probe (job 993691, proven)

`nodes_probe.slurm.sh` in `gpudev` (one node allowed): Qwen3-Coder-Next on vLLM 0.34.0 as a head on GPU 0 and a worker on GPU 1 of daisg209 (B200), the worker through the real `serve_model`/`srun --overlap` path, pipeline parallel over the two, meeting at the node's `ib0` address; NCCL kept off NVLink and shared memory (`NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`), so its data went through the InfiniBand HCAs.

- vLLM's multi-node mode: "DP group leader: node_rank=0 ... master_addr=10.179.39.78 ... world_size=2, local_world_size=1"; the worker "headless", stages Worker_PP0 and Worker_PP1; NCCL initialised over `tcp://10.179.39.78`.
- NCCL in the container: "Using network IB", "GPU Direct RDMA Enabled" for the HCAs, channels "via NET/IB/13" and "via NET/IB/14/GDRDMA".
- Weights loaded in 79 s (74.8 GiB per GPU); ready after 206 s (`--enforce-eager`); `/health` 200; the chat and all five tool-call checks of `probe_endpoint.py` passed (one trial each, short inputs).
- At the end the job's cleanup stopped the worker's step (`993691.0` CANCELLED, the job COMPLETED).

What it does not show: two different nodes over the switch fabric, the worker step on a second exclusive node (GRES, cgroups), and anything about Kimi. The same script on two nodes (partition `gpu`) shows the first two:

```
cd <checkout with this branch>
sbatch -p gpu --qos=g0008 -N 2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:b200:8 --exclusive --mem=0 -c 96 \
  -t 00:30:00 -o /dais/fs/scratch/levinb/policy-finder-kimi/logs/%x.%j.out \
  --export=PF_SCRATCH=/dais/fs/scratch/levinb/policy-finder-kimi,VLLM_SIF=/dais/fs/scratch/levinb/policy-finder-kimi/images/vllm-v0.34.0.sif,EXPECT_VLLM_TAG=vllm/vllm-openai:v0.34.0,SERVER_GPUS=4 \
  infra/dais/nodes_probe.slurm.sh
```

## The profile and the campaigns

`kimi-k3.1-pro.sh`: the official weights at 80aff651, vLLM 0.34.0, parsers `kimi_k3`, `MAX_MODEL_LEN` 1,048,576, `SERVER_NODES=2`, `SERVER_GPUS=8`, `SERVER_NET_IF=ib0`, `MODEL_VLLM_ARGS` with `--trust-remote-code` (the repository's tokenizer), `--language-model-only`, `--kv-cache-dtype fp8`, `--mamba-ssm-cache-dtype float32`, `--max-num-seqs 32` and the maximum reasoning effort as the server's default (`--default-chat-template-kwargs {"reasoning_effort":"max"}`, as DeepSeek's profile does), checked on the running server with `PROMPT_MUST_CONTAIN="thinking_effort=max"`; `VLLM_ENGINE_READY_TIMEOUT_S=5400` and `READY_TIMEOUT=5400`. `export_instance.LOCAL_MODELS`: temperature 1.0, top_p 0.95, context 1,048,576, output 131,072 (the card states no output limit).

Campaigns (DeepSeek's sizes, new prefixes): `dais-kimi-smoke-exp2.yml` (`pfkms2-20261128`), `dais-kimi-exp2.yml` (`pfkm-exp2-20261128`), `dais-kimi-exp1.yml` (`pfkm-exp1-20261128`); stages `infra/dais/stages/node-kimi-2node.txt`: the smoke campaign, then both real campaigns at once on the one server, re-evaluated in 15 and 45 slices.

## Disk

The 1.56 TB has not been downloaded. GPFS charges about twice the apparent size (measured: DeepSeek 476 GB apparent, 951 GB charged; Qwen 149/297; Hy4-FP8 733/1,465; GLM-5.3 704/1,408). Kimi: 1,561,355,967,331 B apparent, about 3.12 TB charged. With about 2.73 TB of quota free after Hy4-FP8 (5.46 of 8.19 TB used), about 0.4 TB more must be freed, plus room for the runs: freeing one of DeepSeek (0.95 TB charged), GLM-5.3 (1.41 TB) or Hy4-FP8 (1.47 TB) is enough; Qwen alone (0.30 TB) is not. After DeepSeek is freed about 0.56 TB would remain.

The download, on the login node (149 GB took 5 minutes there; at that rate about an hour):

```
cd <checkout with this branch>
nohup setsid bash infra/dais/download_model.sh moonshotai/Kimi-K3.1-Pro 80aff651e9b2d2f5b9f70f92a0a7a77054418e16 \
  > /dais/fs/scratch/levinb/policy-finder-agents/logs/download.kimi-k3.1-pro.log 2>&1 &
```

## The runs

All in partition `gpu` (two full nodes), from a checkout with this branch, `PF_SCRATCH` the default `policy-finder-agents`. Both passed `sbatch --test-only` on 2026-11-28 (daisg[202-203]).

A serving proof first (about an hour; the model loads, `/health`, the prompt check, a chat and the tool calls):

```
sbatch -p gpu --qos=g0008 -N 2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:b200:8 --exclusive --mem=0 -c 96 \
  --time=02:00:00 -o /dais/fs/scratch/levinb/policy-finder-agents/logs/%x.%j.out \
  --export=MODEL_PROFILE=kimi-k3.1-pro,PF_SCRATCH=/dais/fs/scratch/levinb/policy-finder-agents,MODE=probe \
  infra/dais/serve.slurm.sh
```

The final run, one job: the smoke campaign, then both real campaigns:

```
sbatch -p gpu --qos=g0008 -N 2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:b200:8 --exclusive --mem=0 -c 96 \
  --time=24:00:00 -o /dais/fs/scratch/levinb/policy-finder-agents/logs/%x.%j.out \
  --export=MODEL_PROFILE=kimi-k3.1-pro,SERVER_CORES=8,DRIVER_CORES=2,AGENT_CORES=24,PF_AGENT_SLOTS=24 \
  infra/dais/campaign.slurm.sh stages infra/dais/stages/node-kimi-2node.txt
```

24 hours is the partition's limit; the campaigns resume from their state files in a resubmitted job with the same stages file (the smoke stage then returns at once).

## Open until the real run

- Whether vLLM 0.34.0 loads and serves Kimi K3.1 Pro at all, and with PP 2 (the recipe names 0.35; fallback: the `kimi-k31` image, built with `vllm_image.slurm.sh VLLM_TAG=kimi-k31`, and `EXPECT_VLLM_TAG` changed in the profile).
- NCCL between two nodes over the switch fabric; the worker step on a second exclusive node.
- Load time (about 780 GB per node from GPFS), memory per GPU, KV cache size, speed.
- That the rendered prompt carries `thinking_effort=max` (the check runs before any agent starts) and that the `kimi_k3` parser returns proper `tool_calls`.
- Whether OpenCode hands the model's reasoning back in later turns, which the card requires for multi-turn tool use.
