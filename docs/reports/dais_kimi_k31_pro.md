# Kimi K3.1 Pro on DAIS

Can `moonshotai/Kimi-K3.1-Pro` (weights released 2026-11-04) be served with vLLM on one DAIS node for the policy-finder pipeline, behind a model profile like the others? Assessed and prepared on 2026-11-28. Profile: `infra/dais/model_profiles/kimi-k3.1-pro.sh`.

**Result: not as released, but as an expert-pruned build.** The official checkpoint needs about 191 GiB of weights per GPU at TP 8, and a B200 has 179 GiB. Nota AI's build with 37.5% of the routed experts removed (`nota-ai/Kimi-K3.1-Pro-Global-Pruned-37`, 948 GiB) fits one node of 8 B200 with room for the KV cache. vLLM 0.34.0 serves it with a two-edit patch for per-layer expert counts. It is a derivative of the model: results with it are results of "Kimi K3.1 Pro, 37.5% of routed experts pruned (Nota)", not of Kimi K3.1 Pro.

Marked below: **verified** (read from the files, the API or the code, or measured here) and **read** (taken from a web page, not checked).

## The model

| | Value | Source |
|---|---|---|
| Licence | "Kimi K3.1-Pro License", modified MIT: free use, copying and modification; conditions only for a Model-as-a-Service business over USD 20M revenue, or a product over 100M monthly users or USD 20M a month; internal use exempt. Research use at a public institute is allowed. | verified, `LICENSE` at revision 80aff651 |
| Gated | no | verified, HF API |
| Revision | `80aff651e9b2d2f5b9f70f92a0a7a77054418e16` (2026-11-25, "Merge plain-text content parts in encoding_k3.py to align tokenization") | verified |
| Architecture | `KimiK31ForConditionalGeneration`: 93 layers (69 KDA linear attention, 24 gated DSA, layer 0 dense), 896 routed experts with top-16 per MoE layer, 2 shared, latent MoE (7168 to 3584), context 1,048,576, vision encoder MoonViT-V2 | verified, `config.json` |
| Weights | 1,561,293,160,568 B in 96 files (1,454.0 GiB): routed experts MXFP4 (4-bit, a scale per 32) 1,446,456,066,048 B (1,347.1 GiB); the rest BF16, 113,866,593,280 B (106.0 GiB); vision 894,717,952 B | verified, all safetensors headers by HTTP range |
| Sampling | temperature 1.0, top_p 0.95, reasoning effort `max` (low/high/max; vLLM 0.34 also accepts medium); `generation_config.json` sets none | verified, card and `generation_config.json` |
| Multi-turn | the assistant message must be passed back whole, `reasoning_content` and `tool_calls` included | read, card section 5 |

`nvidia/Kimi-K3.1-Pro-NVFP4` (revision 3af04629, NVIDIA Open Model Agreement on top of the Kimi licence, ungated) is larger, 1,605,125,645,632 B: its routed experts are a lossless cast of MXFP4 to NVFP4 (4.5 instead of 4.25 bits), KDA and shared experts MXFP8, MLA BF16 (verified, its `conversion_report.json`). Tested on GB300 (read). It does not help here.

## vLLM

- vLLM 0.34.0 (our image `vllm-v0.34.0.sif`) registers `KimiK31ForConditionalGeneration` and has the `kimi_k3` tool and reasoning parsers, the `kimi_k3` renderer, the NVFP4 MLA KV cache (`nvfp4_ds_mla`) and weight offload to the CPU (`uva`, `prefetch`) (verified, in the image).
- The vLLM recipe names `vllm/vllm-openai:kimi-k31` (on Docker Hub since 2026-11-04, verified) and says vLLM ≥ 0.35.0, unreleased, is needed; PyPI's latest is 0.34.0 (verified). The recipe lists B300, GB300, GB200, VR200, MI355X, TPU v7x and PPUs, not B200 or H200, with `vram_minimum_gb` 1874 (read, recipe JSON).
- The blog post "vLLM Delivers Day-0 Support for Kimi K3.1 Pro" gives TP 8 on GB300 and the KV bytes per token and GPU at TP 8 over the 24 full-attention layers: 16,992 with an FP8 latent and FP8 index keys, 10,080 with an NVFP4 latent and MXFP4 index keys (read).

## Route A: one node

8 B200 have 8 × 183,359 MiB = 1,432.5 GiB, less than the weights (1,454.0 GiB). Per GPU it is worse: at TP 8 the router, the latent MoE projections, MLA's `kv_a` and the DSA indexer's `wq_index` are replicated, not sharded (`ReplicatedLinear` in vLLM 0.34's `models/kimi_k3/nvidia`, verified), about 10.7 GiB on every GPU. Weights per GPU: about 191.0 GiB against 179.06 GiB.

**With CPU offload of weights (not chosen).** About 50 GiB per GPU (27% of the experts) would sit in host memory, for 10 GiB of KV cache and about 20 GiB of runtime memory at utilisation 0.95. The prefetch backend copies whole offloaded layers on each forward pass, at least 1 s per step at about 50 GB/s over PCIe; UVA reads only the touched experts, but prefill touches all. An estimate, not measured: decoding 5 to 20 times slower, and whether the MXFP4 MoE kernels work with offloaded weights was not tried. The download (1.56 TB) also does not fit the scratch quota left (about 1.35 TB of apparent size; GPFS charges about twice the apparent size).

**With an expert-pruned build (chosen).** No smaller quantisation of the experts that vLLM serves and that a reputable source made exists (verified, HF search on 2026-11-28: GGUF, EXL3 and MLX builds only otherwise). Pruned builds (sizes verified, HF API; all ungated, Kimi licence):

| Build | Size | Weights per GPU at TP 8 | Evidence |
|---|---|---|---|
| `nota-ai/Kimi-K3.1-Pro-Global-Pruned-37` @ 72f0f57d | 1,018,400,340,440 B (948 GiB); 51,520 of 82,432 routed experts kept, 536 to 896 per layer, all kept in the first two and last two MoE layers | ≈128 GiB | read: GPQA-Diamond 93.43 (card of the original: 94.20); 1M context on 8 B200 at TP 8 with 1,415,974 KV tokens (FP8, utilisation 0.985, DSpark draft and vision on) |
| `nota-ai/Kimi-K3.1-Pro-Global-Pruned-30` @ 28cf33ff | 1,121,808,496,848 B (1,045 GiB); 30.4% removed | ≈140 GiB | read: GPQA-Diamond 94.44; 512K context, 663,070 KV tokens at 0.98 |
| `vllm/Kimi-K3.1-Pro-pruned75` | 475,507,835,896 B; 75% removed | ≈65 GiB | read: "not for quality-sensitive use", GSM8K 43.7% |

Nota's quality evidence is its own and thin: reasoning benchmarks, no agentic or tool-use evaluation. The pruned-37 build was chosen for its KV room (below); the pruned-30 build is closer to the original and leaves room for about half as many sessions.

What fitting costs, for pruned-37 (estimates: KV bytes from the blog post, about 20 GiB per GPU of runtime memory without a draft model, from Nota's two measurements, about 25 GiB with one):

| Utilisation | KV cache per GPU | FP8 KV | NVFP4 KV |
|---|---|---|---|
| 0.90 | ≈13 GiB | ≈0.8M tokens | ≈1.4M tokens |
| 0.95 | ≈22 GiB | ≈1.4M tokens | ≈2.35M tokens |

At a served context of 262,144 and an average live context of 100K tokens per agent session, that is about 14 sessions with FP8 KV and about 23 with NVFP4 KV at 0.95 before vLLM preempts. The profile takes 0.95 and NVFP4 KV (the recipe: GSM8K 96.1 against 96.2 with FP8 on 8 GB300; long-context accuracy unmeasured), the context at 262,144 instead of 1M, no vision encoder and no speculative decoding. Decode context parallelism (DCP 8) would hold about 7 times more tokens (read, blog) at some latency; untested on B200, not used.

### What the profile adds

- `VLLM_PATCH=kimi_k31_nonuniform_experts`: `infra/dais/vllm_patches/kimi_k31_nonuniform_experts.py`, run in the vLLM image at each start, writes patched copies of `models/kimi_k3/nvidia/model.py` (KimiMoE builds the router, its bias and the fused experts with the layer's count from `text_config.num_experts_per_layer`; EPLB refused) and of `specialized_decode/runtime/__init__.py` (the opt-in latency-specialised decode path refuses such a checkpoint), and `serve_model` binds them read-only over the image's files. The edits are the two Nota publishes for the `kimi-k31` image (their script was read, not run); both anchors occur exactly once in vLLM 0.34.0, and the script fails, writing nothing, if one does not (verified, on the image). Nota's checkpoint numbers each layer's experts from 0 (verified, its index against its `pruning_info.json`).
- The tokenizer from the official repository at 80aff651: the pruned repositories carry `encoding_k3.py` from before the official fix of 2026-11-25 (verified: the other tokenizer files are identical blobs, this one differs). Only its small files are downloaded (`DOWNLOAD_ALLOW`).
- `MODEL_ARGS`: `--trust-remote-code` (the tokenizer is the repository's code), the tokenizer, `--language-model-only`, `--kv-cache-dtype nvfp4_ds_mla`, `--mamba-ssm-cache-dtype float32`, `--max-num-seqs 32`.
- `export_instance.LOCAL_MODELS`: temperature 1.0, top_p 0.95, `reasoning_effort` max through `chat_template_kwargs` (vLLM's `kimi_k3` renderer maps it to the encoder's `thinking_effort`), context 262,144, output 49,152 (the compaction reserve, output plus 65,536, must stay below half the context).

Generic additions, empty by default so Qwen and DeepSeek serve with the same arguments as before (verified, a dry run of `serve_model` per profile): `MODEL_ARGS`, `VLLM_PATCH`, `PROBE_ARGS` in `common.sh`; `DOWNLOAD_ALLOW` in `download_model.sh`; `--max-tokens` and `--extra-body` in `probe_endpoint.py` (a reasoning model needs room to think before it answers); a `/health` check in `serve.slurm.sh`'s probe mode.

## Route B: two nodes (assessed, not built)

The official checkpoint on 16 B200 as TP 8 × PP 2 over two nodes is what NVIDIA's Dynamo recipe and InferenceX (#4797, merged with results) run (read), with vLLM's own multi-node mode: `--nnodes 2 --node-rank 0|1 --master-addr <head>`, `--headless` on the worker, `GLOO_SOCKET_IFNAME`/`NCCL_SOCKET_IFNAME`, `VLLM_ENGINE_READY_TIMEOUT_S=3600`; no Ray. DAIS nodes have InfiniBand, 8 × 200 Gbit/s per node (read, MPCDF); the login node's HCA reports InfiniBand at "200 Gb/sec (2X NDR)" (verified); not checked on a B200 node. Apptainer without network namespaces shares the host network, which multi-node serving needs; `--containall` would need `/dev/infiniband` bound, and the image's IB verbs and GPUDirect RDMA are unverified.

What it would add: a two-node Slurm job with one container per node; start, readiness and shutdown across nodes; the server watch over a remote worker; the IB interface and HCA settings. Failure modes: either node, NCCL or the fabric failing stops the server; both nodes must be free at once, which is the whole allowance of two full nodes, so no other campaign runs meanwhile; the worker node's 192 threads stay idle while agents and sims share the head node as now. The 1.56 TB download needs about 0.21 TB of apparent size freed. Effort about 2 to 3 days plus node time; a different setup from the other models.

## Route C: another engine

SGLang and TokenSpeed (the card's other engines) need the same GPU memory, so the full model does not fit one node there either; a CPU-expert hybrid (KTransformers-style) would be a different engine and slow. Not pursued.

## Measured

To be filled by the serving proof (`serve.slurm.sh` with `MODEL_PROFILE=kimi-k3.1-pro` on a full B200 node).
