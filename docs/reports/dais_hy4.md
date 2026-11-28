# Tencent Hy4 (FP8) on DAIS

Serving preparation for `tencent/Hy4-FP8` as a model profile (`MODEL_PROFILE=hy4-fp8`, `infra/dais/model_profiles/hy4-fp8.sh`), beside `dais_feasibility.md`. Done on 2026-11-28 under account `mpib_inst`. What is measured is marked with its job; the rest is read from the model card, the licence, vLLM's source and its recipe, and is marked as such.

## Jobs and measurements

| What | Where | Result |
|---|---|---|
| Download, `download_model.sh tencent/Hy4-FP8 78037597...` | login node, 12:23:45 to 12:27:49 | 245 shards, 786 GB apparent in 4 minutes. GPFS charges twice that (`du` 1.5 TiB; `mmlsquota` 3890 GB of 8192 before the download). |
| vLLM image, `vllm_image.slurm.sh` with `VLLM_TAG=nightly-70b67cf80585b671b57ea748232ff6c68e7ba4d7` | job 993433, `small`, 12:51 min | `vllm 0.34.1rc1.dev319+g70b67cf80`, torch 2.14.0+cu130, 9.8 GB, sha256 `1a78e321...c0db6c1`; label `ai.vllm.image.tag` = `vllm/vllm-openai:nightly-70b67cf8...`, which `assert_vllm_image` reads. Hard-linked into `policy-finder-agents/images/` as well, so the profile resolves with either `PF_SCRATCH`. |
| CPU check in the image (a script in the job's scratch folder, not in the repo) | job 993528, `small` | `HYV4VLForConditionalGeneration` registered; parsers `hy_v4` resolve to `HYV4ToolParser` and `HYV4ReasoningParser`; the tokenizer loads offline (120,832 tokens); the chat template renders a tool-call history with both `reasoning_effort` values (with tools, earlier reasoning is kept in the prompt under `high`); the tool parser turns the model's own tool-call syntax, as the template renders it, into one `tool_calls` entry with typed arguments (`{"command": "ls -la", "timeout_s": 30}`). |
| Full-node serve and probe, `serve.slurm.sh` MODE=probe | not run: `sbatch --test-only` only (job 993530 would start 2026-11-28 16:22 on daisg202) | Loading time, memory, KV cache, a chat completion and a parsed tool call on the GPUs are not measured yet. |

## Read from documentation (not measured)

- **Checkpoint:** `tencent/Hy4-FP8` at revision `78037597c0ed338334957bd94f881f6853cf5dc5`, ungated, 245 safetensors shards, 786.3 GB (733 GiB). The weights are block FP8 W8A8 (compressed-tensors, 128 x 128 blocks); `lm_head`, the vision tower, layer 0's attention and the indexer projections stay BF16 (config.json). Architecture `HYV4VLForConditionalGeneration`: 770B MoE with 49B active, 78 layers, gated sparse MLA (DSA, top-2048, IndexCache: 20 of 78 layers run their own indexer), iHC with 4 residual streams, a 1.23B ViT and a 3.3B DFly2 drafter (model card).
- **Licence:** Tencent Hy License 1.0, Apache 2.0 with Section 10 added: a separate licence is needed only by a MaaS business with more than US$50M revenue or 100M monthly users, and 10.3 exempts internal use. It has no territorial exclusion (the older Hunyuan licences excluded the EU). Research use at MPIB is allowed.
- **vLLM:** no release serves it. v0.34.0 registers only the text-only `HYV4ForCausalLM` of Hy4-preview; the VL class came with PR #68296 (2026-11-19), and vLLM's recipe (`recipes/models/tencent/Hy4.yaml`) names 0.35.0 as the first release. The model card's `vllm/vllm-openai:hy4` image (2026-11-20 03:08) predates PR #68689 (2026-11-20 14:09), which rewrote the streaming path of the `hy_v4` reasoning parser; OpenCode streams. The profile therefore uses the nightly at commit `70b67cf80585b671b57ea748232ff6c68e7ba4d7` (2026-11-28), built with `vllm_image.slurm.sh`. It should move to v0.35.0 once that is released. Parsers: `--tool-call-parser hy_v4 --reasoning-parser hy_v4`.
- **Sampling:** temperature 0.9 and top_p 1.0 (model card, `generation_config.json`). `reasoning_effort` is `high` (the template's default) or `no_think`; any other value raises an error in the chat template. The card gives no output limit; the 65,536 in `LOCAL_MODELS` is our choice.
- **Context:** 1,048,576 positions natively; the recipe evaluated 131,072. It is served at 262,144.
- **Memory at TP 8 on 8 B200 (computed, not measured):** about 100 GB of weights per GPU. At 0.90 utilisation, after about 8 GB of activations and graphs, about 55 GB per GPU is left for the KV cache. The MLA latent cache is replicated on every rank: 78 layers x 576 values x 2 bytes, plus the indexer keys of the 20 indexing layers, about 92 KB per token. That is about 600k tokens per GPU, or 2.3 sequences at the full 262,144. The recipe's minimum is 944 GB of GPU memory in total.
- **Vision encoder:** it is loaded on every rank (about 2.5 GB) whatever the requests contain. Text-only use needs no flag.

## Known problems and open points

- **Not proven on GPUs.** The model has not been loaded; everything about loading time, memory, the attention backend and real tool calls waits for the full-node job.
- **Offline snapshot check of the newer huggingface_hub.** `download_model.sh` skips `*.png`, `*.jpg` and `*.pdf`. The nightly's huggingface_hub refuses such a snapshot with `local_files_only=True` (`IncompleteSnapshotError`, job 993479). I fetched the 7 skipped asset files (5 MB) so that the snapshot is complete. Whether vLLM's own loader would also have refused it is not tested.
- **The probe is written for a model without reasoning.** `probe_endpoint.py` caps answers at 200 and 600 tokens and sends no `chat_template_kwargs`, while Hy4 thinks by default. Its chat and tool checks may end at `finish_reason=length`. This is a generic gap, proposed to the main session, not changed here.
- **`agent_check.slurm.sh` waits 720 s for the server** and ignores `READY_TIMEOUT`, which is too short for 786 GB. So the OpenCode session check cannot run with this model unchanged. This is also generic and proposed, not changed.
- **Reasoning in the history.** With tools, the template puts earlier turns' reasoning back into the prompt (`preserved_thinking`) when the request carries it as `reasoning` or `reasoning_content`. Whether OpenCode 1.19.6 sends it back through `@ai-sdk/openai-compatible` has not been checked; without it the model sees empty think blocks for earlier turns.
- **H200 untested.** The profile is meant for B200 only.
- **DFly2 speculative decoding** (the card's `--speculative-config '{"num_speculative_tokens":5,"method":"dfly"}'`) is not enabled. Turning it on would need a flag the profile cannot pass today.
