"""Patched copies of vLLM files for a Kimi-K3.1 checkpoint whose MoE layers
have different numbers of routed experts (non-uniform expert pruning, such as
nota-ai/Kimi-K3.1-Pro-Global-Pruned-37).

Such a checkpoint lists each decoder layer's routed experts in its config,
`text_config.num_experts_per_layer` (the dense layer 0 has 0), and numbers
every layer's experts from 0. vLLM 0.34.0 builds every MoE layer with
`num_experts` (896). The two edits, the same as the ones Nota publishes for the
`vllm/vllm-openai:kimi-k31` image (its patch/steps/kimi_k31_nonuniform_
experts.py, read, not run), written here for vLLM 0.34.0:
  * models/kimi_k3/nvidia/model.py: KimiMoE builds the router, its bias and
    the fused experts with the layer's count; EPLB is refused.
  * models/kimi_k3/nvidia/specialized_decode/runtime/__init__.py: the
    opt-in latency_specialized decode path, whose kernels assume 896 experts,
    reports such a checkpoint as unsupported.
Without `num_experts_per_layer` the model behaves as before.

Run inside the vLLM image (common.sh, vllm_patch_binds); writes the patched
files under <out> and prints one apptainer bind per file
(<patched file>:<file in the image>:ro). Fails, writing nothing, if an anchor
does not occur exactly once (another vLLM version).

    python3 kimi_k31_nonuniform_experts.py <out> [--vllm-dir <site-packages/vllm>]
"""

import argparse
import importlib.util
import sys
from pathlib import Path

MODEL = "models/kimi_k3/nvidia/model.py"
SPECIALIZED = "models/kimi_k3/nvidia/specialized_decode/runtime/__init__.py"

EDITS = [
    (
        MODEL,
        "        num_experts = config.num_experts\n"
        "        num_experts_per_token = config.num_experts_per_token\n",
        "        num_experts = config.num_experts\n"
        "        # a non-uniformly expert-pruned checkpoint: this layer's experts\n"
        "        num_experts_per_layer = getattr(\n"
        '            config, "num_experts_per_layer", None\n'
        "        )\n"
        "        if num_experts_per_layer is not None:\n"
        "            num_experts = int(num_experts_per_layer[layer_idx])\n"
        "            if vllm_config.parallel_config.enable_eplb:\n"
        "                raise ValueError(\n"
        '                    "EPLB is not supported with num_experts_per_layer"\n'
        "                )\n"
        "        num_experts_per_token = config.num_experts_per_token\n",
    ),
    (
        SPECIALIZED,
        "        (\n"
        '            moe != ("situ", 4.0, 25.0, 896, 16, 1, 1, "sigmoid"),\n',
        "        (\n"
        '            getattr(text, "num_experts_per_layer", None) is not None,\n'
        '            "expert-pruned checkpoints (num_experts_per_layer)",\n'
        "        ),\n"
        "        (\n"
        '            moe != ("situ", 4.0, 25.0, 896, 16, 1, 1, "sigmoid"),\n',
    ),
]


def patch_text(text, old, new, where):
    """`text` with `old` replaced by `new`; `old` must occur exactly once."""
    n = text.count(old)
    if n != 1:
        raise ValueError(f"{where}: anchor found {n} times, expected 1")
    return text.replace(old, new)


def patched_files(vllm_dir):
    """{relative path: patched text} for every file the edits touch."""
    out = {}
    for rel, old, new in EDITS:
        text = out[rel] if rel in out else (vllm_dir / rel).read_text()
        out[rel] = patch_text(text, old, new, rel)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--vllm-dir", type=Path)
    args = ap.parse_args(argv)
    vllm_dir = args.vllm_dir
    if vllm_dir is None:  # without importing vllm (slow, wants CUDA)
        vllm_dir = Path(importlib.util.find_spec("vllm").origin).parent
    try:
        files = patched_files(vllm_dir)
    except (OSError, ValueError) as e:
        print(f"FATAL: {e} in {vllm_dir}", file=sys.stderr)
        return 1
    for rel, text in files.items():
        dst = args.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
        print(f"{dst}:{vllm_dir / rel}:ro")
    return 0


if __name__ == "__main__":
    sys.exit(main())
