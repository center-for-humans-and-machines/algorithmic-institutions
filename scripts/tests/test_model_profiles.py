"""The DAIS model profiles and the vLLM patch one of them needs."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROFILES = sorted((ROOT / "infra/dais/model_profiles").glob("*.sh"))
PATCH = ROOT / "infra/dais/vllm_patches/kimi_k31_nonuniform_experts.py"
REQUIRED = ["MODEL", "MODEL_REV", "VLLM_SIF", "EXPECT_VLLM_TAG", "TOOL_PARSER",
            "MAX_MODEL_LEN", "MOE_BACKEND", "SERVER_GPUS"]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def profile(path):
    """The profile's settings as bash sees them after sourcing it."""
    names = REQUIRED + ["VLLM_PATCH"]
    script = (f"set -euo pipefail; source {path}; "
              + "; ".join(f'echo "{n}=${{{n}:-}}"' for n in names))
    out = subprocess.run(["bash", "-c", script], env={"PF_SCRATCH": "/s"},
                         capture_output=True, text=True, check=True).stdout
    return dict(line.split("=", 1) for line in out.splitlines())


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_a_profile_matches_its_opencode_entry(path):
    conf = profile(path)
    assert all(conf[n] for n in REQUIRED), conf
    assert len(conf["MODEL_REV"]) == 40
    if conf["VLLM_PATCH"]:
        assert (PATCH.parent / f"{conf['VLLM_PATCH']}.py").is_file()
    ex = load(ROOT / "scripts/policy_finder/export_instance.py", "export_instance")
    assert conf["MODEL"] in ex.LOCAL_MODELS
    entry, _ = ex.local_model(conf["MODEL"])  # raises if the context is too small
    assert entry["limit"]["context"] == int(conf["MAX_MODEL_LEN"])


def fake_vllm(tmp_path, times=1):
    patch = load(PATCH, "kimi_patch")
    root = tmp_path / "vllm"
    for rel in {rel for rel, _, _ in patch.EDITS}:
        old = "".join(o for r, o, _ in patch.EDITS if r == rel)
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("head\n" + old * times + "tail\n")
    return patch, root


def test_the_patch_writes_patched_copies_and_their_binds(tmp_path, capsys):
    patch, root = fake_vllm(tmp_path)
    assert patch.main([str(tmp_path / "out"), "--vllm-dir", str(root)]) == 0
    binds = capsys.readouterr().out.split()
    assert len(binds) == 2 and all(b.endswith(":ro") for b in binds)
    model = (tmp_path / "out" / patch.MODEL).read_text()
    assert "num_experts_per_layer[layer_idx]" in model
    assert (root / patch.MODEL).read_text().count("num_experts_per_layer") == 0


@pytest.mark.parametrize("times", [0, 2])
def test_the_patch_refuses_another_vllm(tmp_path, times):
    patch, root = fake_vllm(tmp_path, times)
    assert patch.main([str(tmp_path / "out"), "--vllm-dir", str(root)]) == 1
    assert not (tmp_path / "out").exists()
