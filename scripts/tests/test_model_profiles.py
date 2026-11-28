"""Model profiles (infra/dais/model_profiles/*.sh) against LOCAL_MODELS."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/policy_finder"))
import export_instance as ex  # noqa: E402

PROFILES = sorted((ROOT / "infra/dais/model_profiles").glob("*.sh"))
KEYS = ["MODEL", "MODEL_REV", "VLLM_SIF", "EXPECT_VLLM_TAG", "TOOL_PARSER",
        "REASONING_PARSER", "MAX_MODEL_LEN", "MOE_BACKEND", "SERVER_GPUS",
        "READY_TIMEOUT"]


def profile(path):
    """The variables a profile sets, sourced as common.sh does."""
    script = f'source "{path}"; ' + "; ".join(f'echo "{k}=${{{k}}}"' for k in KEYS)
    out = subprocess.run(["bash", "-euc", script], capture_output=True, text=True,
                         check=True, env={"PF_SCRATCH": "/scratch/x", "PATH": "/usr/bin:/bin"})
    return dict(line.split("=", 1) for line in out.stdout.splitlines())


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_a_profile_is_complete_and_matches_its_opencode_entry(path):
    p = profile(path)
    assert all(p[k] for k in KEYS if k != "REASONING_PARSER"), p
    assert len(p["MODEL_REV"]) == 40  # a pinned commit, not a branch
    assert p["VLLM_SIF"].startswith("/scratch/x/images/")
    assert p["EXPECT_VLLM_TAG"].startswith("vllm/vllm-openai:")
    # the served name has OpenCode settings, for the context the server serves
    assert p["MODEL"] in ex.LOCAL_MODELS
    assert ex.LOCAL_MODELS[p["MODEL"]]["limit"]["context"] == int(p["MAX_MODEL_LEN"])


def test_hy4_sampling_is_the_model_cards(monkeypatch):
    monkeypatch.setenv("PF_LLM_CONTEXT", "262144")
    conf = ex.opencode_config({"model": "tencent/Hy4-FP8", "base_url": "u"})
    model = conf["provider"]["local"]["models"]["tencent/Hy4-FP8"]
    # the chat template accepts only "high" and "no_think"
    assert model["options"] == {"temperature": 0.9, "top_p": 1.0,
                                "chat_template_kwargs": {"reasoning_effort": "high"}}
    assert model["limit"] == {"context": 262144, "input": 262144, "output": 65536}
    assert conf["compaction"]["reserved"] == 65536 + ex.COMPACTION_MARGIN
