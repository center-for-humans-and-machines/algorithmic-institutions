"""The DAIS model profiles (infra/dais/model_profiles) against their OpenCode
entries (export_instance.LOCAL_MODELS)."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROFILES = sorted((ROOT / "infra/dais/model_profiles").glob("*.sh"))
REQUIRED = ["MODEL", "MODEL_REV", "VLLM_SIF", "EXPECT_VLLM_TAG", "TOOL_PARSER",
            "MAX_MODEL_LEN", "MOE_BACKEND", "SERVER_GPUS"]


def profile(path):
    """The profile's settings as bash sees them after sourcing it."""
    names = REQUIRED + ["SERVER_NODES"]
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
    assert int(conf["SERVER_NODES"] or 1) >= 1
    spec = importlib.util.spec_from_file_location(
        "export_instance", ROOT / "scripts/policy_finder/export_instance.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    assert conf["MODEL"] in ex.LOCAL_MODELS
    entry, _ = ex.local_model(conf["MODEL"])  # raises if the context is too small
    assert entry["limit"]["context"] == int(conf["MAX_MODEL_LEN"])
