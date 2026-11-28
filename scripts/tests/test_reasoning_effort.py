"""Tests for infra/dais/reasoning_effort.sh with the DeepSeek profile: a
submission's REASONING_EFFORT, the server default and the prompt check."""

import subprocess
from pathlib import Path

import pytest

DAIS = Path(__file__).resolve().parents[2] / "infra/dais"
PROFILE = DAIS / "model_profiles/deepseek-v4.1-flash-1017.sh"


def resolve(effort=None, profile=PROFILE):
    env = {"PATH": "/usr/bin:/bin", "PF_SCRATCH": "/s"}
    if effort is not None:
        env["REASONING_EFFORT"] = effort
    script = (f"source {profile}; " if profile else "") + (
        f"source {DAIS}/reasoning_effort.sh; resolve_reasoning_effort || exit 2; "
        'printf "%s\\n" "$REASONING_EFFORT" "$MODEL_VLLM_ARGS" "${PROMPT_MUST_CONTAIN:-}"')
    run = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    return run.returncode, run.stdout.splitlines(), run.stderr


@pytest.mark.parametrize("effort, shown, value, prompt", [
    (None, "max", '"max"', "Reasoning Effort: 100 (range 1-100"),  # the default
    ("max", "max", '"max"', "Reasoning Effort: 100 (range 1-100"),
    ("high", "high", '"high"', "Reasoning Effort: 75 (range 1-100"),
    ("low", "low", '"low"', "Reasoning Effort: 50 (range 1-100"),
    ("10", "10", "10", "Reasoning Effort: 10 (range 1-100"),
])
def test_the_effort_sets_the_server_default_and_the_prompt(effort, shown, value, prompt):
    code, (resolved, args, check), _ = resolve(effort)
    assert code == 0 and resolved == shown and check == prompt
    assert args == '--default-chat-template-kwargs {"reasoning_effort":%s}' % value


@pytest.mark.parametrize("effort", ["none", "xhigh", "medium", "0", "101", "7.5", "max "])
def test_an_invalid_effort_fails(effort):
    code, out, err = resolve(effort)
    assert code == 2 and "FATAL: reasoning effort" in err


def test_a_model_without_an_effort_refuses_one():
    assert resolve(profile=None)[:2] == (0, ["", "", ""])
    code, _, err = resolve("high", profile=None)
    assert code == 2 and "has no reasoning effort" in err


GLM = DAIS / "model_profiles/glm-5.3.sh"


@pytest.mark.parametrize("effort, shown, prompt", [
    (None, "max", "Reasoning Effort: Max"),  # the model's own default
    ("high", "high", "Reasoning Effort: High"),
    ("low", "low", "Reasoning Effort: Low"),
])
def test_glm_efforts_are_its_chat_templates(effort, shown, prompt):
    code, (resolved, args, check), _ = resolve(effort, profile=GLM)
    assert code == 0 and resolved == shown and check == prompt
    assert args == '--default-chat-template-kwargs {"reasoning_effort":"%s"}' % shown


@pytest.mark.parametrize("effort", ["50", "xhigh", "none"])
def test_glm_takes_no_other_effort(effort):
    code, _, err = resolve(effort, profile=GLM)
    assert code == 2 and "FATAL: reasoning effort" in err
