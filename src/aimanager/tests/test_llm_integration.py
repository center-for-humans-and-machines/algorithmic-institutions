"""Integration test: the LLM manager against a real vLLM server.

Skipped unless `HOSTED_VLLM_API_BASE` points at a live endpoint, so it is
inert in the ordinary local run and exercised on the cluster inside a serving
job:

    sbatch --gres=gpu:a100:1 scripts/llm_manager/serve_vllm.slurm.sh \\
        ~/algorithmic-institutions/.venv/bin/python -m pytest \\
        src/aimanager/tests/test_llm_integration.py -v

These are the assertions that cannot be made against a stub, because they are
about the server's behaviour rather than ours:

* that Qwen3's default thinking mode really does eat the completion budget and
  truncate, and that `enable_thinking=False` really does stop it -- the trap
  the plan says to verify rather than trust;
* that vLLM honours the decode constraint, so the parse failure rate is zero
  by construction rather than by luck;
* that a full 24-round rollout completes with the trace accumulating, against
  the model that will actually be used.
"""

import os
import re

import pytest
import torch as th

from aimanager.manager.llm_client import ChatClient, DecodeConstraint, integers_regex
from aimanager.manager.llm_manager import LLMManager

API_BASE = os.environ.get("HOSTED_VLLM_API_BASE")
MODEL = os.environ.get("LLM_MANAGER_MODEL", "Qwen/Qwen3-8B")

pytestmark = pytest.mark.skipif(
    not API_BASE,
    reason="needs a live vLLM endpoint in HOSTED_VLLM_API_BASE",
)

N_AGENTS = 8
AGENT_GROUPS = [0, 0, 0, 0, 1, 1, 1, 1]

ASK = (
    "Four players contributed 12, 3, 20 and 7 out of 20. Reply with exactly "
    "four integers between 0 and 30, comma separated, and nothing else."
)


def make_state(n_batch, round_number, generator, prev=None):
    size = (n_batch, N_AGENTS, 1)
    state = {
        "contribution": th.randint(0, 21, size, generator=generator, dtype=th.int64),
        "contribution_valid": th.rand(size, generator=generator) > 0.05,
        "punishment": th.zeros(size, dtype=th.int64),
        "punishment_valid": th.ones(size, dtype=th.bool),
        "round_number": th.full(size, round_number, dtype=th.int64),
        "is_first": th.full(size, round_number == 0, dtype=th.bool),
        "agent_group": th.tensor(AGENT_GROUPS, dtype=th.int64)
        .reshape(1, N_AGENTS, 1)
        .expand(size)
        .contiguous(),
        "common_good": th.zeros(size, dtype=th.float),
    }
    zeros_i = th.zeros(size, dtype=th.int64)
    state["prev_contribution"] = prev["contribution"] if prev else zeros_i
    state["prev_punishment"] = prev["punishment"] if prev else zeros_i
    state["prev_contribution_valid"] = (
        prev["valid"] if prev else th.zeros(size, dtype=th.bool)
    )
    state["prev_common_good"] = th.zeros(size, dtype=th.float)
    return state


def test_the_server_is_serving_the_model_we_asked_for():
    client = ChatClient(model=MODEL, api_base=API_BASE, max_tokens=8)
    result = client.complete([[{"role": "user", "content": "Say OK."}]])[0]
    assert result.ok, result.error
    assert result.prompt_tokens > 0


@pytest.mark.skipif("qwen3" not in MODEL.lower(), reason="Qwen3-only trap")
def test_qwen3_thinking_truncates_by_default_and_the_flag_stops_it():
    """The documented trap, verified on a real completion, not trusted.

    Measured on Qwen/Qwen3-8B, vLLM 0.19.1, 2026-09-22: with the flag absent
    the reply hit `finish_reason: length` at 64 tokens having emitted only the
    opening of a `<think>` block, and never produced an answer. Note the
    `<think>` text arrives in `content`, not `reasoning_content`, unless the
    server was started with a reasoning parser.
    """
    messages = [[{"role": "user", "content": ASK}]]

    thinking = ChatClient(
        model=MODEL, api_base=API_BASE, max_tokens=64, enable_thinking=True
    )
    on = thinking.complete(messages)[0]
    assert on.ok, on.error

    quiet = ChatClient(
        model=MODEL, api_base=API_BASE, max_tokens=64, enable_thinking=False
    )
    off = quiet.complete(messages)[0]
    assert off.ok, off.error

    # With thinking on, the budget goes on reasoning and the answer is cut off.
    assert on.finish_reason == "length"
    assert "<think>" in (on.text or "") or on.reasoning_text
    # With it off, the model answers and stops well inside the budget.
    assert off.finish_reason == "stop"
    assert off.completion_tokens < on.completion_tokens
    assert re.search(r"\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+", off.text)


def test_qwen3_gets_thinking_off_without_being_asked():
    """The default must be the safe one, or every caller has to remember."""
    client = ChatClient(model=MODEL, api_base=API_BASE, max_tokens=64)
    if "qwen3" in MODEL.lower():
        assert client.enable_thinking is False
    result = client.complete([[{"role": "user", "content": ASK}]])[0]
    assert result.ok, result.error
    assert result.finish_reason == "stop"


def test_vllm_honours_the_decode_constraint():
    """The mechanism: the model cannot emit an invalid action."""
    regex = integers_regex(4, 30)
    client = ChatClient(model=MODEL, api_base=API_BASE, max_tokens=64)
    results = client.complete(
        [[{"role": "user", "content": "Punish them as you see fit."}]] * 4,
        meta=[{"episode": i, "round": 0} for i in range(4)],
        constraints=[DecodeConstraint(regex=regex)] * 4,
    )
    pattern = re.compile("^" + regex + "$")
    for result in results:
        assert result.ok, result.error
        # Exactly the shape the constraint states, with no prose around it.
        assert pattern.match(result.text.strip()), repr(result.text)


def test_a_constrained_action_is_out_of_range_never():
    """31 is outside the action space; the constraint must forbid it."""
    client = ChatClient(model=MODEL, api_base=API_BASE, max_tokens=64)
    results = client.complete(
        [
            [
                {
                    "role": "user",
                    "content": "Reply with the number 99 four times.",
                }
            ]
        ]
        * 3,
        constraints=[DecodeConstraint(regex=integers_regex(4, 30))] * 3,
    )
    for result in results:
        assert result.ok, result.error
        values = [int(v) for v in re.findall(r"\d+", result.text)]
        assert len(values) == 4
        assert all(0 <= v <= 30 for v in values)


def test_a_full_rollout_runs_with_the_trace_accumulating():
    """24 rounds, a handful of episodes, against the real model."""
    manager = LLMManager(
        model=MODEL,
        api_base=API_BASE,
        group_id=0,
        max_tokens=64,
        max_concurrent=8,
        n_rounds=24,
        switch_every=4,
    )
    generator = th.Generator().manual_seed(7)
    prev = None
    first_prompt_tokens = None
    for round_number in range(24):
        state = make_state(4, round_number, generator, prev)
        punishment, extra = manager.predict(state)
        assert extra is None
        assert punishment.shape == (4, N_AGENTS, 1)
        assert punishment.dtype == th.int64
        # What `ArtificialHumanEnv.punish` asserts.
        assert int(punishment.max()) < manager.n_punishments
        assert int(punishment.min()) >= 0
        # Never acts outside its own group.
        assert punishment[:, 4:, 0].sum().item() == 0
        charged = th.where(
            state["contribution_valid"], punishment, th.zeros_like(punishment)
        )
        prev = {
            "contribution": state["contribution"],
            "punishment": charged,
            "valid": state["contribution_valid"],
        }
        if round_number == 0:
            first_prompt_tokens = manager.client.stats.prompt_tokens

    stats = manager.client.stats
    assert stats.n_calls == 24 * 4
    assert stats.n_errors == 0
    # Constrained decode means the parser never has to rescue anything.
    assert manager.parse_failures == 0
    assert manager.parse_failure_rate == 0.0
    # Nothing was cut off: the budget is big enough with thinking off.
    assert stats.truncated == 0
    # The trace grew, which is the whole point of the accumulating prompt.
    last_round = stats.prompt_tokens - first_prompt_tokens
    assert last_round > first_prompt_tokens


def test_the_log_records_every_call_with_its_episode_and_round(tmp_path):
    import json

    log_path = tmp_path / "calls.jsonl"
    manager = LLMManager(
        model=MODEL,
        api_base=API_BASE,
        group_id=0,
        max_tokens=64,
        max_concurrent=4,
        log_path=str(log_path),
    )
    generator = th.Generator().manual_seed(3)
    manager.predict(make_state(2, 0, generator))
    manager.predict(make_state(2, 1, generator))

    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    calls = [line for line in lines if line["record"] == "call"]
    assert len(calls) == 4
    assert sorted((c["episode"], c["round"]) for c in calls) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    for call in calls:
        assert call["error"] is None
        assert call["prompt"].startswith("<<user>>")
        assert call["completion"]
        assert call["prompt_tokens"] > 0
        assert call["endpoint"]
