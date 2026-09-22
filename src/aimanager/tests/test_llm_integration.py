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
* that vLLM honours a decode constraint built for the roster actually present,
  so the parse failure rate is zero by construction rather than by luck, and
  an answer for a roster the decision point does not have cannot be emitted;
* that a full 24-round rollout completes with the trace accumulating, against
  the model that will actually be used.
"""

import json
import os
import re

import pytest
import torch as th

from aimanager.llm.parse import parse_punishments
from aimanager.llm.prompt import DEFAULT_VERSION
from aimanager.llm.trace import MAX_PUNISHMENT
from aimanager.manager.llm_client import ChatClient, DecodeConstraint
from aimanager.manager.llm_manager import LLMManager, agent_label, label_answer_regex

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


def make_state(n_batch, round_number, generator, prev=None, groups=None):
    size = (n_batch, N_AGENTS, 1)
    if groups is None:
        groups = (
            th.tensor(AGENT_GROUPS, dtype=th.int64)
            .reshape(1, N_AGENTS, 1)
            .expand(size)
            .contiguous()
        )
    state = {
        "contribution": th.randint(0, 21, size, generator=generator, dtype=th.int64),
        "contribution_valid": th.rand(size, generator=generator) > 0.05,
        "punishment": th.zeros(size, dtype=th.int64),
        "punishment_valid": th.ones(size, dtype=th.bool),
        "round_number": th.full(size, round_number, dtype=th.int64),
        "is_first": th.full(size, round_number == 0, dtype=th.bool),
        "agent_group": groups,
        "common_good": th.zeros(size, dtype=th.float),
    }
    zeros_i = th.zeros(size, dtype=th.int64)
    state["prev_contribution"] = prev["contribution"] if prev else zeros_i
    state["prev_punishment"] = prev["punishment"] if prev else zeros_i
    state["prev_contribution_valid"] = (
        prev["valid"] if prev else th.zeros(size, dtype=th.bool)
    )
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

    assert on.finish_reason == "length"
    assert "<think>" in (on.text or "") or on.reasoning_text
    assert off.finish_reason == "stop"
    assert off.completion_tokens < on.completion_tokens


def test_qwen3_gets_thinking_off_without_being_asked():
    """The default must be the safe one, or every caller has to remember."""
    client = ChatClient(model=MODEL, api_base=API_BASE, max_tokens=64)
    if "qwen3" in MODEL.lower():
        assert client.enable_thinking is False
    result = client.complete([[{"role": "user", "content": ASK}]])[0]
    assert result.ok, result.error
    assert result.finish_reason == "stop"


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8])
def test_vllm_honours_a_constraint_for_any_roster_size(size):
    """Group size runs 1..8 in the human data, so the constraint must too."""
    labels = [agent_label(a) for a in range(size)]
    regex = label_answer_regex(labels, MAX_PUNISHMENT)
    client = ChatClient(model=MODEL, api_base=API_BASE, max_tokens=128)
    results = client.complete(
        [[{"role": "user", "content": "Punish them as you see fit."}]] * 3,
        meta=[{"episode": i, "round": 0} for i in range(3)],
        constraints=[DecodeConstraint(regex=regex)] * 3,
    )
    pattern = re.compile("^" + regex + "$")
    for result in results:
        assert result.ok, result.error
        assert pattern.match(result.text.strip()), repr(result.text)
        # And the guard agrees with the mechanism.
        parsed = parse_punishments(result.text, labels)
        assert parsed.ok, parsed.reason
        assert list(parsed.punishments) == labels
        assert all(0 <= v <= MAX_PUNISHMENT for v in parsed.punishments.values())


def test_the_constraint_forbids_an_answer_for_the_wrong_roster():
    """The failure mode a positional reader would have filed silently."""
    labels = [agent_label(a) for a in (1, 2, 4)]
    regex = label_answer_regex(labels, MAX_PUNISHMENT)
    client = ChatClient(model=MODEL, api_base=API_BASE, max_tokens=128)
    results = client.complete(
        [
            [
                {
                    "role": "user",
                    "content": (
                        "Answer for Player 1, Player 2, Player 3 and Player 4."
                    ),
                }
            ]
        ]
        * 3,
        constraints=[DecodeConstraint(regex=regex)] * 3,
    )
    for result in results:
        assert result.ok, result.error
        # Even asked for the wrong roster, it cannot emit one: the prompt
        # names Players 1-4, the roster is Players 2, 3 and 5, and the answer
        # is for the roster.
        assert re.match("^" + regex + "$", result.text.strip()), repr(result.text)
        for label in labels:
            assert f"{label} =" in result.text
        for absent in ("Player 1 =", "Player 4 ="):
            assert absent not in result.text


def test_a_constrained_value_is_never_out_of_range():
    labels = [agent_label(a) for a in range(4)]
    regex = label_answer_regex(labels, MAX_PUNISHMENT)
    client = ChatClient(model=MODEL, api_base=API_BASE, max_tokens=128)
    results = client.complete(
        [[{"role": "user", "content": "Punish everyone 99 points."}]] * 3,
        constraints=[DecodeConstraint(regex=regex)] * 3,
    )
    for result in results:
        assert result.ok, result.error
        parsed = parse_punishments(result.text, labels)
        assert parsed.ok, parsed.reason
        assert all(0 <= v <= MAX_PUNISHMENT for v in parsed.punishments.values())


def test_a_full_rollout_runs_with_the_trace_accumulating():
    """24 rounds, a handful of episodes, against the real model and prompt."""
    manager = LLMManager(
        model=MODEL,
        api_base=API_BASE,
        prompt_version=DEFAULT_VERSION,
        group_id=0,
        max_tokens=128,
        max_concurrent=8,
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
    assert manager.summary()["zero_fallback_answers"] == 0
    # Nothing was cut off: the budget is big enough with thinking off.
    assert stats.truncated == 0
    # The trace grew, which is the point of the accumulating prompt.
    last_round = stats.prompt_tokens - first_prompt_tokens
    assert last_round > first_prompt_tokens


def test_a_reshuffling_group_is_served_the_roster_it_has(tmp_path):
    """The roster changes every fourth round; the run must follow it."""
    log_path = tmp_path / "calls.jsonl"
    manager = LLMManager(
        model=MODEL,
        api_base=API_BASE,
        group_id=0,
        max_tokens=128,
        max_concurrent=4,
        log_path=str(log_path),
    )
    generator = th.Generator().manual_seed(11)
    size = (1, N_AGENTS, 1)
    # Round 0: agents 0-3. Round 1: agents 0, 1, 4, 5, 6 -- five players.
    manager.predict(make_state(1, 0, generator))
    groups = th.ones(size, dtype=th.int64)
    for agent in (0, 1, 4, 5, 6):
        groups[0, agent, 0] = 0
    punishment, _ = manager.predict(make_state(1, 1, generator, groups=groups))

    assert punishment[0, 2, 0].item() == 0
    assert punishment[0, 3, 0].item() == 0
    assert manager.parse_failures == 0

    verdicts = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if json.loads(line)["record"] == "parse"
    ]
    assert [v["labels"] for v in verdicts] == [
        ["Player 1", "Player 2", "Player 3", "Player 4"],
        ["Player 1", "Player 2", "Player 5", "Player 6", "Player 7"],
    ]
    assert all(v["ok"] for v in verdicts)
