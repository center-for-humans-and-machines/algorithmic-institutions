"""Unit tests for the LLM manager's serving and client layer.

These run anywhere: no GPU, no PyG, no language model. The transport is
exercised against a real HTTP server on localhost that answers the OpenAI
chat-completions protocol, so the request body, the concurrency, the shard
routing and the JSONL log are all tested on the path a real run takes.

The prompt, the trace format and the parser are `aimanager.llm`'s and are
tested there. What is tested here is the half this module owns: that the trace
handed to `build_prompt` is built correctly from `served_state()`, that the
decode constraint is built for the roster actually present, and that a failure
is counted and falls back loudly rather than silently.
"""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import torch as th

from aimanager.llm.parse import parse_punishments, summarise
from aimanager.llm.trace import MAX_PUNISHMENT
from aimanager.manager.llm_client import (
    ChatClient,
    DecodeConstraint,
    integers_regex,
    parse_endpoints,
    wants_thinking_flag,
)
from aimanager.manager.llm_manager import (
    LLMManager,
    agent_label,
    label_answer_regex,
)

N_AGENTS = 8
AGENT_GROUPS = [0, 0, 0, 0, 1, 1, 1, 1]


# ---------------------------------------------------------------------------
# A stub endpoint that speaks the real protocol
# ---------------------------------------------------------------------------


def answer_for(body, value=3):
    """A well-formed answer for whatever roster the request constrains."""
    regex = (body.get("structured_outputs") or {}).get("regex", "")
    labels = re.findall(r"Player\\?\s(\d+)", regex)
    if not labels:
        labels = ["1", "2", "3", "4"]
    return "PUNISHMENT: " + ", ".join(f"Player {n} = {value}" for n in labels)


class _StubServer:
    """An OpenAI-compatible endpoint. `replies` is a list or a callable."""

    def __init__(self, replies=answer_for, status=200):
        self.replies = replies
        self.status = status
        self.bodies = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - name fixed by the base class
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode())
                with outer._lock:
                    outer.bodies.append(body)
                    index = len(outer.bodies) - 1
                if outer.status != 200:
                    self.send_response(outer.status)
                    self.end_headers()
                    self.wfile.write(b'{"error": "stub failure"}')
                    return
                if callable(outer.replies):
                    reply = outer.replies(body)
                else:
                    reply = outer.replies[index % len(outer.replies)]
                payload = json.dumps(
                    {
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": reply},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": len(json.dumps(body["messages"])) // 4,
                            "completion_tokens": len(reply) // 4,
                        },
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._server.shutdown()
        self._server.server_close()

    @property
    def api_base(self):
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def prompts(self):
        return [b["messages"][-1]["content"] for b in self.bodies]


def make_state(
    n_batch, round_number, contributions=None, valid=None, groups=None, prev=None
):
    """A `served_state()`-shaped dict, (B, A, 1), with every key read."""
    size = (n_batch, N_AGENTS, 1)
    if contributions is None:
        contributions = th.full(size, 10, dtype=th.int64)
    if valid is None:
        valid = th.ones(size, dtype=th.bool)
    if groups is None:
        groups = (
            th.tensor(AGENT_GROUPS, dtype=th.int64)
            .reshape(1, N_AGENTS, 1)
            .expand(size)
            .contiguous()
        )
    state = {
        "contribution": contributions,
        "contribution_valid": valid,
        "punishment": th.zeros(size, dtype=th.int64),
        "punishment_valid": th.ones(size, dtype=th.bool),
        "round_number": th.full(size, round_number, dtype=th.int64),
        "is_first": th.full(size, round_number == 0, dtype=th.bool),
        "agent_group": groups,
        "common_good": th.zeros(size, dtype=th.float),
    }
    prev = prev or {}
    state["prev_contribution"] = prev.get(
        "contribution", th.zeros(size, dtype=th.int64)
    )
    state["prev_punishment"] = prev.get("punishment", th.zeros(size, dtype=th.int64))
    state["prev_contribution_valid"] = prev.get("valid", th.zeros(size, dtype=th.bool))
    return state


def make_manager(api_base, **kwargs):
    kwargs.setdefault("group_id", 0)
    kwargs.setdefault("max_concurrent", 8)
    return LLMManager(model="Qwen/Qwen3-8B", api_base=api_base, **kwargs)


# ---------------------------------------------------------------------------
# The rollout seat
# ---------------------------------------------------------------------------


def test_predict_returns_the_shape_and_dtype_the_env_asserts():
    """`ArtificialHumanEnv.punish` asserts int64, (B, A, 1), max < n."""
    with _StubServer() as stub:
        manager = make_manager(stub.api_base)
        punishment, extra = manager.predict(make_state(4, 0))
    assert extra is None
    assert punishment.shape == (4, N_AGENTS, 1)
    assert punishment.dtype == th.int64
    assert int(punishment.max()) < manager.n_punishments
    assert int(punishment.min()) >= 0
    assert manager.parse_failures == 0


def test_predict_punishes_only_its_own_group():
    with _StubServer(lambda b: answer_for(b, 7)) as stub:
        manager = make_manager(stub.api_base, group_id=0)
        punishment, _ = manager.predict(make_state(2, 0))
    assert punishment[:, :4, 0].tolist() == [[7] * 4] * 2
    assert punishment[:, 4:, 0].tolist() == [[0] * 4] * 2


def test_one_call_per_episode_per_round():
    with _StubServer() as stub:
        manager = make_manager(stub.api_base)
        manager.predict(make_state(5, 0))
        assert len(stub.bodies) == 5
        manager.predict(make_state(5, 1))
        assert len(stub.bodies) == 10
    assert manager.client.stats.n_calls == 10


# ---------------------------------------------------------------------------
# A GROUP IS NOT FOUR PLAYERS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 8])
def test_every_group_size_from_one_to_eight_is_served(size):
    """Group size runs 1..8 in the human data and is four only 26% of the
    time, so nothing may assume an arity."""
    state = make_state(1, 0)
    groups = th.ones((1, N_AGENTS, 1), dtype=th.int64)
    groups[0, :size, 0] = 0
    state["agent_group"] = groups

    with _StubServer() as stub:
        manager = make_manager(stub.api_base, group_id=0)
        punishment, _ = manager.predict(state)
        regex = stub.bodies[0]["structured_outputs"]["regex"]
        prompt = stub.prompts()[0]

    expected = [agent_label(a) for a in range(size)]
    # The constraint names exactly this roster, in order.
    assert re.match(
        "^" + regex + "$",
        "PUNISHMENT: " + ", ".join(f"{label} = 3" for label in expected),
    )
    # An answer for a different roster cannot be emitted at all.
    assert not re.match(
        "^" + regex + "$",
        "PUNISHMENT: " + ", ".join(f"Player {i + 1} = 3" for i in range(size + 1)),
    )
    for label in expected:
        assert label in prompt
    assert punishment[0, :size, 0].tolist() == [3] * size
    assert punishment[0, size:, 0].sum().item() == 0


def test_the_roster_is_read_fresh_at_every_decision_point():
    """Members reshuffle; the constraint must follow, not be built once."""
    with _StubServer() as stub:
        manager = make_manager(stub.api_base, group_id=0)
        manager.predict(make_state(1, 0))
        # Agent 0 leaves group 0 and agent 4 joins it.
        groups = th.tensor(AGENT_GROUPS, dtype=th.int64).reshape(1, N_AGENTS, 1).clone()
        groups[0, 0, 0] = 1
        groups[0, 4, 0] = 0
        manager.predict(make_state(1, 1, groups=groups))
        first, second = (b["structured_outputs"]["regex"] for b in stub.bodies)

    assert "Player\\ 1" in first.replace("Player 1", "Player\\ 1") or "Player" in first
    assert re.match(
        "^" + first + "$",
        "PUNISHMENT: " + ", ".join(f"Player {i} = 0" for i in (1, 2, 3, 4)),
    )
    # Round 1's roster is players 2, 3, 4, 5 -- and 1 is no longer admissible.
    assert re.match(
        "^" + second + "$",
        "PUNISHMENT: " + ", ".join(f"Player {i} = 0" for i in (2, 3, 4, 5)),
    )
    assert not re.match(
        "^" + second + "$",
        "PUNISHMENT: " + ", ".join(f"Player {i} = 0" for i in (1, 2, 3, 4)),
    )


def test_label_answer_regex_admits_only_the_stated_roster():
    regex = label_answer_regex(["Player 2", "Player 5"], MAX_PUNISHMENT)
    ok = re.compile("^" + regex + "$")
    assert ok.match("PUNISHMENT: Player 2 = 0, Player 5 = 30")
    assert not ok.match("PUNISHMENT: Player 5 = 0, Player 2 = 30")  # wrong order
    assert not ok.match("PUNISHMENT: Player 2 = 0")  # short
    assert not ok.match("PUNISHMENT: Player 2 = 0, Player 5 = 31")  # out of range
    assert not ok.match("PUNISHMENT: Player 2 = 0, Player 3 = 0")  # wrong roster
    with pytest.raises(ValueError, match="empty roster"):
        label_answer_regex([])


def test_the_constrained_answer_is_what_the_parser_accepts():
    """The constraint and the guard must agree, or one of them is wrong."""
    labels = ["Player 2", "Player 3", "Player 7"]
    regex = label_answer_regex(labels, MAX_PUNISHMENT)
    answer = "PUNISHMENT: Player 2 = 4, Player 3 = 0, Player 7 = 30"
    assert re.match("^" + regex + "$", answer)
    result = parse_punishments(answer, labels)
    assert result.ok
    assert result.punishments == {"Player 2": 4, "Player 3": 0, "Player 7": 30}


# ---------------------------------------------------------------------------
# The trace, built from the env by the rules aimanager.llm states
# ---------------------------------------------------------------------------


def test_a_player_who_gave_no_input_is_never_shown_a_number():
    """Rule 1: the filler in a masked cell must not reach the model."""
    size = (1, N_AGENTS, 1)
    contributions = th.full(size, 9, dtype=th.int64)  # the dataset median
    valid = th.ones(size, dtype=th.bool)
    valid[0, 1, 0] = False
    with _StubServer() as stub:
        manager = make_manager(stub.api_base, group_id=0)
        manager.predict(make_state(1, 0, contributions=contributions, valid=valid))
        prompt = stub.prompts()[0]
    assert "Player 2 gave no input" in prompt
    assert "Player 2 put in 9" not in prompt


def test_the_trace_carries_the_punishment_the_game_charged():
    """Rule 2: the past round shows what was charged, not the raw action."""
    size = (1, N_AGENTS, 1)
    charged = th.zeros(size, dtype=th.int64)
    charged[0, :4, 0] = th.tensor([5, 0, 5, 5])
    prev_valid = th.ones(size, dtype=th.bool)
    prev_valid[0, 1, 0] = False

    with _StubServer() as stub:
        manager = make_manager(stub.api_base, group_id=0)
        manager.predict(make_state(1, 0))
        manager.predict(
            make_state(
                1,
                1,
                prev={
                    "contribution": th.full(size, 11, dtype=th.int64),
                    "punishment": charged,
                    "valid": prev_valid,
                },
            )
        )
        second = stub.prompts()[1]

    assert "Player 1 put in 11, you punished 5" in second
    # Agent 1 timed out: charged zero, and shown as no-input rather than as 11.
    assert "Player 2 gave no input" in second
    assert "Player 2 put in 11" not in second


def test_round_zero_resets_the_trace():
    with _StubServer() as stub:
        manager = make_manager(stub.api_base, group_id=0)
        manager.predict(make_state(1, 0))
        manager.predict(make_state(1, 1))
        manager.predict(make_state(1, 0))
        prompts = stub.prompts()
    # A fresh episode's first prompt has no history, like the very first one.
    assert prompts[0] == prompts[2]
    assert len(prompts[1]) > len(prompts[0])


def test_the_prompt_grows_by_appending_so_the_prefix_cache_holds():
    """Prefix caching is what keeps a long rollout cheap; see the module
    docstring.

    Round r's prompt shares everything up to the last round heading with
    round r-1's. It is not identical beyond that: the block for round r-1 is
    rendered `punishment not set yet` while it is being decided and
    `you punished N` once it has been, so exactly one block is re-prefilled
    per round. Everything before it -- which is the bulk, and all of the
    growth -- is reused.
    """
    with _StubServer() as stub:
        manager = make_manager(stub.api_base, group_id=0)
        for round_number in range(5):
            manager.predict(make_state(1, round_number))
        prompts = stub.prompts()

    for earlier, later in zip(prompts[1:], prompts[2:]):
        cut = earlier.rfind("\nRound ")
        assert cut > 0
        shared = earlier[:cut]
        assert later.startswith(shared)
        # The shared part is most of the prompt, and grows with the trace.
        assert len(shared) > 0.5 * len(earlier)


# ---------------------------------------------------------------------------
# The guard: loud, counted, never silent
# ---------------------------------------------------------------------------


def test_an_unparsable_reply_falls_back_to_zero_and_is_counted():
    with _StubServer(["I would rather not say."]) as stub:
        manager = make_manager(stub.api_base)
        punishment, _ = manager.predict(make_state(2, 0))
    assert punishment.sum().item() == 0
    assert manager.parse_failures == 2
    assert manager.n_decisions == 2
    assert manager.parse_failure_rate == 1.0
    assert manager.summary()["zero_fallback_answers"] == 2
    assert manager.summary()["reason[no_marker]"] == 2


def test_an_answer_for_the_wrong_roster_is_a_failure_not_a_misfiling():
    """The failure the constraint prevents, with the constraint off: three
    numbers for four players must not be filed against three of them."""
    with _StubServer(["PUNISHMENT: Player 1 = 9, Player 2 = 9, Player 3 = 9"]) as stub:
        manager = make_manager(stub.api_base, constrained_decode=False)
        punishment, _ = manager.predict(make_state(1, 0))
    assert punishment.sum().item() == 0
    assert manager.parse_failures == 1
    assert manager.summary()["reason[label_mismatch]"] == 1


def test_a_failing_endpoint_falls_back_to_zero_rather_than_aborting():
    with _StubServer(status=500) as stub:
        manager = make_manager(stub.api_base, max_retries=1, retry_base_delay=0.0)
        punishment, _ = manager.predict(make_state(2, 0))
    assert punishment.sum().item() == 0
    assert manager.call_errors == 2
    assert manager.parse_failures == 2


def test_a_punishment_aimed_at_a_no_input_player_is_recorded_as_wasted():
    """It parses fine and the env discards it, so it is measured, not hidden."""
    size = (1, N_AGENTS, 1)
    valid = th.ones(size, dtype=th.bool)
    valid[0, 2, 0] = False
    with _StubServer(lambda b: answer_for(b, 6)) as stub:
        manager = make_manager(stub.api_base, group_id=0)
        punishment, _ = manager.predict(make_state(1, 0, valid=valid))
    assert manager.wasted_answers == 1
    assert manager.parse_failures == 0
    # `enforce` zeroes it, exactly as the environment would.
    assert punishment[0, 2, 0].item() == 0
    assert punishment[0, 0, 0].item() == 6


def test_the_counters_agree_with_the_shared_summarise():
    """`parse.summarise` is the one place the rate is defined; the manager's
    incremental counters must not drift from it."""
    answers = [
        "PUNISHMENT: Player 1 = 1, Player 2 = 2, Player 3 = 3, Player 4 = 4",
        "nonsense with no marker",
        "PUNISHMENT: 1, 2, 3, 4",
        "PUNISHMENT: Player 1 = 99, Player 2 = 0, Player 3 = 0, Player 4 = 0",
    ]
    labels = [agent_label(a) for a in range(4)]
    with _StubServer(answers) as stub:
        manager = make_manager(stub.api_base, constrained_decode=False)
        manager.predict(make_state(4, 0))

    expected = summarise(parse_punishments(a, labels) for a in answers)
    got = manager.summary()
    for key, value in expected.items():
        assert got[key] == value, key


def test_report_carries_what_a_result_must_be_quoted_with():
    with _StubServer() as stub:
        manager = make_manager(stub.api_base)
        manager.predict(make_state(2, 0))
        report = manager.report()
    for key in (
        "model",
        "temperature",
        "enable_thinking",
        "prompt_version",
        "constrained_decode",
        "parse_failure_rate",
        "completions_per_s",
        "prompt_tokens",
        "answers",
        "failures",
        "zero_fallback_answers",
    ):
        assert key in report, key
    assert report["answers"] == 2
    assert report["parse_failure_rate"] == 0.0


# ---------------------------------------------------------------------------
# The api_manager seat
# ---------------------------------------------------------------------------


def make_create_data(n_groups=2, n_t=3):
    """The `create_data` view: (n_groups, A, T), other-group cells masked."""
    size = (n_groups, N_AGENTS, n_t)
    in_group = th.zeros(size, dtype=th.bool)
    for row in range(n_groups):
        for agent in range(N_AGENTS):
            in_group[row, agent, :] = AGENT_GROUPS[agent] == row
    return {
        "contribution": th.full(size, 8, dtype=th.int64),
        "contribution_valid": in_group.clone(),
        "punishment": th.zeros(size, dtype=th.int64),
        "punishment_valid": in_group.clone(),
        "in_group": in_group,
        "round_number": th.arange(n_t).reshape(1, 1, n_t).expand(size).contiguous(),
    }


def test_get_punishments_fills_the_shape_multimanager_indexes():
    """`MultiManager` slices `v[group_idx, arange, -1]`, so shape must match."""
    data = make_create_data()
    with _StubServer(lambda b: answer_for(b, 6)) as stub:
        manager = make_manager(stub.api_base, group_id=0)
        out = manager.get_punishments(data)
    assert out.shape == data["punishment"].shape
    assert out.dtype == th.int64
    assert out[0, :, -1].tolist() == [6, 6, 6, 6, 0, 0, 0, 0]
    assert out[1].sum().item() == 0


def test_get_punishments_rebuilds_the_trace_from_the_time_axis():
    data = make_create_data(n_t=4)
    with _StubServer() as stub:
        manager = make_manager(stub.api_base, group_id=0)
        manager.get_punishments(data)
        prompt = stub.prompts()[0]
    # Three completed rounds, then the one being decided.
    for round_number in (1, 2, 3, 4):
        assert f"Round {round_number}" in prompt
    assert "punishment not set yet" in prompt


def test_llm_is_registered_as_a_manager_type():
    # `api_manager` imports the GNN stack, which needs PyG; that is only on
    # the cluster, so this one runs there (scripts/remote_test.sh).
    pytest.importorskip("torch_scatter")
    from aimanager.manager.api_manager import MANAGER_CLASS

    assert MANAGER_CLASS["llm"] is LLMManager


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


def test_qwen3_gets_thinking_disabled_without_being_asked():
    """The documented trap: Qwen3 thinks by default and truncates."""
    assert wants_thinking_flag("Qwen/Qwen3-8B")
    assert wants_thinking_flag("hosted_vllm/Qwen/Qwen3-32B")
    assert not wants_thinking_flag("meta-llama/Llama-3.1-8B-Instruct")

    with _StubServer(["ok"]) as stub:
        client = ChatClient(model="Qwen/Qwen3-8B", api_base=stub.api_base)
        assert client.enable_thinking is False
        client.complete([[{"role": "user", "content": "hi"}]])
        body = stub.bodies[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["model"] == "Qwen/Qwen3-8B"


def test_the_provider_prefix_is_accepted_and_stripped():
    with _StubServer(["ok"]) as stub:
        client = ChatClient(model="hosted_vllm/Qwen/Qwen3-8B", api_base=stub.api_base)
        client.complete([[{"role": "user", "content": "hi"}]])
    assert stub.bodies[0]["model"] == "Qwen/Qwen3-8B"


def test_a_non_thinking_model_sends_no_template_kwargs():
    with _StubServer(["ok"]) as stub:
        client = ChatClient(model="meta-llama/Llama-3.1-8B", api_base=stub.api_base)
        assert client.enable_thinking is None
        client.complete([[{"role": "user", "content": "hi"}]])
    assert "chat_template_kwargs" not in stub.bodies[0]


def test_temperature_and_seed_reach_the_wire():
    with _StubServer(["ok"]) as stub:
        client = ChatClient(
            model="Qwen/Qwen3-8B",
            api_base=stub.api_base,
            temperature=0.0,
            seed=42,
            max_tokens=64,
        )
        client.complete([[{"role": "user", "content": "hi"}]])
    body = stub.bodies[0]
    assert body["temperature"] == 0.0
    assert body["seed"] == 42
    assert body["max_tokens"] == 64


def test_legacy_guided_style_uses_the_old_field_names():
    constraint = DecodeConstraint(regex="abc")
    assert constraint.as_body_fields("structured_outputs") == {
        "structured_outputs": {"regex": "abc"}
    }
    assert constraint.as_body_fields("guided") == {"guided_regex": "abc"}
    with pytest.raises(ValueError, match="constraint_style"):
        constraint.as_body_fields("nonsense")


def test_integers_regex_admits_only_well_formed_numbers():
    pattern = re.compile("^" + integers_regex(4, 30) + "$")
    assert pattern.match("0, 0, 0, 0")
    assert pattern.match("30, 7, 0, 12")
    assert not pattern.match("31, 0, 0, 0")
    assert not pattern.match("1, 2, 3")
    assert not pattern.match("00, 1, 2, 3")


def test_batch_order_is_preserved_under_concurrency():
    replies = [f"reply {i}" for i in range(12)]
    with _StubServer(replies) as stub:
        client = ChatClient(
            model="Qwen/Qwen3-8B", api_base=stub.api_base, max_concurrent=6
        )
        conversations = [
            [{"role": "user", "content": f"episode {i}"}] for i in range(12)
        ]
        results = client.complete(conversations)
    by_prompt = {
        body["messages"][0]["content"]: index for index, body in enumerate(stub.bodies)
    }
    for i, result in enumerate(results):
        assert result.text == replies[by_prompt[f"episode {i}"]]


def test_client_requires_an_endpoint(monkeypatch):
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    with pytest.raises(ValueError, match="api_base"):
        ChatClient(model="Qwen/Qwen3-8B")


# ---------------------------------------------------------------------------
# Data-parallel sharding
# ---------------------------------------------------------------------------


def test_endpoints_parse_from_a_list_or_a_comma_separated_string():
    assert parse_endpoints("http://a/v1,http://b/v1") == ["http://a/v1", "http://b/v1"]
    assert parse_endpoints(["http://a/v1/", " http://b/v1 "]) == [
        "http://a/v1",
        "http://b/v1",
    ]


def test_an_episode_always_meets_the_same_server():
    """Sampling differs between instances; a moving episode would read as
    variance in the manager rather than as an artefact of routing."""
    client = ChatClient(
        model="Qwen/Qwen3-8B",
        api_base=["http://a/v1", "http://b/v1", "http://c/v1"],
    )
    assert client.n_endpoints == 3
    first = [client.endpoint_for({"episode": e}) for e in range(9)]
    for _round in range(3):
        assert [client.endpoint_for({"episode": e}) for e in range(9)] == first
    assert first == ["http://a/v1", "http://b/v1", "http://c/v1"] * 3


def test_sharding_routes_a_batch_across_every_server():
    with (
        _StubServer(lambda b: answer_for(b, 1)) as one,
        _StubServer(lambda b: answer_for(b, 2)) as two,
    ):
        manager = make_manager([one.api_base, two.api_base])
        punishment, _ = manager.predict(make_state(6, 0))
    assert len(one.bodies) == 3
    assert len(two.bodies) == 3
    assert punishment[0, :4, 0].tolist() == [1, 1, 1, 1]
    assert punishment[1, :4, 0].tolist() == [2, 2, 2, 2]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_every_call_is_logged_with_its_episode_round_and_verdict(tmp_path):
    log_path = tmp_path / "calls.jsonl"
    with _StubServer() as stub:
        manager = make_manager(stub.api_base, log_path=str(log_path))
        manager.predict(make_state(2, 0))
        manager.predict(make_state(2, 1))
    assert stub.bodies

    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    header = [line for line in lines if line["record"] == "header"][0]
    assert header["temperature"] == 0.0
    assert header["enable_thinking"] is False

    calls = [line for line in lines if line["record"] == "call"]
    assert len(calls) == 4
    assert sorted((c["episode"], c["round"]) for c in calls) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    for call in calls:
        # The rendered prompt, not the pieces it was assembled from: a
        # trace-format bug is invisible in the pieces.
        assert call["prompt"].startswith("<<system>>")
        assert "YOUR DECISION FOR ROUND" in call["prompt"]
        assert call["prompt_version"]
        assert call["prompt_fingerprint"]
        assert call["error"] is None
        assert call["constraint"]

    # The verdict, so a result can name what produced it.
    verdicts = [line for line in lines if line["record"] == "parse"]
    assert len(verdicts) == 4
    for verdict in verdicts:
        assert verdict["ok"] is True
        assert verdict["form"] == "labelled"
        assert verdict["fallback"] is None
        assert verdict["labels"]


def test_a_failure_is_logged_with_its_reason_and_fallback(tmp_path):
    log_path = tmp_path / "calls.jsonl"
    with _StubServer(["no answer here"]) as stub:
        manager = make_manager(stub.api_base, log_path=str(log_path))
        manager.predict(make_state(1, 0))
    assert stub.bodies
    verdict = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if json.loads(line)["record"] == "parse"
    ][0]
    assert verdict["ok"] is False
    assert verdict["reason"] == "no_marker"
    assert verdict["fallback"] == "zero"
