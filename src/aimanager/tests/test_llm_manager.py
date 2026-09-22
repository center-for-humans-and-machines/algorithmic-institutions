"""Unit tests for the LLM manager's serving and client layer.

These run anywhere: no GPU, no PyG, no language model. The transport is
exercised against a real HTTP server on localhost that answers the
OpenAI chat-completions protocol with canned replies, so the request body, the
concurrency and the JSONL log are all tested on the path a real run takes.

What is deliberately NOT tested here is the wording of the prompt or the
parsing of a real reply: those are a sibling's, injected as a strategy. What is
tested is that the boundary holds -- that a strategy is handed a
`RoundContext` with the trace in it, and that whatever it returns is
length-checked, clamped and, when it fails, replaced by zero punishment and
counted.
"""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import torch as th

from aimanager.manager.llm_client import (
    ChatClient,
    DecodeConstraint,
    integers_regex,
    parse_endpoints,
    wants_thinking_flag,
)
from aimanager.manager.llm_manager import (
    DefaultPromptStrategy,
    LLMManager,
    PlayerRound,
    RoundContext,
    get_prompt_strategy,
    register_prompt_strategy,
)

N_AGENTS = 8
AGENT_GROUPS = [0, 0, 0, 0, 1, 1, 1, 1]


# ---------------------------------------------------------------------------
# A stub endpoint that speaks the real protocol
# ---------------------------------------------------------------------------


class _StubServer:
    """An OpenAI-compatible endpoint that replays scripted answers."""

    def __init__(self, replies, status=200):
        self.replies = list(replies)
        self.status = status
        self.bodies = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
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


def make_state(n_batch, round_number, contributions=None, valid=None, prev=None):
    """A `served_state()`-shaped dict, (B, A, 1), with every key read."""
    size = (n_batch, N_AGENTS, 1)
    if contributions is None:
        contributions = th.full(size, 10, dtype=th.int64)
    if valid is None:
        valid = th.ones(size, dtype=th.bool)
    state = {
        "contribution": contributions,
        "contribution_valid": valid,
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
    prev = prev or {}
    state["prev_contribution"] = prev.get(
        "contribution", th.zeros(size, dtype=th.int64)
    )
    state["prev_punishment"] = prev.get("punishment", th.zeros(size, dtype=th.int64))
    state["prev_contribution_valid"] = prev.get("valid", th.zeros(size, dtype=th.bool))
    state["prev_common_good"] = prev.get("common_good", th.zeros(size, dtype=th.float))
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
    with _StubServer(["3, 4, 5, 6"]) as stub:
        manager = make_manager(stub.api_base)
        punishment, extra = manager.predict(make_state(4, 0))
    assert extra is None
    assert punishment.shape == (4, N_AGENTS, 1)
    assert punishment.dtype == th.int64
    assert int(punishment.max()) < manager.n_punishments
    assert int(punishment.min()) >= 0


def test_predict_punishes_only_its_own_group():
    """The harness masks by group, but a manager must not act outside it."""
    with _StubServer(["7, 7, 7, 7"]) as stub:
        manager = make_manager(stub.api_base, group_id=0)
        punishment, _ = manager.predict(make_state(2, 0))
    assert punishment[:, :4, 0].tolist() == [[7] * 4] * 2
    assert punishment[:, 4:, 0].tolist() == [[0] * 4] * 2


def test_one_call_per_episode_per_round():
    """A B-episode round is B completions, issued as one batch."""
    with _StubServer(["1, 1, 1, 1"]) as stub:
        manager = make_manager(stub.api_base)
        manager.predict(make_state(5, 0))
        assert len(stub.bodies) == 5
        manager.predict(make_state(5, 1))
        assert len(stub.bodies) == 10
    assert manager.client.stats.n_calls == 10


def test_the_trace_accumulates_and_carries_what_the_game_charged():
    """Round 1's prompt must contain round 0, with the charged punishment."""
    seen = []

    class Recorder(DefaultPromptStrategy):
        version = "rec"

        def build(self, ctx):
            seen.append(ctx)
            return super().build(ctx)

    size = (1, N_AGENTS, 1)
    with _StubServer(["2, 2, 2, 2"]) as stub:
        manager = make_manager(stub.api_base, strategy=Recorder())
        manager.predict(make_state(1, 0))
        # The env zeroes a punishment aimed at a player who gave no input,
        # and that zero is what `prev_punishment` carries back.
        charged = th.zeros(size, dtype=th.int64)
        charged[0, :4, 0] = th.tensor([2, 0, 2, 2])
        prev_valid = th.ones(size, dtype=th.bool)
        prev_valid[0, 1, 0] = False
        manager.predict(
            make_state(
                1,
                1,
                prev={
                    "contribution": th.full(size, 11, dtype=th.int64),
                    "punishment": charged,
                    "valid": prev_valid,
                    "common_good": th.full(size, 13.5, dtype=th.float),
                },
            )
        )

    first, second = seen[0], seen[1]
    assert first.history == ()
    assert len(second.history) == 1
    record = second.history[0]
    assert record.round_number == 0
    assert [p.agent for p in record.players] == [0, 1, 2, 3]
    assert [p.contribution for p in record.players] == [11] * 4
    # Agent 1 gave no input: charged zero, and marked as such in the trace.
    assert [p.punishment for p in record.players] == [2, 0, 2, 2]
    assert [p.contribution_valid for p in record.players] == [True, False, True, True]
    assert record.common_good == pytest.approx(13.5)


def test_round_zero_resets_the_trace():
    """A new rollout starts from an empty trace without an explicit reset."""
    seen = []

    class Recorder(DefaultPromptStrategy):
        version = "rec"

        def build(self, ctx):
            seen.append(ctx)
            return super().build(ctx)

    with _StubServer(["0, 0, 0, 0"]) as stub:
        manager = make_manager(stub.api_base, strategy=Recorder())
        manager.predict(make_state(1, 0))
        manager.predict(make_state(1, 1))
        manager.predict(make_state(1, 0))
    assert [len(ctx.history) for ctx in seen] == [0, 1, 0]


def test_membership_changes_are_followed():
    """Players switch groups; the managed seats follow `agent_group`."""
    state = make_state(1, 0)
    # Agent 4 moves into group 0, agent 0 moves out.
    groups = state["agent_group"].clone()
    groups[0, 0, 0] = 1
    groups[0, 4, 0] = 0
    state["agent_group"] = groups
    with _StubServer(["5, 5, 5, 5"]) as stub:
        manager = make_manager(stub.api_base, group_id=0)
        punishment, _ = manager.predict(state)
    assert punishment[0, :, 0].tolist() == [0, 5, 5, 5, 5, 0, 0, 0]


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
        "common_good": th.full(size, 12.0, dtype=th.float),
    }


def test_get_punishments_fills_the_shape_multimanager_indexes():
    """`MultiManager` slices `v[group_idx, arange, -1]`, so shape must match."""
    data = make_create_data()
    with _StubServer(["6, 6, 6, 6"]) as stub:
        manager = make_manager(stub.api_base, group_id=0)
        out = manager.get_punishments(data)
    assert out.shape == data["punishment"].shape
    assert out.dtype == th.int64
    # Row 0 is group 0's view; only its own agents are acted on.
    assert out[0, :, -1].tolist() == [6, 6, 6, 6, 0, 0, 0, 0]
    # Row 1 is group 1's view, which this manager does not hold.
    assert out[1].sum().item() == 0


def test_get_punishments_rebuilds_the_trace_from_the_time_axis():
    seen = []

    class Recorder(DefaultPromptStrategy):
        version = "rec"

        def build(self, ctx):
            seen.append(ctx)
            return super().build(ctx)

    data = make_create_data(n_t=4)
    with _StubServer(["1, 1, 1, 1"]) as stub:
        manager = make_manager(stub.api_base, group_id=0, strategy=Recorder())
        manager.get_punishments(data)
    assert len(seen) == 1
    ctx = seen[0]
    # T columns are three completed rounds plus the one being decided.
    assert len(ctx.history) == 3
    assert ctx.round_number == 3
    assert [r.round_number for r in ctx.history] == [0, 1, 2]
    assert [p.agent for p in ctx.current] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# The strategy boundary
# ---------------------------------------------------------------------------


def test_context_holds_the_managed_group_only():
    seen = []

    class Recorder(DefaultPromptStrategy):
        version = "rec"

        def build(self, ctx):
            seen.append(ctx)
            return super().build(ctx)

    with _StubServer(["1, 1, 1, 1"]) as stub:
        manager = make_manager(
            stub.api_base, strategy=Recorder(), objective="MAXIMISE THE POOL"
        )
        manager.predict(make_state(3, 0))

    assert len(seen) == 3
    for index, ctx in enumerate(seen):
        assert isinstance(ctx, RoundContext)
        assert ctx.episode == index
        assert ctx.round_number == 0
        assert ctx.n_players == 4
        assert [p.agent for p in ctx.current] == [0, 1, 2, 3]
        assert all(isinstance(p, PlayerRound) for p in ctx.current)
        assert ctx.objective == "MAXIMISE THE POOL"
        assert ctx.n_punishments == 31
        assert ctx.max_punishment == 30


def test_a_wrong_length_reply_is_a_failure_not_a_partial_action():
    """Three numbers for four players must not be spread over four seats."""

    class ThreeOnly:
        version = "three"

        def build(self, ctx):
            return [{"role": "user", "content": "x"}]

        def parse(self, text, ctx):
            return [1, 2, 3]

    with _StubServer(["irrelevant"]) as stub:
        manager = make_manager(stub.api_base, strategy=ThreeOnly())
        punishment, _ = manager.predict(make_state(1, 0))
    assert punishment.sum().item() == 0
    assert manager.parse_failures == 1
    assert manager.parse_failure_rate == 1.0


def test_out_of_range_values_are_clamped_not_rejected():
    class TooBig:
        version = "big"

        def build(self, ctx):
            return [{"role": "user", "content": "x"}]

        def parse(self, text, ctx):
            return [99, -5, 30, 0]

    with _StubServer(["irrelevant"]) as stub:
        manager = make_manager(stub.api_base, strategy=TooBig())
        punishment, _ = manager.predict(make_state(1, 0))
    assert punishment[0, :4, 0].tolist() == [30, 0, 30, 0]
    assert manager.parse_failures == 0


def test_an_unparsable_reply_falls_back_to_zero_and_is_counted():
    with _StubServer(["I would rather not say."]) as stub:
        manager = make_manager(stub.api_base)
        punishment, _ = manager.predict(make_state(2, 0))
    assert punishment.sum().item() == 0
    assert manager.parse_failures == 2
    assert manager.n_decisions == 2


def test_a_failing_endpoint_falls_back_to_zero_rather_than_aborting():
    with _StubServer(["never reached"], status=500) as stub:
        manager = make_manager(stub.api_base, max_retries=1, retry_base_delay=0.0)
        punishment, _ = manager.predict(make_state(2, 0))
    assert punishment.sum().item() == 0
    assert manager.call_errors == 2
    assert manager.parse_failures == 2


def test_strategy_registry_round_trips():
    class Mine:
        version = "test-v9"

        def build(self, ctx):
            return [{"role": "user", "content": "x"}]

        def parse(self, text, ctx):
            return [0] * len(ctx.current)

    register_prompt_strategy("test-v9", Mine())
    assert get_prompt_strategy("test-v9").version == "test-v9"
    with pytest.raises(ValueError, match="Unknown prompt_version"):
        get_prompt_strategy("no-such-version")
    with pytest.raises(TypeError):
        register_prompt_strategy("bad", object())


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


def test_qwen3_gets_thinking_disabled_without_being_asked():
    """The documented trap: Qwen3 thinks by default and truncates."""
    assert wants_thinking_flag("Qwen/Qwen3-8B")
    assert wants_thinking_flag("hosted_vllm/Qwen/Qwen3-32B")
    assert not wants_thinking_flag("meta-llama/Llama-3.1-8B-Instruct")

    with _StubServer(["1, 1, 1, 1"]) as stub:
        client = ChatClient(model="Qwen/Qwen3-8B", api_base=stub.api_base)
        assert client.enable_thinking is False
        client.complete([[{"role": "user", "content": "hi"}]])
        body = stub.bodies[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    # The provider prefix never reaches the wire.
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


def test_batch_order_is_preserved_under_concurrency():
    replies = [f"{i}, {i}, {i}, {i}" for i in range(12)]
    with _StubServer(replies) as stub:
        client = ChatClient(
            model="Qwen/Qwen3-8B", api_base=stub.api_base, max_concurrent=6
        )
        conversations = [
            [{"role": "user", "content": f"episode {i}"}] for i in range(12)
        ]
        results = client.complete(conversations)
    # The stub answers in arrival order, which concurrency scrambles; what
    # must hold is that result i is the answer to conversation i.
    by_prompt = {
        body["messages"][0]["content"]: index for index, body in enumerate(stub.bodies)
    }
    for i, result in enumerate(results):
        assert result.text == replies[by_prompt[f"episode {i}"]]


def test_every_prompt_and_completion_is_logged_with_episode_and_round(tmp_path):
    log_path = tmp_path / "calls.jsonl"
    with _StubServer(["4, 4, 4, 4"]) as stub:
        manager = make_manager(stub.api_base, log_path=str(log_path))
        manager.predict(make_state(2, 0))
        manager.predict(make_state(2, 1))

    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    header = lines[0]
    assert header["record"] == "header"
    assert header["temperature"] == 0.0
    assert header["enable_thinking"] is False
    assert header["model"] == "Qwen/Qwen3-8B"

    calls = [line for line in lines if line["record"] == "call"]
    assert len(calls) == 4
    assert sorted((c["episode"], c["round"]) for c in calls) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    for call in calls:
        assert call["messages"][0]["role"] == "user"
        assert call["completion"] == "4, 4, 4, 4"
        assert call["prompt_tokens"] > 0
        assert call["error"] is None


def test_report_carries_what_a_result_must_be_quoted_with():
    with _StubServer(["1, 1, 1, 1"]) as stub:
        manager = make_manager(stub.api_base, temperature=0.0)
        manager.predict(make_state(2, 0))
        report = manager.report()
    for key in (
        "model",
        "temperature",
        "enable_thinking",
        "prompt_version",
        "objective",
        "n_decisions",
        "parse_failure_rate",
        "completions_per_s",
        "prompt_tokens",
        "completion_tokens",
    ):
        assert key in report, key
    assert report["n_decisions"] == 2
    assert report["parse_failure_rate"] == 0.0


def test_client_requires_an_endpoint(monkeypatch):
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    with pytest.raises(ValueError, match="api_base"):
        ChatClient(model="Qwen/Qwen3-8B")


# ---------------------------------------------------------------------------
# Constrained decode: the mechanism, with the parser kept as a guard
# ---------------------------------------------------------------------------


def test_integers_regex_admits_only_well_formed_actions():
    pattern = re.compile("^" + integers_regex(4, 30) + "$")
    assert pattern.match("0, 0, 0, 0")
    assert pattern.match("30, 7, 0, 12")
    assert not pattern.match("31, 0, 0, 0")  # out of range
    assert not pattern.match("1, 2, 3")  # too few
    assert not pattern.match("1, 2, 3, 4, 5")  # too many
    assert not pattern.match("00, 1, 2, 3")  # leading zero
    assert not pattern.match("I would punish 1, 2, 3, 4")  # prose


def test_the_constraint_reaches_the_wire_in_the_vllm_field():
    """vLLM 0.19+ reads `structured_outputs`; older builds read `guided_*`."""
    with _StubServer(["1, 2, 3, 4"]) as stub:
        manager = make_manager(stub.api_base)
        manager.predict(make_state(1, 0))
        body = stub.bodies[0]
    assert "structured_outputs" in body
    regex = body["structured_outputs"]["regex"]
    assert re.compile("^" + regex + "$").match("1, 2, 3, 4")
    assert not re.compile("^" + regex + "$").match("1, 2, 3")


def test_legacy_guided_style_uses_the_old_field_names():
    constraint = DecodeConstraint(regex="abc")
    assert constraint.as_body_fields("structured_outputs") == {
        "structured_outputs": {"regex": "abc"}
    }
    assert constraint.as_body_fields("guided") == {"guided_regex": "abc"}
    with pytest.raises(ValueError, match="constraint_style"):
        constraint.as_body_fields("nonsense")


def test_the_constraint_narrows_with_the_group_size():
    """Members switch groups, so the action width changes round to round."""
    state = make_state(1, 0)
    groups = state["agent_group"].clone()
    groups[0, 0, 0] = 1  # agent 0 leaves group 0, leaving three players
    state["agent_group"] = groups
    with _StubServer(["1, 2, 3"]) as stub:
        manager = make_manager(stub.api_base, group_id=0)
        punishment, _ = manager.predict(state)
        regex = stub.bodies[0]["structured_outputs"]["regex"]
    assert re.compile("^" + regex + "$").match("1, 2, 3")
    assert not re.compile("^" + regex + "$").match("1, 2, 3, 4")
    assert punishment[0, :, 0].tolist() == [0, 1, 2, 3, 0, 0, 0, 0]
    assert manager.parse_failures == 0


def test_constrained_decode_can_be_turned_off_to_measure_the_free_rate():
    with _StubServer(["1, 2, 3, 4"]) as stub:
        manager = make_manager(stub.api_base, constrained_decode=False)
        manager.predict(make_state(1, 0))
    assert "structured_outputs" not in stub.bodies[0]
    assert manager.report()["constrained_decode"] is False


def test_a_strategy_without_a_constraint_still_works():
    class NoConstraint:
        version = "free"

        def build(self, ctx):
            return [{"role": "user", "content": "x"}]

        def parse(self, text, ctx):
            return [1] * len(ctx.current)

    with _StubServer(["whatever"]) as stub:
        manager = make_manager(stub.api_base, strategy=NoConstraint())
        punishment, _ = manager.predict(make_state(1, 0))
    assert "structured_outputs" not in stub.bodies[0]
    assert punishment[0, :4, 0].tolist() == [1, 1, 1, 1]


# ---------------------------------------------------------------------------
# Data-parallel sharding
# ---------------------------------------------------------------------------


def test_endpoints_parse_from_a_list_or_a_comma_separated_string():
    assert parse_endpoints("http://a/v1,http://b/v1") == [
        "http://a/v1",
        "http://b/v1",
    ]
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
    # Stable across rounds ...
    for _round in range(3):
        assert [client.endpoint_for({"episode": e}) for e in range(9)] == first
    # ... and an even split.
    assert (
        first
        == [
            "http://a/v1",
            "http://b/v1",
            "http://c/v1",
        ]
        * 3
    )


def test_sharding_routes_a_batch_across_every_server():
    with _StubServer(["1, 1, 1, 1"]) as one, _StubServer(["2, 2, 2, 2"]) as two:
        manager = make_manager([one.api_base, two.api_base])
        punishment, _ = manager.predict(make_state(6, 0))
    assert len(one.bodies) == 3
    assert len(two.bodies) == 3
    # Even episodes hit the first server, odd ones the second.
    assert punishment[0, :4, 0].tolist() == [1, 1, 1, 1]
    assert punishment[1, :4, 0].tolist() == [2, 2, 2, 2]


def test_the_log_records_which_server_answered(tmp_path):
    log_path = tmp_path / "calls.jsonl"
    with _StubServer(["1, 1, 1, 1"]) as one, _StubServer(["2, 2, 2, 2"]) as two:
        manager = make_manager([one.api_base, two.api_base], log_path=str(log_path))
        manager.predict(make_state(4, 0))
    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    header = [line for line in lines if line["record"] == "header"][0]
    assert header["n_endpoints"] == 2
    calls = {line["episode"]: line["endpoint"] for line in lines if "episode" in line}
    assert calls[0] == calls[2] == one.api_base
    assert calls[1] == calls[3] == two.api_base


def test_the_log_carries_the_rendered_prompt_not_only_its_pieces(tmp_path):
    """A trace-format bug is invisible in the pieces it was assembled from."""
    log_path = tmp_path / "calls.jsonl"
    with _StubServer(["1, 1, 1, 1"]) as stub:
        manager = make_manager(stub.api_base, log_path=str(log_path))
        manager.predict(make_state(1, 0))
    call = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if json.loads(line)["record"] == "call"
    ][0]
    assert "prompt" in call
    assert call["prompt"].startswith("<<user>>")
    # The whole rendered text is there, including this round's contributions.
    assert "this round's contributions" in call["prompt"]
    assert call["constraint"] is not None


# ---------------------------------------------------------------------------
# The default strategy, which exists so this layer is testable alone
# ---------------------------------------------------------------------------


def test_default_strategy_round_trips_its_own_format():
    strategy = DefaultPromptStrategy()
    ctx = RoundContext(
        episode=0,
        round_number=3,
        n_players=4,
        n_punishments=31,
        objective="maximise the pool",
        prompt_version="v0",
        history=(),
        current=tuple(
            PlayerRound(
                agent=a, contribution=5 * a, punishment=0, contribution_valid=True
            )
            for a in range(4)
        ),
    )
    messages = strategy.build(ctx)
    assert messages[0]["role"] == "user"
    assert "maximise the pool" in messages[0]["content"]
    assert strategy.parse("The answer is 1, 2, 3, 4", ctx) == [1, 2, 3, 4]
    assert strategy.parse("nothing here", ctx) is None


def test_default_strategy_marks_players_who_gave_no_input():
    strategy = DefaultPromptStrategy()
    ctx = RoundContext(
        episode=0,
        round_number=0,
        n_players=4,
        n_punishments=31,
        objective="o",
        prompt_version="v0",
        history=(),
        current=(
            PlayerRound(agent=0, contribution=5, punishment=0, contribution_valid=True),
            PlayerRound(
                agent=1, contribution=0, punishment=0, contribution_valid=False
            ),
            PlayerRound(agent=2, contribution=7, punishment=0, contribution_valid=True),
            PlayerRound(agent=3, contribution=9, punishment=0, contribution_valid=True),
        ),
    )
    content = strategy.build(ctx)[0]["content"]
    assert re.search(r"player 1: 0 \(no input\)", content)
    assert "cannot be punished" in content
