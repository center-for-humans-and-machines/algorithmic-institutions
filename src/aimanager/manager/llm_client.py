"""Batched chat client for the language-model manager.

The manager talks to vLLM over its OpenAI-compatible HTTP API. One round of one
episode is one chat completion, so a round of a B-episode rollout is B of them
issued together and a 24-round rollout is 24 batched calls rather than 24*B
serial ones. vLLM's continuous batching is what makes the batch cheap;
`max_concurrent` caps how many requests are in flight so a large B cannot bury
the server's queue.

Data parallelism
----------------
`api_base` may be a list (or a comma-separated string) of endpoints. An 8B
model fits on one A100 with room for a large KV cache, so N independent
servers with the episode batch sharded across them beat one tensor-parallel
instance: this workload is throughput-bound, not memory-bound, and tensor
parallelism across small models mostly buys communication overhead.

**The shard assignment is fixed and recorded.** An episode is routed by
`episode % n_endpoints`, so a given episode always meets the same server for
its whole rollout. Two instances of the same weights still differ in sampling
detail, and routing an episode to a different server between rounds would turn
that into episode-level noise indistinguishable from variance in the manager.
The chosen endpoint is written into every log line.

Constrained decode
------------------
The action is a few small integers, which is exactly the case structured
output suits. Passing a `DecodeConstraint` makes vLLM mask the logits so the
model *cannot* emit anything but a well-formed action -- the parse failure
rate is then zero by construction, and a non-zero value is a bug rather than a
property of the model.

vLLM 0.19+ takes this as `structured_outputs: {"regex": ...}` in the request
body; older builds used a top-level `guided_regex`. `constraint_style` picks
which name goes on the wire. Confirmed present in vLLM 0.19.1 via
`vllm.sampling_params.StructuredOutputsParams`, whose fields are `regex`,
`choice`, `json`, `grammar` and `structural_tag`.

Transport backends
------------------
`backend="http"` (the default) speaks the OpenAI chat-completions protocol
directly with `urllib` from the standard library. `backend="litellm"` routes
the same request through LiteLLM's `hosted_vllm/` provider, the convention the
sibling project documents in `doc/vllm-backend.md`. The two are wire
equivalent; `http` is the default only because LiteLLM is not a dependency of
this project and installing it into the shared cluster venv would disturb work
running there. `model` may be written bare (`Qwen/Qwen3-8B`) or with the
provider prefix (`hosted_vllm/Qwen/Qwen3-8B`).

Qwen3 thinking
--------------
Qwen3 turns on a `<think>` reasoning mode by default. Left on it spends the
completion budget on hidden reasoning and truncates the answer. Any model whose
id contains `qwen3` therefore gets
`chat_template_kwargs={"enable_thinking": False}`, which vLLM forwards into the
chat template. Measured on Qwen3-8B, 2026-09-22: with the flag absent the reply
hit `finish_reason: length` at 64 tokens having emitted only the opening of a
`<think>` block; with it false the same prompt answered in 10 tokens and
stopped cleanly. Note that the `<think>` text arrives in `content`, not in
`reasoning_content`, unless the server is started with a reasoning parser -- a
caller watching `reasoning_content` would not see it.

Logging
-------
Every call appends one JSONL object carrying the **fully rendered** prompt as
sent, not the pieces it was assembled from, because a trace-format bug is
invisible in the pieces. Each line also carries the episode, the round, the
endpoint it was routed to, token counts, latency and any error.
"""

import json
import logging
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

PROVIDER_PREFIX = "hosted_vllm/"

DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_BASE_DELAY = 2.0

#: Request-body field carrying the decode constraint, by vLLM generation.
CONSTRAINT_STYLES = ("structured_outputs", "guided")


@dataclass(frozen=True)
class DecodeConstraint:
    """A restriction on what the model is allowed to emit.

    Exactly one of the fields is set. `regex` is the one this project uses:
    the action is N integers in a known range, which a regex states exactly.
    """

    regex: Optional[str] = None
    choice: Optional[Sequence[str]] = None
    json_schema: Optional[Dict[str, Any]] = None
    grammar: Optional[str] = None

    def as_body_fields(self, style: str = "structured_outputs") -> Dict[str, Any]:
        """Render into the request-body fields the server expects."""
        if style not in CONSTRAINT_STYLES:
            raise ValueError(
                f"Unknown constraint_style {style!r}; use one of {CONSTRAINT_STYLES}"
            )
        payload: Dict[str, Any] = {}
        if self.regex is not None:
            payload["regex"] = self.regex
        if self.choice is not None:
            payload["choice"] = list(self.choice)
        if self.json_schema is not None:
            payload["json"] = self.json_schema
        if self.grammar is not None:
            payload["grammar"] = self.grammar
        if not payload:
            return {}
        if style == "structured_outputs":
            return {"structured_outputs": payload}
        # Legacy vLLM: the same options as top-level `guided_*` fields.
        return {f"guided_{key}": value for key, value in payload.items()}


@dataclass
class Completion:
    """One model reply, plus everything needed to account for it."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = None
    latency_s: float = 0.0
    error: Optional[str] = None
    reasoning_text: Optional[str] = None
    endpoint: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class CallStats:
    """Running totals over a client's lifetime, for the throughput report."""

    n_calls: int = 0
    n_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_s: float = 0.0
    truncated: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, completion: "Completion") -> None:
        with self._lock:
            self.n_calls += 1
            self.prompt_tokens += completion.prompt_tokens
            self.completion_tokens += completion.completion_tokens
            if completion.error is not None:
                self.n_errors += 1
            if completion.finish_reason == "length":
                self.truncated += 1

    def as_dict(self) -> Dict[str, Any]:
        rate = self.n_calls / self.wall_s if self.wall_s > 0 else 0.0
        return {
            "n_calls": self.n_calls,
            "n_errors": self.n_errors,
            "truncated": self.truncated,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_s": round(self.wall_s, 3),
            "completions_per_s": round(rate, 3),
        }


def _strip_prefix(model: str) -> str:
    if model.startswith(PROVIDER_PREFIX):
        return model[len(PROVIDER_PREFIX) :]  # noqa: E203
    return model


def _with_prefix(model: str) -> str:
    return model if model.startswith(PROVIDER_PREFIX) else PROVIDER_PREFIX + model


def wants_thinking_flag(model: str) -> bool:
    """Whether this model family reads `chat_template_kwargs.enable_thinking`."""
    return "qwen3" in model.lower()


def parse_endpoints(api_base) -> List[str]:
    """Normalise one endpoint, a list, or a comma-separated string."""
    if api_base is None:
        api_base = os.environ.get("HOSTED_VLLM_API_BASE") or ""
    if isinstance(api_base, str):
        parts = [p.strip() for p in api_base.split(",")]
    else:
        parts = [str(p).strip() for p in api_base]
    return [p.rstrip("/") for p in parts if p]


def integers_regex(n_values: int, max_value: int, separator: str = ", ") -> str:
    """A regex matching exactly `n_values` integers in `[0, max_value]`.

    Each alternative is a whole number, so the constraint cannot be satisfied
    by leading zeros or by a longer number sharing a prefix.
    """
    if n_values < 1:
        raise ValueError("n_values must be at least 1")
    alternatives = "|".join(str(v) for v in range(max_value, -1, -1))
    one = f"(?:{alternatives})"
    sep = separator.replace(" ", r"\s*")
    return one + f"(?:{sep}{one})" * (n_values - 1)


class ChatClient:
    """Issue chat completions in batches against OpenAI-compatible servers.

    Parameters
    ----------
    model:
        Model id, bare or `hosted_vllm/`-prefixed. Must match the model the
        servers were launched with.
    api_base:
        One endpoint, or several for data-parallel sharding. Each must carry
        the `/v1` suffix. Falls back to `$HOSTED_VLLM_API_BASE`, which may
        itself be a comma-separated list.
    api_key:
        Bearer token. A node-local vLLM needs none; `EMPTY` is conventional.
    temperature:
        Fixed for a run and recorded. 0.0 makes the manager reproducible,
        which none of the sampling baselines are.
    max_tokens:
        Completion budget. Small, because a constrained action is a handful
        of tokens -- but only safe to keep small with thinking disabled.
    max_concurrent:
        Requests in flight at once, **per endpoint**.
    enable_thinking:
        `None` means "off for Qwen3, untouched otherwise".
    constraint_style:
        `structured_outputs` for vLLM 0.19+, `guided` for older builds.
    log_path:
        JSONL destination for every rendered prompt and completion.
    """

    def __init__(
        self,
        *,
        model: str,
        api_base=None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_concurrent: int = 64,
        enable_thinking: Optional[bool] = None,
        backend: str = "http",
        constraint_style: str = "structured_outputs",
        timeout: float = 600.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        log_path: Optional[str] = None,
        seed: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        if backend not in ("http", "litellm"):
            raise ValueError(f"Unknown backend {backend!r}; use 'http' or 'litellm'")
        if constraint_style not in CONSTRAINT_STYLES:
            raise ValueError(
                f"Unknown constraint_style {constraint_style!r}; "
                f"use one of {CONSTRAINT_STYLES}"
            )
        self.backend = backend
        self.constraint_style = constraint_style
        self.model = model
        self.endpoints = parse_endpoints(api_base)
        if not self.endpoints:
            raise ValueError(
                "No api_base: pass one (or several) or set HOSTED_VLLM_API_BASE "
                "(each must include the /v1 suffix)"
            )
        self.api_key = api_key or os.environ.get("HOSTED_VLLM_API_KEY") or "EMPTY"
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.max_concurrent = max(1, int(max_concurrent))
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.retry_base_delay = float(retry_base_delay)
        self.seed = seed
        self.extra_body = dict(extra_body or {})
        self.stats = CallStats()
        self._fallback_counter = 0

        if enable_thinking is None:
            self.enable_thinking = False if wants_thinking_flag(model) else None
        else:
            self.enable_thinking = bool(enable_thinking)

        self.log_path = log_path
        self._log_lock = threading.Lock()
        if self.log_path:
            parent = os.path.dirname(os.path.abspath(self.log_path))
            os.makedirs(parent, exist_ok=True)
            self._log({"record": "header", **self.describe()})

    @property
    def api_base(self) -> str:
        """The first endpoint; kept for callers that expect a single one."""
        return self.endpoints[0]

    @property
    def n_endpoints(self) -> int:
        return len(self.endpoints)

    # -- configuration record ------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """The settings a result has to be reported next to."""
        return {
            "model": _strip_prefix(self.model),
            "backend": self.backend,
            "endpoints": list(self.endpoints),
            "n_endpoints": self.n_endpoints,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_concurrent": self.max_concurrent,
            "enable_thinking": self.enable_thinking,
            "constraint_style": self.constraint_style,
            "seed": self.seed,
        }

    # -- routing -------------------------------------------------------

    def endpoint_for(self, meta: Dict[str, Any]) -> str:
        """Fixed shard assignment: an episode always meets the same server.

        Keyed on `meta["episode"]` when the caller supplies it, which the
        manager always does. Without one the assignment falls back to
        round-robin, which is still deterministic within a process but is
        not stable across runs -- so callers that care pass an episode.
        """
        if self.n_endpoints == 1:
            return self.endpoints[0]
        key = meta.get("episode")
        if key is None:
            key = self._fallback_counter
            self._fallback_counter += 1
        return self.endpoints[int(key) % self.n_endpoints]

    # -- public API ----------------------------------------------------

    def complete(
        self,
        conversations: List[List[Dict[str, str]]],
        meta: Optional[List[Dict[str, Any]]] = None,
        constraints: Optional[Sequence[Optional[DecodeConstraint]]] = None,
    ) -> List[Completion]:
        """Run `conversations` as one batch, preserving order.

        `meta[i]` is written into the log line for conversation `i` and picks
        its endpoint; the manager puts the episode and round there.
        `constraints[i]` restricts what the model may emit for that item. A
        failing call yields a `Completion` with `error` set rather than
        raising, so one bad episode cannot abort a rollout.
        """
        if not conversations:
            return []
        metas = list(meta) if meta is not None else [{} for _ in conversations]
        if len(metas) != len(conversations):
            raise ValueError("meta must be the same length as conversations")
        if constraints is None:
            constraints = [None] * len(conversations)
        elif len(constraints) != len(conversations):
            raise ValueError("constraints must be the same length as conversations")

        start = time.time()
        # Concurrency is per endpoint, so N servers take N times the width.
        n_workers = min(self.max_concurrent * self.n_endpoints, len(conversations))
        work = list(zip(conversations, metas, constraints))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(self._one_item, work))
        self.stats.wall_s += time.time() - start
        for result in results:
            self.stats.add(result)
        return results

    # -- internals -----------------------------------------------------

    def _one_item(self, item):
        messages, meta, constraint = item
        return self._one(messages, meta, constraint)

    def _one(
        self,
        messages: List[Dict[str, str]],
        meta: Dict[str, Any],
        constraint: Optional[DecodeConstraint] = None,
    ) -> Completion:
        endpoint = self.endpoint_for(meta)
        t0 = time.time()
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self._dispatch(messages, endpoint, constraint)
                completion.latency_s = time.time() - t0
                completion.endpoint = endpoint
                self._log_call(messages, completion, meta, constraint)
                return completion
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                last_exc = exc
                logger.warning(
                    "chat completion failed (attempt %d/%d, %s): %s",
                    attempt,
                    self.max_retries,
                    endpoint,
                    exc,
                )
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** (attempt - 1))
                    time.sleep(delay * (0.5 + random.random()))
        completion = Completion(
            text="",
            latency_s=time.time() - t0,
            endpoint=endpoint,
            error=f"{type(last_exc).__name__}: {last_exc}",
        )
        self._log_call(messages, completion, meta, constraint)
        return completion

    def _body(
        self,
        messages: List[Dict[str, str]],
        constraint: Optional[DecodeConstraint],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": _strip_prefix(self.model),
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if self.enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        if constraint is not None:
            body.update(constraint.as_body_fields(self.constraint_style))
        body.update(self.extra_body)
        return body

    def _dispatch(
        self,
        messages: List[Dict[str, str]],
        endpoint: str,
        constraint: Optional[DecodeConstraint],
    ) -> Completion:
        if self.backend == "litellm":
            return self._call_litellm(messages, endpoint, constraint)
        return self._call_http(messages, endpoint, constraint)

    def _call_http(
        self,
        messages: List[Dict[str, str]],
        endpoint: str,
        constraint: Optional[DecodeConstraint],
    ) -> Completion:
        payload = json.dumps(self._body(messages, constraint)).encode("utf-8")
        request = urllib.request.Request(
            f"{endpoint}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        return self._from_openai_dict(data)

    @staticmethod
    def _from_openai_dict(data: Dict[str, Any]) -> Completion:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return Completion(
            text=message.get("content") or "",
            reasoning_text=message.get("reasoning_content"),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=choice.get("finish_reason"),
        )

    def _call_litellm(
        self,
        messages: List[Dict[str, str]],
        endpoint: str,
        constraint: Optional[DecodeConstraint],
    ) -> Completion:
        import litellm  # lazy: not a dependency of this project

        body = self._body(messages, constraint)
        body.pop("model")
        extra_body: Dict[str, Any] = {}
        for key in ("chat_template_kwargs", "structured_outputs"):
            if key in body:
                extra_body[key] = body.pop(key)
        for key in [k for k in body if k.startswith("guided_")]:
            extra_body[key] = body.pop(key)
        response = litellm.completion(
            model=_with_prefix(self.model),
            api_base=endpoint,
            api_key=self.api_key,
            extra_body=extra_body or None,
            **body,
        )
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return Completion(
            text=choice.message.content or "",
            reasoning_text=getattr(choice.message, "reasoning_content", None),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            finish_reason=getattr(choice, "finish_reason", None),
        )

    # -- logging -------------------------------------------------------

    def _log(self, record: Dict[str, Any]) -> None:
        if not self.log_path:
            return
        line = json.dumps(record, default=str)
        with self._log_lock:
            with open(self.log_path, "a") as handle:
                handle.write(line + "\n")

    def _log_call(
        self,
        messages: List[Dict[str, str]],
        completion: Completion,
        meta: Dict[str, Any],
        constraint: Optional[DecodeConstraint] = None,
    ) -> None:
        self._log(
            {
                "record": "call",
                "ts": time.time(),
                **meta,
                "endpoint": completion.endpoint,
                # The rendered prompt exactly as sent: a trace-format bug is
                # invisible in the pieces it was assembled from.
                "prompt": "\n\n".join(
                    f"<<{m.get('role')}>>\n{m.get('content')}" for m in messages
                ),
                "messages": messages,
                "constraint": None if constraint is None else constraint.regex,
                "completion": completion.text,
                "reasoning": completion.reasoning_text,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "finish_reason": completion.finish_reason,
                "latency_s": round(completion.latency_s, 4),
                "error": completion.error,
            }
        )
