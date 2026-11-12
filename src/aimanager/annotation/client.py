"""One LLM client for every step.

OpenAI-compatible chat completions over HTTP (or a deterministic fake), with a
top-level json_schema response_format, an exact-result disk cache of validated
replies, a call log and a bounded window of outstanding requests.

Failure handling:
- 5xx, 429, connection errors and malformed 200 replies are retried with
  backoff; a read timeout is retried at most `timeout_retries` times. A call
  that exhausts its attempts becomes a failed Result (the step's failure
  budget decides what that means).
- `max_consecutive_server_errors` failed attempts in a row (across all calls)
  abort the whole map: a dead or hung server ends the step at once.
- `max_client_errors` HTTP 4xx replies abort the map.
- An empty reply with reasoning text means thinking is on: a hard error.
- Any exception in a worker cancels the queued calls and ends the step.
"""

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import ValidationError

from .common import atomic_write_text, log
from .schemas import json_schema


@dataclass
class Call:
    step: str
    item_id: str
    messages: list
    model_cls: Any
    kind: str  # max_tokens key: open_rules, open_notes, propose, assign, judge
    meta: dict = field(default_factory=dict)  # only read by the fake backend


@dataclass
class Result:
    call: Call
    value: Any = None
    error: Optional[str] = None
    info: dict = field(default_factory=dict)  # latency, think_stripped, reasoning

    @property
    def ok(self):
        return self.error is None


class ClientError(Exception):
    """HTTP 4xx: the request itself is wrong, retrying will not help."""


class ServerError(Exception):
    """5xx, connection failure, timeout or malformed reply after all attempts."""


class Abort(Exception):
    """The step must stop: dead server, thinking switched on, too many 4xx."""


THINKING_HELP = (
    "the server returned an empty reply with reasoning text, so the model is "
    "thinking: set the thinking switch for this model under llm.extra_body in "
    "the config (a `models` entry whose `match` occurs in the model id)"
)


class HttpBackend:
    def __init__(self, base_url, cfg, transport=None):
        import httpx

        self.httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.backoff = cfg.get("backoff", [5, 10, 20, 40])
        self.timeout_retries = cfg.get("timeout_retries", 1)
        self.max_consecutive = cfg.get("max_consecutive_server_errors", 6)
        self.lock = threading.Lock()
        self.consecutive = 0
        self.abort = threading.Event()
        n = cfg.get("concurrency", 16)
        self.client = httpx.Client(
            timeout=httpx.Timeout(
                cfg.get("read_timeout", 900), connect=cfg.get("connect_timeout", 10)
            ),
            limits=httpx.Limits(max_connections=n, max_keepalive_connections=n),
            transport=transport,
        )

    def default_model(self):
        try:
            r = self.client.get(self.base_url + "/models")
            r.raise_for_status()
            return r.json()["data"][0]["id"]
        except (self.httpx.HTTPError, ValueError, KeyError, IndexError) as e:
            raise SystemExit(f"cannot list models at {self.base_url}: {e}")

    def _failed_attempt(self, msg):
        with self.lock:
            self.consecutive += 1
            if self.consecutive >= self.max_consecutive:
                self.abort.set()
        if self.abort.is_set():
            raise Abort(
                f"{self.max_consecutive} failed attempts in a row "
                f"(llm.max_consecutive_server_errors); server down? last: {msg}"
            )

    def _ok_attempt(self):
        with self.lock:
            self.consecutive = 0

    @staticmethod
    def _parse(r):
        """The reply fields of a 200, or None if the body is malformed."""
        try:
            j = r.json()
            msg = j["choices"][0]["message"]
            return {
                "content": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                "finish_reason": j["choices"][0].get("finish_reason"),
                "usage": j.get("usage") or {},
            }
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            return None

    def complete(self, body, call):
        last, n_timeouts = None, 0
        for delay in [0] + list(self.backoff):
            # an abort elsewhere ends the backoff wait at once
            if self.abort.is_set() or (delay and self.abort.wait(delay)):
                raise Abort("aborted after repeated server errors")
            try:
                r = self.client.post(self.base_url + "/chat/completions", json=body)
            except self.httpx.ReadTimeout as e:
                n_timeouts += 1
                last = f"ReadTimeout: {e}"
                self._failed_attempt(last)
                if n_timeouts > self.timeout_retries:
                    break
                continue
            except self.httpx.TransportError as e:
                last = f"{type(e).__name__}: {e}"
                self._failed_attempt(last)
                continue
            if r.status_code >= 500 or r.status_code == 429:
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                self._failed_attempt(last)
                continue
            if r.status_code >= 400:
                self._ok_attempt()  # the server is alive
                raise ClientError(f"HTTP {r.status_code}: {r.text[:500]}")
            resp = self._parse(r)
            if resp is None:
                last = f"malformed 200: {r.text[:300]}"
                self._failed_attempt(last)
                continue
            self._ok_attempt()
            return resp
        raise ServerError(last)


class FakeBackend:
    def __init__(self):
        self.abort = threading.Event()

    def default_model(self):
        return "fake"

    def complete(self, body, call):
        from . import fake

        content = fake.respond(call)
        prompt_chars = sum(len(m["content"]) for m in body["messages"])
        return {
            "content": content,
            "reasoning": "",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": prompt_chars // 4,
                "completion_tokens": len(content) // 4,
            },
        }


_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


def strip_think(content):
    """(content without a leading <think>...</think> block, whether one was cut)."""
    out = _THINK.sub("", content, count=1)
    return out, out != content


class LLMClient:
    def __init__(
        self,
        run,
        cfg,
        backend="http",
        base_url=None,
        model=None,
        transport=None,
        use_cache=True,
    ):
        self.run = run
        self.cfg = cfg["llm"]
        if backend == "fake":
            self.backend = FakeBackend()
        else:
            if not base_url:
                raise SystemExit("no endpoint: pass --base-url or set ANNOT_BASE_URL")
            self.backend = HttpBackend(base_url, self.cfg, transport)
        self.model = model or self.backend.default_model()
        # the server's context length (the job passes --max-model-len)
        self.max_model_len = int(
            os.environ.get("ANNOT_MAX_MODEL_LEN")
            or self.cfg.get("max_model_len", 32768)
        )
        self.extra = self._extra_body(self.model)
        self.use_cache = use_cache
        self.log_lock = threading.Lock()
        self.key_locks = {}
        self.n_client_errors = 0
        self.n_think_stripped = 0
        self.logged_steps = set()
        log(f"llm: backend={backend} model={self.model}")

    def _extra_body(self, model):
        eb = self.cfg.get("extra_body", {})
        for entry in eb.get("models", []):
            if entry["match"].lower() in model.lower():
                return entry["body"]
        return eb.get("default", {})

    def _body(self, call, schema, max_tokens):
        body = {
            "model": self.model,
            "messages": call.messages,
            "temperature": self.cfg.get("temperature", 0.0),
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": call.kind, "schema": schema, "strict": True},
            },
        }
        body.update(self.extra)
        return body

    def _cache_path(self, body):
        key = hashlib.sha256(
            json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        return self.run.dir / "cache" / key[:2] / f"{key}.json"

    def _log(self, call, resp, latency, hit, max_tokens, status, stripped=False):
        usage = resp.get("usage", {}) if resp else {}
        row = {
            "ts": round(time.time(), 3),
            "step": call.step,
            "item": call.item_id,
            "kind": call.kind,
            "model": self.model,
            "max_tokens": max_tokens,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "latency_s": round(latency, 3),
            "finish_reason": resp.get("finish_reason") if resp else None,
            "cache_hit": hit,
            "status": status,
            "think_stripped": stripped,
        }
        path = self.run.path("logs", "calls.jsonl")
        with self.log_lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _attempt(self, call, body, max_tokens):
        """One request (or cache hit) -> (Result | None for 'length', resp)."""
        path = self._cache_path(body)
        with self.log_lock:
            lock = self.key_locks.setdefault(path.name, threading.Lock())
        with lock:
            t0 = time.time()
            hit = self.use_cache and path.exists()
            try:
                if hit:
                    resp = json.loads(path.read_text(encoding="utf-8"))
                else:
                    resp = self.backend.complete(body, call)
            except ClientError as e:
                self._log(call, None, time.time() - t0, False, max_tokens, "4xx")
                with self.log_lock:
                    self.n_client_errors += 1
                return Result(call, error=f"client_error: {e}")
            except ServerError as e:
                self._log(call, None, time.time() - t0, False, max_tokens, "5xx")
                return Result(call, error=f"server_error: {e}")
            latency = time.time() - t0
            info = {
                "latency_s": latency,
                "reasoning_chars": len(resp.get("reasoning") or ""),
            }
            content, stripped = strip_think(resp.get("content") or "")
            info["think_stripped"] = stripped
            # checked before `length`: a thinking model usually runs out of
            # tokens while still reasoning
            if not content.strip() and resp.get("reasoning"):
                self._log(call, resp, latency, hit, max_tokens, "thinking")
                raise Abort(f"{call.step} {call.item_id}: {THINKING_HELP}")
            if resp.get("finish_reason") == "length":
                self._log(call, resp, latency, hit, max_tokens, "length")
                if self.use_cache and not hit:
                    atomic_write_text(path, json.dumps(resp, ensure_ascii=False))
                return None
            if stripped:
                with self.log_lock:
                    self.n_think_stripped += 1
            try:
                value = call.model_cls.model_validate_json(content)
            except ValidationError as e:
                self._log(call, resp, latency, hit, max_tokens, "parse_error", stripped)
                return Result(call, error=f"parse_error: {str(e)[:300]}", info=info)
            self._log(call, resp, latency, hit, max_tokens, "ok", stripped)
            if self.use_cache and not hit:
                atomic_write_text(path, json.dumps(resp, ensure_ascii=False))
            return Result(call, value=value, info=info)

    def _one(self, call):
        schema = json_schema(call.model_cls)
        budget = self.cfg["max_tokens"][call.kind]
        for max_tokens in (budget["first"], budget["retry"]):
            res = self._attempt(call, self._body(call, schema, max_tokens), max_tokens)
            if res is not None:
                return res
        return Result(call, error="length")

    def map(self, calls, label=None):
        """Run calls with a continuously refilled window; results in input order."""
        calls = list(calls)
        if not calls:
            return []
        step = calls[0].step
        if step not in self.logged_steps:
            self.logged_steps.add(step)
            log(f"llm: {step}: extra request body {json.dumps(self.extra)}")
        limit = self.cfg.get("max_client_errors", 3)
        results = [None] * len(calls)
        n_fail, stripped0 = 0, self.n_think_stripped
        label = label or step
        every = max(len(calls) // 4, 50)
        ex = ThreadPoolExecutor(max_workers=self.cfg.get("concurrency", 16))
        try:
            futs = {ex.submit(self._one, c): i for i, c in enumerate(calls)}
            for done, fut in enumerate(as_completed(futs), 1):
                res = fut.result()  # a worker exception ends the step here
                results[futs[fut]] = res
                if not res.ok:
                    n_fail += 1
                    log(f"  failed {res.call.item_id}: {res.error[:200]}")
                if self.n_client_errors >= limit:
                    raise Abort(
                        f"{self.n_client_errors} HTTP 4xx errors "
                        f"(llm.max_client_errors); last: {res.error}"
                    )
                if done % every == 0 or done == len(calls):
                    log(f"  {label}: {done}/{len(calls)}, {n_fail} failed")
        except BaseException as e:
            self.backend.abort.set()
            ex.shutdown(wait=True, cancel_futures=True)
            if isinstance(e, Abort):
                raise SystemExit(f"{label}: aborted: {e}")
            raise
        ex.shutdown(wait=True)
        if self.n_think_stripped > stripped0:
            log(
                f"  {label}: stripped a leading <think> block from "
                f"{self.n_think_stripped - stripped0} replies"
            )
        return results
