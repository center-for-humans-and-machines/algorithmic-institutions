"""One LLM client for every step.

OpenAI-compatible chat completions over HTTP (or a deterministic fake), with a
top-level json_schema response_format, an exact-result disk cache, a call log
and a bounded window of outstanding requests.
"""

import hashlib
import json
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

    @property
    def ok(self):
        return self.error is None


class ClientError(Exception):
    """HTTP 4xx: the request itself is wrong, retrying will not help."""


class ServerError(Exception):
    """5xx or connection failure after all backoff attempts."""


class HttpBackend:
    def __init__(self, base_url, cfg, transport=None):
        import httpx

        self.httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.backoff = cfg.get("backoff", [5, 10, 20, 40])
        n = cfg.get("concurrency", 16)
        self.client = httpx.Client(
            timeout=httpx.Timeout(
                cfg.get("read_timeout", 600), connect=cfg.get("connect_timeout", 10)
            ),
            limits=httpx.Limits(max_connections=n, max_keepalive_connections=n),
            transport=transport,
        )

    def default_model(self):
        r = self.client.get(self.base_url + "/models")
        r.raise_for_status()
        return r.json()["data"][0]["id"]

    def complete(self, body, call):
        last = None
        for delay in [0] + list(self.backoff):
            if delay:
                time.sleep(delay)
            try:
                r = self.client.post(self.base_url + "/chat/completions", json=body)
            except self.httpx.TransportError as e:
                last = f"{type(e).__name__}: {e}"
                continue
            if r.status_code >= 500 or r.status_code == 429:
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                continue
            if r.status_code >= 400:
                raise ClientError(f"HTTP {r.status_code}: {r.text[:500]}")
            j = r.json()
            choice = j["choices"][0]
            return {
                "content": choice["message"].get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "usage": j.get("usage") or {},
            }
        raise ServerError(last)


class FakeBackend:
    def default_model(self):
        return "fake"

    def complete(self, body, call):
        from . import fake

        content = fake.respond(call)
        prompt_chars = sum(len(m["content"]) for m in body["messages"])
        return {
            "content": content,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": prompt_chars // 4,
                "completion_tokens": len(content) // 4,
            },
        }


class LLMClient:
    def __init__(
        self, run, cfg, backend="http", base_url=None, model=None, transport=None
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
        self.extra = self._extra_body(self.model)
        self.log_lock = threading.Lock()
        self.key_locks = {}
        self.n_client_errors = 0
        log(f"llm: backend={backend} model={self.model}")

    def _extra_body(self, model):
        eb = self.cfg.get("extra_body", {})
        for entry in eb.get("models", []):
            if entry["match"] in model:
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

    def _log(self, call, resp, latency, hit, max_tokens, status):
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
        }
        path = self.run.path("logs", "calls.jsonl")
        with self.log_lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _fetch(self, call, body, path):
        """Cached response, or one backend call; identical requests run once."""
        with self.log_lock:
            lock = self.key_locks.setdefault(path.name, threading.Lock())
        with lock:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8")), True
            resp = self.backend.complete(body, call)
            atomic_write_text(path, json.dumps(resp, ensure_ascii=False))
            return resp, False

    def _one(self, call):
        schema = json_schema(call.model_cls)
        budget = self.cfg["max_tokens"][call.kind]
        for max_tokens in (budget["first"], budget["retry"]):
            body = self._body(call, schema, max_tokens)
            path = self._cache_path(body)
            t0 = time.time()
            try:
                resp, hit = self._fetch(call, body, path)
            except ClientError as e:
                self._log(call, None, time.time() - t0, False, max_tokens, "4xx")
                with self.log_lock:
                    self.n_client_errors += 1
                return Result(call, error=f"client_error: {e}")
            except ServerError as e:
                self._log(call, None, time.time() - t0, False, max_tokens, "5xx")
                return Result(call, error=f"server_error: {e}")
            latency = time.time() - t0
            if resp.get("finish_reason") == "length":
                self._log(call, resp, latency, hit, max_tokens, "length")
                continue
            try:
                value = call.model_cls.model_validate_json(resp["content"])
            except ValidationError as e:
                self._log(call, resp, latency, hit, max_tokens, "parse_error")
                return Result(call, error=f"parse_error: {str(e)[:300]}")
            self._log(call, resp, latency, hit, max_tokens, "ok")
            return Result(call, value=value)
        return Result(call, error="length")

    def map(self, calls, label=None):
        """Run calls with a continuously refilled window; results in input order."""
        calls = list(calls)
        if not calls:
            return []
        limit = self.cfg.get("max_client_errors", 3)
        results = [None] * len(calls)
        n_fail = 0
        label = label or calls[0].step
        every = max(len(calls) // 4, 50)
        with ThreadPoolExecutor(max_workers=self.cfg.get("concurrency", 16)) as ex:
            futs = {ex.submit(self._one, c): i for i, c in enumerate(calls)}
            for done, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                results[futs[fut]] = res
                if not res.ok:
                    n_fail += 1
                    log(f"  failed {res.call.item_id}: {res.error[:200]}")
                if self.n_client_errors >= limit:
                    for f in futs:
                        f.cancel()
                    raise SystemExit(
                        f"{self.n_client_errors} HTTP 4xx errors; aborting "
                        f"(last: {res.error})"
                    )
                if done % every == 0 or done == len(calls):
                    log(f"  {label}: {done}/{len(calls)}, {n_fail} failed")
        return results
