"""The manager this harness was built against.

A language model is slow, costs money, and did not exist when the evaluation
had to be finished. So the harness was built and validated against a manager
whose battery is known in advance: it returns fixed punishments, and it
carries the same surface the real one will -- `predict(state) -> (p, None)`,
a `telemetry()` of token counts and parse failures, and a parse-failure mode
that falls back to zero punishment for the episode-round that failed.

Two shapes, both deterministic given the state:

  * a constant, whose battery is known exactly: rho is nan (no variance in
    the punishment column), magnitude is 0, and at `punishment=0` it must
    reproduce never-punishing cell for cell;
  * a 21-entry table indexed by the contribution, which can be made to be
    `thr9_p10` exactly, so the harness can be checked against a manager
    whose numbers are published (PR #217, PR #219).

The parse-failure draw uses its own numpy Generator rather than the global
torch RNG, so turning failures on does not shift the environment's stream
and the two runs remain comparable.
"""

import time

import numpy as np
import torch as th

from aimanager.llm_manager.battery import TELEMETRY_KEYS

#: `punishment=10` where the contribution is 9 or less -- the incumbent rule
#: `thr9_p10`, expressed as a table so the stub can wear it.
THR9_P10_TABLE = tuple(10 if c <= 9 else 0 for c in range(21))


class StubManager:
    """Fixed punishments, with an LLM manager's telemetry surface.

    Args:
        punishment: the constant to return, when `table` is None.
        table: 21 punishments indexed by the current contribution.
        parse_failure_rate: probability that an episode-round's answer fails
            to parse. Those cells fall back to 0, exactly as the real client
            will, so the harness's failure accounting is exercised.
        prompt_tokens_per_call / completion_tokens_per_call: tokens charged
            per episode-round prompt, so the token columns carry a number
            whose arithmetic can be checked.
        latency_s: seconds to sleep per batched round. 0 by default; it
            exists to rehearse a throughput budget, not for tests.
    """

    autoregressive = False

    def __init__(
        self,
        punishment=0,
        table=None,
        *,
        parse_failure_rate=0.0,
        prompt_tokens_per_call=0,
        completion_tokens_per_call=0,
        latency_s=0.0,
        seed=0,
        n_punishments=31,
        **_,
    ):
        self.punishment = int(punishment)
        self.table = None if table is None else th.tensor(table, dtype=th.int64)
        assert self.table is None or self.table.numel() == 21, "table is per level"
        self.parse_failure_rate = float(parse_failure_rate)
        assert 0.0 <= self.parse_failure_rate <= 1.0
        self.prompt_tokens_per_call = int(prompt_tokens_per_call)
        self.completion_tokens_per_call = int(completion_tokens_per_call)
        self.latency_s = float(latency_s)
        self.n_punishments = int(n_punishments)
        self.seed = int(seed)
        self.model = None
        self.default_values = {
            "contribution": 0,
            "punishment": 0,
            "contribution_valid": False,
            "punishment_valid": False,
            "in_group": False,
        }
        self.reset_telemetry()

    def to(self, device):
        return self

    def reset_telemetry(self):
        self._rng = np.random.default_rng(self.seed)
        self._t = {k: 0.0 for k in TELEMETRY_KEYS}

    def telemetry(self):
        return dict(self._t)

    def _base(self, state):
        c = state["contribution"]
        if self.table is None:
            return th.full_like(c, self.punishment)
        return self.table.to(c.device)[c.clamp(0, 20)]

    def predict(self, state, **_):
        t0 = time.perf_counter()
        p = self._base(state).clamp(0, self.n_punishments - 1).to(th.int64)
        n_episodes = p.shape[0]
        if self.parse_failure_rate > 0:
            # one prompt per episode, so a failure zeroes that episode's
            # whole group for this round -- the fallback the plan specifies
            failed = th.from_numpy(
                self._rng.random(n_episodes) < self.parse_failure_rate
            ).to(p.device)
            self._t["n_parse_failures"] += float(failed.sum())
            p = th.where(failed.view(-1, *([1] * (p.ndim - 1))), th.zeros_like(p), p)
        if self.latency_s:
            time.sleep(self.latency_s)
        self._t["n_calls"] += 1
        self._t["n_decisions_requested"] += float(n_episodes)
        self._t["prompt_tokens"] += float(self.prompt_tokens_per_call * n_episodes)
        self._t["completion_tokens"] += float(
            self.completion_tokens_per_call * n_episodes
        )
        self._t["manager_wall_clock_s"] += time.perf_counter() - t0
        return p, None


def collect_telemetry(manager):
    """A manager's telemetry, or nothing if it does not keep any.

    Returns `None` rather than zeros for a manager that does not report, so
    "this rule has no tokens" never reads as "the language model spent none".
    """
    fn = getattr(manager, "telemetry", None)
    return dict(fn()) if callable(fn) else None


def reset_telemetry(manager):
    fn = getattr(manager, "reset_telemetry", None)
    if callable(fn):
        fn()
