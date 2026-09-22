"""A manager whose punishment decisions come from a language model.

This module owns the serving-facing half of the LLM manager: the class that
sits in the existing manager seats, the per-episode trace it accumulates, and
the batching of one chat completion per episode per round. It does **not** own
the wording of the prompt or the parsing of the reply. Those are injected, as
a `PromptStrategy`, so that work can proceed against this file without
touching it.

THE BOUNDARY, which a sibling builds against
============================================

A prompt strategy is any object with a `version` string, two required methods
and one optional one::

    class PromptStrategy:
        version: str

        def build(self, ctx: RoundContext) -> List[Dict[str, str]]:
            '''Return an OpenAI-style message list for ONE episode-round.'''

        def parse(self, text: str, ctx: RoundContext) -> Optional[List[int]]:
            '''Return one punishment per managed player, in the order of
            `ctx.current`, or None if the reply could not be parsed.'''

        def constraint(self, ctx: RoundContext) -> Optional[DecodeConstraint]:
            '''Optional. Restrict what the model may emit. Returning None
            leaves the decode free.'''

`build` is called once per episode per round and must be pure: same
`RoundContext` in, same messages out. `parse` receives the raw assistant text
with no preprocessing. A strategy that omits `constraint` still works; one
that supplies it gets the guarantee below.

KEEP THE PROMPT APPEND-ONLY. vLLM has prefix caching on by default
(`CacheConfig.enable_prefix_caching` is True in 0.19.1), so the part of round
r's prompt that round r-1's prompt already contained is not prefilled again.
The trace is the bulk of the prompt and it grows monotonically, so a strategy
whose prompt is `fixed header + trace so far + this round + instruction`
re-prefills only the last two pieces. A strategy that rewrites, reorders or
summarises earlier rounds, or that puts anything varying (a round counter, a
running total) *before* the trace, throws the cache away and pays full prefill
every round. That is the difference between a cheap rollout and an expensive
one, so it is part of the contract rather than an optimisation.

Constrain the decode; keep the parser as a guard
------------------------------------------------
The action is a handful of small integers, so the right mechanism is to stop
the model emitting anything else rather than to let it and catch the mistake.
`constraint` returns a `DecodeConstraint`, vLLM masks the logits, and the
parse failure rate is then **zero by construction**. The parser and the
failure counter stay, but they are a guard: a non-zero `parse_failure_rate`
under a constraint is a bug in the constraint or the server, not a property
of the model, and should be read that way.

This matters because the fallback is not neutral. An unparsable reply falls
back to **zero punishment**, and zero punishment is exactly the policy every
collapsed learned manager in this project converged on. A silent fallback
would therefore bias a run toward the very outcome the experiment is trying to
distinguish from a real decision not to punish. The rate is reported next to
every result for that reason, and a run with a material rate is not a result.

Register a strategy under a name and select it with `prompt_version`::

    from aimanager.manager.llm_manager import register_prompt_strategy
    register_prompt_strategy("v1", MyStrategy())

`DefaultPromptStrategy` ("v0") ships here so this layer is testable on its own.
It is deliberately plain; it is not the experiment's prompt.

What the strategy is handed
---------------------------
`RoundContext` carries the managed group only -- the four-ish players whose
punishments this call decides -- never the other group. Membership changes:
players switch groups every `switch_every` rounds, so `ctx.current` can hold a
different set of agents, and a different number of them, from one round to the
next. `PlayerRound.agent` is the global agent index and is the only stable
identity across rounds; `ctx.history` is the accumulating trace, oldest first,
and is the manager's only memory.

`contribution_valid=False` marks a player who gave no input. The environment
charges such a player nothing and zeroes any punishment aimed at them
(`ArtificialHumanEnv.punish`), so a punishment spent there is wasted. The trace
marks them, and the fallback does not, so a wasted decision shows up in the log
rather than being silently absorbed.

The seats this class fills
==========================

`predict(state, **_) -> (punishment, None)` is the opponent/clone seat of
`rl_manager.run_batch` and `scripts/rl_anneal_local/guard.py::rollout_cells`.
`state` is `ArtificialHumanEnv.served_state()`: a dict of `(B, A, T)` tensors,
with `T == 1` in a rollout. The return is int64 `(B, A, 1)` over **all** A
agents -- the harness masks it by group -- with zeros wherever this manager
does not manage. This is the batched path: B episodes become B concurrent
completions, and a 24-round rollout is 24 such batches.

`get_punishments(data) -> punishment` is the `api_manager` seat, dispatched by
`MultiManager` for `simulation/simulate.py`. `data` is the `create_data` view,
`(n_groups, A, T)` with the full round history along T, so the trace is rebuilt
from it rather than accumulated. That path runs one episode at a time and gets
no batching benefit; it exists so an LLM manager can appear in a `pairings:`
config beside the rule and clone managers.

Episode identity and resets
---------------------------
The trace resets whenever the served `round_number` is 0 (equivalently
`is_first`), or on an explicit `reset()`. A rollout is therefore a fresh set of
episodes every time `env.reset()` runs, and batch row `b` is episode `b` of the
current rollout; `episode_offset` makes those ids unique across rollouts in the
log.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as th

from aimanager.manager.llm_client import ChatClient, DecodeConstraint, integers_regex

logger = logging.getLogger(__name__)

DEFAULT_OBJECTIVE = (
    "You are paid on your own group's common pool: the total your group's "
    "members contribute, multiplied by 1.6, minus the points you deduct by "
    "punishing. Maximise it over the whole game."
)


# ---------------------------------------------------------------------------
# What a prompt strategy is handed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerRound:
    """One player's record for one round, within the managed group.

    `agent` is the global agent index: the only identity that survives a
    group switch. `punishment` is what the game actually charged, which is 0
    for a player who gave no input however much was aimed at them.
    """

    agent: int
    contribution: int
    punishment: int
    contribution_valid: bool
    punishment_valid: bool = True

    @property
    def gave_input(self) -> bool:
        return self.contribution_valid


@dataclass(frozen=True)
class RoundRecord:
    """A completed round of the managed group."""

    round_number: int
    players: Tuple[PlayerRound, ...]
    common_good: Optional[float] = None

    @property
    def total_contribution(self) -> int:
        return sum(p.contribution for p in self.players if p.contribution_valid)

    @property
    def total_punishment(self) -> int:
        return sum(p.punishment for p in self.players)


@dataclass(frozen=True)
class RoundContext:
    """Everything a prompt strategy may use for one episode-round decision."""

    episode: int
    round_number: int
    n_players: int
    n_punishments: int
    objective: str
    prompt_version: str
    history: Tuple[RoundRecord, ...]
    current: Tuple[PlayerRound, ...]
    n_rounds: Optional[int] = None
    switch_every: Optional[int] = None
    max_contribution: int = 20
    multiplier: float = 1.6

    @property
    def max_punishment(self) -> int:
        return self.n_punishments - 1


# ---------------------------------------------------------------------------
# The default strategy: plain, and only so this layer can be tested alone
# ---------------------------------------------------------------------------


class DefaultPromptStrategy:
    """A minimal working strategy. Not the experiment's prompt."""

    version = "v0"

    def build(self, ctx: RoundContext) -> List[Dict[str, str]]:
        lines = [
            f"You manage a group of {ctx.n_players} players in a "
            f"{ctx.n_rounds or 24}-round public goods game.",
            f"Each round every player receives {ctx.max_contribution} points "
            f"and chooses how many to contribute. Contributions are pooled, "
            f"multiplied by {ctx.multiplier}, and shared equally.",
            f"You may then deduct {0} to {ctx.max_punishment} points from each "
            f"player. A deduction costs the pool the same amount.",
            "A player who gave no input cannot be punished.",
            ctx.objective,
            "",
        ]
        if ctx.history:
            lines.append("Play so far:")
            for record in ctx.history:
                lines.append(f"Round {record.round_number + 1}:")
                for player in record.players:
                    mark = "" if player.contribution_valid else " (no input)"
                    lines.append(
                        f"  player {player.agent}: contributed "
                        f"{player.contribution}{mark}, you deducted "
                        f"{player.punishment}"
                    )
                if record.common_good is not None:
                    lines.append(f"  pool per member: {record.common_good:.1f}")
            lines.append("")
        lines.append(f"Round {ctx.round_number + 1}, this round's contributions:")
        for player in ctx.current:
            mark = "" if player.contribution_valid else " (no input)"
            lines.append(f"  player {player.agent}: {player.contribution}{mark}")
        lines.append("")
        lines.append(
            f"Reply with exactly {len(ctx.current)} integers between 0 and "
            f"{ctx.max_punishment}, comma separated, in the order listed, and "
            "nothing else."
        )
        return [{"role": "user", "content": "\n".join(lines)}]

    def parse(self, text: str, ctx: RoundContext) -> Optional[List[int]]:
        numbers = re.findall(r"-?\d+", text or "")
        if len(numbers) < len(ctx.current):
            return None
        return [int(n) for n in numbers[-len(ctx.current) :]]  # noqa: E203

    def constraint(self, ctx: RoundContext) -> Optional[DecodeConstraint]:
        """Exactly `n_players` integers in range, comma separated.

        This is what makes the parse above a guard rather than a gamble: the
        model is not permitted to emit a reply the parser would reject.
        """
        if not ctx.current:
            return None
        return DecodeConstraint(
            regex=integers_regex(len(ctx.current), ctx.max_punishment)
        )


PROMPT_STRATEGIES: Dict[str, Any] = {"v0": DefaultPromptStrategy()}


def register_prompt_strategy(version: str, strategy: Any) -> None:
    """Make `strategy` selectable as `prompt_version=version`."""
    for method in ("build", "parse"):
        if not callable(getattr(strategy, method, None)):
            raise TypeError(f"strategy for {version!r} has no callable {method}()")
    PROMPT_STRATEGIES[version] = strategy


def get_prompt_strategy(version: str) -> Any:
    try:
        return PROMPT_STRATEGIES[version]
    except KeyError:
        known = ", ".join(sorted(PROMPT_STRATEGIES))
        raise ValueError(
            f"Unknown prompt_version {version!r}; registered: {known}"
        ) from None


# ---------------------------------------------------------------------------
# The manager
# ---------------------------------------------------------------------------


@dataclass
class _EpisodeTrace:
    """Accumulating record for one batch row, across a rollout."""

    records: List[RoundRecord] = field(default_factory=list)
    open_round: Optional[int] = None
    open_players: Tuple[PlayerRound, ...] = ()


class LLMManager:
    """Punishments chosen by a language model reading an accumulating trace.

    Parameters
    ----------
    model, api_base, api_key:
        Passed to :class:`ChatClient`. `api_base` must carry the `/v1`
        suffix; both fall back to `HOSTED_VLLM_API_BASE` / `_API_KEY`.
    prompt_version:
        Key into the strategy registry. Ignored if `strategy` is given.
    objective:
        The sentence stating what the manager is paid on. Reaches the
        strategy through `RoundContext.objective`.
    n_punishments:
        Action-space size; punishments are `0 .. n_punishments - 1`.
    group_id:
        Which `agent_group` this manager manages. `None` manages every
        agent, which is what a single-manager rollout and most tests want.
    strategy:
        A prompt strategy object, overriding `prompt_version`.
    client:
        A ready :class:`ChatClient`, overriding the connection arguments.
        Tests pass a stub here.
    """

    #: `MultiManager` reads this; the `create_data` view is what we consume.
    needs_rounds = False

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-8B",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt_version: str = "v0",
        objective: str = DEFAULT_OBJECTIVE,
        n_punishments: int = 31,
        group_id: Optional[int] = None,
        strategy: Any = None,
        client: Optional[ChatClient] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_concurrent: int = 64,
        enable_thinking: Optional[bool] = None,
        backend: str = "http",
        constrained_decode: bool = True,
        constraint_style: str = "structured_outputs",
        log_path: Optional[str] = None,
        seed: Optional[int] = None,
        max_retries: int = 4,
        retry_base_delay: float = 2.0,
        n_rounds: Optional[int] = None,
        switch_every: Optional[int] = None,
        episode_offset: int = 0,
        **_,
    ) -> None:
        self.prompt_version = prompt_version
        self.strategy = strategy or get_prompt_strategy(prompt_version)
        self.objective = objective
        self.n_punishments = int(n_punishments)
        self.group_id = None if group_id is None else int(group_id)
        self.n_rounds = n_rounds
        self.switch_every = switch_every
        self.episode_offset = int(episode_offset)
        self.constrained_decode = bool(constrained_decode)

        self.client = client or ChatClient(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
            enable_thinking=enable_thinking,
            backend=backend,
            constraint_style=constraint_style,
            log_path=log_path,
            seed=seed,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
        )

        # The `api_manager` seat needs these; they mirror DummyManager's.
        self.model = None
        self.default_values = {
            "contribution": 0,
            "punishment": 0,
            "contribution_valid": False,
            "punishment_valid": False,
            "in_group": False,
        }

        self._traces: Dict[int, _EpisodeTrace] = {}
        self.n_decisions = 0
        self.parse_failures = 0
        self.call_errors = 0

    # -- accounting ----------------------------------------------------

    @property
    def parse_failure_rate(self) -> float:
        if not self.n_decisions:
            return 0.0
        return self.parse_failures / self.n_decisions

    def report(self) -> Dict[str, Any]:
        """Everything a run has to be reported with."""
        return {
            **self.client.describe(),
            **self.client.stats.as_dict(),
            "prompt_version": getattr(self.strategy, "version", self.prompt_version),
            "objective": self.objective,
            "n_punishments": self.n_punishments,
            "group_id": self.group_id,
            "constrained_decode": self.constrained_decode,
            "n_decisions": self.n_decisions,
            "parse_failures": self.parse_failures,
            # Under a constraint this must be 0; anything else is a bug in
            # the constraint or the server, not a property of the model.
            "parse_failure_rate": round(self.parse_failure_rate, 5),
            "call_errors": self.call_errors,
        }

    def reset(self) -> None:
        """Drop every accumulated trace."""
        self._traces = {}

    # -- the rollout seat ----------------------------------------------

    def predict(self, state: Dict[str, th.Tensor], **_):
        """Choose punishments for every episode in the batch, for one round.

        Returns `(punishment, None)` with `punishment` int64 `(B, A, 1)` over
        all agents, zero outside the managed group. The second element is
        `None` to match the punisher-model call site, which discards it.
        """
        contribution = state["contribution"]
        device = contribution.device
        n_batch, n_agents = contribution.shape[0], contribution.shape[1]

        round_number = int(state["round_number"][0, 0, -1])
        if round_number == 0:
            self.reset()

        groups = self._column(state.get("agent_group"), n_batch, n_agents, device)
        valid = self._bool_column(state.get("contribution_valid"), n_batch, n_agents)
        contrib = self._column(contribution, n_batch, n_agents, device)
        prev_punishment = self._column(
            state.get("prev_punishment"), n_batch, n_agents, device
        )
        prev_contribution = self._column(
            state.get("prev_contribution"), n_batch, n_agents, device
        )
        prev_valid = self._bool_column(
            state.get("prev_contribution_valid"), n_batch, n_agents
        )
        prev_common_good = state.get("prev_common_good")

        contexts: List[RoundContext] = []
        seats: List[List[int]] = []
        for b in range(n_batch):
            managed = [
                a
                for a in range(n_agents)
                if self.group_id is None or int(groups[b][a]) == self.group_id
            ]
            trace = self._traces.setdefault(b, _EpisodeTrace())
            if round_number > 0:
                pool = None
                if prev_common_good is not None and managed:
                    pool = float(prev_common_good[b, managed[0], -1])
                self._close_round(
                    trace,
                    round_number - 1,
                    prev_contribution[b],
                    prev_punishment[b],
                    prev_valid[b],
                    pool,
                )
            current = tuple(
                PlayerRound(
                    agent=a,
                    contribution=int(contrib[b][a]),
                    punishment=0,
                    contribution_valid=bool(valid[b][a]),
                )
                for a in managed
            )
            seats.append(managed)
            trace.open_round = round_number
            trace.open_players = current
            contexts.append(self._context(b, round_number, current, trace))

        decisions = self._decide(contexts)

        punishment = th.zeros((n_batch, n_agents, 1), dtype=th.int64, device=device)
        for b, (managed, chosen) in enumerate(zip(seats, decisions)):
            for agent, value in zip(managed, chosen):
                punishment[b, agent, 0] = value
        return punishment, None

    # -- the api_manager seat ------------------------------------------

    def get_punishments(self, data: Dict[str, th.Tensor]) -> th.Tensor:
        """Punishments for the `MultiManager` dispatcher.

        `data` is the `create_data` view: `(n_groups, A, T)` with the whole
        round history along T, and, in batch row `g`, every agent outside
        group `g` masked to this manager's defaults. Only the last round's
        column is used by the caller, but the earlier columns are the trace,
        so the whole history is read here.
        """
        contribution = data["contribution"]
        n_rows, n_agents, n_t = contribution.shape
        in_group = data.get("in_group")
        valid = data.get("contribution_valid")
        punishments_hist = data.get("punishment")
        common_good = data.get("common_good")

        contexts: List[RoundContext] = []
        seats: List[List[int]] = []
        rows: List[int] = []
        for row in range(n_rows):
            if self.group_id is not None and row != self.group_id:
                continue
            managed_now = [
                a
                for a in range(n_agents)
                if in_group is None or bool(in_group[row, a, -1])
            ]
            records: List[RoundRecord] = []
            for t in range(n_t - 1):
                members = [
                    a
                    for a in range(n_agents)
                    if in_group is None or bool(in_group[row, a, t])
                ]
                if not members:
                    continue
                records.append(
                    RoundRecord(
                        round_number=t,
                        players=tuple(
                            PlayerRound(
                                agent=a,
                                contribution=int(contribution[row, a, t]),
                                punishment=(
                                    0
                                    if punishments_hist is None
                                    else int(punishments_hist[row, a, t])
                                ),
                                contribution_valid=(
                                    True if valid is None else bool(valid[row, a, t])
                                ),
                            )
                            for a in members
                        ),
                        common_good=(
                            None
                            if common_good is None
                            else float(common_good[row, members[0], t])
                        ),
                    )
                )
            current = tuple(
                PlayerRound(
                    agent=a,
                    contribution=int(contribution[row, a, -1]),
                    punishment=0,
                    contribution_valid=(
                        True if valid is None else bool(valid[row, a, -1])
                    ),
                )
                for a in managed_now
            )
            trace = _EpisodeTrace(records=records)
            contexts.append(self._context(row, n_t - 1, current, trace))
            seats.append(managed_now)
            rows.append(row)

        decisions = self._decide(contexts)

        out = th.zeros_like(data["punishment"])
        for row, managed, chosen in zip(rows, seats, decisions):
            for agent, value in zip(managed, chosen):
                out[row, agent, -1] = value
        return out

    # -- shared machinery ----------------------------------------------

    def _context(
        self,
        batch_index: int,
        round_number: int,
        current: Tuple[PlayerRound, ...],
        trace: _EpisodeTrace,
    ) -> RoundContext:
        return RoundContext(
            episode=self.episode_offset + batch_index,
            round_number=round_number,
            n_players=len(current),
            n_punishments=self.n_punishments,
            objective=self.objective,
            prompt_version=getattr(self.strategy, "version", self.prompt_version),
            history=tuple(trace.records),
            current=current,
            n_rounds=self.n_rounds,
            switch_every=self.switch_every,
        )

    def _decide(self, contexts: Sequence[RoundContext]) -> List[List[int]]:
        """One batched round: build, call, parse, validate, fall back."""
        live = [i for i, ctx in enumerate(contexts) if ctx.current]
        decisions: List[List[int]] = [[] for _ in contexts]
        if not live:
            return decisions

        conversations = [self.strategy.build(contexts[i]) for i in live]
        meta = [
            {
                "episode": contexts[i].episode,
                "round": contexts[i].round_number,
                "prompt_version": contexts[i].prompt_version,
                "n_players": contexts[i].n_players,
                "agents": [p.agent for p in contexts[i].current],
            }
            for i in live
        ]
        constraints = [self._constraint_for(contexts[i]) for i in live]
        completions = self.client.complete(
            conversations, meta=meta, constraints=constraints
        )

        for i, completion in zip(live, completions):
            ctx = contexts[i]
            self.n_decisions += 1
            if not completion.ok:
                self.call_errors += 1
                self.parse_failures += 1
                logger.warning(
                    "episode %d round %d: call failed (%s); punishing zero",
                    ctx.episode,
                    ctx.round_number,
                    completion.error,
                )
                decisions[i] = [0] * len(ctx.current)
                continue
            values = self._validate(self.strategy.parse(completion.text, ctx), ctx)
            if values is None:
                self.parse_failures += 1
                logger.warning(
                    "episode %d round %d: unparsable reply %r; falling back to "
                    "ZERO punishment, which is not a neutral action -- it is "
                    "the policy collapsed managers converge on (constrained "
                    "decode was %s)",
                    ctx.episode,
                    ctx.round_number,
                    (completion.text or "")[:200],
                    "on" if self.constrained_decode else "off",
                )
                decisions[i] = [0] * len(ctx.current)
            else:
                decisions[i] = values
        return decisions

    def _constraint_for(self, ctx: RoundContext) -> Optional[DecodeConstraint]:
        """The strategy's decode constraint, if it has one and it is on.

        A strategy without a `constraint` method is fine: the decode is then
        free and the parser is doing the work alone, which is exactly the
        situation `parse_failure_rate` was built to measure.
        """
        if not self.constrained_decode:
            return None
        hook = getattr(self.strategy, "constraint", None)
        if hook is None:
            return None
        return hook(ctx)

    def _validate(
        self, values: Optional[Sequence[Any]], ctx: RoundContext
    ) -> Optional[List[int]]:
        """Length-check first, then clamp. A wrong length is a failure."""
        if values is None:
            return None
        try:
            as_ints = [int(v) for v in values]
        except (TypeError, ValueError):
            return None
        if len(as_ints) != len(ctx.current):
            return None
        top = self.n_punishments - 1
        return [min(max(v, 0), top) for v in as_ints]

    @staticmethod
    def _close_round(
        trace: _EpisodeTrace,
        round_number: int,
        contribution_row,
        punishment_row,
        valid_row,
        pool: Optional[float],
    ) -> None:
        """Turn the open round into a record, using what the game charged."""
        if trace.open_round is None or trace.open_round != round_number:
            return
        players = tuple(
            PlayerRound(
                agent=p.agent,
                contribution=int(contribution_row[p.agent]),
                punishment=int(punishment_row[p.agent]),
                contribution_valid=bool(valid_row[p.agent]),
            )
            for p in trace.open_players
        )
        trace.records.append(
            RoundRecord(round_number=round_number, players=players, common_good=pool)
        )
        trace.open_round = None
        trace.open_players = ()

    @staticmethod
    def _column(tensor, n_batch: int, n_agents: int, device) -> List[List[int]]:
        if tensor is None:
            return [[0] * n_agents for _ in range(n_batch)]
        return tensor[..., -1].to(th.int64).tolist()

    @staticmethod
    def _bool_column(tensor, n_batch: int, n_agents: int) -> List[List[bool]]:
        if tensor is None:
            return [[True] * n_agents for _ in range(n_batch)]
        return tensor[..., -1].to(th.bool).tolist()
