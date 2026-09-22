"""A manager whose punishment decisions come from a language model.

This module owns the serving-facing half: the class that sits in the existing
manager seats, the per-episode trace it accumulates from the environment, and
the batching of one chat completion per episode per round. The prompt, the
trace format and the parser are NOT owned here -- they are `aimanager.llm`,
and this module injects them exactly as that package's docstring specifies::

    from aimanager.llm.prompt import build_prompt
    from aimanager.llm.parse import enforce, mark_wasted, parse_punishments

    prompt = build_prompt(records, version=self.prompt_version)
    completion = self.client(prompt.messages)
    result = mark_wasted(parse_punishments(completion, prompt.labels), target)
    charged = enforce(result, target)

Building `records` from `served_state()` is this module's job, and
`aimanager.llm` states two rules for it that are not negotiable:

1. a player with `contribution_valid == False` is passed
   `contribution_valid=False` and never their stored contribution -- the
   simulation state fills that cell with the dataset median and the human CSV
   with 0, and the game used neither. `PlayerRound.from_masked` is the only
   constructor used here, so the mask cannot be forgotten;
2. `punishment` on a past round is what the environment CHARGED -- zero on a
   no-input player -- not the raw action. That value is read back out of
   `prev_punishment`, which is the env's own record of what it charged.

GROUP SIZE IS NOT FOUR
======================
Across the 4,512 group-rounds in the human data, group size runs from 1 to 8
and is four in only 26 per cent of them: members reshuffle every fourth round
and nothing rebalances the sides. So the roster is read from `agent_group` at
every decision point and everything downstream is built from it -- the labels,
the prompt, and the decode constraint. Nothing in this module assumes an
arity.

CONSTRAIN THE DECODE, KEEP THE PARSER AS A GUARD
================================================
The answer is a fixed line naming each member:

    PUNISHMENT: Player 1 = 3, Player 2 = 0, Player 5 = 12

`label_answer_regex` builds a regex over exactly the labels the prompt listed,
in order, so vLLM masks the logits and the model *cannot* answer for a roster
the decision point does not have. That failure mode is worse than a crash: the
sibling's audit found 11 of 120 completions answering for the wrong roster,
and a positional reader would have filed those numbers against the wrong
players and produced a policy shape that looked entirely reasonable.

The parser stays, and so does its counter. Under a constraint the failure rate
is zero by construction, and the counter exists to prove that rather than to
absorb failures: a constrained decode can be misconfigured, pointed at the
wrong schema, or silently dropped when a client falls back to an
unconstrained call, and nothing else downstream would notice. Every failure is
logged, carries `fallback="zero"`, and is counted, because zero punishment is
the policy every collapsed learned manager converged on and a quiet fallback
would push a result toward the outcome the comparison exists to distinguish.

KEEP THE PROMPT APPEND-ONLY
===========================
vLLM has prefix caching on by default (`CacheConfig.enable_prefix_caching` is
True in 0.19.1), so the part of round r's prompt that round r-1's already
contained is not prefilled again. Exactly one block moves: the round being
decided is rendered `punishment not set yet` and is re-rendered as
`you punished N` once it has been, so round r re-prefills one round block and
reuses everything before it. Measured 2026-09-22 on Qwen3-8B: at 200 episodes
the per-round wall clock stayed flat at ~1.2 s while the prompt grew from 207
to 1974 tokens; at 1000 it rose from 3.1 s to 32.9 s, because the traces no
longer fit the KV cache. The cache is therefore what sets the affordable
episode count, and it only works while the trace grows by appending.

The seats this class fills
==========================
`predict(state, **_) -> (punishment, None)` is the opponent/clone seat of
`rl_manager.run_batch` and `scripts/rl_anneal_local/guard.py::rollout_cells`.
`state` is `ArtificialHumanEnv.served_state()`: a dict of `(B, A, T)` tensors
with `T == 1` in a rollout. The return is int64 `(B, A, 1)` over **all** A
agents -- the harness masks it by group -- with zeros outside the managed
group. This is the batched path: B episodes become B concurrent completions.

`get_punishments(data) -> punishment` is the `api_manager` seat, dispatched by
`MultiManager` for `simulation/simulate.py`, registered as manager type `llm`.
`data` is the `create_data` view, `(n_groups, A, T)` with the full history
along T, so the trace is rebuilt from it rather than accumulated.

Episode identity and resets
---------------------------
The trace resets whenever the served `round_number` is 0, or on an explicit
`reset()`. Batch row `b` is episode `b`; `episode_offset` makes those ids
unique across rollouts in the log. Labels are derived from the global agent
index and are therefore stable for the whole episode, which is what lets the
model recognise a player who left and came back.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as th

from aimanager.llm.parse import REASONS, enforce, mark_wasted, parse_punishments
from aimanager.llm.prompt import DEFAULT_VERSION, build_prompt
from aimanager.llm.trace import MAX_PUNISHMENT, PlayerRound, RoundRecord
from aimanager.manager.llm_client import ChatClient, DecodeConstraint

logger = logging.getLogger(__name__)


def agent_label(agent: int) -> str:
    """The stable, per-episode name of a seat, as the model sees it."""
    return f"Player {agent + 1}"


def _literal(text: str) -> str:
    """Escape for a regex, but leave a space as a plain space.

    `re.escape` renders a space as `\\ `. Python's engine accepts that; the
    grammar backends behind vLLM's structured output are less forgiving, and
    a space is not special anyway.
    """
    return re.escape(text).replace("\\ ", " ")


def label_answer_regex(labels: Sequence[str], max_punishment=MAX_PUNISHMENT) -> str:
    """A regex admitting exactly the answer line the prompt asks for.

    Built from the labels of the roster PRESENT at this decision point, so an
    answer for a different roster -- the failure a positional reader would
    have filed against the wrong players -- cannot be emitted at all.

    NO UNBOUNDED WHITESPACE. An earlier version wrote the separators as
    `\\s*`, which is the natural thing and is wrong: whitespace is then always
    a legal next token, so a model with nothing better to say can emit it
    forever. Measured on Qwen3-8B, 2026-09-22: asked to punish with no context
    to go on, it produced `PUNISHMENT:` followed by 128 tab tokens and stopped
    only at `finish_reason: length`, having answered nothing. The separators
    are therefore exactly the single spaces the prompt's template shows, which
    makes the shortest legal continuation an actual answer. vLLM offers
    `disable_any_whitespace` for the JSON backends for the same reason.
    """
    labels = list(labels)
    if not labels:
        raise ValueError("cannot constrain an answer for an empty roster")
    number = "(?:" + "|".join(str(v) for v in range(max_punishment, -1, -1)) + ")"
    pairs = [f"{_literal(label)} = {number}" for label in labels]
    return "PUNISHMENT: " + ", ".join(pairs)


@dataclass
class _EpisodeTrace:
    """Accumulating record for one batch row, across a rollout."""

    records: List[RoundRecord] = field(default_factory=list)
    open_round: Optional[int] = None
    open_agents: Tuple[int, ...] = ()
    previous_agents: Tuple[int, ...] = ()


class LLMManager:
    """Punishments chosen by a language model reading an accumulating trace.

    Parameters
    ----------
    model, api_base, api_key:
        Passed to :class:`ChatClient`. `api_base` must carry the `/v1`
        suffix and may be several endpoints; both fall back to
        `HOSTED_VLLM_API_BASE` / `_API_KEY`.
    prompt_version:
        A name in `aimanager.llm.prompt.PROMPT_VERSIONS`, or a
        `PromptVersion`. The result names the version and the version pins
        the text.
    objective:
        Unused -- the objective is part of the prompt version, which owns
        the text. Accepted so a config written against the plan's signature
        still loads, and warned about if it is set to something else.
    n_punishments:
        Action-space size; punishments are `0 .. n_punishments - 1`.
    group_id:
        Which `agent_group` this manager manages. `None` manages every
        agent, which is what a single-manager rollout and most tests want.
    client:
        A ready :class:`ChatClient`, overriding the connection arguments.
    """

    #: `MultiManager` reads this; the `create_data` view is what we consume.
    needs_rounds = False

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-8B",
        api_base=None,
        api_key: Optional[str] = None,
        prompt_version: str = DEFAULT_VERSION,
        objective: Optional[str] = None,
        n_punishments: int = 31,
        group_id: Optional[int] = None,
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
        episode_offset: int = 0,
        prompt_builder=None,
        parser=None,
        **_,
    ) -> None:
        self.prompt_version = prompt_version
        self.n_punishments = int(n_punishments)
        self.group_id = None if group_id is None else int(group_id)
        self.episode_offset = int(episode_offset)
        self.constrained_decode = bool(constrained_decode)
        # Test seams, not a second contract: they default to the real ones.
        self._build_prompt = prompt_builder or build_prompt
        self._parse = parser or parse_punishments

        if objective is not None:
            logger.warning(
                "objective=%r ignored: the objective is part of prompt_version "
                "%r, which owns the text (aimanager.llm.prompt)",
                objective[:60],
                prompt_version,
            )
        if self.n_punishments - 1 != MAX_PUNISHMENT:
            logger.warning(
                "n_punishments=%d implies a maximum of %d, but the prompt and "
                "the parser are pinned to %d",
                self.n_punishments,
                self.n_punishments - 1,
                MAX_PUNISHMENT,
            )

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
        self.call_errors = 0
        self.wasted_answers = 0
        self._counts = self._empty_counts()

    # -- accounting ----------------------------------------------------

    @staticmethod
    def _empty_counts() -> Dict[str, int]:
        """Counters in exactly the shape `aimanager.llm.parse.summarise`
        returns, so the two can be compared and are (see the unit test)."""
        counts = {"answers": 0, "failures": 0, "zero_fallback_answers": 0}
        counts.update({f"reason[{reason}]": 0 for reason in REASONS})
        counts.update({"form[labelled]": 0, "form[positional]": 0})
        return counts

    def _count(self, result) -> None:
        self._counts["answers"] += 1
        if not result.ok:
            self._counts["failures"] += 1
            if result.fallback == "zero":
                self._counts["zero_fallback_answers"] += 1
            key = f"reason[{result.reason}]"
            if key in self._counts:
                self._counts[key] += 1
        if result.form in ("labelled", "positional"):
            self._counts[f"form[{result.form}]"] += 1
        if result.wasted:
            self.wasted_answers += 1

    @property
    def n_decisions(self) -> int:
        return self._counts["answers"]

    @property
    def parse_failures(self) -> int:
        return self._counts["failures"]

    @property
    def parse_failure_rate(self) -> float:
        """Per ANSWER, not per player: counting per player would weight a
        failure by group size."""
        if not self._counts["answers"]:
            return 0.0
        return self._counts["failures"] / self._counts["answers"]

    def summary(self) -> Dict[str, Any]:
        """The parse numbers, in `parse.summarise`'s shape."""
        out = dict(self._counts)
        total = out["answers"]
        out["failure_rate"] = (out["failures"] / total) if total else float("nan")
        return out

    def report(self) -> Dict[str, Any]:
        """Everything a run has to be reported with."""
        return {
            **self.client.describe(),
            **self.client.stats.as_dict(),
            "prompt_version": self.prompt_version,
            "n_punishments": self.n_punishments,
            "group_id": self.group_id,
            "constrained_decode": self.constrained_decode,
            # Under a constraint this must be 0; anything else is a bug in
            # the constraint or the server, not a property of the model.
            "parse_failure_rate": round(self.parse_failure_rate, 5),
            "call_errors": self.call_errors,
            "wasted_answers": self.wasted_answers,
            **self.summary(),
        }

    def reset(self) -> None:
        """Drop every accumulated trace."""
        self._traces = {}

    # -- the evaluation harness's telemetry seat -----------------------
    #
    # `llm_manager.stub.collect_telemetry` reads `telemetry()` off any
    # manager and records NaN for one that has none, so that "this rule has
    # no tokens" can never be read as "the language model spent none". This
    # is the one place the two halves have to agree on a name, so the
    # mapping is spelled out rather than guessed at.
    #
    # `n_calls` MEANS DIFFERENT THINGS on the two managers and the
    # difference is real, not a bug to paper over: the stub answers a whole
    # batch in one call, while this manager issues one HTTP completion per
    # episode-round, so here `n_calls` and `n_decisions_requested` coincide.
    # `n_calls` carries the client's count, which includes calls that
    # errored, so it is the number of requests actually put on the wire.

    def telemetry(self) -> Dict[str, float]:
        stats = self.client.stats
        return {
            "prompt_tokens": float(stats.prompt_tokens),
            "completion_tokens": float(stats.completion_tokens),
            "n_calls": float(stats.n_calls),
            # one prompt per episode-round; `answers` counts results read
            # back, which is what a failure rate has to be a share of.
            "n_decisions_requested": float(self._counts["answers"]),
            "n_parse_failures": float(self._counts["failures"]),
            "manager_wall_clock_s": float(stats.wall_s),
        }

    def reset_telemetry(self) -> None:
        """Zero the counters, and the traces with them.

        Called once per arm by `harness.run_arm`, before its first rollout.
        The traces go too: an arm's first round must not inherit a trace
        from the arm before it.
        """
        stats = self.client.stats
        stats.n_calls = 0
        stats.n_errors = 0
        stats.prompt_tokens = 0
        stats.completion_tokens = 0
        stats.wall_s = 0.0
        stats.truncated = 0
        self.call_errors = 0
        self.wasted_answers = 0
        self._counts = self._empty_counts()
        self.reset()

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

        groups = self._ints(state.get("agent_group"), n_batch, n_agents)
        contrib = self._ints(contribution, n_batch, n_agents)
        valid = self._bools(state.get("contribution_valid"), n_batch, n_agents)
        prev_contribution = self._ints(
            state.get("prev_contribution"), n_batch, n_agents
        )
        prev_punishment = self._ints(state.get("prev_punishment"), n_batch, n_agents)
        prev_valid = self._bools(
            state.get("prev_contribution_valid"), n_batch, n_agents
        )

        jobs: List[Tuple[int, List[int], List[RoundRecord]]] = []
        for b in range(n_batch):
            members = [
                a
                for a in range(n_agents)
                if self.group_id is None or int(groups[b][a]) == self.group_id
            ]
            trace = self._traces.setdefault(b, _EpisodeTrace())
            if round_number > 0:
                self._close_round(
                    trace,
                    round_number - 1,
                    prev_contribution[b],
                    prev_punishment[b],
                    prev_valid[b],
                )
            target = RoundRecord(
                round_number=round_number,
                players=tuple(
                    PlayerRound.from_masked(
                        label=agent_label(a),
                        contribution=contrib[b][a],
                        contribution_valid=valid[b][a],
                        joined=a not in trace.previous_agents and round_number > 0,
                    )
                    for a in members
                ),
                left=tuple(
                    agent_label(a) for a in trace.previous_agents if a not in members
                ),
            )
            trace.open_round = round_number
            trace.open_agents = tuple(members)
            jobs.append((b, members, list(trace.records) + [target]))

        decisions = self._decide(jobs)

        punishment = th.zeros((n_batch, n_agents, 1), dtype=th.int64, device=device)
        for (b, members, _records), charged in zip(jobs, decisions):
            for agent in members:
                punishment[b, agent, 0] = charged.get(agent_label(agent), 0)
        return punishment, None

    # -- the api_manager seat ------------------------------------------

    def get_punishments(self, data: Dict[str, th.Tensor]) -> th.Tensor:
        """Punishments for the `MultiManager` dispatcher.

        `data` is the `create_data` view: `(n_groups, A, T)` with the whole
        round history along T, and, in batch row `g`, every agent outside
        group `g` masked to this manager's defaults. Only the last round's
        column is read by the caller, but the earlier columns are the trace.
        """
        contribution = data["contribution"]
        n_rows, n_agents, n_t = contribution.shape
        in_group = data.get("in_group")
        valid = data.get("contribution_valid")
        history = data.get("punishment")

        def members_at(row, t):
            return [
                a
                for a in range(n_agents)
                if in_group is None or bool(in_group[row, a, t])
            ]

        jobs: List[Tuple[int, List[int], List[RoundRecord]]] = []
        for row in range(n_rows):
            if self.group_id is not None and row != self.group_id:
                continue
            records: List[RoundRecord] = []
            previous: List[int] = []
            for t in range(n_t):
                members = members_at(row, t)
                if not members and t < n_t - 1:
                    continue
                decided = t < n_t - 1
                records.append(
                    RoundRecord(
                        round_number=t,
                        players=tuple(
                            PlayerRound.from_masked(
                                label=agent_label(a),
                                contribution=int(contribution[row, a, t]),
                                contribution_valid=(
                                    True if valid is None else bool(valid[row, a, t])
                                ),
                                punishment=(
                                    int(history[row, a, t])
                                    if decided and history is not None
                                    else None
                                ),
                                joined=t > 0 and a not in previous,
                            )
                            for a in members
                        ),
                        left=tuple(
                            agent_label(a) for a in previous if a not in members
                        ),
                    )
                )
                previous = members
            if records and records[-1].players:
                jobs.append((row, members_at(row, n_t - 1), records))

        decisions = self._decide(jobs)

        out = th.zeros_like(data["punishment"])
        for (row, members, _records), charged in zip(jobs, decisions):
            for agent in members:
                out[row, agent, -1] = charged.get(agent_label(agent), 0)
        return out

    # -- shared machinery ----------------------------------------------

    def _decide(self, jobs) -> List[Dict[str, int]]:
        """One batched round: build, constrain, call, parse, enforce."""
        if not jobs:
            return []
        prompts = [
            self._build_prompt(records, version=self.prompt_version)
            for _b, _members, records in jobs
        ]
        conversations = [prompt.messages for prompt in prompts]
        constraints = [self._constraint_for(prompt) for prompt in prompts]
        meta = [
            {
                "episode": self.episode_offset + b,
                "round": records[-1].round_number,
                "prompt_version": prompt.version,
                "prompt_fingerprint": prompt.fingerprint,
                "labels": list(prompt.labels),
            }
            for (b, _members, records), prompt in zip(jobs, prompts)
        ]
        completions = self.client.complete(
            conversations, meta=meta, constraints=constraints
        )

        charged_all: List[Dict[str, int]] = []
        for (_b, _members, records), prompt, completion, info in zip(
            jobs, prompts, completions, meta
        ):
            target = records[-1]
            if not completion.ok:
                self.call_errors += 1
                logger.warning(
                    "episode %s round %s: call failed (%s); falling back to "
                    "ZERO punishment, which is the policy under test and not "
                    "a neutral default",
                    info["episode"],
                    info["round"],
                    completion.error,
                )
            result = mark_wasted(self._parse(completion.text, prompt.labels), target)
            self._count(result)
            self.client.annotate(
                info,
                {
                    "ok": result.ok,
                    "reason": result.reason,
                    "form": result.form,
                    "wasted": list(result.wasted),
                    "fallback": result.fallback,
                },
            )
            charged_all.append(enforce(result, target))
        return charged_all

    def _constraint_for(self, prompt) -> Optional[DecodeConstraint]:
        """The decode constraint for THIS decision point's roster."""
        if not self.constrained_decode or not prompt.labels:
            return None
        return DecodeConstraint(
            regex=label_answer_regex(prompt.labels, self.n_punishments - 1)
        )

    @staticmethod
    def _close_round(
        trace: _EpisodeTrace,
        round_number: int,
        contribution_row,
        punishment_row,
        valid_row,
    ) -> None:
        """Turn the open round into a record of what the game CHARGED."""
        if trace.open_round is None or trace.open_round != round_number:
            return
        previous = trace.previous_agents
        trace.records.append(
            RoundRecord(
                round_number=round_number,
                players=tuple(
                    PlayerRound.from_masked(
                        label=agent_label(a),
                        contribution=contribution_row[a],
                        contribution_valid=valid_row[a],
                        punishment=int(punishment_row[a]),
                        joined=round_number > 0 and a not in previous,
                    )
                    for a in trace.open_agents
                ),
                left=tuple(
                    agent_label(a) for a in previous if a not in trace.open_agents
                ),
            )
        )
        trace.previous_agents = trace.open_agents
        trace.open_round = None
        trace.open_agents = ()

    @staticmethod
    def _ints(tensor, n_batch: int, n_agents: int) -> List[List[int]]:
        if tensor is None:
            return [[0] * n_agents for _ in range(n_batch)]
        return tensor[..., -1].to(th.int64).tolist()

    @staticmethod
    def _bools(tensor, n_batch: int, n_agents: int) -> List[List[bool]]:
        if tensor is None:
            return [[True] * n_agents for _ in range(n_batch)]
        return tensor[..., -1].to(th.bool).tolist()
