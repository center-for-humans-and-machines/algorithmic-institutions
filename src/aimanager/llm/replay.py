"""Replay the real human games and ask the model what it would have punished.

The point of this harness is to judge a PROMPT without running a rollout. A
manager evaluated only by how it scores conflates two things -- whether the
prompt communicates the game, and whether the model's judgement is any good --
and a rollout takes a GPU, 24 sequential batched calls per episode and a
contribution model that then reacts to the manager, so every number in it
depends on the whole stack.

Replay removes all of that. `experiments/2group_8agent_50ep.csv` holds 50 real
games; a decision point is one (game, manager, round). The trace is built from
what actually happened up to that round, the model is asked for that round's
punishments, and the answer is compared against what the real manager did on
the same state. No simulation, no contribution model, no feedback loop: the
state is real and identical across prompt variants, so a difference between
two variants is a difference between the prompts.

Structure of the data, which sets what a decision point is:

- Every game appears twice with the group labels mirrored (the flip
  augmentation). One copy is kept, as `evaluation_suite.convert.load_human`
  and the linear training pipeline do.
- A game has eight players and two groups, each with its own human manager.
  The manager is attached to a group slot for the whole game; players move
  between slots at the decision rounds, so a manager's group is four members
  at round 0 and can be anything from empty to all eight later.
- 104 of 2256 group-rounds have `manager_no_input`: the human manager timed
  out. Those rounds are NOT decision points -- there is no human decision to
  compare against -- but they stay in the trace, marked, because they are part
  of what the manager saw.
- 280 of 9600 agent-rounds have `player_no_input`. The CSV stores contribution
  0 on them and the simulation stores 9; both are dropped by
  `PlayerRound.from_masked` and shown as "gave no input".

The client is injected: any callable taking the `messages` list and returning
the completion string. `StubClient` implements the same signature from a
policy function, so every test and the whole battery run without a GPU.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from aimanager.llm.parse import enforce, mark_wasted, parse_punishments, summarise
from aimanager.llm.prompt import DEFAULT_VERSION, build_prompt
from aimanager.llm.trace import build_records, label_for

HUMAN_DATA_FILE = "experiments/2group_8agent_50ep.csv"


@dataclass(frozen=True)
class Decision:
    """One (game, manager, round) the model is asked to decide."""

    episode_id: int
    group_id: int
    round_number: int
    records: Tuple
    human: Dict[str, Optional[int]]
    contribution: Dict[str, Optional[int]]

    @property
    def key(self):
        return (self.episode_id, self.group_id, self.round_number)

    @property
    def target(self):
        return self.records[-1]


def load_games(csv_path=HUMAN_DATA_FILE):
    """The human CSV, one copy per game, columns this harness needs."""
    df = pd.read_csv(csv_path)
    if "pair_id" in df.columns:
        keep = df.groupby("pair_id")["episode_id"].transform("min")
        df = df[df["episode_id"] == keep]
    columns = [
        "episode_id",
        "round_number",
        "group_id",
        "player_id",
        "contribution",
        "punishment",
        "player_no_input",
        "manager_no_input",
    ]
    return df[columns].copy()


def _labels(sub):
    """Stable labels in arrival order within this manager's group."""
    first = sub.groupby("player_id")["round_number"].min()
    order = sorted(first.index, key=lambda pid: (first[pid], pid))
    return {pid: label_for(i) for i, pid in enumerate(order)}


def _rows(sub, upto):
    """Trace rows up to and including `upto`; `upto` has no punishments yet."""
    window = sub[sub["round_number"] <= upto]
    return [
        (
            int(r.round_number),
            r.player_id,
            r.contribution,
            not bool(r.player_no_input),
            None if r.round_number == upto else int(r.punishment),
            bool(r.manager_no_input),
        )
        for r in window.itertuples()
    ]


def decisions(df):
    """Every decision point in `df`, in a fixed order.

    A round is skipped when the manager's group is empty (nothing to decide)
    or when the human manager timed out (nothing to compare against). An empty
    round therefore leaves a gap in the trace it precedes; 218 of the 2152
    decision points carry at least one, and the format does not yet say so.
    """
    out = []
    for (episode_id, group_id), sub in df.groupby(["episode_id", "group_id"]):
        sub = sub.sort_values(["round_number", "player_id"])
        labels = _labels(sub)
        present = sorted(sub["round_number"].unique())
        for upto in present:
            here = sub[sub["round_number"] == upto]
            if here.empty or bool(here["manager_no_input"].iloc[0]):
                continue
            records = build_records(_rows(sub, upto), labels)
            target = records[-1]
            out.append(
                Decision(
                    episode_id=int(episode_id),
                    group_id=int(group_id),
                    round_number=int(upto),
                    records=tuple(records),
                    human={
                        labels[r.player_id]: int(r.punishment)
                        for r in here.itertuples()
                    },
                    contribution={p.label: p.contribution for p in target.players},
                )
            )
    return out


class StubClient:
    """A deterministic stand-in for the serving layer.

    `policy(decision) -> {label: punishment}` is asked for the answer, and the
    stub renders it in the format the prompt demands, so the parser is
    exercised on real strings rather than bypassed. `corrupt(i)` may return a
    replacement completion for call `i`, which is how the failure paths are
    tested.
    """

    def __init__(self, policy, corrupt=None):
        self.policy = policy
        self.corrupt = corrupt
        self.calls = []

    def __call__(self, messages, decision=None, **_):
        self.calls.append(messages)
        index = len(self.calls) - 1
        if self.corrupt is not None:
            replacement = self.corrupt(index)
            if replacement is not None:
                return replacement
        values = self.policy(decision)
        body = ", ".join(f"{label} = {int(v)}" for label, v in values.items())
        return f"PUNISHMENT: {body}"


def run_replay(points, client, version=DEFAULT_VERSION, cache_path=None):
    """Ask `client` for every decision in `points`; return a tidy frame.

    One row per (decision, player). `model_punishment` is what the model
    asked for and `charged_punishment` what the environment would charge
    (zero on a player who gave no input), so a wasted decision is visible in
    the difference rather than absorbed.

    `cache_path` is a JSONL of raw completions keyed by version and decision;
    a rerun reads it instead of calling, which makes the battery reproducible
    and keeps a re-scoring free.

    The frame carries `attrs["parse_summary"]`, computed once by
    `parse.summarise` over the answers -- the only place the failure rate is
    calculated, so a report cannot quietly compute a second version of it.
    """
    cache = _load_cache(cache_path)
    rows = []
    results = []
    for point in points:
        prompt = build_prompt(point.records, version=version)
        cache_key = _cache_key(version, point)
        completion = cache.get(cache_key)
        if completion is None:
            completion = client(prompt.messages, decision=point)
            _append_cache(cache_path, cache_key, completion)
        result = mark_wasted(parse_punishments(completion, prompt.labels), point.target)
        results.append(result)
        charged = enforce(result, point.target)
        for label in prompt.labels:
            rows.append(
                {
                    "episode_id": point.episode_id,
                    "group_id": point.group_id,
                    "round_number": point.round_number,
                    "label": label,
                    "contribution": point.contribution[label],
                    "contribution_valid": point.contribution[label] is not None,
                    "human_punishment": point.human.get(label),
                    "model_punishment": result.punishments[label],
                    "charged_punishment": charged[label],
                    "wasted": label in result.wasted,
                    "parse_ok": result.ok,
                    "parse_reason": result.reason,
                    "parse_form": result.form,
                    "prompt_version": prompt.version,
                    "prompt_fingerprint": prompt.fingerprint,
                    "prompt_chars": len(prompt.system) + len(prompt.user),
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["parse_summary"] = summarise(results)
    return frame


def _cache_key(version, point):
    return f"{version}|{point.episode_id}|{point.group_id}|{point.round_number}"


def _load_cache(path):
    if path is None or not Path(path).exists():
        return {}
    out = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                out[record["key"]] = record["completion"]
    return out


def _append_cache(path, key, completion):
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps({"key": key, "completion": completion}) + "\n")


def human_frame(points):
    """The same tidy frame for the real managers, for the reference battery."""
    rows = []
    for point in points:
        for label, contribution in point.contribution.items():
            rows.append(
                {
                    "episode_id": point.episode_id,
                    "group_id": point.group_id,
                    "round_number": point.round_number,
                    "label": label,
                    "contribution": contribution,
                    "contribution_valid": contribution is not None,
                    "model_punishment": point.human.get(label),
                    "charged_punishment": point.human.get(label),
                    "human_punishment": point.human.get(label),
                    "wasted": False,
                    "parse_ok": True,
                    "parse_reason": None,
                    "parse_form": "human",
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "HUMAN_DATA_FILE",
    "Decision",
    "StubClient",
    "decisions",
    "human_frame",
    "load_games",
    "run_replay",
]
