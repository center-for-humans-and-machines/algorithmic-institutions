"""The accumulating record of play the language-model manager reads.

One block per round, own group only, which is what the human managers saw.
The trace is the model's ONLY memory: nothing else carries across rounds, so
anything the manager should be able to condition on has to be in here.

**Typed events, one renderer.** A round is turned into a list of `Event`s and
a `TraceRenderer` turns those into text; nothing in this module concatenates a
line by hand. The history and the round being decided go through the same
renderer, so a past round and the current round cannot drift into different
shapes -- the current round is marked `punishment not set yet` rather than
being recognisable by a clause that is missing, which is the cheap way that
asymmetry leaks. A prompt version is then a renderer configuration rather
than a function full of f-strings, and `TraceRenderer.spec` folds the
templates into the version fingerprint, so editing a template moves the
fingerprint of every version that uses it.

**A player who gave no input is shown as such, never as a number.** The
simulation state fills those cells with `default_values["contribution"]`, the
dataset median 9, and `per_round.parquet` records that 9; the human CSV stores
0 there. Neither is a contribution the game ever used (see
`generic/data.py::MISSING_CONTRIBUTION` and the accounting identity verified
there). Both are wrong to show. `PlayerRound.contribution is None` is the only
representation of "gave no input" this module accepts, and
`PlayerRound.from_masked` is the one place a validity flag is turned into it,
so a caller cannot forget the mask.

**The pool arithmetic is the trade-off, spelled out.** `pool` is
`1.6 * contributed - punished`, the identity that holds on all 2256 human
group-rounds to 3e-14, with punishment counted only on players who gave input
(a punishment aimed at a timed-out player is discarded by the env and was
never charged in the real game). Showing it per round is one of the two prompt
axes; `TraceRenderer(show_pool=False)` drops it and shows contributions alone.

Round numbers are 0-based everywhere in this project and 1-based in the text
the model reads -- converted once, in `_round_events`.
"""

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

MULTIPLIER = 1.6
MAX_CONTRIBUTION = 20
MAX_PUNISHMENT = 30


def sanitise(value):
    """Nothing that could break a line or a quoted field ever reaches the text.

    Our labels and numbers are tame, but the guard is one line and the failure
    it prevents -- a value carrying a newline, splitting one event across two
    rendered lines -- is silent and would look like a malformed trace.
    """
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace("\r", " ").replace("\n", " ").replace('"', "'").strip()


@dataclass(frozen=True)
class Event:
    """One rendered line's worth of the record, before it is text."""

    kind: str
    fields: Tuple[Tuple[str, str], ...] = ()

    @classmethod
    def make(cls, kind, **fields):
        return cls(kind, tuple((k, sanitise(v)) for k, v in sorted(fields.items())))

    @property
    def mapping(self):
        return dict(self.fields)


@dataclass(frozen=True)
class PlayerRound:
    """One player's row in one round of the manager's own group.

    `label` is stable across the whole episode, so a player who leaves and
    comes back is recognisable -- the manager of a reshuffling group can see
    that it has met this player before. `contribution` and `punishment` are
    None for "gave no input" and "not decided yet" respectively; a round block
    for the round currently being decided carries `punishment=None`.
    """

    label: str
    contribution: Optional[int]
    punishment: Optional[int] = None
    joined: bool = False

    @classmethod
    def from_masked(cls, label, contribution, contribution_valid, **kwargs):
        """The only sanctioned way to build one from a (value, valid) pair.

        Whatever filler sits in `contribution` where `contribution_valid` is
        False -- 0 in the human CSV, 9 in the simulation output -- is dropped
        here and never reaches the model.
        """
        value = int(contribution) if contribution_valid else None
        return cls(label=label, contribution=value, **kwargs)

    @property
    def gave_input(self):
        return self.contribution is not None

    @property
    def charged_punishment(self):
        """What the pool was actually charged for this player."""
        if not self.gave_input or self.punishment is None:
            return 0
        return int(self.punishment)


@dataclass(frozen=True)
class RoundRecord:
    """One round of the manager's own group.

    `players` is in the order the model is asked to answer in. `left` names
    players who were in the group the round before and are not in it now;
    `manager_no_input` marks a round the human manager timed out on, which
    happens in the replay data and never in a rollout.
    """

    round_number: int
    players: Tuple[PlayerRound, ...]
    left: Tuple[str, ...] = ()
    manager_no_input: bool = False

    @property
    def contributed(self):
        return sum(p.contribution for p in self.players if p.gave_input)

    @property
    def punished(self):
        return sum(p.charged_punishment for p in self.players)

    @property
    def pool(self):
        return MULTIPLIER * self.contributed - self.punished

    @property
    def decided(self):
        return all(p.punishment is not None for p in self.players)


TEMPLATES = {
    "round": "Round {round}",
    "membership": "  ({changes})",
    "manager_no_input": "  (you gave no input this round, so nobody was punished)",
    "played": "  {label} put in {contribution}, you punished {punishment}",
    "pending": "  {label} put in {contribution}, punishment not set yet",
    "no_input": "  {label} gave no input (put in nothing, cannot be punished)",
    "pool": "  Pool: {pool} (1.6 x {contributed} put in, minus {punished} punished)",
    "pool_pending": (
        "  Pool before your punishment: {gross} (1.6 x {contributed} put in)"
    ),
    "empty": "(nothing yet -- this is the first round.)",
}


def _round_events(record, show_pool):
    events = [Event.make("round", round=record.round_number + 1)]
    arrived = [p.label for p in record.players if p.joined]
    changes = []
    if arrived:
        changes.append(f"{', '.join(arrived)} joined your group")
    if record.left:
        changes.append(f"{', '.join(record.left)} left your group")
    if changes:
        events.append(Event.make("membership", changes="; ".join(changes)))
    if record.manager_no_input:
        events.append(Event.make("manager_no_input"))
    for player in record.players:
        if not player.gave_input:
            events.append(Event.make("no_input", label=player.label))
        elif player.punishment is None:
            events.append(
                Event.make(
                    "pending", label=player.label, contribution=player.contribution
                )
            )
        else:
            events.append(
                Event.make(
                    "played",
                    label=player.label,
                    contribution=player.contribution,
                    punishment=player.punishment,
                )
            )
    if show_pool:
        if record.decided:
            events.append(
                Event.make(
                    "pool",
                    pool=float(record.pool),
                    contributed=float(record.contributed),
                    punished=float(record.punished),
                )
            )
        else:
            events.append(
                Event.make(
                    "pool_pending",
                    gross=MULTIPLIER * record.contributed,
                    contributed=float(record.contributed),
                )
            )
    return events


@dataclass(frozen=True)
class TraceRenderer:
    """Events to text. The whole trace format, as configuration."""

    show_pool: bool = True
    templates: Mapping[str, str] = field(default_factory=lambda: dict(TEMPLATES))

    def events(self, records):
        out = []
        for record in records:
            out.extend(_round_events(record, self.show_pool))
        return out

    def render(self, event):
        return self.templates[event.kind].format(**event.mapping)

    def round(self, record):
        return "\n".join(self.render(e) for e in _round_events(record, self.show_pool))

    def trace(self, records):
        """The whole accumulating record, oldest round first.

        The last record is the round being decided: its players render through
        the `pending` template and its pool line through `pool_pending`.
        """
        if not records:
            return self.templates["empty"]
        return "\n\n".join(self.round(r) for r in records)

    @property
    def spec(self):
        """Canonical text of this configuration, for the version fingerprint."""
        items = "\n".join(f"{k}={self.templates[k]}" for k in sorted(self.templates))
        return f"show_pool={self.show_pool}\n{items}"


DEFAULT_RENDERER = TraceRenderer()


def format_trace(records, show_pool=True):
    return TraceRenderer(show_pool=show_pool).trace(records)


def format_round(record, show_pool=True):
    return TraceRenderer(show_pool=show_pool).round(record)


def label_for(index):
    """Stable player label. Index is per-episode arrival order, 0-based."""
    return f"Player {index + 1}"


def build_records(rows, labels_by_key, show_left=True):
    """Assemble `RoundRecord`s from per-round, per-player tuples.

    `rows` is an iterable of `(round_number, key, contribution,
    contribution_valid, punishment, manager_no_input)`, where `key` identifies
    a player across rounds and `punishment` is None for the round being
    decided. `labels_by_key` maps a key to its stable label, in arrival order.

    `joined` / `left` are derived by differencing consecutive rounds'
    membership, so the caller does not have to track it.
    """
    by_round = {}
    timed_out = {}
    for rnd, key, contribution, valid, punishment, manager_no_input in rows:
        by_round.setdefault(rnd, {})[key] = (contribution, valid, punishment)
        timed_out[rnd] = bool(manager_no_input) or timed_out.get(rnd, False)

    # arrival order, not label text: "Player 10" sorts before "Player 2"
    rank = {key: i for i, key in enumerate(labels_by_key)}
    records = []
    previous = set()
    for rnd in sorted(by_round):
        members = by_round[rnd]
        order = sorted(members, key=lambda k: rank[k])
        players = tuple(
            PlayerRound.from_masked(
                label=labels_by_key[key],
                contribution=members[key][0],
                contribution_valid=members[key][1],
                punishment=members[key][2],
                joined=show_left and bool(previous) and key not in previous,
            )
            for key in order
        )
        left = tuple(
            labels_by_key[k] for k in sorted(previous - set(members), key=rank.get)
        )
        records.append(
            RoundRecord(
                round_number=int(rnd),
                players=players,
                left=left if show_left else (),
                manager_no_input=timed_out[rnd],
            )
        )
        previous = set(members)
    return records


def expected_labels(record):
    """The labels the model must answer for, in the order they are listed."""
    return [p.label for p in record.players]


def punishable(record):
    """Labels the environment will actually charge a punishment for."""
    return [p.label for p in record.players if p.gave_input]


__all__ = [
    "DEFAULT_RENDERER",
    "MAX_CONTRIBUTION",
    "MAX_PUNISHMENT",
    "MULTIPLIER",
    "TEMPLATES",
    "Event",
    "PlayerRound",
    "RoundRecord",
    "TraceRenderer",
    "build_records",
    "expected_labels",
    "format_round",
    "format_trace",
    "label_for",
    "punishable",
    "sanitise",
]
