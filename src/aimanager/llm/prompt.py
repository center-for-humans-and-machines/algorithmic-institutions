"""Immutable, named prompt versions for the language-model manager.

A result has to be able to name the prompt that produced it, so the text lives
in frozen module constants, a version is a frozen selection over them, and
`PromptVersion.fingerprint` is the sha256 of exactly the blocks that version
selects -- including `TraceRenderer.spec`, so a change to the trace format
moves it too. Edit a block and every fingerprint that uses it moves; the run
that quoted the old one is still identifiable. Versions are never edited in
place: a change to the wording is a new name.

**The objective is settled and is not an axis.** The target is the group's
undivided common pool, summed over the game -- what the real managers were
paid on (`reports/basics.md`) and the project's welfare measure. It counts
what contributors produce and charges punishment at full price, so a manager
that raises contributions by punishing enormously does not score on it. That
is the measure working, not a defect in it: measured elsewhere, a
correctly-targeted threshold rule is indistinguishable from never punishing on
the pool (about +0.5, interval spanning zero), while the capped fitted rule of
PR #219 beats that threshold rule by +5.0 pool points. The pool does reward
good management; it does not reward punishment for its own sake.

Three axes, chosen so that every comparison against `v3_explicit_pool` is a
single-factor change:

- `cost`, how explicitly the arithmetic of the pool is stated. `bare` leaves
  it to the rules, which do say punishment leaves the pool but never dwell on
  it; `stated` adds one sentence; `explicit` states the whole trade-off in
  two.
- `show_pool`, whether each round block carries the resulting pool or only the
  contributions and punishments.
- `answer`, whether the model is asked for the number directly or asked to
  reason briefly first. Worth separating because a served Qwen3 has its own
  `<think>` mode disabled by default, so "reason first" has to be asked for in
  the prompt if it is wanted at all.

Deliberately NOT in any version: anything about whom to punish, any threshold,
any example answer with numbers in it, any mention that the contributors are
models. Nor any word that reads as discouraging punishment, and nor any
suggestion that punishing works -- both are conclusions the comparison exists
to watch the model reach or not reach, and either one in the text would make a
good result uninterpretable. The prompt states the accounting and stops. If a
variant ever needs a hint about which players to aim at to produce a sensible
policy, that is a finding about the model and belongs in the log, not here.

Versions become immutable at their first commit: from then on a wording change
is a new name, enforced by the pinned fingerprints in `test_llm_prompt.py`.
"""

import hashlib
from dataclasses import dataclass

from aimanager.llm.trace import (
    MAX_CONTRIBUTION,
    MAX_PUNISHMENT,
    TraceRenderer,
    expected_labels,
)

SYSTEM = (
    "You are the manager of a group in an economic game with real payoffs. "
    "You read the record of play and decide the punishments. "
    "You answer in the exact format you are asked for."
)

# The numbers are written out rather than interpolated, so the prose wraps the
# way it reads; the assertion below is what keeps them tied to the constants.
RULES = """THE GAME

The game runs for 24 rounds. You manage one group. It starts with four members,
and there is a second group you never see.

In every round each member of your group is given 20 points into a private
account and decides how many of those points, any whole number from 0 to 20, to
put into the group's common pool. What is put in is multiplied by 1.6 and then
split equally between the members of the group, regardless of how much each of
them put in.

You then see what every member put in and set a punishment for each of them: a
whole number from 0 to 30. A punishment is taken out of that member's private
account, and the same number of points is taken out of the common pool.

After every fourth round each of the eight players chooses which of the two
groups to play the next rounds in, so members can leave your group and players
from the other group can join it. Your group can end up larger or smaller than
four.

A member who gave no input in a round put nothing in and cannot be punished.
Those members are marked in the record below, and any punishment set for them
is discarded."""

assert f" 0 to {MAX_CONTRIBUTION}, to" in RULES, "endowment drifted from the rules"
assert f" 0 to {MAX_PUNISHMENT}." in RULES, "punishment range drifted from the rules"

OBJECTIVE = """YOUR OBJECTIVE

Your aim is the total of your group's common pool over the whole 24 rounds.
You are paid in proportion to that total. Nothing else you do is paid, and you
are not paid on the members' private accounts."""

COSTS = {
    "bare": "",
    "stated": """THE ARITHMETIC OF THE POOL

Every point put in enters the pool multiplied by 1.6, and every point you
punish leaves it at full price, in the round you set it.""",
    "explicit": """THE ARITHMETIC OF THE POOL

Every point put in enters the pool multiplied by 1.6, and every point you
punish leaves it at full price, in the round you set it. A punishment
therefore pays for itself only through what it changes about the contributions
of the rounds that follow.""",
}

HISTORY_HEADING = "THE RECORD SO FAR"

ANSWERS = {
    "direct": f"""YOUR DECISION FOR ROUND {{round}}

The members of your group this round are: {{listed}}.

Answer with one single line, in exactly this format, and write nothing else:

PUNISHMENT: {{template}}

Each <number> is a whole number from 0 to {MAX_PUNISHMENT}. Give a number for
every member listed, including any who gave no input.""",
    "reason_first": f"""YOUR DECISION FOR ROUND {{round}}

The members of your group this round are: {{listed}}.

First write at most three short sentences saying what you are weighing. Then,
as the last line of your reply and on a line of its own, write exactly:

PUNISHMENT: {{template}}

Each <number> is a whole number from 0 to {MAX_PUNISHMENT}. Give a number for
every member listed, including any who gave no input.""",
}


@dataclass(frozen=True)
class PromptVersion:
    name: str
    cost: str
    show_pool: bool
    answer: str = "direct"

    @property
    def renderer(self):
        return TraceRenderer(show_pool=self.show_pool)

    @property
    def blocks(self):
        return [
            SYSTEM,
            RULES,
            OBJECTIVE,
            COSTS[self.cost],
            ANSWERS[self.answer],
            self.renderer.spec,
        ]

    @property
    def fingerprint(self):
        """sha256 over exactly the text this version selects, 12 hex chars."""
        digest = hashlib.sha256("\n\x00\n".join(self.blocks).encode()).hexdigest()
        return digest[:12]


PROMPT_VERSIONS = {
    v.name: v
    for v in (
        # The cost ladder, everything else held at the v3 setting.
        PromptVersion("v1_bare_pool", cost="bare", show_pool=True),
        PromptVersion("v2_stated_pool", cost="stated", show_pool=True),
        PromptVersion("v3_explicit_pool", cost="explicit", show_pool=True),
        # One factor off v3 each.
        PromptVersion("v4_explicit_nopool", cost="explicit", show_pool=False),
        PromptVersion(
            "v5_explicit_reason", cost="explicit", show_pool=True, answer="reason_first"
        ),
    )
}

DEFAULT_VERSION = "v3_explicit_pool"


def resolve(version):
    """A registry name, or a `PromptVersion` passed straight through."""
    return PROMPT_VERSIONS[version] if isinstance(version, str) else version


@dataclass(frozen=True)
class Prompt:
    version: str
    fingerprint: str
    system: str
    user: str
    labels: tuple

    @property
    def messages(self):
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def build_prompt(records, version=DEFAULT_VERSION):
    """The full prompt for the decision at `records[-1].round_number`.

    `records` is the accumulating trace, oldest first; its last record is the
    round being decided and carries `punishment=None` on every player. Returns
    a `Prompt` carrying the version name and fingerprint, so the caller can
    log which text produced the completion, and the labels the answer must
    cover, which is what `parse.parse_punishments` checks against.
    """
    spec = resolve(version)
    target = records[-1]
    labels = expected_labels(target)
    sections = [RULES, OBJECTIVE]
    if COSTS[spec.cost]:
        sections.append(COSTS[spec.cost])
    sections.append(HISTORY_HEADING + "\n\n" + spec.renderer.trace(records))
    sections.append(
        ANSWERS[spec.answer].format(
            round=target.round_number + 1,
            listed=", ".join(labels),
            template=", ".join(f"{label} = <number>" for label in labels),
        )
    )
    return Prompt(
        version=spec.name,
        fingerprint=spec.fingerprint,
        system=SYSTEM,
        user="\n\n".join(sections),
        labels=tuple(labels),
    )


__all__ = [
    "ANSWERS",
    "COSTS",
    "DEFAULT_VERSION",
    "OBJECTIVE",
    "PROMPT_VERSIONS",
    "Prompt",
    "PromptVersion",
    "build_prompt",
    "resolve",
]
