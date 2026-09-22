"""Strict reader of the manager's answer -- a GUARD, not the mechanism.

The contract the prompt states is one line:

    PUNISHMENT: Player 1 = 3, Player 2 = 0, Player 3 = 0, Player 4 = 12

On a served endpoint the answer should be constrained at the token level
(vLLM guided decoding; four integers in 0..30 suits it exactly), which makes
the failure rate zero by construction. This parser stays anyway, and so does
its counter: a constrained decode can be misconfigured, pointed at the wrong
schema, or silently dropped when the client falls back to an unconstrained
call, and nothing else downstream would notice.

**The fallback is not neutral, which is why it is loud.** A failure returns
zero punishment for that episode-round, and zero punishment is exactly the
policy every collapsed learned manager in this project converged on. A silent
fallback would therefore push a result toward the outcome the whole comparison
exists to distinguish, in proportion to how often it fires. So every failure
logs a warning, carries `fallback="zero"`, and is counted; `summarise()` is
the one place the rate is computed, shared by the replay harness and the
serving layer. A non-zero rate is a bug to be fixed, not a property of the
model to be reported and moved past.

What counts as strict here, stated so the rate means something:

- The labelled form is checked against the exact set of labels the prompt
  listed. A missing, extra, repeated or unknown label is a failure, not a
  best-effort match -- a model that answers for three of four players has not
  answered, and silently zeroing the fourth would hide it.
- A bare positional list (`PUNISHMENT: 3, 0, 0, 12`) is accepted ONLY when it
  carries exactly as many numbers as there are players, because the prompt
  fixes the order and reading it is then determined rather than guessed. It is
  recorded as `form="positional"` so deviation from the requested format is
  visible in the log rather than absorbed into the success rate.
- Values outside 0..30 and non-integers are failures. They are not clamped:
  clamping would turn a model that does not know the action space into a model
  that scores well at the boundary.

`wasted` is separate from all of this. A punishment set for a player who gave
no input parses fine and is a legal answer; the environment discards it. The
prompt says so and the trace marks those players, so a wasted decision is a
measurable property of the policy, and it is recorded rather than quietly
zeroed.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from aimanager.llm.trace import MAX_PUNISHMENT

logger = logging.getLogger(__name__)

MARKER = re.compile(r"PUNISHMENT\s*:", re.IGNORECASE)
PAIR = re.compile(r"([A-Za-z]*\s*\d+)\s*[=:]\s*(-?\d+)")
NUMBER = re.compile(r"-?\d+")
DECIMAL = re.compile(r"\d\.\d")
DECORATION = re.compile(r"[*`#_>]")

REASONS = (
    "empty",
    "no_marker",
    "no_numbers",
    "not_integer",
    "wrong_count",
    "label_mismatch",
    "duplicate_label",
    "out_of_range",
)


class ParseFailure(ValueError):
    """Raised instead of falling back, when the caller asks for it."""


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    punishments: Dict[str, int]
    reason: Optional[str] = None
    form: Optional[str] = None
    detail: str = ""
    wasted: Tuple[str, ...] = ()
    fallback: Optional[str] = None

    @property
    def values(self):
        """Punishments in the order the labels were given to the parser."""
        return list(self.punishments.values())


def _zeros(labels):
    return {label: 0 for label in labels}


def _fail(labels, reason, detail="", strict=False):
    """The fallback, made loud. See the module docstring on why zero is not a
    neutral default in this game."""
    message = f"unparseable manager answer ({reason}): {detail!r}"
    if strict:
        raise ParseFailure(message)
    logger.warning("%s -- falling back to zero punishment for this round", message)
    return ParseResult(
        ok=False,
        punishments=_zeros(labels),
        reason=reason,
        detail=detail,
        fallback="zero",
    )


def _normalise(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _payload(completion):
    """The text the answer is read from: the last marker to the next blank
    line. Anything the model wrote before its final answer is ignored, which
    is what makes a version that reasons before answering parseable at all."""
    text = DECORATION.sub("", completion)
    matches = list(MARKER.finditer(text))
    if not matches:
        return None
    rest = text[matches[-1].end() :]
    return rest.split("\n\n")[0].strip()


def parse_punishments(completion, labels, strict=False):
    """Read the model's answer for exactly `labels`, in that order.

    `strict=True` raises `ParseFailure` instead of falling back to zero, for
    a caller that would rather abort a run than record a zero-punishment
    round it did not mean.
    """
    labels = list(labels)
    if not completion or not completion.strip():
        return _fail(labels, "empty", strict=strict)
    payload = _payload(completion)
    if payload is None:
        return _fail(labels, "no_marker", completion.strip()[-120:], strict)
    if DECIMAL.search(payload):
        return _fail(labels, "not_integer", payload[:120], strict)

    pairs = PAIR.findall(payload)
    if pairs:
        return _from_pairs(pairs, labels, payload, strict)
    return _from_positions(payload, labels, strict)


def _from_pairs(pairs, labels, payload, strict):
    lookup = {}
    for label in labels:
        lookup[_normalise(label)] = label
        digits = re.findall(r"\d+", label)
        if digits:
            lookup.setdefault(digits[-1], label)
    seen = {}
    for key, value in pairs:
        target = lookup.get(_normalise(key))
        if target is None:
            return _fail(labels, "label_mismatch", f"unknown {key.strip()!r}", strict)
        if target in seen:
            return _fail(labels, "duplicate_label", target, strict)
        seen[target] = int(value)
    missing = [label for label in labels if label not in seen]
    if missing:
        return _fail(labels, "label_mismatch", f"missing {', '.join(missing)}", strict)
    return _checked(seen, labels, "labelled", payload, strict)


def _from_positions(payload, labels, strict):
    first_line = payload.split("\n")[0]
    numbers = NUMBER.findall(first_line)
    if not numbers:
        return _fail(labels, "no_numbers", payload[:120], strict)
    if len(numbers) != len(labels):
        detail = f"{len(numbers)} numbers for {len(labels)} players"
        return _fail(labels, "wrong_count", detail, strict)
    values = {label: int(n) for label, n in zip(labels, numbers)}
    return _checked(values, labels, "positional", first_line, strict)


def _checked(values, labels, form, payload, strict):
    bad = [f"{k}={v}" for k, v in values.items() if not 0 <= v <= MAX_PUNISHMENT]
    if bad:
        return _fail(labels, "out_of_range", ", ".join(bad), strict)
    ordered = {label: values[label] for label in labels}
    return ParseResult(ok=True, punishments=ordered, form=form, detail=payload[:120])


def mark_wasted(result, record):
    """Attach the labels punished above zero that the game cannot charge.

    Called with the `RoundRecord` being decided. Returns a new `ParseResult`;
    the punishments are left as the model gave them, because the environment
    is what zeroes them and the log should show what the model tried to do.
    """
    no_input = {p.label for p in record.players if not p.gave_input}
    wasted = tuple(
        label
        for label, value in result.punishments.items()
        if label in no_input and value > 0
    )
    return ParseResult(
        ok=result.ok,
        punishments=result.punishments,
        reason=result.reason,
        form=result.form,
        detail=result.detail,
        wasted=wasted,
        fallback=result.fallback,
    )


def enforce(result, record):
    """What the environment will actually charge: zero on no-input players."""
    no_input = {p.label for p in record.players if not p.gave_input}
    return {
        label: (0 if label in no_input else value)
        for label, value in result.punishments.items()
    }


def summarise(results):
    """The reported numbers, computed in one place.

    `failure_rate` is per ANSWER, not per player: a failure is a failure of
    the whole answer and counting it per player would weight it by group size.
    """
    results = list(results)
    total = len(results)
    failures = [r for r in results if not r.ok]
    summary = {
        "answers": total,
        "failures": len(failures),
        "failure_rate": (len(failures) / total) if total else float("nan"),
        "zero_fallback_answers": sum(1 for r in failures if r.fallback == "zero"),
    }
    for reason in REASONS:
        summary[f"reason[{reason}]"] = sum(1 for r in failures if r.reason == reason)
    for form in ("labelled", "positional"):
        summary[f"form[{form}]"] = sum(1 for r in results if r.form == form)
    return summary


__all__ = [
    "REASONS",
    "ParseFailure",
    "ParseResult",
    "enforce",
    "mark_wasted",
    "parse_punishments",
    "summarise",
]
