"""The answer parser, as a guard. Runs locally (no PyG).

On a served endpoint the answer should be constrained at the token level, so
these are the cases that reach the guard when the constraint is missing or
misconfigured. The fallback they all share is zero punishment, which is not a
neutral default in this game -- it is the policy every collapsed learned
manager converged on -- so the tests pin that a failure is visible in the
result and in the log, never absorbed.
"""

import logging

import pytest

from aimanager.llm.parse import (
    ParseFailure,
    enforce,
    mark_wasted,
    parse_punishments,
    summarise,
)
from aimanager.llm.trace import PlayerRound, RoundRecord

LABELS = ["Player 1", "Player 2", "Player 3", "Player 4"]
GOOD = "PUNISHMENT: Player 1 = 3, Player 2 = 0, Player 3 = 12, Player 4 = 0"


def test_the_requested_format_parses():
    result = parse_punishments(GOOD, LABELS)
    assert result.ok and result.form == "labelled"
    assert result.values == [3, 0, 12, 0]
    assert result.fallback is None


def test_order_comes_from_the_labels_not_the_answer():
    shuffled = "PUNISHMENT: Player 4 = 1, Player 2 = 2, Player 3 = 3, Player 1 = 4"
    result = parse_punishments(shuffled, LABELS)
    assert result.ok
    assert result.punishments == {
        "Player 1": 4,
        "Player 2": 2,
        "Player 3": 3,
        "Player 4": 1,
    }


@pytest.mark.parametrize(
    "completion",
    [
        "**PUNISHMENT:** Player 1 = 3, Player 2 = 0, Player 3 = 12, Player 4 = 0",
        "Here is my answer.\n\n" + GOOD,
        "I weigh the cost.\n" + GOOD + "\n",
        "PUNISHMENT: 1=3, 2=0, 3=12, 4=0",
        "PUNISHMENT: Player 1: 3, Player 2: 0, Player 3: 12, Player 4: 0",
    ],
)
def test_tolerated_surface_variation(completion):
    """Decoration, preamble and the label short form are surface, not content:
    reading them is determined, not guessed."""
    assert parse_punishments(completion, LABELS).ok


def test_reasoning_before_the_answer_uses_the_last_marker():
    text = "PUNISHMENT: is what I must set.\n\nSo:\n\n" + GOOD
    result = parse_punishments(text, LABELS)
    assert result.ok and result.values == [3, 0, 12, 0]


def test_bare_positional_list_is_accepted_but_flagged():
    result = parse_punishments("PUNISHMENT: 3, 0, 12, 0", LABELS)
    assert result.ok and result.form == "positional"
    assert result.values == [3, 0, 12, 0]


@pytest.mark.parametrize(
    "completion, reason",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("I decline to punish anyone.", "no_marker"),
        ("PUNISHMENT: Player 1 = 3.5, Player 2 = 0", "not_integer"),
        ("PUNISHMENT: 3, 0, 12", "wrong_count"),
        ("PUNISHMENT: Player 1 = 3, Player 2 = 0", "label_mismatch"),
        ("PUNISHMENT: Player 9 = 3", "label_mismatch"),
        (
            "PUNISHMENT: Player 1 = 3, Player 1 = 0, Player 3 = 1, Player 4 = 1",
            "duplicate_label",
        ),
        (
            "PUNISHMENT: Player 1 = 31, Player 2 = 0, Player 3 = 0, Player 4 = 0",
            "out_of_range",
        ),
        (
            "PUNISHMENT: Player 1 = -1, Player 2 = 0, Player 3 = 0, Player 4 = 0",
            "out_of_range",
        ),
        ("PUNISHMENT: none at all", "no_numbers"),
    ],
)
def test_failures_are_recorded_not_guessed(completion, reason):
    result = parse_punishments(completion, LABELS)
    assert not result.ok
    assert result.reason == reason
    assert result.values == [0, 0, 0, 0]
    assert result.fallback == "zero"


def test_out_of_range_is_not_clamped():
    """Clamping would turn a model that does not know the action space into
    one that scores well at the boundary."""
    result = parse_punishments("PUNISHMENT: 40, 0, 0, 0", LABELS)
    assert not result.ok and result.values == [0, 0, 0, 0]


def test_a_failure_is_logged(caplog):
    with caplog.at_level(logging.WARNING):
        parse_punishments("nothing", LABELS)
    assert "falling back to zero punishment" in caplog.text


def test_strict_mode_raises_instead_of_falling_back():
    with pytest.raises(ParseFailure):
        parse_punishments("nothing", LABELS, strict=True)


def decided_record():
    return RoundRecord(
        0,
        (
            PlayerRound("Player 1", 10),
            PlayerRound("Player 2", 2),
            PlayerRound.from_masked("Player 3", 9, False),
            PlayerRound("Player 4", 20),
        ),
    )


def test_wasted_punishment_is_recorded_not_zeroed():
    result = mark_wasted(parse_punishments(GOOD, LABELS), decided_record())
    assert result.wasted == ("Player 3",)
    assert result.punishments["Player 3"] == 12  # what the model asked for
    assert enforce(result, decided_record())["Player 3"] == 0  # what is charged


def test_enforce_leaves_punishable_players_alone():
    result = parse_punishments(GOOD, LABELS)
    charged = enforce(result, decided_record())
    assert [charged[label] for label in LABELS] == [3, 0, 0, 0]


def test_summarise_counts_per_answer():
    results = [
        parse_punishments(GOOD, LABELS),
        parse_punishments("PUNISHMENT: 3, 0, 12, 0", LABELS),
        parse_punishments("nope", LABELS),
    ]
    summary = summarise(results)
    assert summary["answers"] == 3
    assert summary["failures"] == 1
    assert summary["failure_rate"] == pytest.approx(1 / 3)
    assert summary["zero_fallback_answers"] == 1
    assert summary["reason[no_marker]"] == 1
    assert summary["form[labelled]"] == 1
    assert summary["form[positional]"] == 1
