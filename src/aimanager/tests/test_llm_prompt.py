"""Prompt versions: immutable, complete, and free of hints. Runs locally.

The fingerprints are pinned. That is the point of the file: a version is
never edited in place, so a wording change has to fail here and be given a
new name, and a result that quotes a fingerprint stays identifiable.
"""

import pytest

from aimanager.llm.prompt import (
    COSTS,
    OBJECTIVE,
    PROMPT_VERSIONS,
    build_prompt,
    resolve,
)
from aimanager.llm.trace import PlayerRound, RoundRecord

FINGERPRINTS = {
    "v1_bare_pool": "5d40be831fdd",
    "v2_stated_pool": "59c1fc1dbd95",
    "v3_explicit_pool": "b211d81b411a",
    "v4_explicit_nopool": "9e1fb13d7a59",
    "v5_explicit_reason": "0edc2e953d18",
}

RECORDS = [
    RoundRecord(0, (PlayerRound("Player 1", 10, 0), PlayerRound("Player 2", 2, 5))),
    RoundRecord(
        1,
        (
            PlayerRound("Player 1", 4),
            PlayerRound.from_masked("Player 2", 9, False),
        ),
    ),
]


@pytest.mark.parametrize("name, fingerprint", FINGERPRINTS.items())
def test_fingerprints_are_pinned(name, fingerprint):
    assert PROMPT_VERSIONS[name].fingerprint == fingerprint


def test_every_version_has_a_distinct_fingerprint():
    prints = [v.fingerprint for v in PROMPT_VERSIONS.values()]
    assert len(set(prints)) == len(prints)


def test_the_objective_is_the_undivided_pool_and_is_not_an_axis():
    """The maintainer settled it: the sum, not the pool per member. One
    objective statement, shared by every version."""
    assert "divided by the number of members" not in OBJECTIVE
    assert "per member" not in OBJECTIVE
    for name in PROMPT_VERSIONS:
        assert OBJECTIVE in build_prompt(RECORDS, name).user
    assert resolve("v3_explicit_pool") is PROMPT_VERSIONS["v3_explicit_pool"]


@pytest.mark.parametrize("name", list(PROMPT_VERSIONS))
def test_every_version_states_the_rules_and_the_objective(name):
    text = build_prompt(RECORDS, name).user
    for required in ("24 rounds", "1.6", "0 to 20", "0 to 30", "every fourth round"):
        assert required in text, required
    assert "total of your group's common pool over the whole 24 rounds" in text
    assert "cannot be punished" in text


@pytest.mark.parametrize("name", list(PROMPT_VERSIONS))
def test_no_version_hints_at_whom_to_punish(name):
    """The framing is the game. A hint here would make the result about the
    hint; if a model needs one, that is a finding for the log, not a patch."""
    text = build_prompt(RECORDS, name).user.lower()
    for forbidden in (
        "free rider",
        "free-rider",
        "low contributor",
        "threshold",
        "punish those",
        "punish players who",
        "should punish",
        "for example",
    ):
        assert forbidden not in text, forbidden


@pytest.mark.parametrize("name", list(PROMPT_VERSIONS))
def test_no_version_says_the_contributors_are_models(name):
    text = build_prompt(RECORDS, name).user.lower()
    for forbidden in ("model", "simulat", "artificial", "agent", "ai "):
        assert forbidden not in text, forbidden


def test_the_cost_ladder_is_a_ladder():
    lengths = [len(COSTS[c]) for c in ("bare", "stated", "explicit")]
    assert lengths[0] == 0 < lengths[1] < lengths[2]
    assert COSTS["stated"] in COSTS["explicit"].replace(" A punishment", "")
    bare, stated, explicit = (
        build_prompt(RECORDS, v).user
        for v in ("v1_bare_pool", "v2_stated_pool", "v3_explicit_pool")
    )
    assert "at full price" not in bare
    assert "at full price" in stated
    assert "pays for itself only through" not in stated
    assert "pays for itself only through" in explicit


def test_pool_axis_only_changes_the_trace():
    with_pool = build_prompt(RECORDS, "v3_explicit_pool").user
    without = build_prompt(RECORDS, "v4_explicit_nopool").user
    assert "Pool:" in with_pool and "Pool:" not in without
    assert "Player 1 put in 10, you punished 0" in with_pool
    assert "Player 1 put in 10, you punished 0" in without


def test_reason_first_still_demands_the_marker_last():
    text = build_prompt(RECORDS, "v5_explicit_reason").user
    assert "three short sentences" in text
    assert "as the last line of your reply" in text
    assert text.count("PUNISHMENT:") == 1


def test_the_answer_block_covers_every_listed_player():
    prompt = build_prompt(RECORDS, "v3_explicit_pool")
    assert prompt.labels == ("Player 1", "Player 2")
    for label in prompt.labels:
        assert f"{label} = <number>" in prompt.user


def test_the_no_input_player_is_still_asked_for():
    """The environment discards it, but a wasted decision has to be visible in
    the log rather than silently absent from the answer."""
    prompt = build_prompt(RECORDS, "v3_explicit_pool")
    assert "including any who gave no input" in prompt.user
    assert "Player 2" in prompt.labels


def test_the_prompt_never_shows_an_imputed_contribution():
    prompt = build_prompt(RECORDS, "v3_explicit_pool")
    trace = prompt.user.split("THE RECORD SO FAR")[1].split("YOUR DECISION")[0]
    current = trace.strip().split("\n\n")[-1]
    assert current.startswith("Round 2")
    assert "Player 2 gave no input" in current
    assert "Player 2 put in" not in current
    # the filler the sources carry (0 in the CSV, 9 in the simulation)
    assert "Player 2 put in 9" not in trace


def test_messages_carry_the_system_text():
    prompt = build_prompt(RECORDS, "v3_explicit_pool")
    assert prompt.messages[0]["role"] == "system"
    assert prompt.messages[1]["content"] == prompt.user


def test_no_version_discourages_or_encourages_punishment():
    """The prompt states the accounting and stops. Either a discouraging word
    or a claim that punishing works would make a good result uninterpretable."""
    for name in PROMPT_VERSIONS:
        text = build_prompt(RECORDS, name).user.lower()
        for forbidden in (
            "not free",
            "expensive",
            "costly",
            "avoid punish",
            "punishment works",
            "deters",
            "deterrent",
            "be careful",
            "sparingly",
        ):
            assert forbidden not in text, (name, forbidden)
