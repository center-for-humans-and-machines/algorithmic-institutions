"""The trace the language-model manager reads. Runs locally (no PyG).

The tests that matter are the three the format exists for: an imputed
contribution never reaches the model, the pool line is the game's own
accounting identity rather than a re-derivation of it, and the round being
decided is rendered by the same templates as the history rather than by a
second code path.
"""

from pathlib import Path

import pandas as pd
import pytest

from aimanager.llm.trace import (
    TEMPLATES,
    Event,
    PlayerRound,
    RoundRecord,
    TraceRenderer,
    build_records,
    format_round,
    format_trace,
    label_for,
    punishable,
    sanitise,
)

REPO = Path(__file__).resolve().parents[3]


def record(round_number=0, **kwargs):
    players = kwargs.pop(
        "players",
        (PlayerRound("Player 1", 10, 0), PlayerRound("Player 2", 2, 5)),
    )
    return RoundRecord(round_number=round_number, players=players, **kwargs)


def test_no_input_player_shows_no_number():
    """The one thing the format must never do: print the filler."""
    for filler in (0, 9, 20):
        player = PlayerRound.from_masked("Player 3", filler, False)
        assert player.contribution is None
        text = format_round(record(players=(player,)), show_pool=False)
        assert "gave no input" in text
        assert str(filler) not in text


def test_no_input_player_is_not_charged_to_the_pool():
    """A punishment on a timed-out player is discarded by the env, so it must
    not appear in the pool arithmetic either (the free-punishment fix)."""
    players = (
        PlayerRound("Player 1", 10, 0),
        PlayerRound.from_masked("Player 2", 9, False, punishment=30),
    )
    r = record(players=players)
    assert (r.contributed, r.punished) == (10, 0)
    assert r.pool == pytest.approx(16.0)


def test_pool_matches_the_human_accounting_identity():
    """pool == 1.6 * contributed - punished, on real group-rounds."""
    df = pd.read_csv(REPO / "experiments/2group_8agent_50ep.csv")
    df = df[df["episode_id"] == df["episode_id"].min()]
    assert len(df) > 0
    for _, sub in df.groupby(["round_number", "group_id"]):
        players = tuple(
            PlayerRound.from_masked(
                label_for(i),
                row.contribution,
                not bool(row.player_no_input),
                punishment=int(row.punishment),
            )
            for i, row in enumerate(sub.itertuples())
        )
        expected = sub["common_good"].iloc[0]
        assert RoundRecord(0, players).pool == pytest.approx(expected)


def test_round_numbers_are_one_based_in_the_text():
    assert format_round(record(round_number=0)).startswith("Round 1")
    assert format_round(record(round_number=23)).startswith("Round 24")


def test_history_and_current_round_share_one_renderer():
    """The current round must differ by an explicit marker, not by a missing
    clause: an asymmetry that is only an absence is the cheap way to leak."""
    renderer = TraceRenderer()
    decided = record(0)
    pending = RoundRecord(
        1, tuple(PlayerRound(p.label, p.contribution) for p in decided.players)
    )
    kinds = {e.kind for e in renderer.events([decided])}
    pending_kinds = {e.kind for e in renderer.events([pending])}
    assert kinds == {"round", "played", "pool"}
    assert pending_kinds == {"round", "pending", "pool_pending"}
    text = renderer.round(pending)
    assert "punishment not set yet" in text
    assert "you punished" not in text
    # every rendered line comes from a template, none is hand-built
    for line in renderer.round(decided).split("\n"):
        assert any(line.startswith(t.split("{")[0]) for t in TEMPLATES.values())


def test_current_round_pool_is_before_punishment():
    r = record(players=(PlayerRound("Player 1", 10), PlayerRound("Player 2", 5)))
    assert "Pool before your punishment: 24" in format_round(r)


def test_show_pool_toggle_removes_every_pool_line():
    records = [record(0), record(1)]
    assert "Pool" in format_trace(records, show_pool=True)
    assert "Pool" not in format_trace(records, show_pool=False)


def test_renderer_spec_changes_with_the_templates():
    """A version fingerprint folds in this spec, so an edited template must
    move it -- that is what stops a version being edited in place."""
    other = TraceRenderer(templates={**TEMPLATES, "played": "  {label} x"})
    assert TraceRenderer().spec != other.spec
    assert TraceRenderer(show_pool=False).spec != TraceRenderer().spec


def test_sanitise_strips_newlines_and_quotes():
    assert sanitise('a\nb"c\r') == "a b'c"
    assert Event.make("round", round="x\ny").mapping["round"] == "x y"


def test_empty_trace_says_so_rather_than_rendering_nothing():
    assert "first round" in format_trace([])


def test_build_records_derives_joined_and_left():
    rows = [
        (0, "a", 5, True, 0, False),
        (0, "b", 5, True, 0, False),
        (1, "a", 5, True, 0, False),
        (1, "c", 5, True, None, False),
    ]
    labels = {"a": "Player 1", "b": "Player 2", "c": "Player 3"}
    records = build_records(rows, labels)
    assert records[0].left == ()
    assert not any(p.joined for p in records[0].players)
    assert records[1].left == ("Player 2",)
    assert [p.joined for p in records[1].players] == [False, True]
    text = format_round(records[1])
    assert "Player 3 joined your group" in text
    assert "Player 2 left your group" in text


def test_players_are_ordered_by_arrival_not_by_label_text():
    """'Player 10' must not sort before 'Player 2'."""
    keys = "abcdefghijk"
    labels = {k: label_for(i) for i, k in enumerate(keys)}
    rows = [(0, k, 1, True, 0, False) for k in reversed(keys)]
    records = build_records(rows, labels)
    assert [p.label for p in records[0].players] == [label_for(i) for i in range(11)]


def test_manager_timeout_round_is_marked():
    rows = [(0, "a", 5, True, 0, True)]
    text = format_round(build_records(rows, {"a": "Player 1"})[0])
    assert "you gave no input this round" in text


def test_punishable_excludes_no_input_players():
    players = (
        PlayerRound("Player 1", 10),
        PlayerRound.from_masked("Player 2", 9, False),
    )
    assert punishable(RoundRecord(0, players)) == ["Player 1"]
