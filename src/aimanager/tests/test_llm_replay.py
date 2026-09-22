"""The replay harness over the 50 real human games. Runs locally (no PyG).

These pin the facts about the human data that the harness depends on, so a
change to the CSV or to the dedup rule fails here rather than quietly moving
every number in the prompt comparison.
"""

from pathlib import Path

import pytest

from aimanager.llm.replay import (
    HUMAN_DATA_FILE,
    StubClient,
    decisions,
    human_frame,
    load_games,
    run_replay,
)

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def games():
    return load_games(REPO / HUMAN_DATA_FILE)


@pytest.fixture(scope="module")
def points(games):
    return decisions(games)


def test_one_copy_per_game(games):
    """The flip augmentation doubles every game; the evaluation suite and the
    linear pipeline keep one copy and so does this."""
    assert games["episode_id"].nunique() == 50
    assert len(games) == 50 * 24 * 8


def test_decision_points_exclude_manager_timeouts(games, points):
    group_rounds = games.groupby(["episode_id", "round_number", "group_id"]).first()
    timeouts = int(group_rounds["manager_no_input"].sum())
    assert timeouts == 104
    assert len(points) == len(group_rounds) - timeouts == 2152
    assert {p.round_number for p in points} == set(range(24))


def test_a_decision_point_carries_the_trace_up_to_its_round(points):
    point = next(p for p in points if p.round_number == 7)
    assert [r.round_number for r in point.records] == list(range(8))
    assert all(r.decided for r in point.records[:-1])
    assert not point.records[-1].decided
    assert set(point.human) == set(point.contribution)


def test_no_input_players_are_masked_at_source(points):
    """The CSV stores 0 on a timed-out player and the simulation stores 9;
    both are dropped before anything is rendered."""
    masked = [p for p in points if any(c is None for c in p.contribution.values())]
    assert masked, "the human data has timed-out players"
    for point in masked[:20]:
        for player in point.records[-1].players:
            if player.contribution is None:
                assert not player.gave_input


def test_group_membership_follows_the_reshuffle(points):
    """A manager's group is four at round 0 and can be anything later."""
    first = [p for p in points if p.round_number == 0]
    assert {len(p.records[-1].players) for p in first} == {4}
    later = {len(p.records[-1].players) for p in points if p.round_number > 3}
    assert later - {4}


def test_stub_round_trip_recovers_the_policy(points):
    """The stub answers through the real prompt and the real parser, so a
    match here means the whole path is lossless."""
    sample = points[:40]

    def policy(decision):
        return {
            p.label: (0 if not p.gave_input else min(30, 20 - p.contribution))
            for p in decision.target.players
        }

    frame = run_replay(sample, StubClient(policy), version="v3_explicit_pool")
    assert frame["parse_ok"].all()
    assert (frame["prompt_version"] == "v3_explicit_pool").all()
    valid = frame[frame["contribution_valid"]]
    assert (valid["model_punishment"] == 20 - valid["contribution"]).all()
    assert (frame.loc[~frame["contribution_valid"], "charged_punishment"] == 0).all()


def test_a_failure_falls_back_to_zero_and_is_flagged(points):
    corrupt = StubClient(
        lambda d: {p.label: 5 for p in d.target.players},
        corrupt=lambda i: "no answer here" if i % 2 == 0 else None,
    )
    frame = run_replay(points[:10], corrupt, version="v3_explicit_pool")
    per_decision = frame.groupby(
        ["episode_id", "group_id", "round_number"], as_index=False
    ).first()
    assert (~per_decision["parse_ok"]).sum() == 5
    failed = frame[~frame["parse_ok"]]
    assert (failed["model_punishment"] == 0).all()
    assert set(failed["parse_reason"]) == {"no_marker"}


def test_cache_replaces_the_client(tmp_path, points):
    cache = tmp_path / "completions.jsonl"
    sample = points[:5]
    client = StubClient(lambda d: {p.label: 1 for p in d.target.players})
    first = run_replay(sample, client, version="v3_explicit_pool", cache_path=cache)
    assert len(client.calls) == 5

    def refuse(messages, decision=None, **_):
        raise AssertionError("cache should have answered")

    second = run_replay(sample, refuse, version="v3_explicit_pool", cache_path=cache)
    assert first["model_punishment"].tolist() == second["model_punishment"].tolist()


def test_the_frame_carries_one_parse_summary(points):
    """The failure rate has exactly one computation site; a report reads it
    from here rather than recomputing a second version of it."""
    corrupt = StubClient(
        lambda d: {p.label: 5 for p in d.target.players},
        corrupt=lambda i: "" if i < 3 else None,
    )
    frame = run_replay(points[:10], corrupt, version="v3_explicit_pool")
    summary = frame.attrs["parse_summary"]
    assert summary["answers"] == 10
    assert summary["failures"] == summary["zero_fallback_answers"] == 3
    assert summary["failure_rate"] == pytest.approx(0.3)
    assert summary["reason[empty]"] == 3


def test_human_frame_matches_the_decision_points(points):
    frame = human_frame(points[:50])
    assert (frame["model_punishment"] == frame["human_punishment"]).all()
    assert frame["parse_ok"].all()
