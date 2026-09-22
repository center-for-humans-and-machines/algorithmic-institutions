"""The replay battery's own functions. Runs locally (no PyG).

These are what the committed tables are made of, so the cases that matter are
the ones where a naive implementation would report a number instead of
refusing: a flat profile has no targeting direction, and a timed-out player
has no contribution to bin.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data_analysis"))

from llm_prompt_replay import (  # noqa: E402
    bin_profile,
    make_client,
    markdown,
    profile_snr,
    tie_structure,
)


def frame(rows):
    """rows: (episode_id, contribution, contribution_valid, punishment)."""
    return pd.DataFrame(
        [
            {
                "episode_id": e,
                "contribution": c,
                "contribution_valid": v,
                "charged_punishment": p,
            }
            for e, c, v, p in rows
        ]
    )


def test_bin_profile_drops_the_imputed_contribution():
    """The simulation stores 9 on a timed-out player and the CSV stores 0;
    either would land in a bin the player never produced."""
    df = frame([(0, 0, True, 5), (0, 9, False, 0), (0, 20, True, 0)])
    means, counts = bin_profile(df)
    assert counts["6-10"] == 0  # the imputed 9 did not get counted
    assert counts["{0}"] == 1 and counts["{20}"] == 1
    assert means["{0}"] == 5 and means["{20}"] == 0


def test_bin_profile_uses_the_evaluation_suites_edges():
    df = frame([(0, c, True, 0) for c in range(21)])
    _, counts = bin_profile(df)
    assert counts.tolist() == [1, 5, 5, 5, 4, 1]


def test_profile_snr_refuses_a_flat_profile():
    """Zero, not infinity: there is nothing here to rank and the gate says so
    rather than reporting an unbounded signal-to-noise."""
    df = frame([(e, c, True, 3) for e in range(4) for c in (0, 10, 20)])
    assert profile_snr(df) == 0.0


def test_profile_snr_rises_with_a_real_spread():
    rows = [(e, c, True, 10 - c // 2) for e in range(6) for c in (0, 10, 20)]
    assert profile_snr(frame(rows)) > 2


def test_profile_snr_needs_more_than_one_episode():
    assert np.isnan(profile_snr(frame([(0, 0, True, 1), (0, 20, True, 0)])))


def test_tie_structure_counts_distinct_bins():
    assert tie_structure([1.0] * 6) == {"distinct_bins": 1, "min_neighbour_gap": 0.0}
    structure = tie_structure([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    assert structure["distinct_bins"] == 6
    assert structure["min_neighbour_gap"] == pytest.approx(1.0)


def test_tie_structure_ignores_empty_bins():
    assert tie_structure([2.0, float("nan"), 1.0])["distinct_bins"] == 2


def test_markdown_renders_without_tabulate():
    table = pd.DataFrame([{"source": "a", "x": 1.5}, {"source": "b", "x": None}])
    text = markdown(table)
    assert text.splitlines()[0] == "| source | x |"
    assert "| a | 1.500 |" in text
    assert "| b |  |" in text


def test_the_cached_client_refuses_rather_than_inventing_an_answer():
    client = make_client("cached", np.random.default_rng(0))

    class Point:
        key = (1, 0, 3)

    with pytest.raises(KeyError):
        client([], decision=Point())
