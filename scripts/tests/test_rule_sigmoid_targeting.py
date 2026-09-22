"""The targeting statistic must measure aim, not force.

A sibling arm's largest apparent shape difference, -11.1 on a difference of
bin means, turned out to be entirely a level artefact: mean punishment down
to a third, punish rate up threefold, the actual contribution-to-punishment
relationship unchanged to the third decimal. These tests pin the property
that would have caught it -- the statistic does not move when a rule
punishes the same people harder.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rule_sigmoid"))

from aggregate import spearman_from_counts  # noqa: E402


def _table(contribution, punishment):
    tab = np.zeros((21, 31))
    np.add.at(tab, (np.asarray(contribution), np.asarray(punishment)), 1)
    return tab


def test_matches_scipy_with_ties():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    c = rng.integers(0, 21, 4000)
    p = np.clip(20 - c + rng.integers(-6, 7, 4000), 0, 30)
    want = scipy_stats.spearmanr(c, p).statistic
    assert abs(spearman_from_counts(_table(c, p)) - want) < 1e-10


def test_invariant_to_how_hard_the_rule_punishes():
    """Same aim, three times the force -- the statistic must not move."""
    rng = np.random.default_rng(1)
    c = rng.integers(0, 21, 3000)
    p = np.clip((20 - c) // 3, 0, 30)
    gentle = spearman_from_counts(_table(c, p))
    fierce = spearman_from_counts(_table(c, np.clip(p * 3, 0, 30)))
    assert abs(gentle - fierce) < 1e-9


def test_invariant_to_any_strictly_monotone_rescaling():
    """Not just a rescaling by a constant: any strictly increasing map.

    Strictly increasing is the condition, and it is the honest one. A map
    that saturates at the top of the action space merges levels that were
    distinct, which really is a different policy, and the statistic is
    supposed to notice that.
    """
    rng = np.random.default_rng(2)
    c = rng.integers(0, 21, 3000)
    p = np.clip(20 - c, 0, 10)
    lut = np.array([0, 1, 3, 4, 7, 9, 12, 16, 19, 25, 30])
    assert (
        abs(
            spearman_from_counts(_table(c, p)) - spearman_from_counts(_table(c, lut[p]))
        )
        < 1e-9
    )


def test_sign_says_which_way_a_rule_aims():
    c = np.arange(21)
    correct = spearman_from_counts(_table(c, np.clip(20 - c, 0, 30)))
    inverted = spearman_from_counts(_table(c, c))
    assert correct == pytest.approx(-1.0)
    assert inverted == pytest.approx(1.0)


def test_a_rule_that_never_punishes_has_no_aim():
    """No variance in the punishment, so nan rather than a spurious 0."""
    c = np.arange(21)
    assert np.isnan(spearman_from_counts(_table(c, np.zeros_like(c))))
