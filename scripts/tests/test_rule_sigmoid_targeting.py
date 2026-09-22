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

from aggregate import (  # noqa: E402
    spearman_from_counts,
    targeting_triple_from_arrays,
)


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


# --------------------------------------------------------------------- #
# The other half: rank alone is blind to flatness, so magnitude and a
# noise gate are reported with it.
# --------------------------------------------------------------------- #


def _blocks(profile, per_bin=40, blocks=50, noise=0.0, seed=0):
    """Per-block punishment sums and counts for a given profile of bin means."""
    rng = np.random.default_rng(seed)
    den = np.full((blocks, len(profile)), float(per_bin))
    mu = np.asarray(profile, dtype=float)[None, :]
    num = den * (mu + rng.normal(0, noise, size=den.shape))
    return num, den


def test_a_flat_profile_is_caught_by_magnitude_not_by_rank():
    """5.00 down to 4.99 ranks the same as 4.76 down to 0.27."""
    flat = targeting_triple_from_arrays(*_blocks([5.00, 5.00, 4.995, 4.99, 4.99, 4.99]))
    steep = targeting_triple_from_arrays(*_blocks([4.76, 3.9, 2.5, 1.2, 0.5, 0.27]))
    assert flat["magnitude"] < 0.01
    assert steep["magnitude"] > 1.0


def test_the_noise_gate_rejects_noise_ranked():
    """A genuinely flat policy plus sampling noise must not pass the gate,
    however confidently its rank correlation comes out."""
    noise_only = targeting_triple_from_arrays(*_blocks([3.0] * 6, noise=0.5, seed=1))
    real = targeting_triple_from_arrays(
        *_blocks([5.0, 4.0, 3.0, 2.0, 1.0, 0.2], noise=0.5, seed=1)
    )
    assert noise_only["noise_gate"] < 6
    assert real["noise_gate"] > 40


def test_magnitude_is_a_relative_spread_not_an_absolute_one():
    """Same shape, a third of the force: the flatness measure must agree,
    because the rule aims the same way -- that is what `mean_p` is for."""
    a = targeting_triple_from_arrays(*_blocks([6.0, 4.0, 2.0]))
    b = targeting_triple_from_arrays(*_blocks([2.0, 4.0 / 3, 2.0 / 3]))
    assert abs(a["magnitude"] - b["magnitude"]) < 1e-9
    assert a["bin_mean_range"] > b["bin_mean_range"]


def test_a_bin_below_the_count_floor_is_dropped():
    num, den = _blocks([5.0, 1.0, 0.0], per_bin=40, blocks=2)
    den[:, 2] = 1.0  # 2 decisions pooled, below the floor
    num[:, 2] = 0.0
    out = targeting_triple_from_arrays(num, den, min_bin_n=20)
    assert out["n_bins"] == 2


def test_a_deterministic_rule_has_an_unbounded_gate_not_a_missing_one():
    """A hard threshold's extreme bins are the same number in every episode,
    so their sampling error is exactly zero. That is maximal evidence, not
    absent evidence."""
    num, den = _blocks([10.0, 10.0, 0.0], per_bin=40, blocks=20)
    out = targeting_triple_from_arrays(num, den)
    assert np.isinf(out["noise_gate"])
    assert out["magnitude"] > 1


def test_a_rule_with_no_spread_at_all_has_no_gate():
    out = targeting_triple_from_arrays(*_blocks([3.0, 3.0, 3.0]))
    assert np.isnan(out["noise_gate"])
    assert out["magnitude"] == 0.0
