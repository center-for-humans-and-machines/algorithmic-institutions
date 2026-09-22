"""The level-versus-shape trap, made into tests. Runs locally (no PyG).

`scripts/rl_param_noise/targeting.py` exists because a difference of bin means
moves when a manager simply punishes with less force, and "it targets
differently" and "it punishes less hard" are the two things this arm most needs
to keep apart.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rl_param_noise.targeting import (  # noqa: E402
    average_ranks,
    targeting,
    weighted_pearson,
)

# experiments/2group_8agent_50ep.csv through the evaluation suite's RPA bins,
# as scripts/rl_two_worlds/measure.py prints them.
HUMAN_MEANS = [4.755, 2.973, 1.672, 0.978, 0.692, 0.267]
HUMAN_COUNTS = [809, 1955, 2614, 1794, 510, 1232]

# rl_new_clones_s42 as reported by the coordinator: monotone the wrong way.
INVERTED_MEANS = [0.08, 1.0, 2.0, 3.0, 4.0, 5.00]
INVERTED_COUNTS = [1000, 1500, 3000, 2000, 1000, 2500]


def test_the_trap_force_moves_contrast_but_not_the_rank_correlation():
    """Same aim, a third of the force. The contrast falls by a factor of
    three; the rank correlation does not move at all."""
    full = targeting(HUMAN_MEANS, HUMAN_COUNTS)
    weak = targeting([m / 3.0 for m in HUMAN_MEANS], HUMAN_COUNTS)

    assert weak["contrast"] == pytest.approx(full["contrast"] / 3.0)
    assert weak["rho_contribution_punishment"] == pytest.approx(
        full["rho_contribution_punishment"]
    )
    assert weak["contrast_over_mean"] == pytest.approx(full["contrast_over_mean"])


def test_the_rank_correlation_survives_any_monotone_rescaling():
    """Not just multiplication: `rho` depends on the ORDER of the bin means
    and nothing else, so no reweighting of force can move it."""
    base = targeting(HUMAN_MEANS, HUMAN_COUNTS)
    squashed = targeting([m**0.3 + 7.0 for m in HUMAN_MEANS], HUMAN_COUNTS)
    assert squashed["rho_contribution_punishment"] == pytest.approx(
        base["rho_contribution_punishment"]
    )
    # ... while contrast_over_mean, which is only invariant to multiplication,
    # is allowed to move and does. That is the difference between the two.
    assert squashed["contrast_over_mean"] != pytest.approx(
        base["contrast_over_mean"], rel=0.05
    )


def test_human_sign_is_negative_and_inverted_is_positive():
    assert targeting(HUMAN_MEANS, HUMAN_COUNTS)[
        "rho_contribution_punishment"
    ] == pytest.approx(-1.0)
    assert targeting(INVERTED_MEANS, INVERTED_COUNTS)[
        "rho_contribution_punishment"
    ] == pytest.approx(1.0)


def test_a_flat_profile_has_no_targeting_direction():
    """Both guard pilots' evaluated policies are the constant action 5. NaN is
    the honest answer; 0.0 would claim a measured absence of relationship."""
    out = targeting([5.0] * 6, [1000] * 6)
    assert out["rho_contribution_punishment"] != out["rho_contribution_punishment"]
    assert out["contrast"] == 0.0


def test_rho_carries_no_magnitude_which_is_why_it_is_never_read_alone():
    barely = targeting([5.00, 4.998, 4.996, 4.994, 4.992, 4.99], [1000] * 6)
    assert barely["rho_contribution_punishment"] == pytest.approx(-1.0)
    assert barely["contrast"] == pytest.approx(0.01)


def test_counts_weight_the_correlation():
    """A bin standing for three agent-rounds must not count as much as one
    standing for three hundred thousand."""
    means = [0.0, 1.0, 2.0, 3.0, 4.0, 100.0]
    heavy = targeting(means, [100000, 100000, 100000, 100000, 100000, 100000])
    light = targeting(means, [100000, 100000, 100000, 100000, 100000, 1])
    assert heavy["mean_punishment"] > light["mean_punishment"]
    # Order is unchanged, so rho is unchanged -- weights move the means, not
    # the ranking.
    assert heavy["rho_contribution_punishment"] == pytest.approx(
        light["rho_contribution_punishment"]
    )


def test_empty_and_degenerate_bins_drop_out():
    out = targeting(
        [4.0, float("nan"), 2.0, 1.0, 0.5, 0.2], [100, 0, 100, 100, 100, 100]
    )
    assert out["rho_contribution_punishment"] == pytest.approx(-1.0)
    assert out["contrast"] == pytest.approx(3.8)


def test_average_ranks_merges_ties_across_groups():
    # Two groups of 2 tied at value 1, then a group of 6 at value 2:
    # the first four observations share rank (4 + 1) / 2 = 2.5.
    ranks = average_ranks([1.0, 1.0, 2.0], [2, 2, 6])
    assert ranks[0] == pytest.approx(2.5)
    assert ranks[1] == pytest.approx(2.5)
    assert ranks[2] == pytest.approx(4 + (6 + 1) / 2)


def test_weighted_pearson_is_nan_on_a_constant():
    assert weighted_pearson([1, 2, 3], [5, 5, 5], [1, 1, 1]) != weighted_pearson(
        [1, 2, 3], [5, 5, 5], [1, 1, 1]
    )
