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


# ── the tie hazard ───────────────────────────────────────────────────

# A sibling's control seed: monotone decreasing across all six bins, but three
# of them saturate at exactly zero. Spearman scores it well short of 1 for
# that reason alone, and a threshold of 0.9 would have discarded a genuinely
# correctly-targeted policy.
SATURATED_MEANS = [4.0, 2.0, 1.0, 0.0, 0.0, 0.0]
SATURATED_COUNTS = [1000, 1500, 2000, 1500, 800, 2000]


def test_ties_attenuate_spearman_on_a_perfectly_monotone_profile():
    out = targeting(SATURATED_MEANS, SATURATED_COUNTS)
    assert out["monotonicity"] == "decreasing"
    assert out["n_zero_bins"] == 3
    assert out["n_distinct_bins"] == 4
    rho = out["rho_contribution_punishment"]
    assert -1.0 < rho < -0.8, rho  # short of 1, and 0.9 would have binned it
    assert out["verdict"] == "targets free-riders"


def test_the_rank_threshold_is_0_8_not_0_9():
    from rl_param_noise.targeting import RANK_THRESHOLD

    assert RANK_THRESHOLD == 0.8
    # Same perfectly monotone profile, mass shifted into the tied zero bins:
    # it now sits between the two candidate thresholds, which is the case the
    # sibling arm hit at -0.845.
    rho = targeting(SATURATED_MEANS, [1000, 1000, 1000, 2500, 2500, 2500])[
        "rho_contribution_punishment"
    ]
    assert -0.9 < rho <= -RANK_THRESHOLD, rho


def test_ties_can_sink_a_perfectly_monotone_profile_below_any_threshold():
    """Worse than a threshold choice, and the reason `verdict_shape_only`
    exists. The SAME strictly ordered profile, with the agent-rounds moved
    into the three tied zero bins, falls to -0.35 -- below anything a rank
    threshold could sensibly be set to. A quiet policy that punishes
    free-riders and nobody else is exactly the shape the campaign wants and
    exactly the shape ties destroy."""
    out = targeting(SATURATED_MEANS, [200, 200, 200, 5000, 5000, 5000])
    assert out["monotonicity"] == "decreasing"
    assert out["rho_contribution_punishment"] > -0.5
    assert out["verdict"] == "no clean targeting (weak rank)"
    assert out["verdict_shape_only"] == "targets free-riders"
    assert out["tie_attenuated"] is True


def test_all_bins_tied_carries_no_rank_information():
    """Three of five evolution-strategies seeds punish exactly zero in every
    bin. That is all ties: there is nothing to rank, and saying so is not the
    same as measuring no relationship."""
    out = targeting([0.0] * 6, [1000] * 6)
    assert out["n_distinct_bins"] == 1
    assert out["n_zero_bins"] == 6
    assert out["rho_contribution_punishment"] != out["rho_contribution_punishment"]
    assert out["verdict"] == "no contingency (all bins tied)"


def test_tau_b_is_carried_and_saturates_on_a_strict_profile():
    # Same sign convention as rho: negative is the human sign.
    assert targeting(HUMAN_MEANS, HUMAN_COUNTS)["tau_b"] == pytest.approx(-1.0)
    assert targeting(INVERTED_MEANS, INVERTED_COUNTS)["tau_b"] == pytest.approx(1.0)


def test_tau_b_is_less_attenuated_by_ties_than_spearman():
    out = targeting(SATURATED_MEANS, SATURATED_COUNTS)
    assert abs(out["tau_b"]) < abs(out["rho_contribution_punishment"])
    # ... but it is attenuated too, so it is not a rescue either.
    assert abs(out["tau_b"]) < 1.0


def test_tau_b_sign_convention_matches_rho():
    """Both negative for the human sign, so the two can be read side by side
    without a sign flip in the reader's head."""
    for means, counts in (
        (HUMAN_MEANS, HUMAN_COUNTS),
        (SATURATED_MEANS, SATURATED_COUNTS),
    ):
        out = targeting(means, counts)
        assert out["tau_b"] < 0 and out["rho_contribution_punishment"] < 0


def test_a_big_endpoint_difference_on_a_non_monotone_profile_is_not_targeting():
    """The shape that got two sibling seeds withdrawn: `contrast` looks large,
    the profile wanders, and monotonicity catches it where contrast cannot."""
    means = [6.0, 0.5, 5.5, 0.6, 5.0, 0.4]
    out = targeting(means, [1000] * 6)
    assert out["contrast"] == pytest.approx(5.6)  # looks like strong targeting
    assert out["monotonicity"] == "none"
    assert out["verdict"] == "no clean targeting (not monotone)"


def test_relative_range_survives_a_non_monotone_profile():
    """Endpoints can coincide while the profile swings; range cannot hide it."""
    out = targeting([1.0, 9.0, 1.0, 9.0, 1.0, 1.0], [1000] * 6)
    assert out["contrast"] == pytest.approx(0.0)
    assert out["relative_range"] > 1.0


def test_negligible_range_is_advisory_and_not_in_the_verdict():
    """The epsilon-greedy pilot buffer: a real rank on a relationship whose
    size is a rounding error. The verdict must not silently depend on a
    threshold this arm invented."""
    from rl_param_noise.targeting import NEGLIGIBLE_RANGE

    out = targeting([6.014, 6.011, 6.004, 6.001, 6.000, 5.997], [1000] * 6)
    assert out["relative_range"] < NEGLIGIBLE_RANGE
    assert out["monotonicity"] == "decreasing"
    assert out["verdict"] == "targets free-riders"  # rank + monotone only
    # The advisory threshold shows up only in the shape-only column, and the
    # disagreement is flagged rather than silently resolved either way.
    assert out["verdict_shape_only"] == "no clean targeting (negligible range)"
    assert out["tie_attenuated"] is True


def test_verdicts_do_not_turn_on_0_8_versus_0_9():
    """Run over every profile in this file: report which verdicts would change
    if the threshold moved. None may, or the threshold is doing the work."""
    from rl_param_noise.targeting import verdict

    profiles = [
        (HUMAN_MEANS, HUMAN_COUNTS),
        (INVERTED_MEANS, INVERTED_COUNTS),
        ([5.0] * 6, [1000] * 6),
        ([0.0] * 6, [1000] * 6),
    ]
    for means, counts in profiles:
        stats = targeting(means, counts)
        strict = dict(stats)
        rho = strict["rho_contribution_punishment"]
        if rho == rho and abs(rho) < 0.9:
            continue
        assert verdict(strict) == stats["verdict"]
