"""Intensity-invariant targeting statistics for a binned punishment profile.

**The trap this exists to avoid.** A difference of bin means -- which is what
`contrast = mean{0} - mean{20}` is -- scales with how hard a manager punishes.
Two managers with *identical* contribution-to-punishment contingency, one
punishing half as hard as the other, differ by a factor of two on it. Measured
elsewhere in this project: the largest apparent shape difference in a sibling's
study, -11.1, was entirely intensity -- mean punishment down to a third, punish
rate tripled, the actual relationship unchanged in the third decimal. Same aim,
less force, more often.

So whatever names *targeting* has to be invariant to force. A rank correlation
between contribution and punishment is; a difference of bin means is not.

Three statistics are returned, and what each is invariant to is stated rather
than assumed:

  rho_contribution_punishment
      Spearman rank correlation between the contribution bin and the
      punishment served, count-weighted. Invariant to ANY strictly monotone
      rescaling of the punishment scale, so it cannot be moved by force at
      all. Negative is the human sign: punish the free-rider, leave the full
      contributor alone. Human managers sit at -1.0 (their profile is strictly
      decreasing over all six bins); a manager with the inversion sits at +1.0.
      It is a direction-and-monotonicity statistic and carries no magnitude:
      a profile running 5.00 to 4.99 scores the same -1.0 as one running 4.76
      to 0.27. Never read it alone.

  contrast_over_mean
      The raw contrast divided by the profile's own count-weighted mean
      punishment. Invariant to MULTIPLICATIVE rescaling only -- weaker than
      the rank correlation, but unlike it, it keeps relative magnitude.

  contrast
      The raw difference of bin means, in punishment points. NOT invariant to
      anything. Kept because it is the interpretable number and because the
      human and clone references are quoted on it, but it must never be
      compared across managers that punish with different force.

**A limitation, stated.** These are computed from per-bin means and counts,
which is what the training loop records, so `rho` is the rank correlation on
the bin-aggregated profile, not on the agent-round joint distribution. It
measures the monotonicity of the six bin means; it cannot see within-bin
structure. Recording the (contribution, punishment) joint histogram per
rollout would give the exact agent-round statistic and is a successor item --
it was not worth discarding 33 GPU-hours of in-flight runs for, since the bin
means and counts are sufficient for everything computed here.

**And a precondition.** `rho` ranks six numbers. If those six numbers differ by
less than their own sampling noise, it ranks noise and will happily return
+-1. `profile_snr` in `guard_report.py` is the guard: the range of the bin
means over their mean standard error across evaluation points. Measured across
ten runs of this campaign at 100 episodes the smallest such gate was 54
against a threshold of 1.65, and these rollouts are ten times larger, so the
gate does not bind: verdicts turn on rank, monotonicity and relative range.

**The tie hazard, which is the sharpest of these.** Quiet policies are common
in this campaign -- three of five evolution-strategies seeds punish exactly
zero in every bin -- and ties attenuate rank statistics. A sibling's control
seed is monotone decreasing across all six bins and scores only -0.845 because
three bins saturate at zero. It is worse than a threshold choice: measured
here on the perfectly monotone profile [4, 2, 1, 0, 0, 0], moving the
agent-rounds into the three tied zero bins drags `rho` from -0.95 to -0.35,
below anything a rank threshold could sensibly be. No threshold fixes that.

So four things are reported together and `verdict` is never read alone:

  monotonicity      tie-proof, and the thing "correctly targeted" means
  tau_b             attenuated differently from `rho`, not a rescue
  relative_range    range over mean, which survives a non-monotone profile
                    where `contrast` does not
  tie_attenuated    set when `verdict` and `verdict_shape_only` disagree,
                    i.e. exactly when ties have eaten the rank statistic

`verdict` applies the campaign's rule (monotone, then |rho| >= 0.8) so the
four arms' tables are comparable; `verdict_shape_only` asks the same question
from monotonicity and range alone. A row where they disagree is read by hand,
not counted.
"""

import numpy as np


def average_ranks(values, weights):
    """Average ranks for grouped observations.

    Group ``g`` stands for ``weights[g]`` tied observations all at
    ``values[g]``. Groups that share a value are merged into one tie block and
    all of them take that block's average rank -- the standard tie correction,
    which is what makes the result a Spearman rather than a Pearson on
    arbitrary rank assignments.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    below = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        block = order[i : j + 1]
        size = weights[block].sum()
        ranks[block] = below + (size + 1.0) / 2.0
        below += size
        i = j + 1
    return ranks


def weighted_pearson(x, y, w):
    x, y, w = (np.asarray(v, dtype=float) for v in (x, y, w))
    total = w.sum()
    mx, my = (w * x).sum() / total, (w * y).sum() / total
    vx, vy = (w * (x - mx) ** 2).sum(), (w * (y - my) ** 2).sum()
    if vx <= 0 or vy <= 0:
        # A flat profile has no targeting direction to report. NaN, not zero:
        # zero would claim "no relationship measured", which is a different
        # statement from "there is nothing here to rank".
        return float("nan")
    return float((w * (x - mx) * (y - my)).sum() / np.sqrt(vx * vy))


def kendall_tau_b(means, counts):
    """Count-weighted Kendall tau-b over the binned profile.

    Carried beside `rho` because **Spearman is dragged down by ties and this
    campaign's policies are full of them.** A sibling's control seed is
    monotone decreasing across all six bins and scores only -0.845, because
    three of its bins saturate at exactly zero: tied bins share a rank, the
    rank variance collapses, and the correlation is attenuated even though the
    tie handling is correct. A threshold of 0.9 would have thrown away a
    genuinely correctly-targeted policy. tau-b divides the tie counts out of
    its own denominator, so it is attenuated differently, and where the two
    disagree the monotonicity flag settles it.

    Exact, not sampled. Every bin has a distinct contribution rank, so between
    bins ``b < b'`` all ``n_b * n_b'`` pairs agree in x and are concordant or
    discordant by the sign of ``m_b' - m_b``; within a bin every pair is tied
    in both.
    """
    m = np.asarray(means, dtype=float)
    n = np.asarray(counts, dtype=float)
    total = n.sum()
    n0 = total * (total - 1.0) / 2.0
    # Ties in x: every within-bin pair.
    n1 = (n * (n - 1.0) / 2.0).sum()
    # Ties in y: bins sharing a mean merge into one tie block.
    n2 = 0.0
    for value in np.unique(m):
        block = n[m == value].sum()
        n2 += block * (block - 1.0) / 2.0
    concordant_minus_discordant = 0.0
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            concordant_minus_discordant += n[i] * n[j] * np.sign(m[j] - m[i])
    denom = np.sqrt((n0 - n1) * (n0 - n2))
    return float(concordant_minus_discordant / denom) if denom > 0 else float("nan")


# Monotonicity tolerance, as a fraction of the profile's own range. A step in
# the wrong direction smaller than this does not count as breaking
# monotonicity.
#
# It is needed, and measured: seed 46 of this arm runs
# [0.999, 0.884, 0.051, 0.000, 0.000, 0.000298] -- a rise of 0.000298 on a
# profile spanning 0.999, which is 0.03% of its range. With an exact test that
# dust flips the verdict from "targets free-riders" to "no clean targeting"
# and changes the arm's headline from 2 of 5 to 1 of 5.
#
# The choice is not load-bearing. In this arm's five seeds the violations are
# 0.000, 0.000, 0.0003, 0.425 and 0.435 of range, so every tolerance between
# 0.001 and 0.42 -- a factor of 400 -- gives identical verdicts.
# `rise_fraction` is reported so that stays checkable rather than asserted.
#
# The violation is measured CUMULATIVELY, as the total movement against the
# profile's direction, not as the largest single step. A per-step tolerance
# lets many small steps accumulate: [0.08, 1, 2, 3, 4, 5] climbs the whole way
# in five steps of about 1, and a per-step slack of 1.5 would call that flat.
MONOTONE_TOLERANCE = 0.02


def violation_fractions(means):
    """Total movement up and total movement down, each over the range."""
    m = np.asarray(means, dtype=float)
    rng = float(m.max() - m.min())
    if rng <= 0:
        return 0.0, 0.0
    d = np.diff(m)
    return float(d[d > 0].sum()) / rng, float(-d[d < 0].sum()) / rng


def rise_fraction(means):
    """How far the profile is from monotone in whichever direction suits it
    better, as a fraction of its range. Zero for a perfectly monotone
    profile. Reported beside the verdict so the tolerance is auditable."""
    up, down = violation_fractions(means)
    return min(up, down)


def monotonicity(means, tolerance=MONOTONE_TOLERANCE):
    """Weak monotonicity of the bin-mean sequence -- tie-proof, unlike a rank
    correlation, and the thing 'correctly targeted' actually means.

    Returns "decreasing" (the human sign: punish the free-rider), "increasing"
    (the inversion), "flat" (no contingency at all) or "none". A profile with a
    large endpoint difference that is NOT monotone is the shape that got two
    sibling seeds withdrawn, so this is checked before any headline.

    Violations smaller than `tolerance` times the profile's range are ignored;
    see `MONOTONE_TOLERANCE` for why and for the evidence that the threshold
    does not decide anything here.
    """
    m = np.asarray(means, dtype=float)
    if float(m.max() - m.min()) <= 0:
        return "flat"
    up, down = violation_fractions(m)
    is_down, is_up = up <= tolerance, down <= tolerance
    if is_down and is_up:
        return "flat"
    if is_down:
        return "decreasing"
    if is_up:
        return "increasing"
    return "none"


# The rank threshold. 0.8, not 0.9, because ties attenuate both rank
# statistics and 0.9 discards genuinely monotone policies -- see
# `kendall_tau_b`. Matched to the sibling arms so verdicts are comparable.
RANK_THRESHOLD = 0.8

# Advisory only, and mine rather than the campaign's: below this the profile's
# spread is a rounding error next to its own level and a rank statistic is
# describing nothing worth describing. The human managers sit at 2.43 and the
# artificial punisher at 2.09; the epsilon-greedy pilot buffer sits at 0.005.
# It is reported as a flag and deliberately NOT folded into `verdict`, so that
# "no verdict turns on this number" stays checkable.
NEGLIGIBLE_RANGE = 0.05


def verdict(stats):
    """The campaign's three-statistic check: monotonicity first, then rank.

    Deliberately the same rule the sibling arms apply -- monotone, then
    ``|rho| >= RANK_THRESHOLD`` -- so the four arms' tables are comparable. A
    large endpoint difference on a profile that is not monotone is not
    targeting, and `contrast` cannot tell.

    **Read it with `tie_attenuated`.** The rank half of this rule is not safe
    on its own when ties carry the mass; see `verdict_shape_only`.
    """
    if stats["n_distinct_bins"] < 2:
        return "no contingency (all bins tied)"
    shape = stats["monotonicity"]
    if shape == "flat":
        return "no contingency (all bins tied)"
    if shape == "none":
        return "no clean targeting (not monotone)"
    rho = stats["rho_contribution_punishment"]
    if rho != rho:
        return "no clean targeting (rank undefined)"
    if rho <= -RANK_THRESHOLD:
        return "targets free-riders"
    if rho >= RANK_THRESHOLD:
        return "inverted"
    return "no clean targeting (weak rank)"


def verdict_shape_only(stats):
    """The same question asked without any rank statistic: monotonicity and
    relative range alone.

    This exists because the tie hazard is worse than a threshold choice.
    Measured on a *perfectly* monotone decreasing profile
    ``[4, 2, 1, 0, 0, 0]``, `rho` moves from -0.95 to -0.35 purely by shifting
    the agent-rounds into the three tied zero bins -- so a quiet policy that
    punishes free-riders and nobody else can score below ANY sensible rank
    threshold while being exactly the shape the campaign is looking for.
    Lowering the threshold from 0.9 to 0.8 does not fix that; nothing about a
    rank threshold can.

    So the rank verdict is kept for comparability and this one is reported
    beside it. Where the two disagree, `tie_attenuated` is set and the row has
    to be read by hand rather than counted.
    """
    if stats["n_distinct_bins"] < 2 or stats["monotonicity"] == "flat":
        return "no contingency (all bins tied)"
    if stats["monotonicity"] == "none":
        return "no clean targeting (not monotone)"
    rng = stats["relative_range"]
    if rng == rng and rng < NEGLIGIBLE_RANGE:
        return "no clean targeting (negligible range)"
    return {
        "decreasing": "targets free-riders",
        "increasing": "inverted",
    }[stats["monotonicity"]]


def targeting(means, counts):
    """Targeting statistics from an ordered binned punishment profile.

    ``means[b]`` is the mean punishment in contribution bin ``b`` and
    ``counts[b]`` the number of agent-rounds behind it. Bins must already be
    in increasing contribution order, which `RPA_LABELS` is.
    """
    m = np.asarray(means, dtype=float)
    n = np.asarray(counts, dtype=float)
    ok = np.isfinite(m) & np.isfinite(n) & (n > 0)
    m, n = m[ok], n[ok]
    empty = {
        "rho_contribution_punishment": float("nan"),
        "tau_b": float("nan"),
        "monotonicity": "none",
        "n_distinct_bins": 0,
        "n_zero_bins": 0,
        "rise_fraction": float("nan"),
        "contrast": float("nan"),
        "contrast_over_mean": float("nan"),
        "relative_range": float("nan"),
        "mean_punishment": float("nan"),
    }
    if len(m) < 2:
        return {
            **empty,
            "verdict": verdict(empty),
            "verdict_shape_only": verdict_shape_only(empty),
            "tie_attenuated": False,
        }
    bins = np.arange(len(m), dtype=float)
    mean_p = float((m * n).sum() / n.sum())
    contrast = float(m[0] - m[-1])
    stats = {
        "rho_contribution_punishment": weighted_pearson(
            average_ranks(bins, n), average_ranks(m, n), n
        ),
        "tau_b": kendall_tau_b(m, n),
        "monotonicity": monotonicity(m),
        # The tie structure, because a rank number quoted without it is not
        # interpretable: three bins saturated at zero is most of the ranking
        # gone, and six of them is all of it.
        "n_distinct_bins": int(len(np.unique(m))),
        "n_zero_bins": int((m == 0.0).sum()),
        "rise_fraction": rise_fraction(m),
        "contrast": contrast,
        "contrast_over_mean": contrast / mean_p if mean_p else float("nan"),
        # Range rather than endpoints: it survives a non-monotone profile,
        # which `contrast` does not, and it is what caught the seed a sibling
        # withdrew.
        "relative_range": (
            float(m.max() - m.min()) / mean_p if mean_p else float("nan")
        ),
        "mean_punishment": mean_p,
    }
    v, vs = verdict(stats), verdict_shape_only(stats)
    return {
        **stats,
        "verdict": v,
        "verdict_shape_only": vs,
        # The two disagree exactly when ties have eaten the rank statistic.
        # Set here rather than left for a reader to notice.
        "tie_attenuated": v != vs,
    }
