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
means over their mean standard error across evaluation points. Do not read
`rho` when it is small.
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
    if len(m) < 2:
        return {
            "rho_contribution_punishment": float("nan"),
            "contrast": float("nan"),
            "contrast_over_mean": float("nan"),
            "mean_punishment": float("nan"),
        }
    bins = np.arange(len(m), dtype=float)
    rho = weighted_pearson(average_ranks(bins, n), average_ranks(m, n), n)
    mean_p = float((m * n).sum() / n.sum())
    contrast = float(m[0] - m[-1])
    return {
        "rho_contribution_punishment": rho,
        "contrast": contrast,
        "contrast_over_mean": contrast / mean_p if mean_p else float("nan"),
        "mean_punishment": mean_p,
    }
