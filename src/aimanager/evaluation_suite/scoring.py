"""Scoring schema for the evaluation metrics (#132).

Normalises every metric row against a human-vs-human noise ceiling:

    score = E_r[d(h_a, s)] / E_r[d(h_a, h_b)]

over R repeats, where each repeat splits the human data into disjoint
episode-level halves h_a / h_b of m = n_human // 2 episodes and draws a
fresh size-m sample s from the pre-simulated pool. Both terms compare
size m against size m, so finite-sample bias cancels; a model matching
the human data scores 1 (>= 1 up to noise: ~1 at the ceiling, 1-2 minor
deviation, 2-5 clear deviation, > 5 not reproduced).

The resampling plan is fixed by one master seed, so every candidate
model is scored on identical splits and simulation draws.
"""

import warnings

import numpy as np


def make_repeats(human_episode_ids, sim_episode_ids, n_repeats, seed):
    """Deterministic resampling plan: a list of (h_a, h_b, s) episode-id
    arrays, one triple per repeat. Ids are sorted before use so the plan
    does not depend on input order; the sim draw is without replacement
    within a repeat."""
    human_ids = np.asarray(sorted(human_episode_ids))
    sim_ids = np.asarray(sorted(sim_episode_ids))
    m = len(human_ids) // 2
    if m < 1:
        raise ValueError("need at least 2 human episodes to split")
    if len(sim_ids) < m:
        raise ValueError(f"sim pool has {len(sim_ids)} episodes, need at least m={m}")
    rng = np.random.default_rng(seed)
    repeats = []
    for _ in range(n_repeats):
        perm = rng.permutation(human_ids)
        draw = rng.choice(sim_ids, size=m, replace=False)
        repeats.append((perm[:m], perm[m : 2 * m], draw))
    return repeats


def subset(df, episode_ids):
    """Canonical-frame subset for one side of a comparison."""
    return df[df["episode_id"].isin(episode_ids)]


def _row_weights(group, name, human):
    """Stratum weights fixed once on the full human reference; plain
    distribution rows have none."""
    if group.KINDS[name] == "distribution":
        return None
    return group.weights(name, human)


def _d_or_none(group, name, a, b, weights):
    """One d evaluation; None when the comparison cannot support every
    stratum (the #134 guard) -- the repeat is then dropped."""
    try:
        return group.d(name, a, b, weights=weights)
    except ValueError:
        return None


def denominators(group, name, human, repeats, weights=None):
    """d(h_a, h_b) per repeat -- the human-vs-human noise ceiling. It
    does not depend on the sim, so the driver computes it once per row
    and reuses it across pairings. None marks an unsupported repeat."""
    if weights is None:
        weights = _row_weights(group, name, human)
    return [
        _d_or_none(group, name, subset(human, h_a), subset(human, h_b), weights)
        for h_a, h_b, _ in repeats
    ]


def score_row(group, name, human, sim, repeats, weights=None, denoms=None):
    """Normalised score for one metric row: mean_r d(h_a, s) over
    mean_r d(h_a, h_b), the same h_a in both terms, numerator and
    denominator averaged separately before dividing.

    Repeats whose sim draw or human half cannot support every stratum
    are dropped; both terms average over the same surviving repeats and
    repeats_used reports how many. If fewer than half survive, the
    score is NaN and a warning is emitted (#134's policy, settled on
    the one real case: lin_ridge's RSA 16+ bin)."""
    if weights is None:
        weights = _row_weights(group, name, human)
    if denoms is None:
        denoms = denominators(group, name, human, repeats, weights=weights)
    nums = [
        _d_or_none(group, name, subset(human, h_a), subset(sim, s), weights)
        for h_a, _, s in repeats
    ]
    pairs = [(n, d) for n, d in zip(nums, denoms) if n is not None and d is not None]
    result = {
        "score": float("nan"),
        "numerator": float("nan"),
        "denominator": float("nan"),
        "repeats_used": len(pairs),
        "n_repeats": len(repeats),
    }
    if not pairs:
        warnings.warn(f"{name}: no repeat supports all strata -- score is NaN")
        return result
    result["numerator"] = sum(n for n, _ in pairs) / len(pairs)
    result["denominator"] = sum(d for _, d in pairs) / len(pairs)
    if len(pairs) < len(repeats) / 2:
        warnings.warn(
            f"{name}: only {len(pairs)}/{len(repeats)} repeats support all "
            "strata -- score is NaN"
        )
        return result
    result["score"] = result["numerator"] / result["denominator"]
    return result
