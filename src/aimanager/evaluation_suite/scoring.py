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
