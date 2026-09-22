"""Parity of the batched RL-training opponent with the simulation path.

`LinearPunisherOpponent` recomputes the punishment bundle's features straight
from the env state instead of going through `build_feature_pool`, so the two
paths could drift. These tests pin them together on the real bundles, with
`sample=False` so the comparison is on the features and the fitted model rather
than on RNG. Runs locally: nothing here imports torch_geometric.
"""

from pathlib import Path

import joblib
import numpy as np
import pytest
import torch as th

from aimanager.manager.linear_opponent import LinearPunisherOpponent
from aimanager.simulation.linear_ah import LinearAHAdapter

ROOT = Path(__file__).resolve().parents[3]
BUNDLES = [
    "punishment_multinomial_ceiling_severity_copula",
    "punishment_multinomial_ceiling",
    "punishment_multinomial_current_contr_severity_copula",
    "punishment_multinomial_timeout_severity_copula",
]
A, T = 8, 12
GROUPS = [0, 0, 0, 0, 1, 1, 1, 1]


def _bundle(name):
    path = ROOT / "artifacts/baselines" / f"{name}.joblib"
    if not path.exists() or path.stat().st_size < 1000:
        pytest.skip(f"{name} not materialised (git lfs pull)")
    return joblib.load(path)


def _episode(seed=0):
    """A random episode: contributions, punishments, timeouts, memberships."""
    rng = np.random.default_rng(seed)
    contribution = rng.integers(0, 21, size=(A, T))
    punishment = rng.integers(0, 31, size=(A, T))
    valid = rng.random((A, T)) > 0.15  # ~15% timeouts
    contribution = np.where(valid, contribution, 0)  # the recorded 0
    groups = np.tile(np.array(GROUPS), (T, 1)).T
    # one reshuffle so memberships are not constant across the episode
    groups[:, T // 2 :] = 1 - groups[:, T // 2 :]
    return contribution, punishment, valid, groups


def _rounds(contribution, punishment, valid, groups, t):
    """The sim path's round-dict history up to and including round t. Round t's
    punishments are not yet known, exactly as simulate.py builds it."""
    out = []
    for s in range(t + 1):
        out.append(
            {
                "round": s,
                "contribution": [int(x) for x in contribution[:, s]],
                "contribution_valid": [bool(x) for x in valid[:, s]],
                "punishment": (
                    [int(x) for x in punishment[:, s]] if s < t else [0] * A
                ),
                "punishment_valid": [s < t] * A,
                "agent_group": [int(g) for g in groups[:, s]],
                "group": [str(g) for g in groups[:, s]],
            }
        )
    return out


def _state(contribution, punishment, valid, groups, t, default_contribution):
    """The env state the RL loop hands the opponent at round t, already served
    (environment.served_state): the recorded 0 for a timeout, the dataset
    default only where no previous round was played."""

    def col(x, dtype=th.int64):
        return th.tensor(x, dtype=dtype).reshape(1, A, 1)

    if t == 0:
        prev_c = np.full(A, default_contribution)
        prev_p = np.zeros(A)
    else:
        prev_c = contribution[:, t - 1]
        prev_p = punishment[:, t - 1]
    return {
        "contribution": col(contribution[:, t]),
        "contribution_valid": col(valid[:, t], th.bool),
        "prev_contribution": col(prev_c),
        "prev_punishment": col(prev_p),
        "round_number": col(np.full(A, t)),
        "agent_group": col(groups[:, t]),
    }


@pytest.mark.parametrize("name", BUNDLES)
def test_matches_simulation_path(name):
    bundle = _bundle(name)
    opp = LinearPunisherOpponent(bundle, sample=False)
    sim = LinearAHAdapter(bundle, sample=False)
    contribution, punishment, valid, groups = _episode()
    default_contribution = float(bundle["default_values"]["contribution"])

    for t in range(T):
        want = sim.get_punishments(
            _rounds(contribution, punishment, valid, groups, t)
        ).numpy()
        got = opp.predict(
            _state(contribution, punishment, valid, groups, t, default_contribution)
        )[0]
        np.testing.assert_array_equal(
            got.reshape(-1).numpy(), want, err_msg=f"{name} round {t}"
        )


def test_batches_independently():
    """Episodes in the batch must not leak into each other: stacking two
    episodes must give each of them its own single-episode answer."""
    bundle = _bundle(BUNDLES[0])
    opp = LinearPunisherOpponent(bundle, sample=False)
    dc = float(bundle["default_values"]["contribution"])
    eps = [_episode(seed=s) for s in (1, 2)]
    t = 5
    singles = [opp.predict(_state(*e, t, dc))[0].reshape(-1) for e in eps]
    stacked = {
        k: th.cat([_state(*e, t, dc)[k] for e in eps], dim=0)
        for k in _state(*eps[0], t, dc)
    }
    both = opp.predict(stacked)[0]
    assert both.shape == (2, A, 1)
    for i, single in enumerate(singles):
        th.testing.assert_close(both[i].reshape(-1), single)


def test_rejects_unsupported_feature():
    bundle = dict(_bundle(BUNDLES[0]))
    bundle["features"] = list(bundle["features"]) + ["win_common_good"]
    with pytest.raises(AssertionError, match="cannot reproduce"):
        LinearPunisherOpponent(bundle)


def test_copula_draw_is_shared_within_a_group():
    """With rho > 0 the two groups draw independent latents but the members of
    one group share theirs, so a group's levels move together. Checked as the
    sampler's contract rather than through the fitted model."""
    bundle = _bundle(BUNDLES[0])
    assert bundle["copula_rho"] > 0
    opp = LinearPunisherOpponent(bundle, sample=True)
    B, L = 512, 31
    P = np.full((B, A, L), 1.0 / L)
    groups = th.tensor(GROUPS, dtype=th.int64).expand(B, A).contiguous()
    th.manual_seed(0)
    lvl = opp._sample(P, groups).to(th.float64)
    within = np.corrcoef(lvl[:, 0].numpy(), lvl[:, 1].numpy())[0, 1]
    across = np.corrcoef(lvl[:, 0].numpy(), lvl[:, 4].numpy())[0, 1]
    assert within > 0.25, within
    assert abs(across) < 0.15, across


def test_constants_match_training():
    import sys

    sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
    from handcrafted_grid import ENDOWMENT

    assert LinearPunisherOpponent.SUPPORTED_FEATURES
    assert ENDOWMENT == 20.0
