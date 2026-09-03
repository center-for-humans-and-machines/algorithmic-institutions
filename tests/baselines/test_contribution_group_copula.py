"""Unit tests for the Gaussian group-copula CONTRIBUTION sampler,
``LinearAHAdapter._sample_levels_gaussian_copula``: exact marginal
preservation, recovery of BOTH correlation weights from the replayed latent
``z``, the fixed 3n-draw RNG stream, the lifetime of the persistent
per-(episode, group) latent, the arrival-group rule for switchers, and the
bit-identical legacy path for bundles without ``copula_rho_p`` /
``copula_rho_t``. The ``__init__`` configuration gate is covered in
tests/baselines/test_gaussian_mlp.py; this module tests the sampler.

Invariants and rationale:
notes/autoresearch_log/contribution-gmlp-group-copula.md (Notes 21-23).

Local test (CPU torch, no PyG):
    .venv/bin/python -m pytest tests/baselines/test_contribution_group_copula.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch as th  # noqa: E402
from scipy.special import ndtr  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # tests/baselines -> repo root
# this checkout's src must win over any installed/editable aimanager, so the
# adapter under test is the one in THIS worktree
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/baselines"))

from aimanager.simulation.linear_ah import LinearAHAdapter  # noqa: E402
from gaussian_regressor import GaussianMLPRegressor  # noqa: E402

N_CONTRIBUTIONS = 21  # contribution levels 0..20
N_AGENTS = 8
GROUPS = [0, 0, 0, 0, 1, 1, 1, 1]  # two groups of four, the real composition
FEATS = ["prev_contribution", "prev_punishment", "round_number", "is_first"]
DEFAULTS = {
    "punishment": 0.0,
    "contribution": 9.0,
    "common_good": 12.333333333333334,
    "agent_group": 0.0,
    "contribution_valid": 0.0,
    "punishment_valid": 0.0,
    "recorded": 0.0,
}
N_DRAW = 4000  # sampling repeats for the marginal test
N_EPISODES = 800  # episodes for the correlation-recovery replay
N_ROUNDS = 5


# --------------------------------------------------------------------------- #
# synthetic bundle (the toy_bundle pattern of test_gaussian_mlp.py, with the
# group-copula fields added)
# --------------------------------------------------------------------------- #
def toy_features(n, seed):
    """Raw feature rows on the real game's scale -- contributions 0-20,
    punishments 0-7, 24 rounds -- so the fitted model is exercised where the
    simulation would exercise it."""
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.integers(0, 21, n).astype(float),
            rng.integers(0, 8, n).astype(float),
            rng.integers(1, 24, n).astype(float),
            np.zeros(n),
        ]
    )


def toy_bundle(**over):
    """A small but real gaussian_mlp contribution bundle: mean reverting to a
    nonlinear function of prev_contribution, spread widening with punishment."""
    from sklearn.preprocessing import StandardScaler

    X = toy_features(500, 21)
    rng = np.random.default_rng(22)
    sd = 0.5 + 0.4 * X[:, 1]
    y = np.clip(
        0.6 * X[:, 0] + 4.0 * np.sin(X[:, 0] / 3.0) + rng.normal(0.0, 1.0, len(X)) * sd,
        0,
        20,
    )
    scaler = StandardScaler().fit(X)
    est = GaussianMLPRegressor(hidden=16, epochs=400, lr=0.05, seed=0).fit(
        scaler.transform(X), y
    )
    bundle = dict(
        model="gaussian_mlp",
        estimator=est,
        scaler=scaler,
        features=list(FEATS),
        target="contribution",
        n_levels=0,
        default_values=dict(DEFAULTS),
        switch_every=4,
    )
    bundle.update(over)
    return bundle


@pytest.fixture(scope="module")
def base_bundle():
    return toy_bundle()


def _bundle(base, rho_p, rho_t):
    """The same fitted estimator with the copula weights swapped in -- refitting
    per parametrisation would dominate the module's runtime."""
    out = dict(base)
    out.update(copula_rho_p=rho_p, copula_rho_t=rho_t)
    return out


def _adapter(bundle, **kw):
    return LinearAHAdapter(bundle, n_agents=8, n_contributions=N_CONTRIBUTIONS, **kw)


# --------------------------------------------------------------------------- #
# env states / episode driving
# --------------------------------------------------------------------------- #
def _state(t, groups=None, prev_groups=None):
    """Minimal env state for `predict()`: the round index, post-arrival
    membership, and the prev_ siblings `_record` folds into the history. The
    prev_ measures are CONSTANTS, so the feature rows -- and hence (mu, sigma)
    -- depend on the round alone, never on the sampled levels: the latent's
    dependence structure is then observable without the closed loop confounding
    it, and every agent shares one (mu, sigma) (none of FEATS is group-valued)."""
    A = N_AGENTS
    groups = GROUPS if groups is None else groups
    ag = th.tensor(list(groups), dtype=th.int64).reshape(1, A, 1)
    st = {
        "round_number": th.full((1, A, 1), t, dtype=th.int64),
        "agent_group": ag,
    }
    if t > 0:
        prev = (
            ag
            if prev_groups is None
            else th.tensor(list(prev_groups), dtype=th.int64).reshape(1, A, 1)
        )
        st["prev_contribution"] = th.full((1, A, 1), 9.0)
        st["prev_punishment"] = th.zeros((1, A, 1))
        st["prev_common_good"] = th.full((1, A, 1), 12.0)
        st["prev_agent_group"] = prev
    return st


def _run_episode(ad, groups_by_round, *, reset=True):
    """Drive `predict` over one episode. Returns the per-round levels and a
    snapshot of the persistent-latent store after each round."""
    levels, snaps = [], []
    for t, g in enumerate(groups_by_round):
        st = _state(t, g, groups_by_round[t - 1] if t else None)
        pred, _ = ad.predict(st, reset_rnn=(reset and t == 0))
        levels.append(pred.reshape(-1).numpy())
        snaps.append(dict(ad._copula_z))
    return levels, snaps


def _mu_sd(bundle, groups_by_round):
    """(mu, sigma) per round, built through the adapter's OWN feature path on a
    deterministic probe adapter -- no RNG consumed, so the replay below can
    reconstruct `predict`'s levels without touching the stream."""
    ad = _adapter(bundle, sample=False)
    out = []
    for t, g in enumerate(groups_by_round):
        st = _state(t, g, groups_by_round[t - 1] if t else None)
        if t == 0:
            ad._reset_history()
        ad._record(st, t)
        pool = ad._build_pool(t)
        X = np.column_stack([pool[f][0, :, t] for f in ad.features])
        Xs = ad.scaler.transform(X)
        out.append((ad.estimator.predict(Xs), ad.estimator.predict_std(Xs)))
    return out


def _replay_z(rho_p, rho_t, groups_by_round, episodes, seed):
    """Regenerate the sampler's latent `z` from the RNG stream alone.

    The draw order is documented and fixed -- `zu`, `zv`, `eps`, each a size-n
    float64 `randn`, unconditionally, once per call -- so re-seeding and
    repeating it in that order reproduces `z` exactly (Note 23). The persistent
    store mirrors `_copula_z`: `setdefault` on the first member of each group
    id, cleared at the start of every episode."""
    n = N_AGENTS
    th.manual_seed(seed)
    zs = np.empty((episodes, len(groups_by_round), n))
    for e in range(episodes):
        store = {}
        for t, groups in enumerate(groups_by_round):
            zu = th.randn(n, dtype=th.float64).numpy()
            zv = th.randn(n, dtype=th.float64).numpy()
            eps = th.randn(n, dtype=th.float64).numpy()
            g = np.asarray(groups).reshape(-1)
            first, pick = {}, np.empty(n, dtype=np.int64)
            for i, gid in enumerate(g):
                pick[i] = first.setdefault(int(gid), i)
            for gid, i in first.items():
                store.setdefault(gid, float(zu[i]))
            u = np.array([store[int(gid)] for gid in g], dtype=np.float64)
            zs[e, t] = (
                np.sqrt(rho_p) * u
                + np.sqrt(rho_t) * zv[pick]
                + np.sqrt(1.0 - rho_p - rho_t) * eps
            )
    return zs


def _levels_and_replay(base, rho_p, rho_t, *, episodes, groups_by_round, seed):
    """`episodes` episodes through `predict`, plus the replayed `z` and the
    levels that `z` implies."""
    ad = _adapter(_bundle(base, rho_p, rho_t))
    th.manual_seed(seed)
    levels = np.stack(
        [np.stack(_run_episode(ad, groups_by_round)[0]) for _ in range(episodes)]
    )
    z = _replay_z(rho_p, rho_t, groups_by_round, episodes, seed)
    mu_sd = _mu_sd(base, groups_by_round)
    implied = np.stack(
        [
            np.stack(
                [
                    np.clip(np.rint(mu + sd * z[e, t]), 0, N_CONTRIBUTIONS - 1).astype(
                        np.int64
                    )
                    for t, (mu, sd) in enumerate(mu_sd)
                ]
            )
            for e in range(episodes)
        ]
    )
    return levels, z, implied


# --------------------------------------------------------------------------- #
# (a) marginal preservation -- the invariant the whole change rests on
# --------------------------------------------------------------------------- #
def _analytic_marginal(mu, sd):
    """P(clip(rint(N(mu, sigma)), 0, 20) == l) per agent: the exact discrete law
    the INDEPENDENT path draws from, and the one the copula must preserve."""
    lo = np.arange(N_CONTRIBUTIONS, dtype=float) - 0.5
    hi = np.arange(N_CONTRIBUTIONS, dtype=float) + 0.5
    lo[0], hi[-1] = -np.inf, np.inf
    zl = (lo[None, :] - mu[:, None]) / sd[:, None]
    zh = (hi[None, :] - mu[:, None]) / sd[:, None]
    return ndtr(zh) - ndtr(zl)


def _draw_many(ad, Xs, seed, *, copula):
    """N_DRAW independent draws of the 8 agents. `_reset_history()` before each
    one makes every draw a fresh EPISODE: the persistent latent is redrawn, so
    the histogram estimates the marginal the simulation realises over episodes
    (repeated calls on one adapter would all share a single frozen u_g)."""
    th.manual_seed(seed)
    out = np.empty((N_DRAW, N_AGENTS), dtype=np.int64)
    for i in range(N_DRAW):
        ad._reset_history()
        out[i] = (
            ad._sample_levels_gaussian_copula(Xs, N_CONTRIBUTIONS, GROUPS)
            if copula
            else ad._sample_levels(Xs, N_CONTRIBUTIONS)
        )
    return out


def _freqs(draws):
    return np.stack(
        [
            np.bincount(draws[:, a], minlength=N_CONTRIBUTIONS) / len(draws)
            for a in range(N_AGENTS)
        ]
    )


@pytest.fixture(scope="module")
def marginal_reference(base_bundle):
    """The analytic per-agent marginal on eight varied feature rows, plus TWO
    independent-sampler runs at different seeds. The gap between those two is
    the statistic's own noise floor, which is what the copula's gap has to be
    judged against (Note 21: ~0.0065 at 40,000 draws, so any fixed threshold
    below ~0.01 flakes -- it is measured here, at this module's N_DRAW, rather
    than guessed)."""
    ad = _adapter(base_bundle)
    Xs = ad.scaler.transform(toy_features(N_AGENTS, 33))
    mu = ad.estimator.predict(Xs)
    sd = ad.estimator.predict_std(Xs)
    ind_a = _freqs(_draw_many(ad, Xs, 101, copula=False))
    ind_b = _freqs(_draw_many(ad, Xs, 202, copula=False))
    return dict(
        Xs=Xs,
        P=_analytic_marginal(mu, sd),
        ind=ind_a,
        floor=float(np.abs(ind_a - ind_b).max()),
    )


@pytest.mark.parametrize("rho_p, rho_t", [(0.3, 0.0), (0.0, 0.3), (0.2, 0.2)])
def test_marginals_preserved(base_bundle, marginal_reference, rho_p, rho_t):
    """Two independent checks, because the tolerance must not be a guess.

    (1) A properly-sized statistical test against the ANALYTIC marginal: each
    per-agent bin frequency is Binomial(N_DRAW, p) -- draws are iid across
    repeats, the latent being redrawn each time -- so the deviation is asserted
    in binomial SEs (4 SEs, bins with fewer than 20 expected counts skipped,
    the normal approximation not being usable there). This is the house pattern
    of test_punishment_copula.py::test_marginals_match_model and needs no
    hand-picked number at all.

    (2) The same-noise-floor comparison Note 21 asks for: the max abs bin
    frequency gap against the independent sampler, judged against the gap
    between two independent-sampler runs at different seeds. The factor 2.5 is
    slack on a maximum-of-many-bins statistic estimated from a single pair of
    runs, not a tolerance on the effect: a marginal that actually moved would
    do so by far more (an unpreserved rho_p = 0.3 shifts whole histograms)."""
    ad = _adapter(_bundle(base_bundle, rho_p, rho_t))
    P, ind, floor = (
        marginal_reference["P"],
        marginal_reference["ind"],
        marginal_reference["floor"],
    )
    cop = _freqs(_draw_many(ad, marginal_reference["Xs"], 303, copula=True))

    worst_se, worst_bin = 0.0, (None, None)
    for a in range(N_AGENTS):
        for lvl in range(N_CONTRIBUTIONS):
            p = P[a, lvl]
            if p * N_DRAW < 20:
                continue
            se = np.sqrt(p * (1.0 - p) / N_DRAW)
            if abs(cop[a, lvl] - p) / se > worst_se:
                worst_se, worst_bin = abs(cop[a, lvl] - p) / se, (a, lvl)
    assert worst_se < 4.0, f"analytic gap {worst_se:.2f} SEs at {worst_bin}"

    gap = float(np.abs(cop - ind).max())
    assert gap <= 2.5 * floor, f"copula-vs-independent {gap:.5f} vs floor {floor:.5f}"


def test_mean_contribution_unchanged(base_bundle, marginal_reference):
    """Coarser but blunt: the per-agent mean level is what the simulation's
    common good is built from, and it must not move at any (rho_p, rho_t)."""
    P = marginal_reference["P"]
    lvls = np.arange(N_CONTRIBUTIONS)
    target = (P * lvls).sum(1)
    sd = np.sqrt((P * lvls**2).sum(1) - target**2)
    for rho_p, rho_t in [(0.3, 0.0), (0.0, 0.3), (0.2, 0.2)]:
        ad = _adapter(_bundle(base_bundle, rho_p, rho_t))
        got = _draw_many(ad, marginal_reference["Xs"], 404, copula=True).mean(0)
        assert np.all(
            np.abs(got - target) < 4.0 * sd / np.sqrt(N_DRAW)
        ), f"({rho_p}, {rho_t}): {got} vs {target}"


# --------------------------------------------------------------------------- #
# (b) correlation recovery, from the REPLAYED latent
# --------------------------------------------------------------------------- #
def _pairs(groups, same_group):
    g = np.asarray(groups)
    return [
        (i, j)
        for i in range(N_AGENTS)
        for j in range(N_AGENTS)
        if i != j and ((g[i] == g[j]) == same_group)
    ]


def _pooled_corr(z, pairs, lag):
    """One correlation over all pairs pooled: every z has the same marginal
    (exactly N(0, 1)), so pooling is legitimate and much less noisy than
    averaging per-pair correlations."""
    T = z.shape[1]
    a = np.concatenate([z[:, : T - lag, i].ravel() for i, _ in pairs])
    b = np.concatenate([z[:, lag:, j].ravel() for _, j in pairs])
    return float(np.corrcoef(a, b)[0, 1])


@pytest.mark.parametrize("rho_p, rho_t", [(0.1, 0.1), (0.15, 0.0)])
def test_correlation_recovery_from_replayed_z(base_bundle, rho_p, rho_t):
    """The dependence structure, read off `z` itself rather than off levels
    (Note 21: `rint`/`clip` attenuate a level correlation by ~8 %, so a 0.02
    tolerance on levels would be far too tight; on `z` it is loose).

    The replay's bit-identity assertion comes FIRST: if the reconstruction did
    not reproduce `predict`'s levels the recovered numbers would be measuring
    something else entirely. `(0.15, 0.0)` is the declared candidate's own
    shape -- rho_t exactly zero -- which is the configuration Note 22 found
    untested, and where a `zv` draw skipped on a zero weight desynchronises the
    replay."""
    groups_by_round = [GROUPS] * N_ROUNDS
    levels, z, implied = _levels_and_replay(
        base_bundle,
        rho_p,
        rho_t,
        episodes=N_EPISODES,
        groups_by_round=groups_by_round,
        seed=7,
    )
    assert np.array_equal(levels, implied), "replay does not reproduce predict"

    within = _pooled_corr(z, _pairs(GROUPS, True), 0)
    lagged = _pooled_corr(z, _pairs(GROUPS, True), 1)
    cross = _pooled_corr(z, _pairs(GROUPS, False), 0)
    assert within == pytest.approx(rho_p + rho_t, abs=0.02), f"within {within:.4f}"
    assert lagged == pytest.approx(rho_p, abs=0.02), f"cross-round {lagged:.4f}"
    assert cross == pytest.approx(0.0, abs=0.02), f"cross-group {cross:.4f}"


def test_z_is_marginally_standard_normal(base_bundle):
    """The mechanism behind marginal preservation, checked directly: whatever
    the weights, the replayed z is N(0, 1) per agent -- the weights square to 1
    only if the idiosyncratic term carries 1 - rho_p - rho_t."""
    z = _replay_z(0.2, 0.2, [GROUPS] * N_ROUNDS, N_EPISODES, 7)
    per_agent = z.reshape(-1, N_AGENTS)
    # the shared components make the effective sample size the EPISODE count,
    # not the row count, so the mean's own SE is ~0.02 here -- hence 0.08, not
    # the 1/sqrt(rows) that the raw row count would suggest
    assert np.abs(per_agent.mean(0)).max() < 0.08, per_agent.mean(0)
    assert np.abs(per_agent.std(0) - 1.0).max() < 0.05, per_agent.std(0)


# --------------------------------------------------------------------------- #
# (c) the RNG stream: exactly 3n float64 draws, composition-stable
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rho_p, rho_t", [(0.3, 0.0), (0.0, 0.3), (0.2, 0.2)])
def test_rng_consumption_is_three_size_n_draws(base_bundle, rho_p, rho_t):
    """Exactly 3n float64 draws per call, taken UNCONDITIONALLY -- also when a
    weight is 0.0. The step-8 preflight arms are only comparable if they share
    one RNG stream, so a stream that depends on the weights confounds the very
    comparison it exists to make (Note 22, mutation (a))."""
    ad = _adapter(_bundle(base_bundle, rho_p, rho_t))
    Xs = ad.scaler.transform(toy_features(N_AGENTS, 55))

    th.manual_seed(9)
    ad._sample_levels_gaussian_copula(Xs, N_CONTRIBUTIONS, GROUPS)
    got_next = th.randn(1).item()

    th.manual_seed(9)
    for _ in range(3):
        th.randn(N_AGENTS, dtype=th.float64)
    assert th.randn(1).item() == got_next, "sampler is not 3n float64 draws"


@pytest.mark.parametrize("rho_p, rho_t", [(0.3, 0.0), (0.2, 0.2)])
def test_group_composition_does_not_shift_the_stream(base_bundle, rho_p, rho_t):
    """A round with all eight agents in one group must leave the stream exactly
    where a 4/4 round does, so a switch can never desynchronise the RNG."""
    ad = _adapter(_bundle(base_bundle, rho_p, rho_t))
    Xs = ad.scaler.transform(toy_features(N_AGENTS, 56))
    after = []
    for groups in (GROUPS, [0] * N_AGENTS, [0, 1, 0, 1, 0, 1, 0, 1]):
        ad._reset_history()
        th.manual_seed(13)
        ad._sample_levels_gaussian_copula(Xs, N_CONTRIBUTIONS, groups)
        after.append(th.randn(1).item())
    assert len(set(after)) == 1, after


def test_groups_length_is_checked(base_bundle):
    ad = _adapter(_bundle(base_bundle, 0.2, 0.1))
    Xs = ad.scaler.transform(toy_features(N_AGENTS, 57))
    with pytest.raises(AssertionError, match="groups has"):
        ad._sample_levels_gaussian_copula(Xs, N_CONTRIBUTIONS, [0, 0, 1])


# --------------------------------------------------------------------------- #
# (d) the persistent latent's lifetime
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rho_p, rho_t", [(0.15, 0.0), (0.2, 0.1)])
def test_persistent_latent_constant_within_episode(base_bundle, rho_p, rho_t):
    """u_g is drawn ONCE per (episode, group) and held: the store's contents
    must be identical after every round of an episode. Redrawing it each round
    would silently turn the experiment into the transient-only arm (Note 22,
    mutation (b))."""
    ad = _adapter(_bundle(base_bundle, rho_p, rho_t))
    th.manual_seed(17)
    _, snaps = _run_episode(ad, [GROUPS] * 6)
    assert set(snaps[0]) == {0, 1}, snaps[0]
    for t, snap in enumerate(snaps[1:], start=1):
        assert snap == snaps[0], f"round {t}: {snap} != {snaps[0]}"


@pytest.mark.parametrize("reset", [True, False])
def test_persistent_latent_redrawn_per_episode(base_bundle, reset):
    """A new episode gets new latents, on `reset_rnn` and on `t == 0` alike
    (`predict` resets the history on either)."""
    ad = _adapter(_bundle(base_bundle, 0.15, 0.0))
    th.manual_seed(19)
    _, first = _run_episode(ad, [GROUPS] * 3)
    _, second = _run_episode(ad, [GROUPS] * 3, reset=reset)
    assert set(second[0]) == set(first[0])
    for gid in first[-1]:
        assert second[-1][gid] != first[-1][gid], f"group {gid} kept its latent"


def test_group_that_empties_and_reforms_resumes_its_latent(base_bundle):
    """A group id keeps its stored u_g for the whole episode: emptying it for a
    round must not cost it (or refresh) its latent."""
    ad = _adapter(_bundle(base_bundle, 0.15, 0.0))
    th.manual_seed(23)
    _, snaps = _run_episode(ad, [GROUPS, [0] * N_AGENTS, GROUPS])
    assert set(snaps[0]) == {0, 1}
    assert snaps[1] == snaps[0], "an empty group lost its latent"
    assert snaps[2] == snaps[0], "a re-formed group was given a fresh latent"


def test_switcher_draws_from_the_receiving_group(base_bundle):
    """The arrival-group rule: `groups` is post-arrival membership, so at the
    round of the switch the mover already reads the RECEIVING group's u_g.

    At rho_p -> 1 the idiosyncratic term vanishes and, since every agent shares
    one (mu, sigma) here, a group's members all land on one level -- so the
    mover's level identifies the group whose latent it used."""
    ad = _adapter(_bundle(base_bundle, 0.999999, 0.0))
    switched = [1, 0, 0, 0, 1, 1, 1, 1]  # agent 0 moves 0 -> 1
    th.manual_seed(4)
    levels, snaps = _run_episode(ad, [GROUPS, switched])
    assert snaps[1] == snaps[0], "the switch redrew a persistent latent"

    before, after = levels[0], levels[1]
    assert len(set(before[:4].tolist())) == 1 and len(set(before[4:].tolist())) == 1
    assert before[0] != before[4], "the two groups' latents coincide; seed unusable"
    assert after[0] == after[4], "the switcher did not use the receiving latent"
    assert after[0] != after[1], "the switcher stayed on its departing latent"


# --------------------------------------------------------------------------- #
# (e) the legacy path stays byte-identical
# --------------------------------------------------------------------------- #
def _legacy_levels(bundle, mu_sd):
    """The sampling committed BEFORE this change, reimplemented from
    src/aimanager/simulation/linear_ah.py::_sample_levels (gaussian branch):
    ONE float32 randn of size n per call."""
    out = []
    for mu, sd in mu_sd:
        yhat = mu + th.randn(len(mu)).numpy() * sd
        out.append(np.clip(np.rint(yhat), 0, N_CONTRIBUTIONS - 1).astype(np.int64))
    return out


@pytest.mark.parametrize(
    "over", [{}, {"copula_rho_p": 0.0, "copula_rho_t": 0.0}, {"copula_rho_p": None}]
)
def test_legacy_bundles_bit_identical(base_bundle, over):
    """Fields absent, both explicitly 0.0, and None: three consecutive rounds
    under one seed must reproduce the independent sampler's levels AND leave
    the RNG in the same place (an extra draw would desynchronise round 2)."""
    groups_by_round = [GROUPS] * 3
    bundle = dict(base_bundle)
    bundle.update(over)
    ad = _adapter(bundle)
    assert ad.copula_rho_p == 0.0 and ad.copula_rho_t == 0.0

    mu_sd = _mu_sd(base_bundle, groups_by_round)  # no RNG: hoisted out anyway
    th.manual_seed(42)
    want = _legacy_levels(bundle, mu_sd)
    want_next = th.randn(1).item()
    th.manual_seed(42)
    got, _ = _run_episode(ad, groups_by_round)
    got_next = th.randn(1).item()

    for t, (w, g) in enumerate(zip(want, got)):
        assert np.array_equal(w, g), f"round {t}: {w.tolist()} != {g.tolist()}"
    assert got_next == want_next, "RNG consumption changed"
    assert ad._copula_z == {}, "the legacy path drew a persistent latent"


# --------------------------------------------------------------------------- #
# (f) sample=False and determinism
# --------------------------------------------------------------------------- #
def test_sample_false_ignores_both_fields_and_draws_nothing(base_bundle):
    """`sample=False` returns the rounded mean and consumes no RNG, exactly as
    the legacy deterministic path does, whatever the weights."""
    groups_by_round = [GROUPS] * 3
    ad = _adapter(_bundle(base_bundle, 0.3, 0.2), sample=False)
    want = [
        np.clip(np.rint(mu), 0, N_CONTRIBUTIONS - 1).astype(np.int64)
        for mu, _ in _mu_sd(base_bundle, groups_by_round)
    ]

    th.manual_seed(42)
    got, _ = _run_episode(ad, groups_by_round)
    got_next = th.randn(1).item()
    th.manual_seed(42)
    want_next = th.randn(1).item()

    for t, (w, g) in enumerate(zip(want, got)):
        assert np.array_equal(w, g), f"round {t}: {w.tolist()} != {g.tolist()}"
    assert got_next == want_next, "the deterministic path consumed the RNG"
    assert ad._copula_z == {}

    # ...and the sampler itself is deterministic when called directly
    Xs = ad.scaler.transform(toy_features(N_AGENTS, 58))
    th.manual_seed(3)
    direct = ad._sample_levels_gaussian_copula(Xs, N_CONTRIBUTIONS, GROUPS)
    direct_next = th.randn(1).item()
    th.manual_seed(3)
    assert th.randn(1).item() == direct_next
    assert np.array_equal(
        direct,
        np.clip(np.rint(ad.estimator.predict(Xs)), 0, N_CONTRIBUTIONS - 1).astype(
            np.int64
        ),
    )


def test_determinism_under_manual_seed(base_bundle):
    ad = _adapter(_bundle(base_bundle, 0.2, 0.1))
    groups_by_round = [GROUPS] * 4
    th.manual_seed(42)
    first, _ = _run_episode(ad, groups_by_round)
    th.manual_seed(42)
    again, _ = _run_episode(ad, groups_by_round)
    th.manual_seed(43)
    other, _ = _run_episode(ad, groups_by_round)
    for a, b in zip(first, again):
        assert np.array_equal(a, b)
    assert any(not np.array_equal(a, b) for a, b in zip(first, other))
    # genuinely drawing, not collapsed onto the rounded mean
    assert np.stack(first).std(0).max() > 0.0
