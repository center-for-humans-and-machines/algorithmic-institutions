"""Unit tests for the severity-copula punishment sampler (autoresearch
punisher-severity-copula, plan step 11).

The copula path adds ONE shared severity latent per group to the multinomial
punisher: marginals stay exactly what the fitted model says, but a group's
punishments co-move, because one human manager decides them together. What has
to hold:

  * the inverse CDF is the discrete quantile F_i(a-1) < u <= F_i(a);
  * the per-agent marginals are unchanged versus the independent sampler;
  * levels correlate WITHIN a group at rho > 0 and not ACROSS groups;
  * a bundle without `copula_rho` (or with 0.0) is bit-identical to the
    pre-change code path, RNG consumption included -- the legacy sampling is
    reimplemented here from the committed two-liner, so the comparison does not
    depend on the code under test;
  * seeding is deterministic, and the __init__ gate refuses a rho it cannot
    honour.

Local test (CPU torch, no PyG):
    .venv/bin/python -m pytest tests/baselines/test_punishment_copula.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch as th  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # tests/baselines -> repo root
# this checkout's src must win over any installed/editable aimanager, so the
# adapter under test is the one in THIS worktree
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/baselines"))

from aimanager.simulation import linear_ah  # noqa: E402
from aimanager.simulation.linear_ah import LinearAHAdapter  # noqa: E402

REAL_BUNDLE = ROOT / "artifacts/baselines/punishment_multinomial_severity_copula.joblib"
N_LEVELS = 31
RHO = 0.35
N_AGENTS = 8
GROUPS = [0, 0, 0, 0, 1, 1, 1, 1]
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
N_DRAW = 4000  # sampling repeats for the marginal / correlation tests


# --------------------------------------------------------------------------- #
# synthetic bundles
# --------------------------------------------------------------------------- #
def toy_features(n, seed):
    """Raw feature rows in the ranges the real game produces -- contributions
    0-20, punishments 0-7, 24 rounds. The toy model must be trained on and
    evaluated at the SAME scale, or the standardised logistic saturates and P
    collapses onto one level (which would make every sampling test vacuous)."""
    rng = np.random.default_rng(seed)
    prev_contribution = rng.integers(0, 21, n).astype(float)
    prev_punishment = rng.integers(0, 8, n).astype(float)
    round_number = rng.integers(1, 24, n).astype(float)
    return np.column_stack(
        [prev_contribution, prev_punishment, round_number, np.zeros(n)]
    )


def toy_bundle(**over):
    """A small but real multinomial punishment bundle: logistic regression of a
    punish-the-low-contributor rule with noise, four punishment levels,
    standardised inputs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = toy_features(600, 7)
    rng = np.random.default_rng(8)
    score = (10.0 - X[:, 0]) / 6.0 + 0.4 * X[:, 1] + rng.normal(0.0, 0.8, len(X))
    y = np.array([0, 4, 9, 16])[np.digitize(score, [-0.4, 0.8, 2.0])]
    scaler = StandardScaler().fit(X)
    est = LogisticRegression(max_iter=500).fit(scaler.transform(X), y)
    bundle = dict(
        model="multinomial",
        estimator=est,
        scaler=scaler,
        features=list(FEATS),
        target="punishment",
        n_levels=N_LEVELS,
        default_values=dict(DEFAULTS),
        switch_every=4,
    )
    bundle.update(over)
    return bundle


class FixedProba:
    """Estimator stub with a hand-built predict_proba. The rows are dyadic, so
    the floor-and-renormalise in _class_probs is exact and the cumulative
    boundaries are exactly representable -- which is what lets the test pin the
    <= convention at u = F(a) instead of merely near it."""

    def __init__(self, rows, classes):
        self.rows = np.asarray(rows, dtype=float)
        self.classes_ = np.asarray(classes)

    def predict_proba(self, Xs):
        assert len(Xs) == len(self.rows)
        return self.rows.copy()


def toy_rounds(n_rounds=2, n_agents=N_AGENTS, groups=None):
    """Round dicts as the simulation hands them to get_punishments."""
    groups = list(GROUPS if groups is None else groups)[:n_agents]
    return [
        dict(
            contribution=[float(4 + (a + t) % 7) for a in range(n_agents)],
            punishment=[float((a * 3 + t) % 5) for a in range(n_agents)],
            agent_group=list(groups),
            contribution_valid=[True] * n_agents,
        )
        for t in range(n_rounds)
    ]


def adapter_Xs(ad, rounds):
    """The standardised feature matrix get_punishments would build (feature
    path untouched by this change)."""
    pool = ad._pool_from_rounds(rounds)
    X = np.column_stack([pool[f][0, :, len(rounds) - 1] for f in ad.features])
    return ad.scaler.transform(X)


def legacy_levels(bundle, Xs):
    """The sampling committed BEFORE this change, reimplemented verbatim from
    src/aimanager/simulation/linear_ah.py::_sample_levels (multinomial branch,
    temperature 1.0): floor, scatter, renormalise, ONE th.multinomial draw."""
    est = bundle["estimator"]
    P = np.full((len(Xs), bundle["n_levels"]), 1e-12)
    P[:, est.classes_] = est.predict_proba(Xs)
    P /= P.sum(1, keepdims=True)
    return th.multinomial(th.from_numpy(P), 1).reshape(-1).numpy().astype(np.int64)


# --------------------------------------------------------------------------- #
# (a) inverse-CDF correctness on a hand-built P
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "u, expected",
    [
        (1e-12, 0),  # far below F(0)
        (0.25, 0),
        (0.5, 0),  # exactly F(0) -> still level 0 (F(a-1) < u <= F(a))
        (0.5 + 1e-9, 1),  # just above F(0) -> next level
        (0.75 - 1e-9, 1),
        (0.75 + 1e-9, 2),  # just above F(1)
        (1.0 - 1e-12, 2),  # top level, no overflow past n_levels - 1
    ],
)
def test_inverse_cdf_convention(monkeypatch, u, expected):
    from scipy.special import ndtri

    rho = 0.5
    est = FixedProba([[0.5, 0.25, 0.25]], [0, 1, 2])  # cumsum 0.5 / 0.75 / 1.0
    ad = LinearAHAdapter(
        dict(
            model="multinomial",
            estimator=est,
            scaler=None,
            features=list(FEATS),
            target="punishment",
            n_levels=3,
            default_values=dict(DEFAULTS),
            copula_rho=rho,
        )
    )
    # drive the sampler at a prescribed u: the group latent carries
    # ndtri(u)/sqrt(rho) and the idiosyncratic draw is zero
    latent = np.array([ndtri(u) / np.sqrt(rho)])
    seen = []

    def fake_randn(n, dtype=None):
        seen.append(n)
        vals = latent if len(seen) == 1 else np.zeros(n)
        return th.tensor(vals, dtype=th.float64)

    monkeypatch.setattr(linear_ah.th, "randn", fake_randn)
    lvl = ad._sample_levels_copula(np.zeros((1, len(FEATS))), 3, [0])
    assert seen == [1, 1], "expected exactly 2 randn calls of size A"
    assert lvl.tolist() == [expected]
    assert lvl.dtype == np.int64


def test_deterministic_path_ignores_rho():
    """sample=False keeps the deterministic argmax, copula or not."""
    est = FixedProba([[0.25, 0.5, 0.25]], [0, 1, 2])
    ad = LinearAHAdapter(
        dict(
            model="multinomial",
            estimator=est,
            scaler=None,
            features=list(FEATS),
            target="punishment",
            n_levels=3,
            default_values=dict(DEFAULTS),
            copula_rho=RHO,
        ),
        sample=False,
    )
    lvl = ad._sample_levels_copula(np.zeros((1, len(FEATS))), 3, [0])
    assert lvl.tolist() == [1]


# --------------------------------------------------------------------------- #
# sampling fixtures for (b), (c), (d)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def samples():
    """N_DRAW draws of the 8 agents, from the copula sampler at RHO, from the
    copula sampler at rho=0, and from the independent sampler -- plus the
    analytic P every one of them should reproduce per agent."""
    X = toy_features(N_AGENTS, 3)
    out = {}
    for name, rho in (("copula", RHO), ("rho0", 0.0)):
        ad = LinearAHAdapter(toy_bundle(copula_rho=rho))
        Xs = ad.scaler.transform(X)
        th.manual_seed(11)
        out[name] = np.stack(
            [ad._sample_levels_copula(Xs, N_LEVELS, GROUPS) for _ in range(N_DRAW)]
        )
    ad = LinearAHAdapter(toy_bundle())
    Xs = ad.scaler.transform(X)
    out["P"] = ad._class_probs(Xs, N_LEVELS)
    th.manual_seed(11)
    out["independent"] = np.stack(
        [ad._sample_levels(Xs, N_LEVELS) for _ in range(N_DRAW)]
    )
    # identical-feature variant: every agent shares one P, so a level
    # correlation can only come from the shared latent
    ad = LinearAHAdapter(toy_bundle(copula_rho=RHO))
    Xs_same = ad.scaler.transform(np.repeat(X[:1], N_AGENTS, axis=0))
    th.manual_seed(12)
    out["same_copula"] = np.stack(
        [ad._sample_levels_copula(Xs_same, N_LEVELS, GROUPS) for _ in range(N_DRAW)]
    )
    ad0 = LinearAHAdapter(toy_bundle(copula_rho=0.0))
    th.manual_seed(12)
    out["same_rho0"] = np.stack(
        [ad0._sample_levels_copula(Xs_same, N_LEVELS, GROUPS) for _ in range(N_DRAW)]
    )
    return out


# --------------------------------------------------------------------------- #
# (b) marginals preserved
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("source", ["copula", "independent"])
def test_marginals_match_model(samples, source):
    """Per-agent level frequencies must match the analytic P within 4 binomial
    SEs (SE = sqrt(p(1-p)/N)); levels with fewer than 20 expected counts are
    skipped, the normal approximation not being usable there."""
    P, draws = samples["P"], samples[source]
    worst = 0.0
    for agent in range(N_AGENTS):
        counts = np.bincount(draws[:, agent], minlength=N_LEVELS)
        freq = counts / N_DRAW
        for lvl in range(N_LEVELS):
            p = P[agent, lvl]
            if p * N_DRAW < 20:
                continue
            se = np.sqrt(p * (1.0 - p) / N_DRAW)
            worst = max(worst, abs(freq[lvl] - p) / se)
    assert worst < 4.0, f"{source}: worst deviation {worst:.2f} binomial SEs"


def test_marginals_match_independent_sampler(samples):
    """Copula vs independent sampler, agent by agent: the two histograms are
    two noisy estimates of the same P, so they must agree within 4 SEs of
    their difference."""
    cop, ind, P = samples["copula"], samples["independent"], samples["P"]
    worst = 0.0
    for agent in range(N_AGENTS):
        fc = np.bincount(cop[:, agent], minlength=N_LEVELS) / N_DRAW
        fi = np.bincount(ind[:, agent], minlength=N_LEVELS) / N_DRAW
        for lvl in range(N_LEVELS):
            p = P[agent, lvl]
            if p * N_DRAW < 20:
                continue
            se = np.sqrt(2.0 * p * (1.0 - p) / N_DRAW)  # difference of two
            worst = max(worst, abs(fc[lvl] - fi[lvl]) / se)
    assert worst < 4.0, f"worst copula-vs-independent gap {worst:.2f} SEs"


def test_mean_punishment_unchanged(samples):
    """Coarser but blunt: the per-agent mean level is the quantity the
    simulation's P rows feed, and it must not move."""
    P = samples["P"]
    target = (P * np.arange(N_LEVELS)).sum(1)
    for source in ("copula", "independent"):
        got = samples[source].mean(0)
        sd = np.sqrt((P * (np.arange(N_LEVELS) ** 2)).sum(1) - target**2)
        assert np.all(np.abs(got - target) < 4.0 * sd / np.sqrt(N_DRAW))


# --------------------------------------------------------------------------- #
# (c) within-group correlation, (d) cross-group independence
# --------------------------------------------------------------------------- #
def _mean_pair_corr(draws, pairs):
    return float(
        np.mean([np.corrcoef(draws[:, i], draws[:, j])[0, 1] for i, j in pairs])
    )


WITHIN = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7)]
CROSS = [(0, 4), (0, 5), (1, 6), (2, 7), (3, 4), (3, 7)]


def test_within_group_correlation_induced(samples):
    """At rho=0.35 the level correlation lands near 0.25 -- lower than rho
    itself, because discretising the latent costs dependence. Asserted as
    "many SEs above zero" (SE of a zero correlation is 1/sqrt(N))."""
    within = _mean_pair_corr(samples["same_copula"], WITHIN)
    assert within > 6.0 / np.sqrt(N_DRAW), f"within-group corr {within:.4f}"


def test_cross_group_correlation_absent(samples):
    cross = _mean_pair_corr(samples["same_copula"], CROSS)
    assert abs(cross) < 4.0 / np.sqrt(N_DRAW), f"cross-group corr {cross:.4f}"


def test_rho_zero_leaves_agents_independent(samples):
    within = _mean_pair_corr(samples["same_rho0"], WITHIN)
    assert abs(within) < 4.0 / np.sqrt(N_DRAW), f"corr {within:.4f} at rho=0"


def test_within_beats_cross(samples):
    within = _mean_pair_corr(samples["same_copula"], WITHIN)
    cross = _mean_pair_corr(samples["same_copula"], CROSS)
    assert within > cross + 5.0 / np.sqrt(N_DRAW)


# --------------------------------------------------------------------------- #
# (e) legacy path bit-identical (rho absent / rho = 0)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("over", [{}, {"copula_rho": 0.0}])
def test_legacy_bundles_bit_identical(over):
    """Three consecutive calls under one seed: identical values AND identical
    RNG consumption (an extra draw would desynchronise call 2)."""
    bundle = toy_bundle(**over)
    ad = LinearAHAdapter(bundle)
    rounds = toy_rounds()
    Xs = adapter_Xs(ad, rounds)

    th.manual_seed(42)
    want = [legacy_levels(bundle, Xs) for _ in range(3)]
    want_next = th.randn(1).item()  # where the legacy path leaves the stream
    th.manual_seed(42)
    got = [ad.get_punishments(rounds).numpy() for _ in range(3)]
    got_next = th.randn(1).item()

    assert ad.copula_rho == 0.0
    for i, (w, g) in enumerate(zip(want, got)):
        assert np.array_equal(w, g), f"call {i}: {w.tolist()} != {g.tolist()}"
    assert got_next == want_next, "RNG consumption changed"


# --------------------------------------------------------------------------- #
# (f) determinism
# --------------------------------------------------------------------------- #
def test_copula_determinism_under_seed():
    ad = LinearAHAdapter(toy_bundle(copula_rho=RHO))
    rounds = toy_rounds()
    th.manual_seed(42)
    first = [ad.get_punishments(rounds).numpy() for _ in range(20)]
    th.manual_seed(42)
    again = [ad.get_punishments(rounds).numpy() for _ in range(20)]
    for a, b in zip(first, again):
        assert np.array_equal(a, b)
    # ...and the sampler is genuinely drawing: 20 calls are not all one vector
    # (consecutive calls CAN coincide -- this P is concentrated on two levels)
    assert not all(np.array_equal(first[0], x) for x in first[1:])


def test_group_composition_does_not_shift_the_stream():
    """2A draws per call whatever the composition: a one-group round consumes
    the same stream as a two-group round, so the two differ only through the
    latent assignment, never through a desynchronised RNG."""
    ad = LinearAHAdapter(toy_bundle(copula_rho=RHO))
    Xs = adapter_Xs(ad, toy_rounds())
    th.manual_seed(7)
    ad._sample_levels_copula(Xs, N_LEVELS, GROUPS)
    after_two = th.randn(1).item()
    th.manual_seed(7)
    ad._sample_levels_copula(Xs, N_LEVELS, [0] * N_AGENTS)
    after_one = th.randn(1).item()
    assert after_two == after_one


def test_single_group_shares_one_latent():
    """All agents in one group read one latent: with identical feature rows the
    sampled levels are then identical too (same u, same CDF)."""
    ad = LinearAHAdapter(toy_bundle(copula_rho=0.999999))
    Xs = ad.scaler.transform(np.repeat(toy_features(1, 5), 4, axis=0))
    th.manual_seed(3)
    lvl = ad._sample_levels_copula(Xs, N_LEVELS, [0, 0, 0, 0])
    assert len(set(lvl.tolist())) == 1


# --------------------------------------------------------------------------- #
# (g) the __init__ gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "over, match",
    [
        ({"copula_rho": 1.0}, r"\[0, 1\)"),
        ({"copula_rho": -0.1}, r"\[0, 1\)"),
        ({"copula_rho": RHO, "target": "contribution"}, "multinomial punishment"),
        ({"copula_rho": RHO, "model": "ridge"}, "multinomial punishment"),
    ],
)
def test_gate_rejects(over, match):
    with pytest.raises(AssertionError, match=match):
        LinearAHAdapter(toy_bundle(**over))


@pytest.mark.parametrize("over", [{}, {"copula_rho": None}, {"copula_rho": 0.0}])
def test_gate_accepts_absent_rho(over):
    """A rho-free bundle (any target) stays on the independent path."""
    assert LinearAHAdapter(toy_bundle(**over)).copula_rho == 0.0
    assert LinearAHAdapter(toy_bundle(target="contribution", **over)).copula_rho == 0.0


def test_gate_accepts_the_real_bundle():
    ad = LinearAHAdapter(_real_bundle())
    assert 0.0 < ad.copula_rho < 1.0
    assert ad.is_punishment and ad.model_type == "multinomial"


# --------------------------------------------------------------------------- #
# integration: the calibrated artifact through get_punishments
# --------------------------------------------------------------------------- #
def _real_bundle():
    import joblib

    if not REAL_BUNDLE.exists():  # pragma: no cover
        pytest.skip(f"{REAL_BUNDLE} missing")
    return joblib.load(REAL_BUNDLE)


def test_real_bundle_get_punishments():
    bundle = _real_bundle()
    assert bundle["copula_estimator"] == "pairwise_mle"
    ad = LinearAHAdapter(bundle)
    assert ad.copula_rho == pytest.approx(bundle["copula_rho"])
    rounds = toy_rounds(n_rounds=3)

    th.manual_seed(42)
    first = ad.get_punishments(rounds)
    th.manual_seed(42)
    again = ad.get_punishments(rounds)

    assert first.shape == (N_AGENTS,)
    assert first.dtype == th.int64
    assert int(first.min()) >= 0 and int(first.max()) <= bundle["n_levels"] - 1
    assert th.equal(first, again)


def test_real_bundle_group_spread_beats_independent():
    """End-to-end mechanism check on the real artifact: replayed over many
    rounds, the copula must spread group MEAN punishments more than the
    independent sampler does, at equal marginals. This is the adapter-level
    echo of the calibration script's pre-flight."""
    bundle = _real_bundle()
    rounds = toy_rounds(n_rounds=3)
    ad_cop = LinearAHAdapter(bundle)
    ad_ind = LinearAHAdapter({k: v for k, v in bundle.items() if k != "copula_rho"})
    assert ad_ind.copula_rho == 0.0

    spread, mean = {}, {}
    for name, ad in (("copula", ad_cop), ("independent", ad_ind)):
        th.manual_seed(5)
        draws = np.stack([ad.get_punishments(rounds).numpy() for _ in range(600)])
        gm = np.stack([draws[:, :4].mean(1), draws[:, 4:].mean(1)]).reshape(-1)
        spread[name] = float(gm.std(ddof=1) / draws.reshape(-1).std(ddof=1))
        mean[name] = float(draws.mean())
    assert spread["copula"] > spread["independent"] * 1.05, spread
    assert mean["copula"] == pytest.approx(mean["independent"], rel=0.05), mean
