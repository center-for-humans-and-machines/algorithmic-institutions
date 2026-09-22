"""The battery's arithmetic, on tables whose answers are known by hand.

No torch_geometric and no models: every input here is synthetic, so a
failure is the statistic's and not the artificial humans'.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, spearmanr

from aimanager.llm_manager import battery as bat
from aimanager.manager.paired_rollout import RPA_LABELS


def _expand(counts):
    """A contingency table back to the raw pairs, for scipy to check."""
    c, p = np.nonzero(counts)
    return (
        np.repeat(c, counts[c, p].astype(int)),
        np.repeat(p, counts[c, p].astype(int)),
    )


def _table(pairs, n_c=21, n_p=31):
    t = np.zeros((n_c, n_p))
    for c, p, n in pairs:
        t[c, p] += n
    return t


# --------------------------------------------------------------------- #
# the rank, and what ties do to it
# --------------------------------------------------------------------- #
def test_spearman_matches_scipy_with_heavy_ties():
    rng = np.random.default_rng(0)
    counts = np.zeros((21, 31))
    for _ in range(200):
        counts[rng.integers(0, 21), rng.integers(0, 31)] += rng.integers(1, 40)
    c, p = _expand(counts)
    assert bat.spearman_from_counts(counts) == pytest.approx(
        spearmanr(c, p).statistic, abs=1e-9
    )


def test_a_constant_manager_has_no_rank_at_all():
    # every decision is the same punishment: no variance, so nan not 0
    assert np.isnan(bat.spearman_from_counts(_table([(c, 0, 10) for c in range(21)])))


def test_rho_floor_is_attained_by_the_countermonotone_table():
    # lowest contributions get the highest punishment, one level each
    t = _table([(c, 20 - c, 10) for c in range(21)])
    assert bat.spearman_from_counts(t) == pytest.approx(bat.rho_floor_from_counts(t))


def test_the_bounds_hold_both_ways_for_every_joint_with_those_margins():
    rng = np.random.default_rng(1)
    for _ in range(20):
        t = rng.integers(0, 12, size=(21, 31)).astype(float)
        rho = bat.spearman_from_counts(t)
        assert bat.rho_floor_from_counts(t) - 1e-9 <= rho
        assert rho <= bat.rho_ceiling_from_counts(t) + 1e-9


def test_rho_rel_keeps_the_sign_and_stays_inside_one():
    """An inverted rule must not score beyond 1, nor a correct one above 0.

    Normalising by the floor alone does both: the bound in the wrong
    direction is the wrong size (measured: `band16_p20` reaches rho +0.498
    against a floor of -0.458, which would read as 1.086) and dividing a
    negative rho by a negative floor reports correct targeting as +1.
    """
    correct = _table([(c, 20 - c, 10) for c in range(21)])
    inverted = _table([(c, c + 5, 10) for c in range(21)])
    for t, sign in ((correct, -1), (inverted, +1)):
        rho = bat.spearman_from_counts(t)
        rel = bat.rho_relative(
            rho, bat.rho_floor_from_counts(t), bat.rho_ceiling_from_counts(t)
        )
        assert np.sign(rel) == sign
        assert abs(rel) == pytest.approx(1.0)
    # a quiet inverted rule: bounded well short of 1 on rank, but at its own
    # ceiling, which is exactly what rho_rel is for
    quiet_inv = _table(
        [(c, 10, 5) for c in range(17, 21)] + [(c, 0, 100) for c in range(17)]
    )
    assert bat.spearman_from_counts(quiet_inv) < 0.6
    assert bat.rho_relative(
        bat.spearman_from_counts(quiet_inv),
        bat.rho_floor_from_counts(quiet_inv),
        bat.rho_ceiling_from_counts(quiet_inv),
    ) == pytest.approx(1.0)


def test_ties_attenuate_the_rank_and_rho_rel_undoes_it():
    """The same aim, told quietly and loudly, scores differently on rank.

    Both managers punish exactly the bottom four contribution levels and
    nobody else; the quiet one is simply asked fewer questions about them,
    so more of its column is tied at zero. Rank falls, `rho_rel` does not.
    """
    loud = _table([(c, 5, 100) for c in range(4)] + [(c, 0, 100) for c in range(4, 21)])
    quiet = _table([(c, 5, 5) for c in range(4)] + [(c, 0, 100) for c in range(4, 21)])
    r_loud, r_quiet = (bat.spearman_from_counts(t) for t in (loud, quiet))
    assert r_quiet > r_loud  # less negative: attenuated
    rel = [bat.targeting(t, _shape_frame(t))["rho_rel"] for t in (loud, quiet)]
    assert rel[0] == pytest.approx(-1.0) and rel[1] == pytest.approx(-1.0)
    assert (
        bat.tie_structure(quiet)["zero_share"] > bat.tie_structure(loud)["zero_share"]
    )


def test_tie_structure_is_the_share_of_tied_pairs():
    t = _table([(0, 0, 2), (1, 1, 2)])  # n = 4
    s = bat.tie_structure(t)
    # 2 pairs tied on contribution out of 6, same on punishment
    assert s["tie_frac_c"] == pytest.approx(2 / 6)
    assert s["tie_frac_p"] == pytest.approx(2 / 6)
    assert s["zero_share"] == pytest.approx(0.5)


# --------------------------------------------------------------------- #
# magnitude and the noise gate
# --------------------------------------------------------------------- #
def _shape_frame(counts=None, means=None, n_per_bin=100, episodes=10, jitter=0.0):
    """A per-episode frame carrying only the six policy-shape bins."""
    if means is None:
        means = _bin_means_of(counts)
    rng = np.random.default_rng(3)
    den = np.full((episodes, 6), float(n_per_bin))
    num = den * np.asarray(means)[None, :]
    if jitter:
        num = num + rng.normal(0, jitter * n_per_bin, size=num.shape)
    return pd.DataFrame(
        {
            **{f"rpa_p_{lab}": num[:, i] for i, lab in enumerate(RPA_LABELS)},
            **{f"rpa_n_{lab}": den[:, i] for i, lab in enumerate(RPA_LABELS)},
        }
    )


def _bin_means_of(counts):
    edges = [(0, 0), (1, 5), (6, 10), (11, 15), (16, 19), (20, 20)]
    lv = np.arange(counts.shape[1])
    out = []
    for lo, hi in edges:
        block = counts[lo : hi + 1]
        n = block.sum()
        out.append((block * lv[None, :]).sum() / n if n else 0.0)
    return out


def test_a_flat_policy_is_caught_by_magnitude_not_by_rank():
    flat = _shape_frame(means=[5.00, 5.00, 5.00, 4.99, 4.99, 4.99], jitter=0.0)
    t = bat.targeting_triple_from_arrays(
        flat[[f"rpa_p_{x}" for x in RPA_LABELS]].to_numpy(float),
        flat[[f"rpa_n_{x}" for x in RPA_LABELS]].to_numpy(float),
    )
    assert t["magnitude"] < 0.01


def test_a_deterministic_rule_has_an_unbounded_gate():
    t = bat.targeting_triple_from_arrays(
        *[
            _shape_frame(means=[10, 10, 4, 0, 0, 0])[
                [f"rpa_{k}_{x}" for x in RPA_LABELS]
            ].to_numpy(float)
            for k in ("p", "n")
        ]
    )
    assert np.isinf(t["noise_gate"]) and t["bin_mean_range"] == pytest.approx(10.0)


def test_a_bin_below_the_minimum_count_is_dropped_not_guessed():
    f = _shape_frame(means=[1, 1, 1, 1, 1, 1], n_per_bin=100)
    f["rpa_n_{20}"] = 1.0
    m, _ = bat._bin_means_and_se(
        f[[f"rpa_p_{x}" for x in RPA_LABELS]].to_numpy(float),
        f[[f"rpa_n_{x}" for x in RPA_LABELS]].to_numpy(float),
        min_bin_n=20,
    )
    assert np.isnan(m[-1]) and not np.isnan(m[0])


# --------------------------------------------------------------------- #
# the leaver diagnostic: ordering only
# --------------------------------------------------------------------- #
def test_c_gap_is_pooled_over_episodes_not_averaged_per_episode():
    ep = pd.DataFrame(
        {
            "lv_n": [1.0, 3.0],
            "lv_c": [2.0, 30.0],
            "st_n": [4.0, 4.0],
            "st_c": [8.0, 24.0],
        }
    )
    row = bat._leaver_row(ep)
    assert row["c_leavers"] == pytest.approx(32 / 4)
    assert row["c_stayers"] == pytest.approx(32 / 8)
    assert row["c_gap"] == pytest.approx(4.0)
    assert row["leave_rate"] == pytest.approx(4 / 12)


def test_leaver_ordering_ranks_and_never_calls_a_sign():
    b = pd.DataFrame({"arm": ["a", "b", "c"], "c_gap": [+1.7, -3.5, -0.1]})
    o = bat.leaver_ordering(b)
    assert list(o["arm"]) == ["b", "c", "a"]
    assert list(o["rank"]) == [1, 2, 3]
    # b and c are far apart, c and a are too; nothing anywhere is a sign test
    assert bool(o.loc[0, "separated_from_next"])
    assert not any("sign" in c or "inverted" in c or "correct" in c for c in o.columns)


def test_neighbours_inside_the_noise_floor_are_not_separated():
    b = pd.DataFrame({"arm": ["a", "b"], "c_gap": [-1.20, -1.00]})
    o = bat.leaver_ordering(b)
    assert not bool(o.loc[0, "separated_from_next"])
    assert bat.C_GAP_NOISE_FLOOR == pytest.approx(0.577)


# --------------------------------------------------------------------- #
# noise floor and the minimum detectable difference
# --------------------------------------------------------------------- #
def _symmetric_episodes(n=400, seed=0, bias=0.0):
    rng = np.random.default_rng(seed)
    d = {}
    for q in bat.HEADLINE:
        shared = rng.normal(0, 1, n)
        d[f"focal_{q}"] = 10 + shared + rng.normal(0, 1, n) + bias
        d[f"rival_{q}"] = 10 + shared + rng.normal(0, 1, n)
    return pd.DataFrame(d)


def test_a_symmetric_control_finds_no_seat_bias():
    f = bat.noise_floor(_symmetric_episodes())
    assert set(f["quantity"]) == set(bat.HEADLINE)
    for _, r in f.iterrows():
        assert r["seat_bias_lo"] < 0 < r["seat_bias_hi"]


def test_the_symmetric_control_does_find_a_bias_when_there_is_one():
    f = bat.noise_floor(_symmetric_episodes(bias=1.0))
    assert (f["seat_bias_lo"] > 0).all()


def test_which_design_is_cheaper_is_a_per_quantity_fact():
    """Sharing an episode helps; partitioning eight players between the
    seats hurts, and the real harness does both at once.

    Positively correlated seats give `sd_diff < sqrt(2) * sd_episode`;
    seats whose values must sum to a constant give exactly
    `sd_diff = 2 * sd_episode`, which is dearer, not cheaper. `members`
    really is the second case in this game.
    """
    shared = bat.noise_floor(_symmetric_episodes())
    assert (shared["sd_diff"] < np.sqrt(2) * shared["sd_episode"]).all()

    rng = np.random.default_rng(5)
    n = 500
    d = {}
    for q in bat.HEADLINE:
        f = rng.normal(4, 1, n)
        d[f"focal_{q}"], d[f"rival_{q}"] = f, 8 - f
    split = bat.noise_floor(pd.DataFrame(d))
    assert split["sd_diff"].to_numpy() == pytest.approx(
        2 * split["sd_episode"].to_numpy()
    )
    m = bat.mdd_table(split, (200,))
    assert (m["mdd_within_run"] > m["mdd_unpaired"]).all()


def test_mdd_is_the_textbook_formula_and_scales_as_one_over_root_n():
    f = pd.DataFrame([{"quantity": "pool", "sd_episode": 10.0, "sd_diff": 6.0}])
    t = bat.mdd_table(f, episode_counts=(50, 200))
    z = norm.ppf(0.975) + norm.ppf(0.8)
    got = t.set_index("episodes")
    assert got.loc[50, "mdd_unpaired"] == pytest.approx(
        z * np.sqrt(2) * 10.0 / np.sqrt(50)
    )
    assert got.loc[50, "mdd_within_run"] == pytest.approx(z * 6.0 / np.sqrt(50))
    assert got.loc[50, "mdd_unpaired"] / got.loc[200, "mdd_unpaired"] == pytest.approx(
        2.0
    )


def test_detectability_inverts_the_mdd():
    """`n` from `detectability` is the `n` at which the MDD meets the effect."""
    f = pd.DataFrame([{"quantity": "pool", "sd_episode": 10.0, "sd_diff": 6.0}])
    effects = [{"effect": "capped", "quantity": "pool", "size": 5.0, "source": "x"}]
    d = bat.detectability(f, effects).iloc[0]
    n = int(d["n_episodes_unpaired"])
    assert bat.mdd_table(f, episode_counts=(n,)).iloc[0]["mdd_unpaired"] <= 5.0
    assert bat.mdd_table(f, episode_counts=(n - 1,)).iloc[0]["mdd_unpaired"] > 5.0
    # the paired design is cheaper here, because sd_diff < sqrt(2) * sd
    assert d["n_episodes_within_run"] < d["n_episodes_unpaired"]


def test_a_bigger_effect_needs_fewer_episodes():
    f = pd.DataFrame([{"quantity": "pool", "sd_episode": 10.0, "sd_diff": 6.0}])
    small, big = (
        bat.detectability(
            f, [{"effect": "e", "quantity": "pool", "size": s, "source": "x"}]
        ).iloc[0]["n_episodes_unpaired"]
        for s in (5.0, 32.75)
    )
    assert big < small


def test_the_pool_leads_every_table_and_nothing_is_averaged_with_it():
    assert bat.PRIMARY_OBJECTIVE == "pool"
    assert bat.HEADLINE[0] == "pool"
    assert bat.HEADLINE[1] == "contribution"
    # the power table is sized on the objective first
    assert bat.REFERENCE_EFFECTS[0]["quantity"] == "pool"
    f = bat.noise_floor(_symmetric_episodes())
    assert list(f["quantity"]) == list(bat.HEADLINE)
    assert list(bat.mdd_table(f, (50,))["quantity"]) == list(bat.HEADLINE)


def test_contrasts_show_the_route_not_just_the_objective():
    """Two arms level on the pool, arrived at by opposite routes.

    `a` raises contributions and pays for them; `b` does nothing. The pool
    interval crosses zero for both and the contribution interval does not,
    which is the distinction the objective alone cannot make.
    """
    rng = np.random.default_rng(0)
    n = 600
    rows = []
    for arm, contr in (("never", 40.0), ("a", 52.0), ("b", 40.0)):
        d = {"arm": arm}
        d["focal_pool"] = 60.0 + rng.normal(0, 4, n)
        d["focal_contribution"] = contr + rng.normal(0, 4, n)
        for q in bat.HEADLINE[2:]:
            d[f"focal_{q}"] = rng.normal(0, 1, n)
        rows.append(pd.DataFrame(d))
    ep = pd.concat(rows, ignore_index=True)
    ep["rival"] = "clone"
    c = bat.contrasts(ep, "never").set_index(["arm", "quantity"])
    assert list(c.reset_index()["quantity"].unique()) == list(bat.HEADLINE)
    assert bool(c.loc[("a", "pool"), "crosses_zero"])
    assert bool(c.loc[("b", "pool"), "crosses_zero"])
    assert not bool(c.loc[("a", "contribution"), "crosses_zero"])
    assert bool(c.loc[("b", "contribution"), "crosses_zero"])


def test_an_arm_that_faced_another_rival_is_not_contrasted():
    """A symmetric control differs in the other seat, not in the manager."""
    rng = np.random.default_rng(0)
    rows = []
    for arm, riv in (("never", "clone"), ("a", "clone"), ("never_vs_never", "never")):
        d = {"arm": arm, "rival": riv}
        for q in bat.HEADLINE:
            d[f"focal_{q}"] = rng.normal(10, 2, 200)
        rows.append(pd.DataFrame(d))
    c = bat.contrasts(pd.concat(rows, ignore_index=True), "never")
    assert set(c["arm"]) == {"a"}


def test_the_published_budgets_span_the_existing_baselines():
    # #217 ran at 300 episodes and #219's validation at 6,144; a table that
    # stopped at 500 would not reach the range the baselines actually use
    assert bat.EPISODE_BUDGETS[0] == 50 and bat.EPISODE_BUDGETS[-1] == 3000
    assert {e["quantity"] for e in bat.REFERENCE_EFFECTS} <= set(bat.HEADLINE)
