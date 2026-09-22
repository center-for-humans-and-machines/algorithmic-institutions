"""The harness end to end, with the artificial humans replaced by stand-ins.

These are about the harness -- which seat gets which manager, how episodes
are chunked, what the telemetry counts, whether a symmetric control comes
out symmetric -- and not about the models, so they carry no
torch_geometric and run locally. The validation against the published
numbers needs the real stack and is in `scripts/llm_manager/`.
"""

import numpy as np
import pytest
import torch as th

from aimanager.llm_manager import battery as bat
from aimanager.llm_manager.harness import (
    THR9_P10,
    build_manager,
    contingency_frame,
    run_arm,
    run_battery,
)
from aimanager.llm_manager.stub import THR9_P10_TABLE, StubManager

DEFAULTS = {
    "contribution": 9,
    "punishment": 0,
    "round_number": 0,
    "is_first": False,
    "contribution_valid": False,
    "punishment_valid": False,
    "common_good": 0.0,
    "contributor_payoff": 0.0,
    "agent_group": 0,
    "does_switch": False,
    "switch_mask": False,
}


class Fake:
    autoregressive = False

    def __init__(self, fn):
        self.fn = fn
        self.default_values = DEFAULTS

    def to(self, device):
        return self

    def predict(self, state, reset_rnn=False, edge_index=None, **_):
        return self.fn(state), None


def _contributor(state):
    """Contribute less the harder you were punished last round."""
    base = th.full_like(state["contribution"], 12)
    return (base - state["prev_punishment"]).clamp(0, 20)


def _all_valid(state):
    return th.ones_like(state["contribution"])


def _timeout_agent_three(state):
    v = th.ones_like(state["contribution"])
    v[:, 3] = 0
    return v


def _noisy_contributor(state):
    """As above, plus a draw -- so the world has a seed worth setting.

    It consumes the global torch RNG exactly as a real artificial human
    does, which is what makes "same seed, same world" a real claim rather
    than a property of a deterministic stand-in.
    """
    base = th.full_like(state["contribution"], 12) - state["prev_punishment"]
    jitter = th.randint(0, 9, base.shape, device=base.device) - 4
    return (base + jitter).clamp(0, 20)


def _switch_the_punished(state):
    return (state["prev_punishment"] > 0).to(th.int64)


def _models(valid_fn=_all_valid, contribution_fn=_contributor):
    return {
        "contribution_model": Fake(contribution_fn),
        "valid_model": Fake(valid_fn),
        "switch_model": Fake(_switch_the_punished),
    }


def _noisy_models(valid_fn=_all_valid):
    return _models(valid_fn=valid_fn, contribution_fn=_noisy_contributor)


# --------------------------------------------------------------------- #
# dispatch: the seat really holds the manager the arm names
# --------------------------------------------------------------------- #
def test_each_seat_carries_its_own_manager():
    """Focal punishes 7 flat, rival punishes nothing. Neither leaks."""
    ep, counts, _, _ = run_arm(
        "seven",
        StubManager(punishment=7),
        "never",
        _models(),
        episodes=8,
        seeds=42,
    )
    # every focal decision is a 7, and the rival seat never spends
    assert counts[:, 7].sum() == counts.sum() > 0
    assert ep["rival_punishment"].sum() == 0
    assert ep["focal_punishment"].sum() > 0


def test_the_stub_wearing_thr9_p10_is_the_sigmoid_rule_cell_for_cell():
    """The stub's table and the sigmoid family's `tau -> 0` are one rule.

    Both arms are seeded identically and the seats act identically, so the
    two rollouts are the same world and every recorded number matches.
    """
    kw = dict(models=_models(), episodes=8, seeds=42)
    a, ca, _, _ = run_arm("stub", StubManager(table=THR9_P10_TABLE), "never", **kw)
    b, cb, _, _ = run_arm("rule", "thr9_p10", "never", **kw)
    assert np.array_equal(ca, cb)
    for c in ("focal_contribution", "focal_pool", "focal_members", "lv_c", "st_n"):
        assert a[c].to_numpy() == pytest.approx(b[c].to_numpy())


def test_the_capped_rule_never_issues_more_than_its_ceiling():
    _, counts, _, _ = run_arm(
        "capped", "capped_sigmoid", "never", _models(), episodes=8, seeds=42
    )
    highest = np.nonzero(counts.sum(0))[0].max()
    assert highest <= 10  # P_max = 9.34, rounded


def test_a_manager_object_is_passed_straight_through():
    stub = StubManager(punishment=3)
    assert build_manager(stub, 4, {}, th.device("cpu")) is stub


def test_thr9_p10_is_a_step_at_nine():
    m = build_manager("thr9_p10", 2, {}, th.device("cpu"))
    c = th.arange(21).view(1, 21, 1).repeat(2, 1, 1)
    p, _ = m.predict({"contribution": c, "round_number": th.zeros_like(c)})
    assert p[0, :, 0].tolist() == [10] * 10 + [0] * 11
    assert THR9_P10["p_max"] == 10.0


# --------------------------------------------------------------------- #
# what the harness masks, and what it refuses to average
# --------------------------------------------------------------------- #
def test_a_timed_out_cell_is_not_a_decision():
    """Agent 3 never plays, so nothing it was served reaches the shape.

    Its recorded contribution is the imputed default, which is 9 and would
    land in the `6-10` bin; the mask is applied at source so the rule's
    timeout behaviour is not filed as policy shape.
    """
    _, counts, _, _ = run_arm(
        "constant",
        StubManager(punishment=5),
        "never",
        _models(valid_fn=_timeout_agent_three),
        episodes=8,
        seeds=42,
    )
    # the env zeroes punishment on an invalid cell, so a leaked timeout row
    # would show up as a (contribution 9, punishment 0) cell
    assert counts[9, 0] == 0
    assert counts[:, 5].sum() == counts.sum()


def test_the_battery_keeps_the_pool_and_the_contribution_apart():
    out = run_battery(
        {"never": "never", "seven": StubManager(punishment=7)},
        _models(),
        rival="never",
        episodes=8,
        verbose=False,
    )
    b = out["battery"]
    assert {"focal_pool", "focal_contribution"} <= set(b.columns)
    banned = ("objective", "score", "combined", "overall", "mean_outcome")
    assert not [c for c in b.columns if any(x in c.lower() for x in banned)]
    # and they really do disagree here: punishing costs pool, buys nothing
    # from a contributor that only reacts downwards
    row = b.set_index("arm")
    assert row.loc["seven", "focal_pool"] < row.loc["never", "focal_pool"]


def test_the_reported_ratio_is_pooled_not_an_average_of_ratios():
    """Episodes differ in how many member-rounds they contain.

    The level a paired arm in this project reports is the pooled ratio; the
    average of the per-episode ratios is a different number (11% apart on
    the clone's spend at 6,144 episodes), and both are emitted so nobody has
    to guess which one a table carries.
    """
    ep, counts, _, _ = run_arm(
        "seven",
        StubManager(punishment=7),
        "never",
        _noisy_models(valid_fn=_timeout_agent_three),
        episodes=16,
        seeds=42,
    )
    row = bat.battery_row("seven", ep, counts)
    for q, (num, den) in bat.POOLED_RATIOS.items():
        pooled = ep[f"focal_{num}"].sum() / ep[f"focal_{den}"].sum()
        assert row[f"focal_{q}"] == pytest.approx(pooled)
        assert row[f"focal_{q}_episodemean"] == pytest.approx(ep[f"focal_{q}"].mean())
    # the spread is still taken over episodes, which is the independent unit
    assert row["focal_pool_per_member_sd"] == pytest.approx(
        ep["focal_pool_per_member"].std(ddof=1)
    )


# --------------------------------------------------------------------- #
# episodes, chunking and seeds
# --------------------------------------------------------------------- #
def test_a_partial_chunk_still_produces_exactly_the_episodes_asked_for():
    ep, _, _, _ = run_arm(
        "c", StubManager(punishment=1), "never", _models(), episodes=7, chunk=3
    )
    assert len(ep) == 7
    assert list(ep["episode"]) == list(range(7))
    assert list(ep["rep"]) == [0, 0, 0, 1, 1, 1, 2]


def test_the_same_seed_is_the_same_world():
    kw = dict(models=_noisy_models(), episodes=6, seeds=7)
    a, _, _, _ = run_arm("a", StubManager(punishment=0), "never", **kw)
    b, _, _, _ = run_arm("b", "never", "never", **kw)
    assert a["focal_contribution"].to_numpy() == pytest.approx(
        b["focal_contribution"].to_numpy()
    )


def test_a_different_seed_is_a_different_world():
    kw = dict(models=_noisy_models(), episodes=6)
    a, _, _, _ = run_arm("a", StubManager(punishment=0), "never", seeds=7, **kw)
    b, _, _, _ = run_arm("b", StubManager(punishment=0), "never", seeds=8, **kw)
    assert not np.allclose(
        a["focal_contribution"].to_numpy(), b["focal_contribution"].to_numpy()
    )


# --------------------------------------------------------------------- #
# telemetry, which is the whole point of the stub
# --------------------------------------------------------------------- #
def test_token_counts_are_one_prompt_per_episode_per_round():
    ep, counts, tel, wall = run_arm(
        "t",
        StubManager(
            punishment=2, prompt_tokens_per_call=100, completion_tokens_per_call=8
        ),
        "never",
        _models(),
        episodes=5,
        seeds=42,
    )
    assert tel["n_calls"] == 24
    assert tel["n_decisions_requested"] == 5 * 24
    assert tel["prompt_tokens"] == 100 * 5 * 24
    row = bat.battery_row("t", ep, counts, tel, wall)
    assert row["total_tokens"] == 108 * 5 * 24
    assert row["tokens_per_episode"] == pytest.approx(108 * 24)
    assert row["parse_failure_rate"] == 0.0
    assert row["wall_clock_s"] > 0


def test_a_parse_failure_falls_back_to_zero_and_is_counted():
    stub = StubManager(punishment=9, parse_failure_rate=0.5, seed=3)
    ep, counts, tel, wall = run_arm(
        "f", stub, "never", _models(), episodes=40, seeds=42
    )
    row = bat.battery_row("f", ep, counts, tel, wall)
    assert 0.3 < row["parse_failure_rate"] < 0.7
    # the failed episode-rounds punished 0; the rest punished 9
    assert set(np.nonzero(counts.sum(0))[0].tolist()) == {0, 9}
    assert tel["n_parse_failures"] == pytest.approx(row["parse_failure_rate"] * 40 * 24)


def test_a_rule_reports_no_telemetry_rather_than_zero_telemetry():
    ep, counts, tel, wall = run_arm(
        "rule", "thr9_p10", "never", _models(), episodes=4, seeds=42
    )
    assert tel is None
    row = bat.battery_row("rule", ep, counts, tel, wall)
    assert np.isnan(row["prompt_tokens"]) and np.isnan(row["parse_failure_rate"])


def test_the_failure_draw_does_not_move_the_environment():
    """Turning parse failures on must not reshuffle the world itself.

    The stub draws from its own generator, so the only thing that changes
    between these two runs is the punishment the manager issued.
    """
    kw = dict(models=_noisy_models(), episodes=6, seeds=42)
    a, _, _, _ = run_arm("a", StubManager(punishment=0), "never", **kw)
    b, _, _, _ = run_arm(
        "b", StubManager(punishment=0, parse_failure_rate=0.9, seed=1), "never", **kw
    )
    assert a["focal_contribution"].to_numpy() == pytest.approx(
        b["focal_contribution"].to_numpy()
    )


# --------------------------------------------------------------------- #
# the symmetric control
# --------------------------------------------------------------------- #
def test_the_same_manager_in_both_seats_reports_no_difference():
    ep, _, _, _ = run_arm(
        "sym",
        StubManager(punishment=4),
        StubManager(punishment=4),
        _noisy_models(),
        episodes=64,
        seeds=42,
    )
    f = bat.noise_floor(ep, arm="sym")
    assert set(f["quantity"]) == set(bat.HEADLINE)
    for _, r in f.iterrows():
        assert r["seat_bias_lo"] <= 0 <= r["seat_bias_hi"], r["quantity"]
    m = bat.mdd_table(f, episode_counts=(50, 200, 500))
    assert len(m) == len(bat.HEADLINE) * 3
    assert (m["mdd_unpaired"] >= 0).all() and m["mdd_unpaired"].notna().all()
    # a constant manager's spend and headcount are the same every episode
    # here, so only the stochastic quantities have a floor above zero
    pool = m[m["quantity"] == "pool"].set_index("episodes")
    assert pool.loc[50, "mdd_unpaired"] > 0
    assert pool.loc[50, "mdd_unpaired"] / pool.loc[
        200, "mdd_unpaired"
    ] == pytest.approx(2.0)


def test_run_battery_carries_the_symmetric_control_as_an_arm():
    out = run_battery(
        {
            "never": "never",
            "sym": ("never", "never"),
        },
        _models(),
        rival="never",
        episodes=6,
        verbose=False,
    )
    assert list(out["battery"]["arm"]) == ["never", "sym"]
    assert set(out["contingency"]) == {"never", "sym"}
    cf = contingency_frame(out["contingency"])
    assert set(cf.columns) == {"arm", "contribution", "punishment", "count"}


def test_seeds_are_the_unit_of_replication_and_pool_into_one_frame():
    """Three seeds at 100 episodes is #217's 300-episode budget."""
    ep, _, _, _ = run_arm(
        "s",
        StubManager(punishment=0),
        "never",
        _noisy_models(),
        episodes=3,
        seeds=(7, 8),
    )
    assert len(ep) == 6
    assert list(ep["seed"]) == [7, 7, 7, 8, 8, 8]
    assert list(ep["episode"]) == list(range(6))
    # the two seeds really are different worlds
    a = ep[ep["seed"] == 7]["focal_contribution"].to_numpy()
    b = ep[ep["seed"] == 8]["focal_contribution"].to_numpy()
    assert not np.allclose(a, b)
