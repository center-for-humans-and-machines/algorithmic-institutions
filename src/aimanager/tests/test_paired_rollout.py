"""The batched paired rollout and the quantities read off it.

The artificial humans are replaced by deterministic stand-ins, so these tests
are about the harness -- which seat gets which manager, what a timed-out cell
costs, how a seat total is formed -- and not about the models. No
torch_geometric, so they run locally.
"""

import torch as th

from aimanager.manager.environment import ArtificialHumanEnv
from aimanager.manager.paired_rollout import (
    RPA_LABELS,
    BatchedPairEnv,
    contingency,
    make_env,
    rollout,
    summarise,
)
from aimanager.manager.sigmoid_rule import ConstantManager

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
    """A stand-in artificial human: a fixed function of the served state."""

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
    """Agent 3 never gives an input; everyone else always does."""
    v = th.ones_like(state["contribution"])
    v[:, 3] = 0
    return v


def _switch_the_punished(state):
    """Leave whenever you were punished last round."""
    return (state["prev_punishment"] > 0).to(th.int64)


def _env(valid_fn=_all_valid, switch_fn=_switch_the_punished, batch_size=4, cls=None):
    kw = dict(
        contribution_model=Fake(_contributor),
        valid_model=Fake(valid_fn),
        switch_model=Fake(switch_fn),
        batch_size=batch_size,
        device=th.device("cpu"),
    )
    if cls is None:
        return make_env(**kw)
    return cls(
        artifical_humans=kw["contribution_model"],
        artifical_humans_valid=kw["valid_model"],
        artifical_humans_switch=kw["switch_model"],
        switch_every=4,
        batch_size=batch_size,
        n_agents=8,
        agent_groups=[0, 0, 0, 0, 1, 1, 1, 1],
        n_groups=2,
        n_contributions=21,
        n_punishments=31,
        n_rounds=24,
        device=th.device("cpu"),
    )


def test_edge_index_matches_base():
    """The cached index is the base class's index, not a different graph."""
    fast = _env(cls=BatchedPairEnv, batch_size=3)
    base = _env(cls=ArtificialHumanEnv, batch_size=3)
    assert th.equal(fast.batch_edge_index, base.batch_edge_index)


def test_cached_index_survives_a_reshuffle():
    env = _env(batch_size=2)
    before = env.batch_edge_index
    env.apply_switch(th.tensor([[1, 0, 0, 0, 0, 0, 0, 1]] * 2, dtype=th.bool))
    assert th.equal(env.batch_edge_index, before)
    # the membership mask, which DOES depend on the groups, has moved
    assert env.agent_group_mask[0, 0, 1] == 1


def test_each_seat_gets_its_own_manager():
    """The dispatch check: the focal seat's signature never appears on the
    rival's cells, and a switcher is punished by the seat they are in."""
    env = _env()
    rec = rollout(env, ConstantManager(7), ConstantManager(3))
    focal, rival = rec["group"] == 0, rec["group"] == 1
    assert set(rec["punishment"][focal].tolist()) == {7}
    assert set(rec["punishment"][rival].tolist()) == {3}
    # the switch predictor moved people, so both seats saw membership change
    assert not th.equal(rec["group"][:, :, 0], rec["group"][:, :, -1])


def test_punishment_aimed_at_a_timed_out_player_is_zeroed():
    """`auto/free-punishment-fix`, from the harness's point of view."""
    env = _env(valid_fn=_timeout_agent_three)
    rec = rollout(env, ConstantManager(7), ConstantManager(3))
    assert rec["punishment"][~rec["valid"]].sum() == 0
    assert (rec["punishment"][rec["valid"]] > 0).all()


def _rec(contribution, punishment, valid, group):
    return {
        "contribution": th.tensor(contribution),
        "punishment": th.tensor(punishment),
        "valid": th.tensor(valid),
        "group": th.tensor(group),
    }


def test_seat_totals_zero_the_imputed_contribution():
    """One episode, two agents, two rounds; agent 1 times out in round 1 and
    its recorded contribution is the imputed 9, which must not be counted."""
    rec = _rec(
        contribution=[[[10, 10], [4, 9]]],
        punishment=[[[2, 2], [0, 0]]],
        valid=[[[True, True], [True, False]]],
        group=[[[0, 0], [0, 0]]],
    )
    s = summarise(rec)
    assert s["focal_members"].item() == 2.0
    assert s["focal_n_valid"].item() == 1.5  # (2 + 1) / 2 rounds
    assert s["focal_contribution"].item() == (14 + 10) / 2
    assert s["focal_punishment"].item() == 2.0
    assert abs(s["focal_pool"].item() - (1.6 * 24 - 4) / 2) < 1e-4
    assert s["rival_members"].item() == 0.0


def test_seats_are_split_by_membership_not_by_index():
    rec = _rec(
        contribution=[[[10], [4]]],
        punishment=[[[2], [6]]],
        valid=[[[True], [True]]],
        group=[[[1], [0]]],
    )
    s = summarise(rec)
    assert s["focal_contribution"].item() == 4.0
    assert s["rival_contribution"].item() == 10.0
    assert s["focal_punishment"].item() == 6.0


def test_policy_shape_uses_the_evaluation_suite_bins():
    """One cell per bin edge case, all on the focal seat."""
    contribution = [[[0], [1], [5], [10], [15], [19], [20]]]
    rec = _rec(
        contribution=contribution,
        punishment=[[[1], [2], [3], [4], [5], [6], [7]]],
        valid=[[[True]] * 7],
        group=[[[0]] * 7],
    )
    s = summarise(rec)
    counts = {label: s[f"rpa_n_{label}"].item() for label in RPA_LABELS}
    assert counts == {"{0}": 1, "1-5": 2, "6-10": 1, "11-15": 1, "16-19": 1, "{20}": 1}
    assert s["rpa_p_{0}"].item() == 1.0
    assert s["rpa_p_1-5"].item() == 5.0  # the c = 1 and c = 5 cells


def test_policy_shape_drops_timed_out_cells():
    """Their recorded 9 would otherwise land in the `6-10` column."""
    rec = _rec(
        contribution=[[[9], [9]]],
        punishment=[[[4], [0]]],
        valid=[[[True], [False]]],
        group=[[[0], [0]]],
    )
    s = summarise(rec)
    assert s["rpa_n_6-10"].item() == 1.0
    assert s["rpa_p_6-10"].item() == 4.0


def test_leaver_gap_is_negative_when_the_low_contributors_leave():
    """Decision rounds are s with (s + 1) % 4 == 0, so round 3 of four."""
    contribution = [[[0, 0, 0, 2], [0, 0, 0, 18]]]
    rec = _rec(
        contribution=contribution,
        punishment=[[[0, 0, 0, 10], [0, 0, 0, 0]]],
        valid=[[[True] * 4] * 2],
        group=[[[0, 0, 0, 0], [0, 0, 0, 0]]],
    )
    # agent 0 leaves after the decision round, agent 1 stays
    rec["group"] = th.tensor([[[0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]])
    for k in ("contribution", "punishment", "valid"):
        rec[k] = th.cat([rec[k], rec[k][:, :, -1:]], dim=-1)
    s = summarise(rec)
    assert s["lv_n"].item() == 1 and s["st_n"].item() == 1
    assert s["lv_c"].item() == 2.0 and s["st_c"].item() == 18.0
    assert s["lv_p"].item() == 10.0 and s["st_p"].item() == 0.0


def test_leaver_diagnostic_ignores_non_decision_rounds():
    """Only s with (s + 1) % 4 == 0 counts: this agent moves out and back
    across rounds 1-3 and is still a stayer, because the only decision it
    is read at is round 3 -> 4."""
    rec = _rec(
        contribution=[[[5, 5, 5, 5, 5]]],
        punishment=[[[0] * 5]],
        valid=[[[True] * 5]],
        group=[[[0, 1, 1, 0, 0]]],
    )
    s = summarise(rec)
    assert s["lv_n"].item() == 0.0
    assert s["st_n"].item() == 1.0


def test_leaver_diagnostic_reads_only_the_focal_seat():
    """A member of the rival seat at the decision round is neither."""
    rec = _rec(
        contribution=[[[5, 5, 5, 5, 5]]],
        punishment=[[[0] * 5]],
        valid=[[[True] * 5]],
        group=[[[0, 0, 0, 1, 1]]],
    )
    s = summarise(rec)
    assert s["lv_n"].item() == 0.0 and s["st_n"].item() == 0.0


def test_contingency_counts_only_real_decisions():
    """Two design points, one episode each; the timed-out cell is excluded."""
    rec = _rec(
        contribution=[[[3], [9]], [[20], [20]]],
        punishment=[[[10], [0]], [[0], [0]]],
        valid=[[[True], [False]], [[True], [True]]],
        group=[[[0], [0]], [[0], [0]]],
    )
    cnt = contingency(rec, th.tensor([0, 1]), 2)
    assert cnt.shape == (2, 21, 31)
    assert cnt[0].sum() == 1 and cnt[0, 3, 10] == 1
    assert cnt[1].sum() == 2 and cnt[1, 20, 0] == 2


def test_contingency_ignores_the_rival_seat():
    rec = _rec(
        contribution=[[[5], [5]]],
        punishment=[[[7], [2]]],
        valid=[[[True], [True]]],
        group=[[[0], [1]]],
    )
    cnt = contingency(rec, th.tensor([0]), 1)
    assert cnt.sum() == 1 and cnt[0, 5, 7] == 1
