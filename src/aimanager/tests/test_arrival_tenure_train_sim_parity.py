"""Train/sim parity for the `rounds_since_arrival` node feature (plan step 3).

Runs locally on macOS with plain pytest:

    PYTHONPATH=$PWD/src .venv/bin/python -m pytest \
        src/aimanager/tests/test_arrival_tenure_train_sim_parity.py -q

Context: notes/autoresearch_log/contribution-arrival-tenure.md, steps 1-3.

THE HAZARD. The feature is built through two independent implementations with
no shared code path. Training builds it in pandas
(`generic/data.py::parse_agent_rounds`: a within-player `group_id` shift, a
forward-filled last-arrival round, `round_number - last`, `fillna(4)`,
`clip(upper=4)`). Simulation builds it in torch
(`manager/environment.py::update_rounds_since_arrival`: a comparison of
`agent_group` against `prev_agent_group`, then `where(arrived, 0, (t+1).clamp(
max=4))`). Two different arithmetics -- an absolute difference of round
numbers against an incremental counter -- for one number. If they disagree the
model trains on one feature and is simulated on another, and the whole
experiment is void with no visible symptom: no crash, no shape error, just a
silently different input column.

This file is the check. The expected tenure table below is written out by hand
from the *specification* ("0 on the arrival round, then 1, 2, 3, capped at 4,
and 4 for an agent that has not arrived anywhere yet"), not derived from either
implementation, so it is an independent third oracle rather than a restatement
of one of the two sides.

WHEN THE ENV STATE IS READ. `step()` rolls every `prev_*` key, then calls
`apply_switch` (rebinding `agent_group`), then `update_contribution` -- whose
first line is `update_rounds_since_arrival`. So the only moment the feature is
meaningful is *inside* the contribution model's `predict`, which is where the
stub below snapshots it. That is the exact tensor the contributor's
`IntEncoder` reads. Reading between `punish()` and `step()` would sample a
round-old value.

THE ROUND-0 GUARD. `reset_state` fills `prev_agent_group` from
`default_values["agent_group"]`, which is 0. Without the explicit round-0
branch in `update_rounds_since_arrival`, every agent that starts in group 1
compares 1 != 0 and reads tenure 0 at round 0 -- "everyone in group 1 just
arrived". The scenario below therefore starts four agents in group 1, and
`test_round_zero_is_the_cap_in_both_groups` is the assertion that fails if the
guard is removed. (Verified by removing it: that test and the bulk parity test
both fail.)

WHAT NEEDS PyG, AND WHY NONE OF IT MATTERS HERE. Neither
`aimanager.generic.data` nor `aimanager.manager.environment` imports
`torch_scatter` or `torch_geometric`. The only import needing a stand-in is
`aimanager.generic.encoder`, because that module's top-level `from
torch_scatter import scatter_mean` runs before `IntEncoder` is reachable;
`IntEncoder` never calls it. The stand-in is therefore never exercised, and the
last test in this file proves that rather than asserting it in a comment. On
Raven, where real PyG is installed, no stand-in is installed at all
(`STAND_INS` is empty) and that is checked too.
"""

import importlib
import sys
import types

import pandas as pd
import pytest
import torch as th

# --------------------------------------------------------------------------- #
# stand-in (macOS only) -- installed only when the real package is absent, the
# same discipline as test_group_size_train_sim_parity.py. Every call it
# receives is recorded so the final test can prove nothing under test here
# routed through it.
# --------------------------------------------------------------------------- #
_SCATTER_MEAN_CALLS = []


def _scatter_mean(src, index, dim=0, dim_size=None):
    _SCATTER_MEAN_CALLS.append(dim)
    assert dim == 0, "the stand-in only implements dim=0, which is all graph.py uses"
    index = index.reshape(-1).to(th.int64)
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() else 0
    out = th.zeros((dim_size, *src.shape[1:]), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    count = th.zeros(dim_size, dtype=src.dtype, device=src.device)
    count.index_add_(0, index, th.ones_like(index, dtype=src.dtype))
    shape = (dim_size,) + (1,) * (src.dim() - 1)
    return out / count.reshape(shape).clamp(min=1.0)


def _install_stand_ins():
    installed = []
    try:
        importlib.import_module("torch_scatter")
    except ImportError:
        scatter = types.ModuleType("torch_scatter")
        scatter.scatter_mean = _scatter_mean
        sys.modules.setdefault("torch_scatter", scatter)
        installed.append("torch_scatter")
    return installed


STAND_INS = _install_stand_ins()

from aimanager.generic.data import (  # noqa: E402
    create_torch_data,
    get_default_values,
    parse_agent_rounds,
)
from aimanager.generic.encoder import IntEncoder  # noqa: E402
from aimanager.manager.environment import ArtificialHumanEnv  # noqa: E402

N_AGENTS = 8
N_GROUPS = 2
SWITCH_EVERY = 4  # the real cadence: decisions at 3, 7, 11; arrivals at 4, 8, 12
N_ROUNDS = 14
CAP = 4

# --------------------------------------------------------------------------- #
# the scenario
#
#   agent 0  arrives at round 4 and AGAIN at round 12 -- the second switch after
#            tenure has run all the way to the cap and held there
#   agent 1  arrives at round 4 and AGAIN at round 8 -- the second switch at the
#            minimum possible gap, resetting from 3 straight back to 0
#   agent 4  arrives at round 8 only -- a first arrival for an agent that STARTS
#            in group 1, so its round-0 cell exercises the guard
#   agents 2, 3 (group 0) and 5, 6, 7 (group 1) never switch: the cap throughout
#
# Four agents start in group 1 (4, 5, 6, 7), which is what makes the round-0
# guard observable at all.
# --------------------------------------------------------------------------- #
INITIAL_GROUPS = [0, 0, 0, 0, 1, 1, 1, 1]
_M_0_3 = [0, 0, 0, 0, 1, 1, 1, 1]
_M_4_7 = [1, 1, 0, 0, 1, 1, 1, 1]  # agents 0, 1 -> group 1
_M_8_11 = [1, 0, 0, 0, 0, 1, 1, 1]  # agents 1, 4 -> group 0
_M_12_13 = [0, 0, 0, 0, 0, 1, 1, 1]  # agent 0 -> group 0
MEMBERSHIP = {
    **{r: _M_0_3 for r in range(0, 4)},
    **{r: _M_4_7 for r in range(4, 8)},
    **{r: _M_8_11 for r in range(8, 12)},
    **{r: _M_12_13 for r in range(12, 14)},
}
ARRIVAL_ROUNDS = (4, 8, 12)

# Written from the spec, by hand, not generated from either implementation.
#                         a0  a1  a2  a3  a4  a5  a6  a7
EXPECTED_TENURE = {
    0: [4, 4, 4, 4, 4, 4, 4, 4],
    1: [4, 4, 4, 4, 4, 4, 4, 4],
    2: [4, 4, 4, 4, 4, 4, 4, 4],
    3: [4, 4, 4, 4, 4, 4, 4, 4],
    4: [0, 0, 4, 4, 4, 4, 4, 4],  # first arrival for agents 0 and 1
    5: [1, 1, 4, 4, 4, 4, 4, 4],
    6: [2, 2, 4, 4, 4, 4, 4, 4],
    7: [3, 3, 4, 4, 4, 4, 4, 4],
    8: [4, 0, 4, 4, 0, 4, 4, 4],  # a0 reaches the cap; a1 resets from 3; a4 arrives
    9: [4, 1, 4, 4, 1, 4, 4, 4],
    10: [4, 2, 4, 4, 2, 4, 4, 4],
    11: [4, 3, 4, 4, 3, 4, 4, 4],  # a0 has held the cap for four rounds
    12: [0, 4, 4, 4, 4, 4, 4, 4],  # a0 resets from the cap; a1 and a4 reach it
    13: [1, 4, 4, 4, 4, 4, 4, 4],
}
NEVER_SWITCHES = (2, 3, 5, 6, 7)


# --------------------------------------------------------------------------- #
# training side: a synthetic raw human-shaped dataframe through data.py
# --------------------------------------------------------------------------- #
def _raw_row(round_number, player_id, group_id):
    return dict(
        episode_id=0,
        round_number=round_number,
        player_id=player_id,
        global_group_id="g0",
        group_id=group_id,
        player_no_input=0,
        manager_no_input=0,
        contribution=10.0,
        punishment=0.0,
        common_good=0.0,
    )


def build_raw_df():
    """One row per player per round -- GAPLESS on purpose, see
    `test_the_gapless_frame_is_a_precondition_not_an_accident`."""
    rows = [
        _raw_row(round_number, player, groups[player])
        for round_number, groups in MEMBERSHIP.items()
        for player in range(N_AGENTS)
    ]
    return pd.DataFrame(rows)


def training_side():
    """Returns (parsed dataframe, rounds_since_arrival tensor of shape
    (1, n_agents, n_rounds))."""
    raw = build_raw_df()
    parsed = parse_agent_rounds(raw.copy(), switch_every=SWITCH_EVERY)
    data, _, _ = create_torch_data(raw.copy(), switch_every=SWITCH_EVERY)
    tenure = data["rounds_since_arrival"]
    assert tenure.shape == (1, N_AGENTS, N_ROUNDS)
    assert tenure.dtype == th.int64
    return parsed, tenure


# --------------------------------------------------------------------------- #
# simulation side: a real ArtificialHumanEnv, driven through real step() calls
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "punishment": 0,
    "contribution": 10,
    "contribution_valid": False,
    "punishment_valid": False,
    "common_good": 0.0,
    "agent_group": 0,  # the value that makes the round-0 guard necessary
    "does_switch": False,
    "switch_mask": False,
    "own_grp_prev_mean_contr": 10.0,
    "rounds_since_arrival": 4,
}


class _RecordingContribution:
    """Stub artificial_humans: deterministic forward, and it snapshots the
    state it is handed. The snapshot is taken inside `predict` -- after
    `update_contribution` has called `update_rounds_since_arrival`, and after
    `apply_switch` on an arrival round -- so it is exactly the tensor the real
    contributor's `IntEncoder` would read."""

    def __init__(self):
        self.default_values = {"contribution": 10, "punishment": 0}
        self.seen = {}

    def predict(self, state, reset_rnn=False, edge_index=None):
        round_ = int(state["round_number"][0, 0, 0])
        assert round_ not in self.seen, "a round was contributed in twice"
        self.seen[round_] = {
            "rounds_since_arrival": state["rounds_since_arrival"].clone(),
            "agent_group": state["agent_group"].clone(),
        }
        return (th.full_like(state["contribution"], 10),)


class _ScriptedSwitch:
    """Returns the scripted leavers for the NEXT round. `step()` calls the
    predictor every round to keep an RNN warm and only uses the output when
    the next round is an arrival round, so the off-rounds are no-ops by the
    environment's own contract."""

    def predict(self, state, reset_rnn=False, edge_index=None):
        round_ = int(state["round_number"][0, 0, 0])
        switch = th.zeros_like(state["contribution"], dtype=th.bool)
        nxt = MEMBERSHIP.get(round_ + 1)
        if nxt is not None:
            now = MEMBERSHIP[round_]
            for agent in range(N_AGENTS):
                if now[agent] != nxt[agent]:
                    switch[:, agent, :] = True
        return switch, None


def simulation_side(default_values=None):
    """Drives a real Environment through the whole episode and returns the
    per-round snapshots the contribution model was handed."""
    contribution = _RecordingContribution()
    env = ArtificialHumanEnv(
        artifical_humans=contribution,
        artifical_humans_valid=None,
        artifical_humans_switch=_ScriptedSwitch(),
        switch_every=SWITCH_EVERY,
        batch_size=1,
        n_agents=N_AGENTS,
        n_contributions=21,
        n_punishments=31,
        n_rounds=N_ROUNDS,
        n_groups=N_GROUPS,
        device="cpu",
        agent_groups=INITIAL_GROUPS,
        default_values=DEFAULTS if default_values is None else default_values,
    )
    done = False
    while not done:
        env.punish(th.zeros((1, N_AGENTS, 1), dtype=th.int64))
        _, _, done = env.step()
    assert sorted(contribution.seen) == list(range(N_ROUNDS))
    return env, contribution.seen


def _sim_tenure(seen, round_):
    tensor = seen[round_]["rounds_since_arrival"]
    assert tensor.dtype == th.int64
    assert tensor.shape == (1, N_AGENTS, 1)
    return tensor.squeeze(-1)[0].tolist()


def _sim_groups(seen, round_):
    return seen[round_]["agent_group"].squeeze(-1)[0].tolist()


# --------------------------------------------------------------------------- #
# the parity assertions
# --------------------------------------------------------------------------- #
def test_the_scenario_really_plays_out_in_the_sim():
    """Guard on everything below: if the scripted switches did not land, the
    two sides could agree on a scenario in which nothing ever happens."""
    _, seen = simulation_side()
    for round_ in range(N_ROUNDS):
        assert _sim_groups(seen, round_) == MEMBERSHIP[round_], (
            f"membership diverged at round {round_}: "
            f"{_sim_groups(seen, round_)} != {MEMBERSHIP[round_]}"
        )
    for round_ in ARRIVAL_ROUNDS:
        assert MEMBERSHIP[round_] != MEMBERSHIP[round_ - 1]


def test_rounds_since_arrival_matches_between_training_and_sim():
    """The core claim, agent for agent and round for round, against a hand
    written expectation that neither implementation produced."""
    _, train = training_side()
    _, seen = simulation_side()
    for round_ in range(N_ROUNDS):
        sim = _sim_tenure(seen, round_)
        trn = train[0, :, round_].tolist()
        assert trn == sim, (
            f"train/sim rounds_since_arrival disagree at round {round_}: "
            f"{trn} != {sim}"
        )
        assert sim == EXPECTED_TENURE[round_], (
            f"both sides agree at round {round_} but on the WRONG value: "
            f"{sim} != {EXPECTED_TENURE[round_]}"
        )


def test_arrival_round_reads_zero_then_one_two_three():
    """The step at the arrival round and the ramp that follows it -- the whole
    shape the onehot encoding exists to express. Agent 0 arrives at round 4."""
    _, train = training_side()
    _, seen = simulation_side()
    agent = 0
    ramp_sim = [_sim_tenure(seen, r)[agent] for r in range(3, 9)]
    ramp_train = [train[0, agent, r].item() for r in range(3, 9)]
    assert ramp_sim == [4, 0, 1, 2, 3, 4]
    assert ramp_train == ramp_sim
    # and the arrival really is where the membership changes, so the 0 is not
    # coming from somewhere else
    assert MEMBERSHIP[4][agent] != MEMBERSHIP[3][agent]
    assert MEMBERSHIP[5][agent] == MEMBERSHIP[4][agent]


def test_second_switch_resets_the_counter():
    """Two resets, from the two places a reset can come from. Agent 1 switches
    again at round 8 with tenure at 3 (the minimum gap); agent 0 switches again
    at round 12 with tenure sitting at the cap."""
    _, train = training_side()
    _, seen = simulation_side()

    # agent 1: 3 -> 0
    assert _sim_tenure(seen, 7)[1] == 3
    assert _sim_tenure(seen, 8)[1] == 0
    assert [train[0, 1, r].item() for r in (7, 8, 9)] == [3, 0, 1]

    # agent 0: cap -> 0
    assert _sim_tenure(seen, 11)[0] == CAP
    assert _sim_tenure(seen, 12)[0] == 0
    assert [train[0, 0, r].item() for r in (11, 12, 13)] == [4, 0, 1]


def test_tenure_reaches_the_cap_and_holds_it():
    """Agent 0 arrives at round 4 and does not move again until 12, so rounds
    8, 9, 10 and 11 must all read 4 -- the incremental `clamp(max=4)` in the
    env and the absolute `clip(upper=4)` in pandas have to agree not just at
    the cap but for every round it is held."""
    _, train = training_side()
    _, seen = simulation_side()
    held = list(range(8, 12))
    assert [_sim_tenure(seen, r)[0] for r in held] == [CAP] * len(held)
    assert [train[0, 0, r].item() for r in held] == [CAP] * len(held)
    assert train.max().item() == CAP
    for round_ in range(N_ROUNDS):
        assert max(_sim_tenure(seen, round_)) <= CAP


def test_an_agent_who_never_switches_is_the_cap_throughout():
    _, train = training_side()
    _, seen = simulation_side()
    for agent in NEVER_SWITCHES:
        assert train[0, agent, :].tolist() == [CAP] * N_ROUNDS
        assert [_sim_tenure(seen, r)[agent] for r in range(N_ROUNDS)] == (
            [CAP] * N_ROUNDS
        )


def test_round_zero_is_the_cap_in_both_groups():
    """THE GUARD. `reset_state` fills `prev_agent_group` with
    `default_values["agent_group"]` = 0, so without the explicit round-0 branch
    in `update_rounds_since_arrival` the four agents that START in group 1 would
    compare 1 != 0 and read tenure 0 -- a phantom arrival for half the
    population on the very first contribution of every episode. Removing that
    branch makes this test fail."""
    parsed, train = training_side()
    _, seen = simulation_side()

    assert set(INITIAL_GROUPS) == {0, 1}, "both groups must be populated at round 0"
    assert DEFAULTS["agent_group"] == 0, "the default that creates the hazard"

    assert _sim_groups(seen, 0) == INITIAL_GROUPS
    assert _sim_tenure(seen, 0) == [CAP] * N_AGENTS
    assert train[0, :, 0].tolist() == [CAP] * N_AGENTS
    assert parsed.loc[
        parsed["round_number"] == 0, "rounds_since_arrival"
    ].unique().tolist() == [CAP]
    assert get_default_values(parsed)["rounds_since_arrival"] == CAP


def test_pre_arrival_rounds_are_the_cap_not_zero():
    """Rounds 0-3 are static by design, so the earliest possible arrival is
    round 4 and an agent that has not arrived anywhere yet reads the SETTLED
    cell, never the arrival cell."""
    _, train = training_side()
    _, seen = simulation_side()
    for round_ in range(min(ARRIVAL_ROUNDS)):
        assert _sim_tenure(seen, round_) == [CAP] * N_AGENTS
        assert train[0, :, round_].tolist() == [CAP] * N_AGENTS


def test_zero_appears_only_on_arrival_rounds_and_only_for_arrivers():
    """Both directions: no spurious zeros, and no missing ones."""
    _, train = training_side()
    _, seen = simulation_side()
    for round_ in range(N_ROUNDS):
        expected_arrivers = (
            []
            if round_ == 0
            else [
                a
                for a in range(N_AGENTS)
                if MEMBERSHIP[round_][a] != MEMBERSHIP[round_ - 1][a]
            ]
        )
        sim_zeros = [a for a, t in enumerate(_sim_tenure(seen, round_)) if t == 0]
        train_zeros = [a for a in range(N_AGENTS) if train[0, a, round_].item() == 0]
        assert sim_zeros == expected_arrivers, f"round {round_}"
        assert train_zeros == expected_arrivers, f"round {round_}"


def test_the_gapless_frame_is_a_precondition_not_an_accident():
    """A DOCUMENTED ASYMMETRY, deliberately not part of the parity case above.

    `data.py` marks an arrival whenever a player's `group_id` differs from that
    player's previous *recorded* row, however many rounds back that row is. The
    env compares against the immediately preceding round. On a frame with a
    dropped row the two therefore CAN disagree -- here the row for agent 0 at
    round 4 is removed, and pandas moves the arrival marker to round 5 while the
    env (which has no notion of a missing round) would still place it at 4.

    This cannot happen in the simulation, where every agent is present in every
    round, and it does not happen in the human data either (one row per player
    per round). The main parity case is therefore built on a gapless frame by
    construction, and this test exists so the limit is written down rather than
    discovered later as a mystery."""
    gapped = build_raw_df()
    gapped = gapped[
        ~((gapped["player_id"] == 0) & (gapped["round_number"] == 4))
    ].copy()
    parsed = parse_agent_rounds(gapped, switch_every=SWITCH_EVERY)
    agent0 = parsed[parsed["player_idx"] == 0].set_index("round_number")
    assert agent0.loc[5, "rounds_since_arrival"] == 0  # marker slid to round 5
    assert agent0.loc[6, "rounds_since_arrival"] == 1
    # the gapless frame, the one the parity case uses, puts it at round 4
    ungapped, _ = training_side()
    ungapped = ungapped[ungapped["player_idx"] == 0].set_index("round_number")
    assert ungapped.loc[4, "rounds_since_arrival"] == 0
    assert ungapped.loc[5, "rounds_since_arrival"] == 1


@pytest.mark.parametrize("level", list(range(5)))
def test_encoder_maps_levels_to_the_five_unit_vectors(level):
    """`{name: rounds_since_arrival, n_levels: 5, encoding: onehot}` -- the
    step-4 config's encoding. Five independent cells, so the arrival step, the
    1-3 plateau and the settled cap each get their own weight and no slope is
    imposed between them."""
    encoder = IntEncoder(encoding="onehot", name="rounds_since_arrival", n_levels=5)
    assert encoder.size == 5
    enc = encoder(rounds_since_arrival=th.tensor([[level]], dtype=th.int64))
    assert enc.shape == (1, 1, 5)
    expected = [0.0] * 5
    expected[level] = 1.0
    assert enc[0, 0].tolist() == expected


def test_encoder_covers_every_value_either_side_can_produce():
    """No level the two implementations emit falls outside the 5-wide map, and
    every one of the five is reachable somewhere in the scenario."""
    encoder = IntEncoder(encoding="onehot", name="rounds_since_arrival", n_levels=5)
    _, train = training_side()
    _, seen = simulation_side()

    produced = set(train.flatten().tolist())
    for round_ in range(N_ROUNDS):
        produced.update(_sim_tenure(seen, round_))
    assert produced == {0, 1, 2, 3, 4}

    enc = encoder(rounds_since_arrival=train)
    assert enc.shape == (1, N_AGENTS, N_ROUNDS, 5)
    assert enc.sum(-1).eq(1.0).all()


def test_environment_runs_without_the_key_in_default_values():
    """The step-11 control's situation: the parent's artifact names no
    `rounds_since_arrival` default. `reset_state`'s `prev_` comprehension is
    membership-guarded, so nothing raises -- and the feature itself is
    unchanged, which is what licenses the bit-identical control run."""
    without = {k: v for k, v in DEFAULTS.items() if k != "rounds_since_arrival"}
    _, seen = simulation_side(default_values=without)
    for round_ in range(N_ROUNDS):
        assert _sim_tenure(seen, round_) == EXPECTED_TENURE[round_]


def test_the_pyg_stand_in_is_never_exercised():
    """The stand-in discipline, asserted rather than asserted-by-comment: this
    test drives the entire parity scenario and then requires that the local
    `_scatter_mean` substitute received no calls. On Raven no stand-in is
    installed at all, and that is checked too."""
    training_side()
    simulation_side()
    IntEncoder(encoding="onehot", name="rounds_since_arrival", n_levels=5)(
        rounds_since_arrival=th.tensor([[4]], dtype=th.int64)
    )
    assert _SCATTER_MEAN_CALLS == [], (
        "the PyG stand-in was exercised; something under test now depends on "
        "real torch_scatter and this file's local result is not trustworthy"
    )
    if not STAND_INS:
        assert importlib.util.find_spec("torch_scatter") is not None
