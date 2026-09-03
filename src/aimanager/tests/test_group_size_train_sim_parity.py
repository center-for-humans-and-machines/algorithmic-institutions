"""Train/sim parity for the `own_group_size` node feature (plan step 3).

Runs locally on macOS with plain pytest:

    PYTHONPATH=$PWD/src uv run pytest \
        src/aimanager/tests/test_group_size_train_sim_parity.py -q

Context: notes/autoresearch_log/contribution-group-size.md.

THE HAZARD. The feature is built through two independent implementations with
no shared code path: training builds it in pandas
(`generic/data.py::parse_agent_rounds`, a
`groupby(episode, round, group_id).transform("size")`), simulation builds it
in torch (`manager/environment.py::update_own_group_size`, a masked sum over
`agent_group_mask` gathered back to each agent). The precedent this follows,
`update_own_grp_prev_mean_contr`, keeps the two in sync by a COMMENT that
nothing verifies. This file is that verification: it asserts the
`own_group_size` column `data.py` derives equals, agent for agent and round
for round, the `state["own_group_size"]` a real `ArtificialHumanEnv` holds AT
CONTRIBUTION TIME -- the tensor the contributor's `IntEncoder` actually reads.

THE ROUND THAT MATTERS. `Environment.step()` calls `apply_switch` (which
rebuilds `agent_groups` and `agent_group_mask`) and only then
`update_contribution`. So on an arrival round the feature must describe the
agent's NEW group. Every other round is insensitive to that ordering, which
is why the arrival round gets its own separately named assertion below
(`test_arrival_round_...`) rather than being one cell inside a bulk compare.

WHAT NEEDS PyG, AND WHY NONE OF IT MATTERS HERE. `aimanager.generic.data` and
`aimanager.manager.environment` import neither `torch_scatter` nor
`torch_geometric`. The one import that needs a stand-in is
`aimanager.generic.encoder`, purely because that module's top-level `from
torch_scatter import scatter_mean` runs before `IntEncoder` is reachable;
`IntEncoder` never calls it. The stand-in is therefore never exercised, and
this is asserted rather than asserted-by-comment: `_scatter_mean` records
every call it receives, and the last test in this file drives the whole parity
scenario and then requires that record to be empty. On Raven, where real PyG
is present, no stand-in is installed at all (`STAND_INS` is empty) and the
same test confirms that too -- so nothing here is left owed to the cluster.
"""

import importlib
import sys
import types

import pandas as pd
import pytest
import torch as th

# --------------------------------------------------------------------------- #
# stand-in (macOS only) -- installed only when the real package is absent, the
# same discipline as test_joint_exodus_train_sim_parity.py. It exists solely so
# `import aimanager.generic.encoder` succeeds; every call it receives is
# recorded so `test_the_pyg_stand_in_is_never_exercised` can prove nothing
# under test here routed through it.
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
SWITCH_EVERY = 2
N_ROUNDS = 5
# (r + 1) % SWITCH_EVERY == 0 -> decisions at r = 1 and r = 3, arrivals at the
# rounds that follow them. Round 0 is static and carries the 4-4 default.
FIRST_ARRIVAL = 2  # 4-4 -> 1-7: the singleton appears here
SECOND_ARRIVAL = 4  # 1-7 -> 0-8: the full merge, group 0 empties

# Membership per round. Agents 1-3 leave group 0 at the first arrival (leaving
# agent 0 alone against a seven); agent 0 follows at the second, merging all
# eight and emptying group 0 entirely.
MEMBERSHIP = {
    0: [0, 0, 0, 0, 1, 1, 1, 1],
    1: [0, 0, 0, 0, 1, 1, 1, 1],
    2: [0, 1, 1, 1, 1, 1, 1, 1],
    3: [0, 1, 1, 1, 1, 1, 1, 1],
    4: [1, 1, 1, 1, 1, 1, 1, 1],
}
EXPECTED_SIZES = {
    0: [4, 4, 4, 4, 4, 4, 4, 4],
    1: [4, 4, 4, 4, 4, 4, 4, 4],
    2: [1, 7, 7, 7, 7, 7, 7, 7],
    3: [1, 7, 7, 7, 7, 7, 7, 7],
    4: [8, 8, 8, 8, 8, 8, 8, 8],
}
# Members who time out / give no input, and must STILL be counted: membership
# is not validity. Agent 0 is the timed-out singleton (step 1 measured 17 such
# rows among the human singletons) and agent 5 a timed-out member of the seven,
# so both "my own count" and "a peer's count of me" are covered.
TIMEOUTS = {FIRST_ARRIVAL: [0, 5]}


# --------------------------------------------------------------------------- #
# training side: a synthetic raw human-shaped dataframe through data.py
# --------------------------------------------------------------------------- #
def _raw_row(round_number, player_id, group_id, no_input):
    return dict(
        episode_id=0,
        round_number=round_number,
        player_id=player_id,
        global_group_id="g0",
        group_id=group_id,
        player_no_input=no_input,
        manager_no_input=0,
        contribution=10.0,
        punishment=0.0,
        common_good=0.0,
    )


def build_raw_df():
    rows = []
    for round_number, groups in MEMBERSHIP.items():
        for player in range(N_AGENTS):
            rows.append(
                _raw_row(
                    round_number,
                    player,
                    groups[player],
                    int(player in TIMEOUTS.get(round_number, ())),
                )
            )
    return pd.DataFrame(rows)


def training_side():
    """Returns (parsed dataframe, own_group_size tensor of shape
    (1, n_agents, n_rounds))."""
    raw = build_raw_df()
    parsed = parse_agent_rounds(raw.copy(), switch_every=SWITCH_EVERY)
    data, _, _ = create_torch_data(raw.copy(), switch_every=SWITCH_EVERY)
    own_group_size = data["own_group_size"]
    assert own_group_size.shape == (1, N_AGENTS, N_ROUNDS)
    return parsed, own_group_size


# --------------------------------------------------------------------------- #
# simulation side: a real ArtificialHumanEnv, driven through real step() calls
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "punishment": 0,
    "contribution": 10,
    "contribution_valid": False,
    "punishment_valid": False,
    "common_good": 0,
    "agent_group": 0,
    "does_switch": False,
    "switch_mask": False,
    "own_group_size": 4,
}


class _RecordingContribution:
    """Stub artificial_humans: deterministic forward, and it snapshots the
    state it is handed. The snapshot is taken inside `predict`, i.e. after
    `update_contribution` has called `update_own_group_size` and after
    `apply_switch` on an arrival round -- so it is exactly what the real
    contributor's `IntEncoder` would read."""

    def __init__(self):
        self.default_values = {"contribution": 10, "punishment": 0}
        self.seen = {}

    def predict(self, state, reset_rnn, edge_index):
        round_ = int(state["round_number"][0, 0, 0])
        self.seen[round_] = {
            "own_group_size": state["own_group_size"].clone(),
            "agent_group": state["agent_group"].clone(),
            "prev_contribution_valid": state["prev_contribution_valid"].clone(),
        }
        return (th.full_like(state["contribution"], 10),)


class _ScriptedValid:
    """Marks the scripted timeouts invalid, so the sim really does contain
    no-input members rather than only asserting about them."""

    def predict(self, state, reset_rnn, edge_index):
        round_ = int(state["round_number"][0, 0, 0])
        valid = th.ones_like(state["contribution"], dtype=th.bool)
        for agent in TIMEOUTS.get(round_, ()):
            valid[:, agent, :] = False
        return (valid,)


class _ScriptedSwitch:
    """Returns the scripted leavers on each decision round. `step()` calls the
    predictor every round to keep an RNN warm, so every other round is a
    no-op, matching `manager/environment.py: step`'s own contract."""

    def predict(self, state, reset_rnn, edge_index):
        round_ = int(state["round_number"][0, 0, 0])
        switch = th.zeros_like(state["contribution"], dtype=th.bool)
        # a switcher is an agent whose group differs at the NEXT round
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
        artifical_humans_valid=_ScriptedValid(),
        artifical_humans_switch=_ScriptedSwitch(),
        switch_every=SWITCH_EVERY,
        batch_size=1,
        n_agents=N_AGENTS,
        n_contributions=21,
        n_punishments=31,
        n_rounds=N_ROUNDS,
        n_groups=2,
        device="cpu",
        agent_groups=MEMBERSHIP[0],
        default_values=DEFAULTS if default_values is None else default_values,
    )
    done = False
    while not done:
        env.punish(th.zeros((1, N_AGENTS, 1), dtype=th.int64))
        _, _, done = env.step()
    assert sorted(contribution.seen) == list(range(N_ROUNDS))
    return env, contribution.seen


def _sim_sizes(seen, round_):
    return seen[round_]["own_group_size"].squeeze(-1)[0].tolist()


# --------------------------------------------------------------------------- #
# the parity assertions
# --------------------------------------------------------------------------- #
def test_own_group_size_matches_between_training_and_sim():
    """The core claim, agent for agent and round for round."""
    _, train = training_side()
    _, seen = simulation_side()
    for round_ in range(N_ROUNDS):
        assert train[0, :, round_].tolist() == _sim_sizes(seen, round_), (
            f"train/sim own_group_size disagree at round {round_}: "
            f"{train[0, :, round_].tolist()} != {_sim_sizes(seen, round_)}"
        )
        assert _sim_sizes(seen, round_) == EXPECTED_SIZES[round_]


def test_arrival_round_uses_the_new_group_not_the_old_one():
    """The one round where a wrong ordering shows up: at the first arrival the
    membership becomes 1-7, so the feature must read [1, 7, ...]. If
    `update_own_group_size` ran BEFORE `apply_switch` it would still read
    [4, 4, ...] here -- the pre-switch value -- and nowhere else would tell."""
    _, train = training_side()
    _, seen = simulation_side()
    sim = _sim_sizes(seen, FIRST_ARRIVAL)
    stale = EXPECTED_SIZES[FIRST_ARRIVAL - 1]

    assert sim == [1, 7, 7, 7, 7, 7, 7, 7]
    assert sim == train[0, :, FIRST_ARRIVAL].tolist()
    assert sim != stale, "the arrival round is reading the PRE-switch membership"
    # and the membership really did change, so agreeing is not accidental
    assert seen[FIRST_ARRIVAL]["agent_group"].squeeze(-1)[0].tolist() == (
        MEMBERSHIP[FIRST_ARRIVAL]
    )
    assert MEMBERSHIP[FIRST_ARRIVAL] != MEMBERSHIP[FIRST_ARRIVAL - 1]


def test_round_zero_is_four_on_both_sides():
    parsed, train = training_side()
    _, seen = simulation_side()
    assert train[0, :, 0].tolist() == [4] * N_AGENTS
    assert _sim_sizes(seen, 0) == [4] * N_AGENTS
    assert parsed.loc[
        parsed["round_number"] == 0, "own_group_size"
    ].unique().tolist() == [4]
    assert get_default_values(parsed)["own_group_size"] == 4


def test_timed_out_members_still_count():
    """Membership is not validity: a member who gave no input is still in the
    group. Agent 0 is the timed-out singleton, agent 5 a timed-out member of
    the seven -- and both counts are unaffected."""
    parsed, train = training_side()
    _, seen = simulation_side()

    # the scenario really does contain invalid members at that round ...
    invalid = parsed.loc[
        (parsed["round_number"] == FIRST_ARRIVAL) & ~parsed["contribution_valid"],
        "player_idx",
    ].tolist()
    assert sorted(invalid) == TIMEOUTS[FIRST_ARRIVAL]

    # ... yet the timed-out singleton counts itself, and the seven counts its
    # timed-out member, on both sides
    assert train[0, 0, FIRST_ARRIVAL].item() == 1
    assert train[0, 5, FIRST_ARRIVAL].item() == 7
    assert _sim_sizes(seen, FIRST_ARRIVAL) == [1, 7, 7, 7, 7, 7, 7, 7]

    # and the round AFTER, where the sim carries the invalidity in state as
    # prev_contribution_valid, is still keyed to membership alone
    prev_valid = seen[FIRST_ARRIVAL + 1]["prev_contribution_valid"]
    assert prev_valid.squeeze(-1)[0].tolist() == [
        agent not in TIMEOUTS[FIRST_ARRIVAL] for agent in range(N_AGENTS)
    ]
    assert _sim_sizes(seen, FIRST_ARRIVAL + 1) == [1, 7, 7, 7, 7, 7, 7, 7]


def test_full_merge_reads_eight_and_no_agent_ever_reads_zero():
    """When a group empties, its count is 0 -- but no agent points at it, so
    the feature can never emit 0 and index 0 of the n_levels=9 map is
    unreachable."""
    _, train = training_side()
    _, seen = simulation_side()

    assert MEMBERSHIP[SECOND_ARRIVAL].count(0) == 0, "group 0 must be empty here"
    assert _sim_sizes(seen, SECOND_ARRIVAL) == [8] * N_AGENTS
    assert train[0, :, SECOND_ARRIVAL].tolist() == [8] * N_AGENTS

    assert train.min().item() >= 1
    for round_ in range(N_ROUNDS):
        assert min(_sim_sizes(seen, round_)) >= 1


def test_group_sizes_sum_to_eight_over_distinct_groups():
    """THE TRAP: the invariant is over DISTINCT GROUPS, not over the 8 agent
    rows. Summing `own_group_size` across a cell's agent rows gives the sum of
    k**2 (32 for 4-4, 50 for 7-1, 64 for 8-0), so the naive check appears to
    fail on every cell."""
    parsed, _ = training_side()
    per_group = parsed.groupby(["episode_id", "round_number", "agent_group"])[
        "own_group_size"
    ].first()
    per_cell = per_group.groupby(level=[0, 1]).sum()
    assert len(per_cell) == N_ROUNDS
    assert (per_cell == N_AGENTS).all(), per_cell[per_cell != N_AGENTS]

    # the naive version, recorded so the next reader does not rediscover it
    naive = parsed.groupby(["episode_id", "round_number"])["own_group_size"].sum()
    assert naive.tolist() == [32, 32, 50, 50, 64]


@pytest.mark.parametrize("size", list(range(1, N_AGENTS + 1)))
def test_encoder_maps_sizes_to_eighths(size):
    """`n_levels: 9` with numeric encoding is `IntEncoder`'s
    `linspace(0, 1, 9)[v] = v / 8` -- the convention `round_number` and the
    joint head's k / 8 already use."""
    encoder = IntEncoder(encoding="numeric", name="own_group_size", n_levels=9)
    assert encoder.size == 1
    enc = encoder(own_group_size=th.tensor([[size]], dtype=th.int64))
    assert enc.shape == (1, 1, 1)
    assert enc.item() == pytest.approx(size / 8.0)


def test_encoder_index_zero_exists_but_is_unreachable():
    """Completes the previous test: the map does have a slot for 0 (mapping to
    0.0), and nothing in either implementation can ever index it."""
    encoder = IntEncoder(encoding="numeric", name="own_group_size", n_levels=9)
    assert encoder(
        own_group_size=th.tensor([[0]], dtype=th.int64)
    ).item() == pytest.approx(0.0)
    _, train = training_side()
    assert (train == 0).sum().item() == 0


def test_environment_accepts_defaults_without_the_key():
    """The control run's situation, and step 10's licence to compare anything:
    the parent's artifact carries no `own_group_size` default. `reset_state`'s
    prev_ comprehension is membership-guarded, so no `prev_own_group_size` is
    created and no KeyError is raised -- and the feature itself is unaffected."""
    without = {k: v for k, v in DEFAULTS.items() if k != "own_group_size"}
    env, seen = simulation_side(default_values=without)
    assert "prev_own_group_size" not in env.state
    assert "own_group_size" in env.state
    for round_ in range(N_ROUNDS):
        assert _sim_sizes(seen, round_) == EXPECTED_SIZES[round_]

    env_with, _ = simulation_side()
    assert "prev_own_group_size" in env_with.state


def test_the_pyg_stand_in_is_never_exercised():
    """The stand-in discipline, asserted rather than asserted-by-comment: this
    test drives the entire parity scenario and then requires that the local
    `_scatter_mean` substitute received no calls. On Raven no stand-in is
    installed at all, and that is checked too."""
    training_side()
    simulation_side()
    IntEncoder(encoding="numeric", name="own_group_size", n_levels=9)(
        own_group_size=th.tensor([[4]], dtype=th.int64)
    )
    assert _SCATTER_MEAN_CALLS == [], (
        "the PyG stand-in was exercised; something under test now depends on "
        "real torch_scatter and this file's local result is not trustworthy"
    )
    if not STAND_INS:
        assert importlib.util.find_spec("torch_scatter") is not None
