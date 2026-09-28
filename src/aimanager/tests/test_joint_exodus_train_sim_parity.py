"""Train/sim parity for the joint exodus mechanism (plan step 5).

Runs locally on macOS with plain pytest:

    PYTHONPATH=$PWD/src uv run pytest \
        src/aimanager/tests/test_joint_exodus_train_sim_parity.py -q

Context: notes/autoresearch_log/switch-joint-exodus.md.

THE HAZARD. Node features and group-level quantities are built through two
independent implementations with no shared code path: training builds them in
pandas (`generic/data.py::parse_agent_rounds`), simulation builds them in
torch (`manager/environment.py::ArtificialHumanEnv`). The only precedent for
keeping the two in sync, `update_own_grp_prev_mean_contr`, does so by a
COMMENT ("mirrors data.py's training-time column so sim matches training")
that nothing verifies. This file is that verification for the joint exodus
head: it asserts that the group membership and the leaver-count pair
`train.joint_exodus_counts` derives from `data.py`-parsed training tensors
agree, agent for agent and label for label, with what a real
`ArtificialHumanEnv` holds at decision time -- the same membership
`GraphNetwork.predict_independent`'s joint-exodus branch reads via
`encoded["agent_group"]` and step 4's sampler spends its drawn pair on.

WHAT NEEDS PyG, AND WHY NONE OF IT MATTERS HERE. `aimanager.generic.data`,
`aimanager.manager.environment` and `aimanager.generic.joint_exodus` (the
module that defines `pool_by_group`, the function both the training loss and
the simulation head call) import neither `torch_scatter` nor
`torch_geometric` -- they are plain pandas/torch and are demonstrably
importable on macOS (see the check in this suite's own CI history). The one
thing that DOES need a stand-in is *importing* `aimanager.artificial_humans
.train` at all, purely because that module's top-level `from
aimanager.artificial_humans import AH_MODELS` pulls in `GraphNetwork`, which
imports `torch_scatter.scatter_mean` and `torch_geometric.nn.MetaLayer` for
its (unrelated, message-passing) machinery. `joint_exodus_counts` itself
never calls either symbol -- it only calls `pool_by_group`, a masked
`index_add_`. So the stand-ins below exist ONLY to satisfy the import; they
are never exercised by anything this file asserts, and the invariant this
file checks is fully verified on macOS with no PyG present at all.

WHAT THIS FILE DOES NOT COVER, AND WHERE THAT LIVES INSTEAD. The claim that
the *real* `JointExodusHead`, wired into a real `GraphNetwork` forward pass
over message-passed embeddings, pools with `pool_by_group` and reports a `k`
consistent with its own `agent_group` input is already covered -- with PyG
stand-ins locally, with real PyG on Raven -- by
`tests/switch/test_joint_exodus_loss.py
::test_training_counts_match_the_heads_own_pooling` and by the runtime
assert in `train.joint_exodus_loss` itself. The claim that the simulation's
drawn pair is spent on the group it was drawn for is covered the same way by
`tests/switch/test_joint_exodus_sampling.py` and the runtime assert in
`GraphNetwork._predict_encoded_joint_exodus`. This file's job is the piece
neither of those touches: whether `environment.py`'s own bookkeeping of
`agent_group` across a real `Environment.step()` call produces the same
membership, keyed the same way, as `data.py`'s independent pandas derivation
of the same underlying scenario. Because none of the functions this file
exercises needs real PyG, there is nothing left here to re-verify on Raven --
`scripts/remote_test.sh` will simply run this file with the real packages
installed and no stand-in path taken.
"""

import importlib
import sys
import types

import pandas as pd
import pytest
import torch as th


# --------------------------------------------------------------------------- #
# stand-ins (macOS only) -- installed only when the real packages are absent,
# the same discipline as tests/switch/test_joint_exodus_loss.py and its
# siblings. They exist solely so `import aimanager.artificial_humans.train`
# succeeds; nothing under test here calls into them.
# --------------------------------------------------------------------------- #
def _scatter_mean(src, index, dim=0, dim_size=None):
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


class _MetaLayer(th.nn.Module):
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None):
        row, col = edge_index
        if self.edge_model is not None:
            edge_attr = self.edge_model(
                x[row], x[col], edge_attr, u, batch if batch is None else batch[row]
            )
        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, batch)
        if self.global_model is not None:
            u = self.global_model(x, edge_index, edge_attr, u, batch)
        return x, edge_attr, u


class _Tqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self._iterable = [] if iterable is None else iterable

    def __iter__(self):
        return iter(self._iterable)

    def set_postfix(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _install_stand_ins():
    installed = []
    try:
        importlib.import_module("torch_scatter")
        importlib.import_module("torch_geometric.nn")
    except ImportError:
        scatter = types.ModuleType("torch_scatter")
        scatter.scatter_mean = _scatter_mean
        sys.modules.setdefault("torch_scatter", scatter)
        geometric = types.ModuleType("torch_geometric")
        geometric_nn = types.ModuleType("torch_geometric.nn")
        geometric_nn.MetaLayer = _MetaLayer
        geometric.nn = geometric_nn
        sys.modules.setdefault("torch_geometric", geometric)
        sys.modules.setdefault("torch_geometric.nn", geometric_nn)
        installed.append("torch_geometric")
    try:
        importlib.import_module("tqdm")
    except ImportError:
        tqdm_mod = types.ModuleType("tqdm")
        tqdm_mod.tqdm = _Tqdm
        sys.modules.setdefault("tqdm", tqdm_mod)
        installed.append("tqdm")
    return installed


STAND_INS = _install_stand_ins()

from aimanager.artificial_humans.train import joint_exodus_counts  # noqa: E402
from aimanager.generic.data import create_torch_data, parse_agent_rounds  # noqa: E402
from aimanager.manager.environment import ArtificialHumanEnv  # noqa: E402


N_AGENTS = 8
SWITCH_EVERY = 2
# (r + 1) % 2 == 0 -> r = 1 is the decision round, with round 0 static before
# it and round 2 the arrival that realises the switch (data.py's convention:
# the decision round s is legal, the arrival is s + 1).
DECISION_ROUND = 1
N_ROUNDS = 3  # rounds 0 (static), 1 (decision), 2 (arrival)

# Four scenarios, one per batch row, chosen to hit every case the plan lists:
#   A: unbalanced 5-3, switchers from the larger group only.
#   B: a fully merged round (8-0), un-merging.
#   C: unbalanced 5-3, switchers in BOTH groups at once.
#   D: a SYMMETRIC 4-4 split with a symmetric 2-2 exchange -- the case where
#      swapping the two group labels leaves the (m, k) PAIR unchanged, so a
#      transposition bug would be invisible to a totals-only comparison and
#      only a per-agent identity check catches it (see
#      test_group_labels_are_not_merely_count_equal below).
SCENARIOS = {
    "A_unbalanced_one_sided": (
        [0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
    ),
    "B_fully_merged": (
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1],
    ),
    "C_unbalanced_both_sides": (
        [0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
    ),
    "D_symmetric_swap_blind": (
        [0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
    ),
}
SCENARIO_NAMES = list(SCENARIOS)
PRE = [SCENARIOS[name][0] for name in SCENARIO_NAMES]
POST = [SCENARIOS[name][1] for name in SCENARIO_NAMES]
N_BATCH = len(SCENARIO_NAMES)


# --------------------------------------------------------------------------- #
# training side: a synthetic raw human-shaped dataframe through data.py
# --------------------------------------------------------------------------- #
def _raw_row(episode_id, round_number, player_id, group_id):
    return dict(
        episode_id=episode_id,
        round_number=round_number,
        player_id=player_id,
        global_group_id=f"g{episode_id}",
        group_id=group_id,
        player_no_input=0,
        manager_no_input=0,
        contribution=10.0,
        punishment=0.0,
        common_good=0.0,
    )


def build_raw_df():
    """One row per (episode, player, round). Round 0 and round 1 carry the
    PRE-switch membership (round 0 is not a decision round and is only there
    so round 1 has a predecessor); round 2 carries the POST-switch membership
    -- the arrival `does_switch` at round 1 is labelled against."""
    rows = []
    for ep, (pre, post) in enumerate(zip(PRE, POST)):
        for player in range(N_AGENTS):
            rows.append(_raw_row(ep, 0, player, pre[player]))
            rows.append(_raw_row(ep, 1, player, pre[player]))
            rows.append(_raw_row(ep, 2, player, post[player]))
    return pd.DataFrame(rows)


def training_side():
    """Returns (m, k) at the decision round, and the (episode -> tensor row)
    map, plus the raw agent_group tensor for the per-agent identity check."""
    raw = build_raw_df()

    # `global_group_id` is "g<episode_id>" for single-digit episode ids, so
    # dense-rank order matches episode order -- verified explicitly (not
    # assumed) via `parse_agent_rounds`'s own `group_idx` column.
    parsed = parse_agent_rounds(raw.copy(), switch_every=SWITCH_EVERY)
    ep_to_row = (
        parsed.drop_duplicates("episode_id")
        .set_index("episode_id")["group_idx"]
        .to_dict()
    )
    assert ep_to_row == {ep: ep for ep in range(N_BATCH)}, (
        "test scaffolding assumption broken: episode id must equal the "
        f"tensor's batch row, got {ep_to_row}"
    )

    data, _, _ = create_torch_data(raw.copy(), switch_every=SWITCH_EVERY)
    agent_group = data["agent_group"]  # (n_batch, n_agents, n_rounds)
    does_switch = data["does_switch"]
    switch_valid = data["switch_valid"]

    n_batch, n_agents, n_rounds = agent_group.shape
    assert (n_batch, n_agents, n_rounds) == (N_BATCH, N_AGENTS, N_ROUNDS)

    flat = lambda t: t.reshape(n_batch * n_agents, n_rounds)  # noqa: E731
    batch_index = th.tensor(
        [b for b in range(n_batch) for _ in range(n_agents)], dtype=th.int64
    )
    m, k = joint_exodus_counts(
        flat(does_switch),
        flat(switch_valid),
        flat(agent_group),
        batch_index,
        n_batch=n_batch,
    )
    return m[:, DECISION_ROUND], k[:, DECISION_ROUND], agent_group[:, :, DECISION_ROUND]


# --------------------------------------------------------------------------- #
# simulation side: a real ArtificialHumanEnv, driven through a real step()
# --------------------------------------------------------------------------- #
class _ConstantContribution:
    """Stub artificial_humans: never invalid, constant contribution."""

    def __init__(self):
        self.default_values = {
            "punishment": 0,
            "contribution": 10,
            "round_number": 0,
            "is_first": False,
            "contribution_valid": False,
            "punishment_valid": False,
            "common_good": 0,
            "contributor_payoff": 0,
            "manager_payoff": 0,
            "reward": 0,
        }

    def predict(self, state, reset_rnn, edge_index):
        return (th.full_like(state["contribution"], 10),)


class _AlwaysValid:
    def predict(self, state, reset_rnn, edge_index):
        return (th.ones_like(state["contribution"], dtype=th.bool),)


class _ScriptedSwitch:
    """Records the membership `state["agent_group"]` at the moment it is
    called for the decision round -- exactly the tensor
    `GraphNetwork.predict_independent` would see -- and returns the scripted
    does_switch pair only on that round. `Environment.step()` calls this
    every round to keep an RNN warm, so every other call must be a no-op,
    matching `manager/environment.py: step`'s own contract."""

    def __init__(self, decision_round, does_switch_at_decision):
        self.decision_round = decision_round
        self.does_switch_at_decision = does_switch_at_decision
        self.captured_at_decision = None
        self.calls = []

    def predict(self, state, reset_rnn, edge_index):
        round_ = int(state["round_number"][0, 0, 0])
        self.calls.append(round_)
        if round_ == self.decision_round:
            # THE ASSERTION THIS TEST IS BUILT AROUND: this snapshot is taken
            # BEFORE `apply_switch` runs (env.step calls the predictor first),
            # so it is the PRE-switch membership -- the same anchoring
            # data.py's `agent_group` column uses (round r's own group_id,
            # not round r+1's).
            self.captured_at_decision = state["agent_group"].clone()
            return self.does_switch_at_decision.clone(), None
        return th.zeros_like(state["contribution"], dtype=th.bool), None


def simulation_side():
    """Drives a real Environment through the decision round and returns the
    captured pre-switch membership, plus the post-switch membership one round
    later (to make the pre/post distinction concrete rather than assumed)."""
    pre = th.tensor(PRE, dtype=th.int64)
    post = th.tensor(POST, dtype=th.int64)
    does_switch = (pre != post).unsqueeze(-1)  # (n_batch, n_agents, 1)

    switch = _ScriptedSwitch(DECISION_ROUND, does_switch)
    env = ArtificialHumanEnv(
        artifical_humans=_ConstantContribution(),
        artifical_humans_valid=_AlwaysValid(),
        artifical_humans_switch=switch,
        switch_every=SWITCH_EVERY,
        batch_size=N_BATCH,
        n_agents=N_AGENTS,
        n_contributions=21,
        n_punishments=21,
        n_rounds=N_ROUNDS,
        n_groups=2,
        device="cpu",
        reward_mode="avg",
        default_values={
            "punishment": 0,
            "contribution": 0,
            "round_number": 0,
            "is_first": False,
            "contribution_valid": False,
            "punishment_valid": False,
            "common_good": 0,
            "contributor_payoff": 0,
            "manager_payoff": 0,
            "reward": 0,
        },
    )
    # The constructor only accepts a single membership broadcast across the
    # batch; each scenario needs its OWN row, so the per-batch membership is
    # set directly and `reset()` is re-run to apply it exactly the way
    # `Environment.reset()` applies `initial_agent_groups` at the start of any
    # real episode.
    env.initial_agent_groups = pre.clone()
    env.reset()
    assert th.equal(env.state["agent_group"].squeeze(-1), pre)

    done = False
    post_switch_capture = None
    while not done:
        _, _, done = env.step()
        if int(env.round_number[0, 0, 0]) == DECISION_ROUND + 1:
            post_switch_capture = env.state["agent_group"].clone()

    assert switch.calls == list(range(N_ROUNDS)), switch.calls
    assert switch.captured_at_decision is not None
    return switch.captured_at_decision.squeeze(-1), post_switch_capture.squeeze(-1)


def _counts_from_sim_shaped(agent_group, switch):
    """(m, k) via `joint_exodus_counts` on sim-shaped (n_batch, n_agents)
    membership/switch tensors -- reshaped to the (N, R) convention the
    function actually expects (R = 1, N = n_batch * n_agents, batch-major
    exactly like `data.py`'s tensors flatten), with every decider valid (the
    simulation never produces a timeout)."""
    n_batch, n_agents = agent_group.shape
    flat_group = agent_group.reshape(n_batch * n_agents, 1)
    flat_switch = switch.reshape(n_batch * n_agents, 1).to(th.bool)
    flat_mask = th.ones_like(flat_switch, dtype=th.bool)
    batch_index = th.tensor(
        [b for b in range(n_batch) for _ in range(n_agents)], dtype=th.int64
    )
    m, k = joint_exodus_counts(
        flat_switch, flat_mask, flat_group, batch_index, n_batch=n_batch
    )
    return m[:, 0], k[:, 0]


# --------------------------------------------------------------------------- #
# the parity assertions
# --------------------------------------------------------------------------- #
def test_pre_switch_membership_matches_between_training_and_sim():
    """The core claim: the PRE-switch membership `data.py` derives for the
    decision round (its `agent_group` column at round r) is, agent for agent,
    the membership a real `Environment.step()` holds when it calls the switch
    predictor for that same round."""
    _, _, agent_group_train = training_side()
    agent_group_sim, _ = simulation_side()
    assert th.equal(agent_group_train, agent_group_sim), (
        "training's data.py-derived pre-switch membership and the "
        "environment's own state at decision time disagree"
    )


def test_pre_switch_is_not_post_switch():
    """Guards the anchoring claim from the trivial way it could pass by
    accident: if PRE and POST membership happened to coincide, agreeing with
    either would look like success. They must differ here, and the captured
    snapshot must be the PRE value, never the POST one."""
    agent_group_sim, post_switch_sim = simulation_side()
    pre = th.tensor(PRE, dtype=th.int64)
    post = th.tensor(POST, dtype=th.int64)
    assert not th.equal(pre, post), "scenario is not exercising a real switch"
    assert th.equal(agent_group_sim, pre)
    assert th.equal(post_switch_sim, post)
    assert not th.equal(agent_group_sim, post_switch_sim)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_counts_match_per_scenario(name):
    """(m, k) from `train.joint_exodus_counts` on the data.py-parsed tensors
    equals (m, k) from the same function on the environment-captured
    membership, scenario by scenario -- unbalanced (A, C), fully merged (B),
    switchers in both groups at once (C), and the label-symmetric case (D)."""
    row = SCENARIO_NAMES.index(name)
    m_train, k_train, _ = training_side()
    agent_group_sim, _ = simulation_side()

    switch_sim = th.tensor(PRE, dtype=th.int64) != th.tensor(POST, dtype=th.int64)
    m_sim, k_sim = _counts_from_sim_shaped(agent_group_sim, switch_sim)

    assert k_train[row].tolist() == k_sim[row].tolist()
    assert m_train[row].tolist() == m_sim[row].tolist()


def test_counts_hit_every_declared_case():
    """Sanity on the scenario design itself, so the parametrised check above
    is known to actually exercise what the plan asks for."""
    m_train, k_train, _ = training_side()
    k = {name: k_train[i].tolist() for i, name in enumerate(SCENARIO_NAMES)}
    m = {name: m_train[i].tolist() for i, name in enumerate(SCENARIO_NAMES)}
    assert k["A_unbalanced_one_sided"] == [5, 3] and m["A_unbalanced_one_sided"] == [
        2,
        0,
    ]
    assert k["B_fully_merged"] == [0, 8] and m["B_fully_merged"] == [0, 3]
    assert k["C_unbalanced_both_sides"] == [5, 3] and m["C_unbalanced_both_sides"] == [
        2,
        1,
    ]
    assert k["D_symmetric_swap_blind"] == [4, 4] and m["D_symmetric_swap_blind"] == [
        2,
        2,
    ]


def test_group_labels_are_not_merely_count_equal():
    """The plan's explicit warning: a test that only compares totals would
    pass even if the two groups were swapped. Scenario D is built exactly so
    that swapping labels leaves the (m, k) PAIR unchanged (k = (4, 4), m =
    (2, 2) either way) -- so only a per-agent identity check can tell the
    correct mapping from a transposed one. This test shows the count check
    alone is blind here, and that the per-agent check this file leads with is
    not."""
    row = SCENARIO_NAMES.index("D_symmetric_swap_blind")
    _, _, agent_group_train = training_side()
    agent_group_sim, _ = simulation_side()
    true_train = agent_group_train[row]
    true_sim = agent_group_sim[row]
    swapped_sim = 1 - true_sim  # the transposition bug this test must catch

    switch_sim = th.tensor(PRE, dtype=th.int64) != th.tensor(POST, dtype=th.int64)

    def counts_for(agent_group_row):
        agent_group = agent_group_sim.clone()
        agent_group[row] = agent_group_row
        m, k = _counts_from_sim_shaped(agent_group, switch_sim)
        return m[row].tolist(), k[row].tolist()

    # the swap is invisible to a totals-only comparison ...
    assert counts_for(true_sim) == counts_for(swapped_sim)
    # ... but not to a per-agent identity check, which is what this file
    # actually asserts in test_pre_switch_membership_matches_between_
    # training_and_sim.
    assert not th.equal(true_train, swapped_sim)
    assert th.equal(true_train, true_sim)
