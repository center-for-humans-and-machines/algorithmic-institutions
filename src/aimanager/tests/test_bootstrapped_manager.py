"""Bootstrapped DQN for the RL manager (Osband, Blundell, Pritzel, Van Roy 2016).

Runs on Raven: `manager.manager` imports GraphNetwork -> torch_scatter.

The load-bearing test is `test_k1_reproduces_the_existing_agent`. Everything
else here describes the new mechanism; that one asserts the mechanism is
*added* rather than the agent changed -- at K=1 with the bootstrap off, the
parameters, the chosen actions and the loss are bit-identical to the
single-head epsilon-greedy agent this project has always trained.
"""

import numpy as np
import pytest
import torch as th

N_PUNISHMENTS = 31
N_CONTRIBUTIONS = 21
E, A, T, G = 6, 4, 3, 2

MODEL_ARGS = {
    "hidden_size": 16,
    "add_rnn": True,
    "add_edge_model": True,
    "add_global_model": False,
    "x_encoding": [
        {"name": "contribution", "n_levels": N_CONTRIBUTIONS, "encoding": "numeric"},
        {"name": "prev_punishment", "n_levels": N_PUNISHMENTS, "encoding": "numeric"},
        {"etype": "bool", "name": "contribution_valid"},
        {"etype": "bool", "name": "in_group"},
    ],
    "b_encoding": [
        {"name": "round_number", "n_levels": 32, "encoding": "onehot"},
    ],
}


def _state(seed=0):
    g = th.Generator().manual_seed(seed)
    return {
        "contribution": th.randint(0, N_CONTRIBUTIONS, (E, A, T), generator=g),
        "prev_punishment": th.randint(0, N_PUNISHMENTS, (E, A, T), generator=g),
        "punishment": th.randint(0, N_PUNISHMENTS, (E, A, T), generator=g),
        "contribution_valid": th.ones((E, A, T), dtype=th.bool),
        "agent_group": th.randint(0, G, (E, A, T)),
        "round_number": th.arange(T).view(1, 1, T).expand(E, A, T).contiguous(),
    }


def _manager(seed=0, **kwargs):
    from aimanager.manager.manager import ArtificalManager

    th.manual_seed(seed)
    return ArtificalManager(
        n_contributions=N_CONTRIBUTIONS,
        n_punishments=N_PUNISHMENTS,
        n_groups=G,
        default_values={"punishment": 0, "contribution": 0},
        device=th.device("cpu"),
        model_args=MODEL_ARGS,
        opt_args={"lr": 2.0e-4},
        gamma=0.98,
        target_update_freq=1000,
        eps=0.1,
        **kwargs,
    )


def _legacy_loss(manager, state, action, reward, rl_group_id):
    """The pre-bootstrap TD loss, copied verbatim from `ArtificalManager.update`
    at 0ff44a9 and reduced to the value it returned. Nothing here knows about
    heads."""
    exp_obs = manager.expand_obs_for_groups(state, G)
    in_group = exp_obs["in_group"].reshape(E, G, A, T).float()
    manager.policy_model.train()
    encoded = manager.policy_model.encode(exp_obs, y_encode=False)
    current_q = manager.policy_model(encoded, reset_rnn=True).reshape(E, G, A, T, -1)
    current_q = current_q.gather(-1, action.unsqueeze(1).unsqueeze(-1))
    current_q_group = th.einsum("egari,egar->egri", current_q, in_group)
    next_q_values = manager.target_model(encoded, reset_rnn=True).reshape(
        E, G, A, T, -1
    )
    max_next = next_q_values[:, :, :, 1:].max(-1)[0].detach()
    max_next_group = th.einsum("egar,egar->egr", max_next, in_group[:, :, :, 1:])
    next_v = th.zeros_like(reward)
    next_v[:, :, :-1] = max_next_group
    expected_q = (next_v * manager.gamma) + reward
    current_q_group = current_q_group[:, rl_group_id : rl_group_id + 1]
    expected_q = expected_q[:, rl_group_id : rl_group_id + 1]
    return th.nn.functional.smooth_l1_loss(current_q_group, expected_q.unsqueeze(-1))


def _legacy_action(manager, state):
    """The pre-bootstrap epsilon-greedy action, copied verbatim."""
    n_batch, n_agents, n_rounds = list(state.values())[0].shape
    exp_state = manager.expand_obs_for_groups(state, G)
    encoded = manager.policy_model.encode(exp_state, edge_index=None)
    with th.no_grad():
        q_values = manager.policy_model(encoded, reset_rnn=True)
        q_values = q_values.reshape(n_batch, G, n_agents, n_rounds, -1)
        n_actions = q_values.shape[-1]
        greedy_action = q_values.argmax(-1)
        agent_group = state["agent_group"].unsqueeze(1)
        greedy_action = greedy_action.gather(1, agent_group).squeeze(1)
        random_actions = th.randint(0, n_actions, size=greedy_action.shape)
        random_numbers = th.rand(size=greedy_action.shape)
        select_random = random_numbers < manager.eps
        return th.where(select_random, random_actions, greedy_action)


# --------------------------------------------------------------------- #
# 1. K = 1 reproduces the existing agent
# --------------------------------------------------------------------- #


def test_k1_builds_bit_identical_parameters():
    """The head machinery draws no RNG and changes no shape at K=1, so a
    seeded build with `n_heads` absent and one with `n_heads=1` agree
    parameter for parameter."""
    from aimanager.generic.graph import GraphNetwork

    th.manual_seed(7)
    without = GraphNetwork(
        y_name="punishment", y_levels=N_PUNISHMENTS, default_values={}, **MODEL_ARGS
    )
    th.manual_seed(7)
    with_one = GraphNetwork(
        y_name="punishment",
        y_levels=N_PUNISHMENTS,
        default_values={},
        n_heads=1,
        **MODEL_ARGS,
    )
    a = dict(without.named_parameters())
    b = dict(with_one.named_parameters())
    assert a.keys() == b.keys()
    for k in a:
        assert th.equal(a[k], b[k]), k


def test_k1_reproduces_the_existing_agent():
    """Actions and loss, bit for bit, against the verbatim pre-bootstrap code."""
    manager = _manager(seed=3, n_heads=1)
    assert manager.exploration == "eps_greedy"
    state = _state(1)

    th.manual_seed(11)
    got = manager.get_action(state, first=True, greedy=False)[0]
    th.manual_seed(11)
    want = _legacy_action(manager, state)
    assert th.equal(got, want)

    action = th.randint(0, N_PUNISHMENTS, (E, A, T))
    reward = th.randn(E, G, T)
    want_loss = _legacy_loss(manager, state, action, reward, rl_group_id=0)
    got_loss = manager.update(1, action=action, reward=reward, rl_group_id=0, **state)
    assert th.equal(got_loss.detach(), want_loss.detach())


def test_k1_with_all_ones_mask_matches_the_unmasked_loss():
    """The masking path with nothing masked out is the plain mean, so turning
    the mechanism on without using it changes no number."""
    manager = _manager(seed=3, n_heads=1, exploration="bootstrap", bootstrap_p=1.0)
    state = _state(1)
    action = th.randint(0, N_PUNISHMENTS, (E, A, T))
    reward = th.randn(E, G, T)
    want = _legacy_loss(manager, state, action, reward, rl_group_id=0)
    got = manager.update(
        1,
        action=action,
        reward=reward,
        rl_group_id=0,
        head_mask=th.ones(E, 1, dtype=th.bool),
        **state,
    )
    assert th.allclose(got.detach(), want.detach(), atol=0, rtol=1e-6)


# --------------------------------------------------------------------- #
# 2. Head independence
# --------------------------------------------------------------------- #


def test_heads_are_separate_parameters_and_disagree_at_init():
    manager = _manager(seed=5, n_heads=8, exploration="bootstrap", bootstrap_p=0.5)
    state = _state(2)
    _, q = manager.get_action(state, first=True, greedy=True)
    assert q.shape == (E, G, A, T, 8, N_PUNISHMENTS)
    per_head = q.argmax(-1)
    spread = per_head.max(-1)[0] - per_head.min(-1)[0]
    assert (spread > 0).float().mean().item() > 0.5, (
        "freshly initialised heads already agree everywhere -- with no "
        "dithering this ensemble would never explore"
    )
    # Perturbing one head's readout slice moves that head's Q values only.
    weight = manager.policy_model.op2.node_model.node_mlp.weight
    assert weight.shape[0] == 8 * N_PUNISHMENTS
    with th.no_grad():
        weight[3 * N_PUNISHMENTS : 4 * N_PUNISHMENTS] += 5.0
    _, q2 = manager.get_action(state, first=True, greedy=True)
    moved = (q2 - q).abs().sum(dim=(0, 1, 2, 3, 5))
    assert moved[3].item() > 0
    assert moved.sum().item() == pytest.approx(moved[3].item(), rel=1e-5)


# --------------------------------------------------------------------- #
# 3. The bootstrap mask
# --------------------------------------------------------------------- #


def test_mask_is_bernoulli_and_independent_of_the_head_draw():
    manager = _manager(seed=5, n_heads=10, exploration="bootstrap", bootstrap_p=0.5)
    th.manual_seed(0)
    n = 20000
    mask = manager.draw_masks(n).reshape(n, 10).float()
    head = manager.draw_heads(n)
    assert abs(mask.mean().item() - 0.5) < 0.02
    corr = np.corrcoef(mask.numpy().T)
    off = corr[~np.eye(10, dtype=bool)]
    assert np.abs(off).max() < 0.05, "heads share a mask draw"
    # Which head generated an episode must say nothing about which heads
    # learn from it: that independence is what stops the shared replay
    # buffer from quietly correlating the ensemble.
    for k in range(10):
        own = (head == k).float().numpy()
        assert abs(np.corrcoef(own, mask[:, k].numpy())[0, 1]) < 0.05


def test_a_masked_out_head_receives_no_gradient():
    manager = _manager(seed=5, n_heads=4, exploration="bootstrap", bootstrap_p=0.5)
    weight = manager.policy_model.op2.node_model.node_mlp.weight
    before = weight.detach().clone()
    mask = th.zeros(E, 4, dtype=th.bool)
    mask[:, 1] = True  # only head 1 learns from these episodes
    manager.update(
        1,
        action=th.randint(0, N_PUNISHMENTS, (E, A, T)),
        reward=th.randn(E, G, T),
        rl_group_id=0,
        head_mask=mask,
        **_state(4),
    )
    delta = (weight.detach() - before).abs().reshape(4, N_PUNISHMENTS, -1)
    per_head = delta.sum(dim=(1, 2))
    assert per_head[1].item() > 0
    for k in (0, 2, 3):
        assert per_head[k].item() == 0.0, f"masked-out head {k} moved"


# --------------------------------------------------------------------- #
# 4. Episode-consistent head selection
# --------------------------------------------------------------------- #


def test_behaviour_follows_the_drawn_head_per_episode():
    manager = _manager(seed=5, n_heads=6, exploration="bootstrap", bootstrap_p=0.5)
    state = _state(2)
    head = th.arange(E) % 6
    action, q = manager.get_action(state, first=True, greedy=False, head=head)
    from aimanager.manager.head_probe import gather_to_own_group

    per_head = gather_to_own_group(q.argmax(-1), state["agent_group"])  # (E,A,T,K)
    want = per_head.gather(-1, head.view(E, 1, 1, 1).expand(E, A, T, 1)).squeeze(-1)
    assert th.equal(action, want)


def test_run_batch_draws_one_head_per_rollout_and_holds_it():
    """The head is drawn at episode start and followed for the whole episode:
    `run_batch` must draw once, not once per round."""
    from aimanager import rl_manager

    env = _StubEnv()
    manager = _StubManager(n_heads=5)
    rl_manager.replay_keys = ["contribution", "agent_group"]
    metrics = rl_manager.run_batch(manager, env, on_policy=False, update_step=0)
    assert manager.n_head_draws == 1
    assert manager.n_mask_draws == 1
    assert len({id(h) for h in manager.heads_seen}) == 1
    assert len(metrics) == env.n_rounds
    assert {m["sampling"] for m in metrics} == {"bootstrap-head"}

    manager2 = _StubManager(n_heads=5)
    ev = rl_manager.run_batch(manager2, _StubEnv(), on_policy=True, update_step=0)
    assert manager2.n_head_draws == 0
    assert all(h is None for h in manager2.heads_seen)
    assert {m["sampling"] for m in ev} == {"greedy"}


def test_per_rollout_assignment_uses_a_single_head():
    m1 = _manager(
        seed=5, n_heads=8, exploration="bootstrap", head_assignment="per_rollout"
    )
    th.manual_seed(0)
    assert m1.draw_heads(500).unique().numel() == 1
    m2 = _manager(seed=5, n_heads=8, exploration="bootstrap")
    th.manual_seed(0)
    assert m2.draw_heads(500).unique().numel() == 8


# --------------------------------------------------------------------- #
# 5. The consensus rule
# --------------------------------------------------------------------- #


def test_consensus_is_mean_of_q_not_a_vote():
    from aimanager.manager.head_probe import consensus_action, vote_action

    # Three heads: two agree on action 1 by a hair, one prefers action 0 by a
    # mile. The vote says 1; the head-averaged Q says 0. The evaluation
    # policy must follow the head-averaged Q.
    q = th.zeros(1, 1, 1, 1, 3, 3)
    q[..., 0, :] = th.tensor([9.0, 0.0, 0.0])
    q[..., 1, :] = th.tensor([0.0, 0.1, 0.0])
    q[..., 2, :] = th.tensor([0.0, 0.1, 0.0])
    agent_group = th.zeros(1, 1, 1, dtype=th.long)
    head_acts = q.argmax(-1)[:, 0]  # (E, A, T, K)
    assert vote_action(head_acts, 3).item() == 1
    assert consensus_action(q, agent_group).item() == 0


def test_greedy_path_uses_the_consensus():
    manager = _manager(seed=5, n_heads=4, exploration="bootstrap")
    from aimanager.manager.head_probe import consensus_action

    state = _state(2)
    action, q = manager.get_action(state, first=True, greedy=True)
    assert th.equal(action, consensus_action(q, state["agent_group"]))


# --------------------------------------------------------------------- #
# 6. The shape probe bins exactly as the evaluation suite does
# --------------------------------------------------------------------- #


def test_rpa_bins_match_pandas_cut():
    import pandas as pd
    from aimanager.evaluation_suite.metrics import RPA_EDGES, RPA_LABELS
    from aimanager.manager.head_probe import rpa_bins

    c = th.arange(0, 21)
    want = pd.cut(pd.Series(c.numpy()), RPA_EDGES, labels=RPA_LABELS).astype(str)
    got = [RPA_LABELS[i] for i in rpa_bins(c).tolist()]
    assert got == list(want)


def test_shape_metrics_are_the_binned_means():
    from aimanager.manager.head_probe import shape_metrics

    contribution = th.tensor([[[0], [20], [0], [7]]])  # (1, 4, 1)
    punishment = th.tensor([[[10], [2], [4], [6]]])
    mask = th.ones_like(contribution, dtype=th.bool)
    out = shape_metrics(punishment, contribution, mask)
    assert out["rpa_mean_b0"] == pytest.approx(7.0)  # (10 + 4) / 2
    assert out["rpa_n_b0"] == 2.0
    assert out["rpa_mean_b5"] == pytest.approx(2.0)
    assert out["rpa_slope"] == pytest.approx(-5.0)  # human sign is negative
    mask2 = th.tensor([[[True], [True], [False], [False]]])
    out2 = shape_metrics(punishment, contribution, mask2)
    assert out2["rpa_mean_b0"] == pytest.approx(10.0)
    assert np.isnan(out2["rpa_mean_b2"])


# --------------------------------------------------------------------- #
# Stubs for the run_batch wiring test
# --------------------------------------------------------------------- #


class _StubEnv:
    n_rounds = 4
    n_agents = 4

    def __init__(self, n_batch=5):
        self.n_batch = n_batch
        self.batch_size = n_batch
        self.round_number = 0
        self.agent_groups = th.zeros(n_batch, self.n_agents, 1, dtype=th.long)
        self.batch_edge_index = None
        self.batch = None
        self.batch_agent_group_mask = None

    def _state(self):
        return {
            "contribution": th.zeros(self.n_batch, self.n_agents, 1, dtype=th.long),
            "punishment": th.zeros(self.n_batch, self.n_agents, 1, dtype=th.long),
            "common_good": th.zeros(self.n_batch, self.n_agents, 1),
            "contributor_payoff": th.zeros(self.n_batch, self.n_agents, 1),
            "group_payoff": th.zeros(self.n_batch, 2, 1),
            "group_payoff_sum": th.zeros(self.n_batch, 2, 1),
            "agent_group": self.agent_groups,
        }

    def reset(self):
        self.round_number = 0
        return self._state()

    def served_state(self):
        return self._state()

    def punish(self, punishment):
        state = self._state()
        state["punishment"] = punishment
        return state

    def step(self):
        self.round_number += 1
        done = self.round_number >= self.n_rounds
        return None, th.zeros(self.n_batch, 2, 1), done


class _StubManager:
    exploration = "bootstrap"
    head_assignment = "per_episode"
    bootstrap_p = 0.5

    def __init__(self, n_heads):
        self.n_heads = n_heads
        self.n_head_draws = 0
        self.n_mask_draws = 0
        self.heads_seen = []

    def draw_heads(self, n_batch, device=None):
        self.n_head_draws += 1
        return th.zeros(n_batch, dtype=th.long)

    def draw_masks(self, n_batch, device=None):
        self.n_mask_draws += 1
        return th.ones(n_batch, 1, 1, self.n_heads, dtype=th.bool)

    def get_action(self, state, first=False, greedy=False, head=None, **_):
        self.heads_seen.append(head)
        shape = state["agent_group"].shape
        q = th.zeros(shape[0], 2, shape[1], shape[2], self.n_heads, N_PUNISHMENTS)
        return th.zeros(shape, dtype=th.long), q


def test_episode_budget_counts_every_rollout():
    """The comparison's budget is equal environment episodes. One `run_batch`
    is one `env.reset()` and `env.batch_size` complete episodes, whatever the
    exploration mechanism -- which is why this arm matches the reference at
    the same `n_update_steps`."""
    from aimanager import rl_manager

    for k in rl_manager.EPISODE_BUDGET:
        rl_manager.EPISODE_BUDGET[k] = 0
    rl_manager.replay_keys = ["contribution", "agent_group"]
    env = _StubEnv()
    rl_manager.run_batch(_StubManager(3), env, on_policy=False, update_step=0)
    rl_manager.run_batch(_StubManager(3), _StubEnv(), on_policy=True, update_step=0)
    budget = rl_manager.EPISODE_BUDGET
    assert budget["rollouts"] == 2
    assert budget["episodes"] == 2 * env.batch_size
    assert budget["behaviour_episodes"] == env.batch_size
    assert budget["eval_episodes"] == env.batch_size
    assert budget["episode_rounds"] == 2 * env.batch_size * env.n_rounds
