import torch as th
from aimanager.generic.graph import GraphNetwork

# Exploration modes. `eps_greedy` is the agent this project has always run:
# one value head, an action resampled uniformly over all 31 punishment levels
# with probability `eps` at every agent-round. `bootstrap` is the
# bootstrapped-DQN alternative (Osband, Blundell, Pritzel and Van Roy, 2016):
# K heads on a shared torso, each trained on its own Bernoulli bootstrap of
# the replay, one head drawn per episode and followed greedily for that whole
# episode, and no per-action dithering at all.
EPS_GREEDY = "eps_greedy"
BOOTSTRAP = "bootstrap"
EXPLORATION_MODES = (EPS_GREEDY, BOOTSTRAP)

# How the K heads map onto the 1000 episodes the env runs in parallel.
# `per_episode` gives every episode in the batch its own head, so one rollout
# exercises the whole ensemble; `per_rollout` draws a single head for all 1000
# episodes, the literal port of the single-env algorithm, kept so the two can
# be compared rather than assumed equivalent.
PER_EPISODE = "per_episode"
PER_ROLLOUT = "per_rollout"
HEAD_ASSIGNMENTS = (PER_EPISODE, PER_ROLLOUT)


class ArtificalManager:
    def __init__(
        self,
        *,
        n_contributions,
        n_punishments,
        default_values,
        n_groups=1,
        model_args=None,
        policy_model=None,
        opt_args=None,
        gamma=None,
        target_update_freq=None,
        eps=None,
        n_heads=1,
        bootstrap_p=1.0,
        exploration=EPS_GREEDY,
        head_assignment=PER_EPISODE,
        device,
    ):
        self.device = device
        self.n_groups = n_groups

        assert (
            exploration in EXPLORATION_MODES
        ), f"exploration must be one of {EXPLORATION_MODES}, got {exploration!r}"
        assert head_assignment in HEAD_ASSIGNMENTS, (
            f"head_assignment must be one of {HEAD_ASSIGNMENTS}, "
            f"got {head_assignment!r}"
        )
        assert (
            0.0 < bootstrap_p <= 1.0
        ), f"bootstrap_p must be in (0, 1], got {bootstrap_p}"
        self.exploration = exploration
        self.head_assignment = head_assignment
        self.bootstrap_p = bootstrap_p

        if policy_model:
            self.policy_model = policy_model
            # A checkpoint carries its own head count; trust the module over
            # the argument so a loaded manager cannot be told the wrong K.
            n_heads = getattr(policy_model, "n_heads", 1)
        else:
            assert "n_heads" not in (
                model_args or {}
            ), "n_heads belongs in manager_args, not manager_args.model_args"
            self.policy_model = GraphNetwork(
                y_name="punishment",
                y_levels=n_punishments,
                default_values=default_values,
                n_heads=n_heads,
                **model_args,
            ).to(device)
        self.n_heads = n_heads

        if opt_args:
            assert model_args is not None
            assert gamma is not None
            assert target_update_freq is not None
            assert eps is not None
            self.target_model = GraphNetwork(
                y_name="punishment",
                y_levels=n_punishments,
                default_values=default_values,
                n_heads=n_heads,
                **model_args,
            ).to(device)

            self.target_model.eval()
            self.optimizer = th.optim.RMSprop(
                self.policy_model.parameters(), **opt_args
            )
            self.gamma = gamma
            self.target_update_freq = target_update_freq
        else:
            assert model_args is None
            assert gamma is None
            assert target_update_freq is None
            assert eps is None
        self.n_contributions = n_contributions
        self.n_punishments = n_punishments
        self.default_values = default_values
        self.eps = eps

    # ---- bootstrap bookkeeping ---------------------------------------- #

    def draw_heads(self, n_batch, device=None):
        """One head index per parallel episode, drawn at episode start and
        held for the whole episode. `per_rollout` draws once and broadcasts."""
        device = self.device if device is None else device
        if self.head_assignment == PER_ROLLOUT:
            one = th.randint(0, self.n_heads, (1,), device=device)
            return one.expand(n_batch).clone()
        return th.randint(0, self.n_heads, (n_batch,), device=device)

    def draw_masks(self, n_batch, device=None):
        """The bootstrap mask, shape (n_batch, 1, 1, K): head k trains on
        episode b iff mask[b, 0, 0, k]. Drawn i.i.d. Bernoulli(bootstrap_p),
        once per episode, and stored with the episode in the replay buffer so
        it is a fixed property of the data rather than resampled on every
        draw. Drawn independently of `draw_heads`, so which head generated an
        episode says nothing about which heads learn from it -- that
        independence is what keeps the shared buffer from correlating the
        ensemble."""
        device = self.device if device is None else device
        u = th.rand((n_batch, 1, 1, self.n_heads), device=device)
        return u < self.bootstrap_p

    def encode(self, state, edge_index, **_):
        return self.policy_model.encode(state, edge_index=edge_index)

    def get_action(self, state, first=False, edge_index=None, greedy=False, head=None):
        """Select punishments.

        `greedy=True` is the evaluation policy and is the same in every arm of
        the exploration comparison: every exploration mechanism off. For an
        ensemble that means the **consensus**, defined here as
        `argmax_a mean_k Q_k(s, a)` -- the head-averaged Q function, acted on
        greedily. Mean-of-Q, not a vote over per-head argmaxes: on an ordinal
        action space of 31 levels a plurality vote fragments across
        neighbouring levels and its winner depends on tie-breaking, while the
        head-averaged Q is the ensemble's own value function and moves
        smoothly with the heads. With K=1 the mean over one head is that head,
        exactly, so this path is bit-identical to the single-head greedy
        policy.

        `greedy=False` is behaviour. Under `eps_greedy` it is the historical
        per-action dither. Under `bootstrap` it is head `head[b]`'s own greedy
        action, with no dithering -- the coherent alternative policy this arm
        exists to test.
        """
        n_batch, n_agents, n_rounds = list(state.values())[0].shape
        exp_state = self.expand_obs_for_groups(state, self.n_groups)
        encoded = self.policy_model.encode(exp_state, edge_index=edge_index)
        with th.no_grad():
            q_values = self.policy_model(encoded, reset_rnn=first)
            q_values = q_values.reshape(
                n_batch, self.n_groups, n_agents, n_rounds, self.n_heads, -1
            )

            n_actions = q_values.shape[-1]
            if (not greedy) and self.exploration == BOOTSTRAP:
                assert head is not None, "bootstrap behaviour needs a head draw"
                idx = head.view(n_batch, 1, 1, 1, 1, 1).expand(
                    n_batch, self.n_groups, n_agents, n_rounds, 1, n_actions
                )
                q_sel = q_values.gather(-2, idx).squeeze(-2)
            else:
                q_sel = q_values.mean(-2)
            greedy_action = q_sel.argmax(-1)
            agent_group = state["agent_group"].unsqueeze(1)  # (E, 1, A, T)
            greedy_action = greedy_action.gather(1, agent_group)  # (E, 1, A, T)
            greedy_action = greedy_action.squeeze(1)  # (E, A, T)
            if greedy or self.exploration == BOOTSTRAP:
                return greedy_action, q_values
            random_actions = th.randint(
                0, n_actions, size=greedy_action.shape, device=self.device
            )
            random_numbers = th.rand(size=greedy_action.shape, device=self.device)
            select_random = random_numbers < self.eps
            picked_action = th.where(select_random, random_actions, greedy_action)
            return picked_action, q_values

    def expand_obs_for_groups(self, obs, n_groups):
        exclude_keys = ["group_payoff"]

        E, A, T = obs["agent_group"].shape[:3]

        obs_group = {
            k: v.unsqueeze(1).expand(E, n_groups, *v.shape[1:])  # -> (E, G, A, T, ...)
            for k, v in obs.items()
            if k not in exclude_keys and v.shape[0] == E
        }
        group_idx = th.arange(n_groups, device=self.device).view(1, n_groups, 1, 1)
        obs_group["group"] = group_idx.expand(E, n_groups, A, T)
        obs_group["in_group"] = obs_group["group"] == obs_group["agent_group"]

        obs_group = {
            k: v.reshape(E * n_groups, *v.shape[2:]) for k, v in obs_group.items()
        }
        return obs_group

    def update(
        self, update_step, action, reward, rl_group_id=None, head_mask=None, **obs
    ):
        if update_step % self.target_update_freq == 0:
            # copy policy net to target net
            self.target_model.load_state_dict(self.policy_model.state_dict())

        E, A, T = action.shape
        G = self.n_groups
        K = self.n_heads
        if head_mask is not None:
            # The replay stores the mask per episode, broadcast over the
            # round axis; any round carries the same draw, so take round 0.
            head_mask = head_mask.reshape(E, -1, K)[:, 0]
        exp_obs = self.expand_obs_for_groups(obs, self.n_groups)
        in_group = (
            exp_obs["in_group"].reshape(E, G, A, T).float()
        )  # episodes, groups, agents, round

        self.policy_model.train()
        encoded = self.policy_model.encode(exp_obs, y_encode=False)
        current_q = self.policy_model(
            encoded, reset_rnn=True
        )  # episode*groups*agents, round, heads*actions
        current_q = current_q.reshape(
            E, G, A, T, K, -1
        )  # episodes, groups, agents, round, heads, actions
        current_q = current_q.gather(
            -1, action.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(E, 1, A, T, K, 1)
        )  # episodes, 1, agents, round, heads, 1

        current_q_group = th.einsum(
            "egarki,egar->egrki", current_q, in_group
        )  # episodes, groups, rounds, heads, 1

        # we skip the first observation and set the future value for the terminal
        # state to 0. Each head bootstraps off its OWN target head: head k's
        # TD target is built from max_a Q_target_k, never from the ensemble.
        next_q_values = self.target_model(
            encoded, reset_rnn=True
        )  # episodes*groups*agents, round, heads*actions
        next_q_values = next_q_values.reshape(
            E, G, A, T, K, -1
        )  # episodes, groups, agents, round, heads, actions

        max_next_q_value = (
            next_q_values[:, :, :, 1:].max(-1)[0].detach()
        )  # episodes, groups, agents, round-1, heads
        max_next_q_value_group = th.einsum(
            "egark,egar->egrk", max_next_q_value, in_group[:, :, :, 1:]
        )
        next_v = th.zeros((*reward.shape, K), device=self.device, dtype=reward.dtype)
        next_v[:, :, :-1] = max_next_q_value_group  # episodes, groups, round, heads

        # Compute the expected Q values
        expected_q = (next_v * self.gamma) + reward.unsqueeze(
            -1
        )  # episodes, groups, round, heads

        # Two-manager training: restrict the TD-error to the RL manager's
        # own group. Replay still stores per-group reward for diagnostics;
        # we only train on the slice we control.
        if rl_group_id is not None:
            current_q_group = current_q_group[:, rl_group_id : rl_group_id + 1]
            expected_q = expected_q[:, rl_group_id : rl_group_id + 1]

        # Compute Huber loss. With no mask this is exactly the mean-reduced
        # Huber loss the single-head agent computed, element for element.
        elementwise = th.nn.functional.smooth_l1_loss(
            current_q_group, expected_q.unsqueeze(-1), reduction="none"
        )
        if head_mask is None:
            loss = elementwise.mean()
        else:
            # head_mask: (E, K). Head k's loss sees only the episodes its own
            # bootstrap mask kept; the normaliser counts exactly those terms,
            # so a head is not penalised for having been dealt fewer episodes.
            w = head_mask.to(elementwise.dtype).view(E, 1, 1, K, 1)
            loss = (elementwise * w).sum() / w.expand_as(elementwise).sum().clamp(
                min=1.0
            )

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def save(self, filename):
        to_save = {
            "policy_model": self.policy_model.to(th.device("cpu")),
            "n_contributions": self.n_contributions,
            "n_punishments": self.n_punishments,
            "n_groups": self.n_groups,
            "default_values": self.default_values,
            "exploration": self.exploration,
            "bootstrap_p": self.bootstrap_p,
            "head_assignment": self.head_assignment,
        }
        th.save(to_save, filename)

    @classmethod
    def load(cls, filename, device):
        to_load = th.load(filename, map_location=device)
        ah = cls(**to_load, device=device)
        return ah
