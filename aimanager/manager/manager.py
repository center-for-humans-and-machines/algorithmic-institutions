import torch as th
from aimanager.generic.graph import GraphNetwork


class ArtificalManager:
    def __init__(
        self,
        *,
        n_contributions,
        n_punishments,
        n_groups,
        default_values,
        model_args=None,
        policy_model=None,
        opt_args=None,
        gamma=None,
        target_update_freq=None,
        eps=None,
        device
    ):
        self.device = device
        self.n_groups = n_groups

        if policy_model:
            self.policy_model = policy_model
        else:
            self.policy_model = GraphNetwork(
                y_name="punishment",
                y_levels=n_punishments,
                default_values=default_values,
                **model_args
            ).to(device)

        if opt_args:
            assert model_args is not None
            assert gamma is not None
            assert target_update_freq is not None
            assert eps is not None
            self.target_model = GraphNetwork(
                y_name="punishment",
                y_levels=n_punishments,
                default_values=default_values,
                **model_args
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

    def encode(self, state, edge_index, **_):
        return self.policy_model.encode(state, edge_index=edge_index)

    def encode_pure(self, state, **_):
        return self.policy_model.encode_pure(state)

    def get_action(self, state, first=False, edge_index=None, greedy=False):
        n_batch, n_agents, n_rounds = list(state.values())[0].shape
        exp_state = self.expand_obs_for_groups(state, self.n_groups)
        encoded = self.policy_model.encode(exp_state, edge_index=edge_index)
        with th.no_grad():
            q_values = self.policy_model(encoded, reset_rnn=first)
            q_values = q_values.reshape(n_batch, self.n_groups, n_agents, n_rounds, -1)

            n_actions = q_values.shape[-1]
            greedy_action = q_values.argmax(-1)
            agent_group = state["agent_group"].unsqueeze(1)       # (E, 1, A, T)
            greedy_action = greedy_action.gather(1, agent_group)   # (E, 1, A, T)
            greedy_action = greedy_action.squeeze(1)         # (E, A, T)
            if not greedy:
                random_actions = th.randint(
                    0, n_actions, size=greedy_action.shape, device=self.device
                )
                random_numbers = th.rand(size=greedy_action.shape, device=self.device)
                select_random = (random_numbers < self.eps)
                picked_action = th.where(select_random, random_actions, greedy_action)
                return picked_action, q_values
            else:
                return greedy_action, q_values

    def get_punishment(self, **state):
        n_batch, n_agents, n_rounds = list(state.values())[0].shape
        first = state["round_number"].max() == 0
        exp_state = self.expand_obs_for_groups(state, self.n_groups)
        encoded = self.policy_model.encode_pure(exp_state)
        q_values = self.policy_model(encoded, reset_rnn=first)
        q_values = q_values.reshape(n_batch, self.n_groups, n_agents, n_rounds, -1)
        greedy_actions = q_values.argmax(-1)  # (E, G, A, T)
        agent_group = state["agent_group"].unsqueeze(1)       # (E, 1, A, T)
        greedy_actions = greedy_actions.gather(1, agent_group)   # (E, 1, A, T)
        greedy_actions = greedy_actions.squeeze(1)         # (E, A, T)
        return greedy_actions


def expand_obs_for_groups(self, obs, n_groups):
    exclude_keys = ["group_payoff"]

    E, A, T = obs["agent_group"].shape[:3]

    obs_group = {
        k: v.unsqueeze(1).expand(
            E, n_groups, *v.shape[1:]   # -> (E, G, A, T, ...)
        ) for k, v in obs.items() if k not in exclude_keys
    }
    group_idx = th.arange(n_groups, device=self.device).view(1, n_groups, 1, 1)
    obs_group["group"] = group_idx.expand(E, n_groups, A, T)
    obs_group["in_group"] = obs_group["group"] == obs_group["agent_group"]

    obs_group = {k: v.reshape(E * n_groups, *v.shape[2:]) for k, v in obs_group.items()}
    return obs_group

    def update(self, update_step, action, reward, **obs):
        if update_step % self.target_update_freq == 0:
            # copy policy net to target net
            self.target_model.load_state_dict(self.policy_model.state_dict())

        E, A, T = action.shape
        G = self.n_groups
        exp_obs = self.expand_obs_for_groups(obs, self.n_groups)
        in_group = exp_obs['in_group'].reshape(E,G,A,T) # episodes, groups, agents, round

        self.policy_model.train()
        encoded = self.policy_model.encode(exp_obs, y_encode=False)
        current_q = self.policy_model(
            encoded, reset_rnn=True
        )  # episode*groups*agents, round, actions
        current_q = current_q.reshape(
            E, G, A, T, -1
        )  # episodes, groups, agents, round, actions
        current_q = current_q.gather(
            -1, action.unsqueeze(1).unsqueeze(-1)
        )  # episodes, groups, agents, round, 1

        current_q_group = th.einsum('egari,egar->egri', current_q, in_group) # episodes, groups, rounds, 1

        # we skip the first observation and set the future value for the terminal
        # state to 0
        next_q_values = self.target_model(
            encoded, reset_rnn=True
        )  # episodes*groups*agents, round, actions
        next_q_values = next_q_values.reshape(
            E, G, A, T, -1
        )  # episodes, groups, agents, round, actions

        max_next_q_value = next_q_values[:, :, :, 1:].max(-1)[0].detach() # episodes, groups, agents, round-1
        max_next_q_value_group = th.einsum('egar,egar->egr', max_next_q_value, in_group[:,:,:,1:])
        next_v = th.zeros_like(reward, device=self.device)
        next_v[:, :, :-1] = max_next_q_value_group # episodes, groups, round

        # Compute the expected Q values
        expected_q = (next_v * self.gamma) + reward # episodes, groups, round

        # Compute Huber loss
        loss = th.nn.functional.smooth_l1_loss(current_q_group, expected_q.unsqueeze(-1))

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
            "default_values": self.default_values,
        }
        th.save(to_save, filename)

    @classmethod
    def load(cls, filename, device):
        to_load = th.load(filename, map_location=device)
        ah = cls(**to_load, device=device)
        return ah
