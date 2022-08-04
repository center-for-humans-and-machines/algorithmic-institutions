import torch as th
from aimanager.generic.graph import GraphNetwork


class ArtificalManager():
    def __init__(
            self, *, n_contributions, n_punishments, default_values, model_args=None, policy_model=None, opt_args=None,
            gamma=None, target_update_freq=None, eps=0,  device):
        self.device = device

        if policy_model:
            self.policy_model = policy_model
        else:
            self.policy_model = GraphNetwork(
                y_name='punishments', y_levels=n_punishments, default_values=default_values,
                **model_args).to(device)

        if opt_args:
            self.target_model = GraphNetwork(
                y_name='punishments', y_levels=n_punishments, default_values=default_values,
                **model_args).to(device)

            self.target_model.eval()
            self.optimizer = th.optim.RMSprop(self.policy_model.parameters(), **opt_args)
            self.gamma = gamma
            self.target_update_freq = target_update_freq
        self.n_contributions = n_contributions
        self.n_punishments = n_punishments
        self.eps = eps

    def encode(self, state, edge_index, **_):
        return self.policy_model.encode(state, edge_index=edge_index)

    def encode_pure(self, state, **_):
        return self.policy_model.encode_pure(state)

    def get_q(self, manager_observations, first=False):
        with th.no_grad():
            return self.policy_model(manager_observations, reset_rnn=first)

    def eps_greedy(self, q_values):
        """
        Args:
            q_values: Tensor of type `th.float` and arbitrary shape, last dimension reflect the actions.
            eps: fraction of actions sampled at random
        Returns:
            actions: Tensor of type `th.long` and the same dimensions then q_values, besides of the last.
        """
        n_actions = q_values.shape[-1]
        actions_shape = q_values.shape[:-1]

        greedy_actions = q_values.argmax(-1)
        random_actions = th.randint(0, n_actions, size=actions_shape, device=self.device)

        # random number which determine whether to take the random action
        random_numbers = th.rand(size=actions_shape, device=self.device)
        select_random = (random_numbers < self.eps).long()
        picked_actions = select_random * random_actions + (1 - select_random) * greedy_actions

        return picked_actions

    def update(self, update_step, action, reward, **obs):
        if (update_step % self.target_update_freq == 0):
            # copy policy net to target net
            self.target_model.load_state_dict(self.policy_model.state_dict())

        self.policy_model.train()
        current_state_action_values = self.policy_model(
            obs, reset_rnn=True).gather(-1, action.unsqueeze(-1))

        next_state_values = th.zeros_like(reward, device=self.device)

        # we skip the first observation and set the future value for the last
        # observation to 0
        next_state_values[:, :-1] \
            = self.target_model(obs, reset_rnn=True)[:, 1:].max(-1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward

        # Compute Huber loss
        loss = th.nn.functional.smooth_l1_loss(
            current_state_action_values, expected_state_action_values.unsqueeze(-1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self, filename):
        to_save = {
            'policy_model': self.policy_model,
            'n_contributions': self.n_contributions,
            'n_punishments': self.n_punishments
        }
        th.save(to_save, filename)

    @classmethod
    def load(cls, filename):
        to_load = th.load(filename)
        ah = cls(**to_load)
        return ah
