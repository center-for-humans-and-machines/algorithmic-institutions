import torch as th
from aimanager.generic.mlp import MultiLayer

class ArtificalManager():
    def __init__(
            self, *, n_contributions, n_punishments, model_args=None, policy_model=None, opt_args=None, 
            gamma=None, target_update_freq=None, device):
        self.device = device
        input_size = n_contributions + 1
        if policy_model:
            self.policy_model = policy_model
        else:
            self.policy_model = MultiLayer(
                output_size=n_punishments, input_size=input_size, **model_args).to(device)

        if opt_args:
            self.target_model = MultiLayer(
                output_size=n_punishments, input_size=input_size, **model_args).to(device)

            self.target_model.eval()
            self.optimizer = th.optim.RMSprop(self.policy_model.parameters(), **opt_args)
            self.gamma = gamma
            self.target_update_freq = target_update_freq
        self.n_contributions = n_contributions
        self.n_punishments = n_punishments

    def init_episode(self, episode):
        if (episode % self.target_update_freq == 0):
            # copy policy net to target net
            self.target_model.load_state_dict(self.policy_model.state_dict())

        # TODO: add for rnn
        # self.policy_model.reset()
        # self.target_model.reset()

    def encode_obs(self, contributions, episode_step, **_):
        oh_cont = th.nn.functional.one_hot(contributions, num_classes=self.n_contributions).float()
        es = episode_step.unsqueeze(0).tile((contributions.shape[0],1)).float() / 16
        return th.cat([oh_cont, es], dim=-1)

    def get_q(self, manager_observations, **_):
        with th.no_grad():
            return self.policy_model(manager_observations)

    def act(self, **state):
        obs = self.encode_obs(**state)
        q = self.get_q(manager_observations=obs)
        return q.argmax(dim=-1)

    def eps_greedy(self, q_values, eps):
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
        select_random = (random_numbers < eps).long()
        picked_actions = select_random * random_actions + (1 - select_random) * greedy_actions

        return picked_actions


    def update(self, actions, rewards, current_obs, next_obs, **_):
        self.policy_model.train()
        current_state_action_values = self.policy_model(
            current_obs).gather(-1, actions.unsqueeze(-1))

        next_state_values = th.zeros_like(rewards, device=self.device)
        next_state_values[:,:-1] = self.target_model(next_obs[:,:-1]).max(-1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + rewards

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