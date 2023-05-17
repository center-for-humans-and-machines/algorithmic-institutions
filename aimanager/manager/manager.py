import torch as th
from aimanager.generic.graph import GraphNetwork


class ArtificalManager:
    def __init__(
        self,
        *,
        n_contributions,
        n_punishments,
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

    def get_q(self, state, first=False, edge_index=None):
        n_batch, n_agents, n_rounds = list(state.values())[0].shape
        encoded = self.policy_model.encode(state, edge_index=edge_index)
        with th.no_grad():
            q_values = self.policy_model(encoded, reset_rnn=first)
            q_values = q_values.reshape(n_batch, n_agents, n_rounds, -1)
        return q_values

    def get_punishment(self, **state):
        first = state["round_number"].max() == 0
        encoded = self.policy_model.encode_pure(state)
        q_values = self.policy_model(encoded, reset_rnn=first)
        greedy_actions = q_values.argmax(-1)
        return greedy_actions

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
        random_actions = th.randint(
            0, n_actions, size=actions_shape, device=self.device
        )

        # random number which determine whether to take the random action
        random_numbers = th.rand(size=actions_shape, device=self.device)
        select_random = (random_numbers < self.eps).long()
        picked_actions = (
            select_random * random_actions + (1 - select_random) * greedy_actions
        )

        return picked_actions

    def update(self, update_step, action, reward, **obs):
        if update_step % self.target_update_freq == 0:
            # copy policy net to target net
            self.target_model.load_state_dict(self.policy_model.state_dict())

        self.policy_model.train()
        encoded = self.policy_model.encode(obs, y_encode=False)
        current_q = self.policy_model(
            encoded, reset_rnn=True
        )  # episodes*agents, round, actions
        current_q = current_q.reshape(
            *action.shape, -1
        )  # episodes, agents, round, actions
        current_q = current_q.gather(
            -1, action.unsqueeze(-1)
        )  # episodes, agents, round, 1

        next_v = th.zeros_like(reward, device=self.device)

        # we skip the first observation and set the future value for the terminal
        # state to 0
        next_q_values = self.target_model(
            encoded, reset_rnn=True
        )  # episodes*agents, round, actions
        next_q_values = next_q_values.reshape(
            *action.shape, -1
        )  # episodes, agents, round, actions

        next_v[:, :, :-1] = next_q_values[:, :, 1:].max(-1)[0].detach()

        # Compute the expected Q values
        expected_q = (next_v * self.gamma) + reward

        # Compute Huber loss
        loss = th.nn.functional.smooth_l1_loss(current_q, expected_q.unsqueeze(-1))

        # Compute the loss for each agent and round
        loss_ur = th.nn.functional.smooth_l1_loss(
            current_q,
            expected_q.unsqueeze(-1),
            reduction="none",
        )
        loss_ur = loss_ur.mean(dim=0).mean(dim=0)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss_ur

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
