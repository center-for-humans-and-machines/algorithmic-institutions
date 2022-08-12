from aimanager.generic.graph_encode import create_fully_connected

import torch as th
from torch_scatter import scatter_sum


class ArtificialHumanEnv():
    """
    Environment that runs the virtual humans and calculuates the value of the common good.

    Indices:
        t: agent types [0..1]
    """
    state_dimensions = {
        'punishments': ['agent'],
        'contributions': ['agent'],
        'payoffs': ['agent'],
        'valid': ['agent'],
        'common_good': ['agent'],
        'round_number': ['agent'],
        'player_id': ['agent'],
    }

    def __init__(
            self, *, artifical_humans, batch_size, n_agents, n_contributions, n_punishments, n_rounds, device):
        """
        Args:
            asdasd
        """
        self.batch_size = batch_size
        self.n_rounds = n_rounds
        self.device = device
        self.n_contributions = n_contributions
        self.n_punishments = n_punishments
        self.artifical_humans = artifical_humans
        self.n_agents = n_agents
        self.edge_index = create_fully_connected(n_agents)
        self.batch_edge_index = th.tensor(
            [[a+(i*self.n_agents), b+(i*self.n_agents)]
             for i in range(self.batch_size)
             for a in range(self.n_agents)
             for b in range(self.n_agents)
             ], device=self.device, dtype=th.int64).T
        self.batch = th.tensor(
            [i
             for i in range(self.batch_size)
             for a in range(self.n_agents)
             ], device=self.device, dtype=th.int64)
        self.groups = [[(i*self.n_agents + a) for a in range(self.n_agents)]
                       for i in range(self.batch_size)
                       ]

        self.reset_state()

    def reset_state(self):
        state = {
            'punishments': th.zeros((self.batch_size * self.n_agents, 1), dtype=th.int64, device=self.device),
            'contributions': th.zeros((self.batch_size * self.n_agents, 1), dtype=th.int64, device=self.device),
            'round_number': th.zeros((self.batch_size * self.n_agents, 1), dtype=th.int64, device=self.device),
            'valid': th.zeros((self.batch_size * self.n_agents, 1), dtype=th.bool, device=self.device),
            'manager_valid': th.zeros((self.batch_size * self.n_agents, 1), dtype=th.bool, device=self.device),
            'common_good': th.zeros((self.batch_size * self.n_agents, 1), dtype=th.float, device=self.device),
            'contributor_payoff': th.zeros((self.batch_size * self.n_agents, 1), dtype=th.float, device=self.device),
            'manager_payoff': th.zeros((self.batch_size * self.n_agents, 1), dtype=th.float, device=self.device),
            'reward': th.zeros((self.batch_size * self.n_agents, 1), dtype=th.float, device=self.device),
            'group': th.tensor([i for i, g in enumerate(self.groups) for a in g], dtype=th.int64, device=self.device),
            'agent': th.tensor([a for i, g in enumerate(self.groups) for a in g], dtype=th.int64, device=self.device),
        }
        default_values = self.artifical_humans.default_values

        prev_state = {
            f'prev_{k}': th.full_like(state[k], fill_value=default_values[k])
            for k, t in state.items() if k in default_values
        }
        self.state = {**prev_state, **state}

    def __getattr__(self, name):
        if 'state' in self.__dict__:
            state = self.__dict__['state']
            return state[name]

    def __setattr__(self, name, value):
        if 'state' in self.__dict__:
            if name in self.__dict__['state']:
                self.state[name] = value
            else:
                object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def update_common_good(self):
        masked_contribution = th.where(self.valid, self.contributions, 0)
        masked_punishments = th.where(self.valid, self.punishments, 0)
        sum_contribution = scatter_sum(
            masked_contribution, self.batch, dim=0, dim_size=self.batch_size)
        sum_punishments = scatter_sum(
            masked_punishments, self.batch, dim=0, dim_size=self.batch_size)
        sum_valid = scatter_sum(
            self.valid.to(th.float), self.batch, dim=0, dim_size=self.batch_size)
        common_good = (sum_contribution * 1.6 - sum_punishments) / sum_valid
        self.common_good = common_good[self.batch]

    def update_payoff(self):
        contributor_payoff = 20 - self.contributions - self.punishments + self.common_good
        self.contributor_payoff = th.where(self.valid, contributor_payoff, 0)
        self.manager_payoff = self.common_good / 4

    def update_reward(self):
        if self.done:
            self.reward = - self.prev_punishments.to(th.float) / 32
        else:
            self.reward = (self.contributions * 1.6 - self.prev_punishments) / 32

    def update_contributions(self):
        state = {**self.state, **self.get_batch_structure()}
        encoded = self.artifical_humans.encode_pure(state, mask=None, y_encode=False)
        contributions = self.artifical_humans.predict_pure(
            encoded, reset_rnn=self.round_number[0][0] == 0)[0]
        self.contributions = contributions
        self.valid = th.ones_like(self.valid)

    def reset(self):
        self.round_number = th.zeros_like(self.round_number)
        self.done = False
        self.reset_state()
        self.update_contributions()
        return self.state

    def punish(self, punishments):
        assert punishments.max() < self.n_punishments
        assert punishments.dtype == th.int64
        self.punishments = punishments
        self.update_common_good()
        # self.update_payoff()
        return self.state

    def step(self):
        self.round_number += 1
        if self.done:
            raise ValueError('Environment is done already.')
        if (self.round_number[0, 0] == (self.n_rounds)):
            self.done = True
        else:
            for k in self.state:
                if k[:4] == 'prev':
                    self.state[k] = self.state[k[5:]]
            self.update_contributions()
        self.update_reward()
        return self.state, self.reward, self.done

    def get_batch_structure(self):
        return {
            'edge_index': self.batch_edge_index,
            'batch': self.batch
        }
