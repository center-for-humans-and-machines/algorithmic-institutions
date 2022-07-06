from aimanager.generic.graph_encode import create_fully_connected

import torch as th


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
            self, *, artifical_humans, n_agents, n_contributions, n_punishments, episode_steps, device):
        """
        Args:
            asdasd
        """
        self.episode = 0
        self.episode_steps = episode_steps
        self.device = device
        self.n_contributions = n_contributions
        self.n_punishments = n_punishments
        self.artifical_humans = artifical_humans
        self.n_agents = n_agents
        self.reward_baseline = (artifical_humans.default_values['common_good'] / 4)
        self.reward_scale = 1 / 20*1.6
        self.edge_index = create_fully_connected(n_agents)
        self.reset_state()


    def reset_state(self):
        state = {
            'punishments': th.zeros(self.n_agents, dtype=th.int64),
            'contributions': th.zeros(self.n_agents, dtype=th.int64),
            'round_number': th.zeros(self.n_agents, dtype=th.int64),
            'valid': th.zeros(self.n_agents, dtype=th.bool),
            'common_good': th.zeros(self.n_agents, dtype=th.float),
            'payoffs': th.zeros(self.n_agents, dtype=th.float),
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

    @staticmethod
    def calc_common_good(contributions, punishments):
        return (contributions * 1.6 - punishments).sum() * th.ones_like(punishments, dtype=th.float)

    @staticmethod
    def calc_payout(contributions, punishments, commond_good, valid):
        payout = 20 - contributions - punishments + commond_good / valid.sum()
        payout[~valid] = 0
        return payout

    def calc_contributions(self):
        state = {k: v.unsqueeze(0).unsqueeze(-1) for k, v in self.state.items()}
        encoded = self.artifical_humans.encode(state, mask=False, y_encode=False, edge_index=self.edge_index)
        contributions = self.artifical_humans.predict_one(encoded[0], reset_rnn=self.round_number[0] == 0)[0]
        self.contributions = contributions.squeeze(-1)
        self.valid = th.ones_like(self.valid)

    def init_episode(self):
        self.episode += 1
        self.round_number = th.zeros_like(self.round_number)


        self.reset_state()
        self.calc_contributions()
        return self.state

    def punish(self, punishments):

        assert punishments.max() < self.n_punishments
        assert punishments.dtype == th.int64

        self.punishments = punishments
        self.common_good = self.calc_common_good(self.contributions, self.punishments)
        # self.payoffs = self.calc_payout(self.contributions, self.punishments, self.common_good, self.valid)
        return self.state

    def step(self):
        if (self.round_number[0] == (self.episode_steps - 1)):
            reward = - self.prev_punishments.to(th.float)
            done = True
        elif self.round_number[0] >= self.episode_steps:
            raise ValueError('Environment is done already.')
        else:
            for k in self.state:
                if k[:4] == 'prev':
                    self.state[k] = self.state[k[5:]]
            self.calc_contributions()
            reward = self.contributions.to(th.float) * 1.6 - self.prev_punishments.to(th.float) 
            done = False
        self.payoffs = reward
        reward = (reward - self.reward_baseline) * self.reward_scale
        self.round_number += 1
        return self.state, reward, done


        