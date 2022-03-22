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
        'episode_step': ['agent'],
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
        self.reset_state()


    def reset_state(self):
        self.state = {
            'punishments': th.zeros(self.n_agents, dtype=th.int64),
            'contributions': th.zeros(self.n_agents, dtype=th.int64),
            'valid': th.zeros(self.n_agents, dtype=th.bool),
            'prev_punishments': th.zeros(self.n_agents, dtype=th.int64),
            'prev_contributions': th.zeros(self.n_agents, dtype=th.int64),
            'prev_valid': th.zeros(self.n_agents, dtype=th.bool),
            'payoffs': th.zeros(self.n_agents, dtype=th.float),
            'common_good': th.tensor(self.n_agents, dtype=th.float),
            'episode_step': th.tensor(0, dtype=th.int64),
            'player_id': th.arange(4, dtype=th.int64)
        }


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
        return (contributions * 1.6 - punishments)

    @staticmethod
    def calc_payout(contributions, punishments, commond_good):
        # TODO: check how to handle missing values
        return 20 - contributions - punishments + commond_good

    def calc_contributions(self):
        self.contributions = self.artifical_humans.act(**self.state)
        self.valid = th.ones_like(self.valid)

    def init_episode(self):
        self.episode += 1
        self.episode_step = 0
        self.reset_state()
        self.calc_contributions()
        return self.state

    def punish(self, punishments):

        assert punishments.max() < self.n_punishments
        assert punishments.dtype == th.int64

        self.punishments = punishments
        self.common_good = self.calc_common_good(self.contributions, self.punishments)
        self.payoffs = self.calc_payout(self.contributions, self.punishments, self.common_good)
        return self.state

    def step(self):

        self.prev_contributions = self.contributions
        self.prev_punishments = self.punishments
        self.prev_valid = self.valid

        self.calc_contributions()
        reward = self.contributions * 1.6 - self.prev_punishments

        self.episode_step += 1
        if (self.episode_step == (self.episode_steps - 1)):
            done = True
        elif self.episode_step >= self.episode_steps:
            raise ValueError('Environment is done already.')
        else:
            done = False
        return self.state, reward, done

