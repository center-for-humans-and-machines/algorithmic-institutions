import os
import collections
import numpy as np
import torch as th
from aimanager.utils.utils import make_dir


class Memory():
    def __init__(self, device, n_episodes, n_episode_steps, output_file=None):
        self.memory = None
        self.n_episodes = n_episodes
        self.n_episode_steps = n_episode_steps
        self.device = device
        self.output_file = output_file
        self.current_row = 0
        self.episode = 0
        self.episode_queue = collections.deque([], maxlen=self.n_episodes)

    def init_store(self, state):
        self.memory = {**{
            k: th.zeros((self.n_episodes, t.shape[0], self.n_episode_steps, *t.shape[2:]),
                        dtype=t.dtype, device=self.device)
            for k, t in state.items()
        },
            'episode': th.empty((self.n_episodes, self.n_episode_steps), dtype=th.int64, device=self.device),
            'episode_steps': th.empty((self.n_episodes, self.n_episode_steps), dtype=th.int64, device=self.device)
        }

    def next_episode(self, episode):
        self.episode = episode
        if self.current_row == (self.n_episodes - 1):
            self.write()
        self.current_row = (self.current_row + 1) % self.n_episodes
        self.episode_queue.appendleft(self.current_row)

    def add(self, episode_step, **state):
        if self.memory is None:
            self.init_store(state)
        self.memory['episode'][self.current_row, episode_step] = self.episode
        self.memory['episode_steps'][self.current_row, episode_step] = episode_step
        for k, t in state.items():
            self.memory[k][self.current_row, :, [episode_step]] = t

    def sample(self, batch_size, horizon=None, **kwargs):
        if horizon is None:
            horizon = len(self)
        eff_horizon = min(len(self), horizon)
        if eff_horizon < batch_size:
            return None
        relative_episode = np.random.choice(eff_horizon, batch_size, replace=False)
        return self.get_relative(relative_episode, **kwargs)

    def get_random(self, device=None):
        episode_idx = np.random.randint(low=0, high=len(self))
        data = {k: v[episode_idx] for k, v in self.memory.items()}
        if device is not None:
            data = {k: v.to(device) for k, v in data.items()}
        return data

    def last(self, batch_size, **kwargs):
        assert batch_size <= self.n_episodes
        relative_episodes = np.arange(batch_size)
        return self.get_relative(relative_episodes, **kwargs)

    def get_relative(self, relative_episode, keys=None):
        if keys is None:
            keys = self.memory.keys()
        hist_idx = th.tensor(
            [self.episode_queue[rp] for rp in relative_episode], dtype=th.int64, device=self.device)
        return {k: v[hist_idx] for k, v in self.memory.items() if k in keys}

    def rec(self, state, episode, episode_steps):
        if self.memory is None:
            self.init_store(state)
        self.add_state(state, episode, episode_steps)

    def __len__(self):
        return len(self.episode_queue)

    def write(self):
        if self.output_file:
            dirname = os.path.dirname(self.output_file)
            make_dir(dirname)
            th.save(
                {
                    k: t[:self.current_row] for k, t in self.memory.items()
                },
                f'{self.output_file}.{self.episode + 1}.pt'
            )

    def __del__(self):
        if hasattr(self, 'memory') and (self.current_row != 0):
            self.write()
