import collections
import numpy as np
import torch as th
from torch_geometric.data import Data, Batch


def merge_data(eps_data):
    no_merge = ['idx', 'group_idx', 'num_nodes', 'edge_index', 'ptr', 'batch', 'player_idx']
    merge = list(set(eps_data[0].keys) - set(no_merge))
    no_merge_values = {k: eps_data[0][k] for k in no_merge}
    merged_values = {k: th.concat([d[k]for d in eps_data], dim=1) for k in merge}
    return Data(**no_merge_values, **merged_values)


class GraphMemory():
    def __init__(self, device, n_batch, n_nodes, n_episode_steps, batch_size, sample_size=None):
        self.n_batch = n_batch
        self.n_episode_steps = n_episode_steps
        self.sample_size = sample_size
        self.batch_size = batch_size

        self.n_episodes = self.n_batch * self.batch_size

        self.episode_queue = collections.deque([], maxlen=self.n_episodes)
        self.reset_batch_memory()
        self.memory = {
            'action': th.empty((self.n_episodes, n_nodes, self.n_episode_steps), dtype=th.int64, device=device),
            'reward':  th.empty((self.n_episodes, n_nodes, self.n_episode_steps), dtype=th.float, device=device),
            'obs': [None]*self.n_episodes,
        }

    def reset_batch_memory(self):
        self.batch_memory = {
            'action': [],
            'reward': [],
            'obs': [],
        }

    def next_batch(self, update_step):
        start_row = (update_step * self.batch_size) % self.n_episodes
        end_row = start_row + self.batch_size

        # TODO: the merging here needs to be changed
        raise NotImplementedError('')

        self.memory['obs'][start_row:end_row] = merge_data(self.batch_memory['obs'])
        self.memory['action'][start_row:end_row] = th.stack(self.batch_memory['action'], dim=1)
        self.memory['reward'][start_row:end_row] = th.stack(self.batch_memory['reward'], dim=1)
        self.episode_queue.extendleft(range(start_row, end_row))
        self.reset_batch_memory()

    def add(self, action, reward, obs):
        self.batch_memory['action'].append(action)
        self.batch_memory['reward'].append(reward)
        self.batch_memory['obs'].append(obs)

    def sample(self, batch_size=None, horizon=None, **kwargs):
        if batch_size is None:
            batch_size = self.sample_size
        if horizon is None:
            horizon = self.n_episodes
        eff_horizon = min(len(self), horizon)
        if eff_horizon < batch_size:
            return None
        relative_episode = np.random.choice(eff_horizon, batch_size, replace=False)
        return self.get_relative(relative_episode, **kwargs)

    def last(self, batch_size, **kwargs):
        assert batch_size <= self.n_episodes
        relative_episodes = np.arange(batch_size)
        return self.get_relative(relative_episodes, **kwargs)

    def get_relative(self, relative_episode):
        hist_idx = [self.episode_queue[rp] for rp in relative_episode]
        action = self.memory['action'][hist_idx]
        reward = self.memory['reward'][hist_idx]
        # flatten first two dimensions
        action = action.reshape((-1, *action.shape[2:]))
        reward = reward.reshape((-1, *reward.shape[2:]))
        return {
            'action': action,
            'reward': reward,
            'obs': Batch.from_data_list([self.memory['obs'][idx] for idx in hist_idx]),
        }

    def __len__(self):
        return len(self.episode_queue)
