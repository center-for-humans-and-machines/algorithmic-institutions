import os
import collections
import numpy as np
import torch as th
from aimanager.utils.utils import make_dir


class Memory():
    def __init__(
            self, device, n_batches, n_rounds, batch_size, group_size, output_file=None):
        self.memory = None
        self.n_batches = n_batches
        self.size = n_batches * batch_size * group_size
        self.batch_size = batch_size
        self.n_rounds = n_rounds
        self.device = device
        self.output_file = output_file
        self.start_row = None
        self.rewind = 0
        self.group_que = collections.deque([], maxlen=self.n_batches*batch_size)

    @property
    def last_valid_row(self):
        lvr = self.start_row + self.batch_size
        return lvr if lvr >= 0 else None

    def init_store(self, state):
        self.memory = {
            k: th.zeros((self.size, self.n_rounds, *t.shape[2:]),
                        dtype=t.dtype, device=self.device)
            for k, t in state.items() if t is not None
        }

    def start_batch(self, groups):
        if self.start_row is None:
            self.start_row = 0
        else:
            self.start_row = (self.start_row + self.batch_size)
        self.end_row = self.start_row + sum(len(g) for g in groups)
        self.current_group = [[r+self.start_row for r in g] for g in groups]

    def finish_batch(self):
        self.group_que.extendleft(self.current_group)
        if self.end_row >= (self.size + 1):
            self.write()
            self.start_row = None
            self.rewind += 1

    def add(self, round_number, **state):
        if self.memory is None:
            self.init_store(state)

        for k, t in state.items():
            if t is not None:
                self.memory[k][self.start_row:self.end_row, round_number] = t[:, 0].to(self.device)

    def sample(self, **kwargs):
        assert self.batch_size is not None, 'No sample size defined.'
        if len(self) < self.batch_size:
            return None
        relative_episode = np.random.choice(len(self), self.batch_size, replace=False)
        return self.get_relative(relative_episode, **kwargs)

    def last(self, **kwargs):
        assert self.batch_size is not None, 'No sample size defined.'
        if len(self) < self.batch_size:
            return None
        relative_episodes = np.arange(self.batch_size)
        return self.get_relative(relative_episodes, **kwargs)

    def get_relative(self, relative_episode, keys=None, device=None):
        if keys is None:
            keys = self.memory.keys()
        hist_idx = th.tensor(
            [row for rp in relative_episode for row in self.group_que[rp]],
            dtype=th.int64, device=self.device)
        sample = {k: v[hist_idx] for k, v in self.memory.items() if k in keys}
        if device is not None:
            sample = {k: v.to(device) for k, v in sample.items()}
        return sample

    def __len__(self):
        return len(self.group_que)

    def write(self):
        if self.output_file and self.last_valid_row is not None:
            dirname = os.path.dirname(self.output_file)
            make_dir(dirname)
            th.save(
                {
                    k: t[:self.last_valid_row] for k, t in self.memory.items()
                },
                f'{self.output_file}.{self.rewind}.pt'
            )

    def __del__(self):
        if self.memory is not None:
            self.write()
