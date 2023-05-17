import os
import collections
import numpy as np
import torch as th
from aimanager.utils.utils import make_dir


class Memory:
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
        self.memory = {
            **{
                k: th.zeros(
                    (
                        self.n_episodes,
                        *t.shape[0:2],
                        self.n_episode_steps,
                        *t.shape[3:],
                    ),
                    dtype=t.dtype,
                    device=self.device,
                )
                for k, t in state.items()
            },
            "episode": th.empty(
                (self.n_episodes, self.n_episode_steps),
                dtype=th.int64,
                device=self.device,
            ),
            "episode_steps": th.empty(
                (self.n_episodes, self.n_episode_steps),
                dtype=th.int64,
                device=self.device,
            ),
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
        self.memory["episode"][self.current_row, episode_step] = self.episode
        self.memory["episode_steps"][self.current_row, episode_step] = episode_step
        for k, t in state.items():
            self.memory[k][self.current_row, :, :, [episode_step]] = t.to(self.device)

    def get_random(self, n_episodes=1, device=None):
        episode_idx = np.random.randint(low=0, high=len(self), size=n_episodes)
        data = {k: v[episode_idx] for k, v in self.memory.items()}
        if device is not None:
            data = {k: v.to(device) for k, v in data.items()}
        data = {k: v.reshape(-1, *v.shape[2:]) for k, v in data.items()}
        return data

    def __len__(self):
        return len(self.episode_queue)

    def write(self):
        if self.output_file:
            dirname = os.path.dirname(self.output_file)
            make_dir(dirname)
            th.save(
                {k: t[: self.current_row] for k, t in self.memory.items()},
                f"{self.output_file}.{self.episode + 1}.pt",
            )

    def __del__(self):
        if hasattr(self, "memory") and (self.current_row != 0):
            self.write()
