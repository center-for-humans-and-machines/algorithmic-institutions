import torch as th
from itertools import count
from aimanager.manager.memory import Memory


class ManagerEvaluator():
    def __init__(self, n_test_episodes, n_episode_steps, output_file=None, n_episodes=1, eval_freq=1):
        self.device = th.device('cpu')
        self.n_test_episodes = n_test_episodes
        self.eval_freq = eval_freq
        self.t_episode = 0
        self.recorder = Memory(
            n_episodes=n_episodes // eval_freq * n_test_episodes, n_episode_steps=n_episode_steps, output_file=output_file, device=self.device)

    def eval_episode(self, manager, env, episode):
        if episode % self.eval_freq == 0:
            episode_ = th.tensor(episode, device=self.device)
            self.eval(manager, env, episode=episode_)

    def eval(self, manager, env, **info):
        for i in range(self.n_test_episodes):
            state = env.init_episode()
            for step in count():
                punishment = manager.get_action(state, env.edge_index, first=step == 0)
                state = env.punish(punishment)
                self.recorder.add(**state, episode_step=step, **info)

                # pass actions to environment and advance by one step
                state, reward, done = env.step()
                if done:
                    break

            self.recorder.next_episode(self.t_episode)
            self.t_episode += 1
