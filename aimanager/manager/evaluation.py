import torch as th
from itertools import count
from aimanager.manager.memory import Memory


class ManagerEvaluator():
    def __init__(self, n_rounds, batch_size, output_file=None, test_period=1):
        self.device = th.device('cpu')
        self.test_period = test_period
        self.t_episode = 0
        self.recorder = Memory(
            n_episodes=n_rounds // test_period * batch_size,
            n_n_rounds=n_rounds, output_file=output_file, device=self.device)

    def eval_update(self, manager, env, update_step):
        if update_step % self.test_period == 0:
            update_step = th.tensor(update_step, device=self.device)
            self.eval(manager, env, update_step=update_step)

    def eval(self, manager, env, **info):
        state = env.init_episode()
        for step in count():
            punishment = manager.get_action(state, env.edge_index, first=step == 0)
            state = env.punish(punishment)
            self.recorder.add(**state, episode_step=step, **info)

            # pass actions to environment and advance by one step
            state, reward, done = env.step()
            if done:
                break

        self.recorder.next_batch()
        # self.t_episode += 1
