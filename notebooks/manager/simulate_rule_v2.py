#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Parameters
from aimanager.artificial_humans import AH_MODELS
from aimanager.manager.environment_v3 import ArtificialHumanEnv
from aimanager.utils.utils import make_dir
from itertools import count
import pandas as pd
import numpy as np
import torch as th
import os
artificial_humans = {
    "complex": "../../data/artificial_humans/ah_1_1/data/model.pt",
    "simple": "../../data/artificial_humans/ah_1_1_simple/data/model.pt",
}
artificial_humans_model = "graph"
output_path = "../../data/manager/simulate_rule/v2/dev"
n_episode_steps = 16
manager_args = {"s": 0, "b": 0, "c": 0}
n_episodes = 2
agents = None
round_numbers = None


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


output_path = os.path.join(output_path, 'data')

make_dir(output_path)


# In[3]:


class RuleManager:
    def __init__(self, s, b, c, agents=None, round_numbers=None):
        self.s = s
        self.b = b
        self.c = c
        self.agents = agents
        self.round_numbers = round_numbers

    def get_punishment(self, contributions, round_number,  **_):
        punishments = th.zeros_like(contributions)
        if (self.round_numbers is None) or (round_number[0] in self.round_numbers):
            punishments = (20-contributions) * self.s + \
                (20 != contributions).to(th.float) * self.c - self.b
            punishments = punishments.round().to(th.int64)
            punishments = th.minimum(th.maximum(punishments, th.zeros_like(
                punishments)), th.full_like(punishments, 30))
        if self.agents is not None:
            punishments_ = th.zeros_like(contributions)
            punishments_[self.agents] = punishments[self.agents]
            punishments = punishments_

        return punishments


# In[4]:


device = th.device('cpu')

rec_keys = ['punishments', 'contributions', 'common_good', 'contributor_payoff', 'manager_payoff']
metric_list = []

for ah_name, ah in artificial_humans.items():
    ah = AH_MODELS[artificial_humans_model].load(ah).to(device)
    env = ArtificialHumanEnv(
        artifical_humans=ah, n_agents=4, n_contributions=21, n_punishments=31, batch_size=n_episodes, n_rounds=16, device=device)
    for s in np.arange(0, 5.1, 0.2):
        args = {**manager_args, 's': s}
        manager = RuleManager(agents=agents, round_numbers=round_numbers, **args)
        state = env.reset()
        for round_number in count():
            action = manager.get_punishment(**state)
            state = env.punish(action)

            metrics = {
                k: state[k].to(th.float).mean().item() for k in rec_keys}

            metrics = {**metrics, **args, 'artificial_humans': ah_name}

            # pass actions to environment and advance by one step
            state, reward, done = env.step()

            metrics['next_reward'] = reward.mean().item()
            metrics['round_number'] = round_number
            metric_list.append(metrics)
            # break
            if done:
                break


# In[5]:


state['prev_punishments'].shape


# In[6]:


id_vars = ['round_number', 's', 'c', 'b', 'artificial_humans']

df = pd.DataFrame.from_records(metric_list)

value_vars = list(set(df.columns) - set(id_vars))
df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='metric')

df.to_parquet(os.path.join(output_path, f'metrics.parquet'))
