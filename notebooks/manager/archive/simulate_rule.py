#!/usr/bin/env python
# coding: utf-8

# In[90]:


# Parameters
from aimanager.utils.array_to_df import using_multiindex, add_labels
from aimanager.artificial_humans import AH_MODELS
from aimanager.manager.environment import ArtificialHumanEnv
from aimanager.manager.archive.memory import Memory
from aimanager.utils.utils import make_dir
from itertools import count
import numpy as np
import torch as th
import os
artificial_humans = "../../data/artificial_humans/ah_1_1/data/model.pt"
artificial_humans_model = "graph"
output_path = "../../data/manager/simulate_rule/v1_comp_fixed/"
n_episode_steps = 16
s = 0
b = 0
c = 0
n_episodes = 1000
agents = None
round_numbers = None


# In[91]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


output_path = os.path.join(output_path, 'data')


# In[92]:


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


# In[93]:


rec_keys = [
    'punishments', 'contributions', 'common_good', 'contributor_payoff', 'manager_payoff']


def run_batch(manager, env):
    state = env.reset()
    metric_list = []
    for round_number in count():
        encoded = manager.encode_pure(state)

        batch_structure = env.get_batch_structure()

        obs = {**encoded, **batch_structure}

        # Get q values from controller
        action = manager.punishments(obs, first=round_number == 0)

        state = env.punish(action)

        metrics = {k: state[k].to(th.float).mean().item() for k in rec_keys}

        # pass actions to environment and advance by one step
        state, reward, done = env.step()

        metrics['next_reward'] = reward.mean().item()
        metrics['round_number'] = round_number
        metric_list.append(metrics)

        if done:
            break
    return metric_list


# In[94]:


device = th.device('cpu')
rec_device = th.device('cpu')
artifical_humans = AH_MODELS[artificial_humans_model].load(artificial_humans).to(device)


s_values = list(np.arange(0, 4.1, 0.2))

env = ArtificialHumanEnv(
    artifical_humans=artifical_humans, n_agents=4, n_contributions=21, n_punishments=31, episode_steps=n_episode_steps, device=device)

recorder = Memory(n_episodes=n_episodes*len(s_values),
                  n_episode_steps=n_episode_steps, output_file=None, device=device)


for s in s_values:
    for episode in range(n_episodes):
        manager = RuleManager(s=s, b=b, c=c, agents=agents, round_numbers=round_numbers)
        state = env.init_episode()
        for step in count():
            action = manager.get_punishment(**state)
            state = env.punish(action)

            s_th = th.full_like(action, fill_value=s, dtype=th.float)

            recorder.add(**state, episode_step=step, s=s_th)
            state, reward, done = env.step()
            if done:
                break
        recorder.next_episode(episode)


# In[95]:


punishments = using_multiindex(recorder.memory['punishments'].numpy(), columns=[
                               'idx', 'round_number', 'agent'], value_name='punishments')
common_good = using_multiindex(recorder.memory['common_good'].numpy(), columns=[
                               'idx', 'round_number', 'agent'], value_name='common_good')
contributions = using_multiindex(recorder.memory['contributions'].numpy(), columns=[
                                 'idx', 'round_number', 'agent'], value_name='contributions')
payoffs = using_multiindex(recorder.memory['payoffs'].numpy(), columns=[
                           'idx', 'round_number', 'agent'], value_name='payoffs')
s = using_multiindex(recorder.memory['s'].numpy(), columns=[
                     'idx', 'round_number', 'agent'], value_name='s')

df = punishments.merge(common_good).merge(contributions).merge(payoffs).merge(s)

df = df.drop(columns=['idx'])

df = df.groupby(['round_number', 'agent', 's']).mean().reset_index()
# df = add_labels(df, labels=labels)


make_dir(output_path)
df.to_csv(os.path.join(output_path, 'trace.csv'))


# In[96]:


df
