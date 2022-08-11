#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Parameters
from aimanager.utils.array_to_df import add_labels
from aimanager.utils.utils import make_dir
from aimanager.manager.manager import ArtificalManager
from aimanager.artificial_humans import AH_MODELS
from aimanager.manager.environment_v3 import ArtificialHumanEnv
from aimanager.manager.memory import Memory
from itertools import count
import os
import pandas as pd
import torch as th
artificial_humans = "../../data/training/ah_10/data/model.pt"
artificial_humans_model = "graph"
data_dir = "../../train_data/manager/v2/dev/"
manager_args = {
    "opt_args": {"lr": 0.01},
    "gamma": 1.0,
    "eps": 0.2,
    "target_update_freq": 50,
    "model_args": {
        "hidden_size": 5,
        "add_rnn": True,
        "add_edge_model": True,
        "add_global_model": False,
        "x_encoding": [
            # {"name": "contributions", "n_levels": 21, "encoding": "numeric"},
            # {"name": "prev_punishments", "n_levels": 31, "encoding": "numeric"},
        ],
        "u_encoding": [
            {"name": "round_number", "n_levels": 16, "encoding": "onehot"},
            # {"name": "prev_common_good", "norm": 128, "etype": "float"},
        ],
    },
}
replay_memory_args = {"n_batches": 1000}
n_update_steps = 250
eval_period = 5
env_args = {
    "n_agents": 4,
    "n_contributions": 21,
    "n_punishments": 31,
    "n_rounds": 16,
    "batch_size": 1000,
}
device = "cpu"
job_id = "dev"
labels = {}


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


metrics_dir = os.path.join(data_dir, 'metrics')
model_dir = os.path.join(data_dir, 'model')
make_dir(metrics_dir)
make_dir(model_dir)


# In[3]:


rec_keys = [
    'punishments', 'contributions', 'common_good', 'contributor_payoff', 'manager_payoff']


def run_batch(manager, env, replay_mem=None, on_policy=True, update_step=None):

    state = env.reset()
    metric_list = []
    for round_number in count():
        encoded = manager.encode_pure(state)

        batch_structure = env.get_batch_structure()

        obs = {**encoded, **batch_structure}

        # Get q values from controller
        q_values = manager.get_q(obs, first=round_number == 0)

        if on_policy:
            action = q_values.argmax(-1)
        else:
            # Sample a action
            action = manager.eps_greedy(q_values=q_values)

        state = env.punish(action)

        metrics = {k: state[k].to(th.float).mean().item() for k in rec_keys}

        # pass actions to environment and advance by one step
        state, reward, done = env.step()
        if replay_mem is not None:
            replay_mem.add(
                round_number=round_number, action=action, reward=reward,
                **encoded)

        metrics['next_reward'] = reward.mean().item()
        metrics['q_min'] = q_values.min().item()
        metrics['q_max'] = q_values.max().item()
        metrics['q_mean'] = q_values.mean().item()
        metrics['round_number'] = round_number
        metrics['sampling'] = 'greedy' if on_policy else 'eps-greedy'
        metrics['update_step'] = update_step
        metric_list.append(metrics)

        if done:
            break
    return metric_list


# In[4]:


device = th.device(device)
cpu = th.device('cpu')

artifical_humans = AH_MODELS[artificial_humans_model].load(artificial_humans).to(device)

env = ArtificialHumanEnv(
    artifical_humans=artifical_humans, device=device, **env_args)

manager = ArtificalManager(
    n_contributions=env.n_contributions, n_punishments=env.n_punishments,
    default_values=artifical_humans.default_values, device=device, **manager_args)

replay_mem = Memory(
    n_rounds=env.n_rounds, device=cpu, batch_size=env.batch_size, group_size=env.n_agents,
    **replay_memory_args)

metrics_list = []

for update_step in range(n_update_steps):
    replay_mem.start_batch(env.groups)

    # here we sample one batch of episodes and add them to the replay buffer
    off_policy_metrics = run_batch(manager, env, replay_mem,
                                   on_policy=False, update_step=update_step)

    replay_mem.finish_batch()

    # allow manager to update itself
    sample, groups = replay_mem.sample(device=device)
    graph = env.get_batch_structure()

    if sample is not None:
        loss = manager.update(update_step, **sample, **graph)

    if (update_step % eval_period) == 0:
        metrics_list.extend([{**m, 'loss': l.item()} for m, l in zip(off_policy_metrics, loss)])
        metrics_list.extend(
            run_batch(manager, env, replay_mem=None, on_policy=True, update_step=update_step))

model_file = os.path.join(model_dir, f'{job_id}.parquet')

manager.save(model_file)


# In[5]:


# lm = ArtificalManager.load(model_file, device=th.device('cpu'))


# In[6]:


id_vars = ['round_number', 'sampling', 'update_step']
value_vars = ['punishments', 'contributions', 'common_good', 'contributor_payoff',
              'manager_payoff', 'next_reward', 'q_min', 'q_max', 'q_mean', 'loss']

df = pd.DataFrame.from_records(metrics_list)

df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='metric')

df = add_labels(df, {**labels, 'job_id': job_id})

df.to_parquet(os.path.join(metrics_dir, f'{job_id}.parquet'))
