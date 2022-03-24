import torch as th
import numpy as np
import random

def shift(tensor):
    tensor = th.roll(tensor, 1, 1)
    tensor[:,0] = 0
    return tensor


def create_torch_data(df):
    n_episodes = df['episode_id'].max() + 1
    n_steps = df['round_number'].max() + 1
    n_agents = df['player_id'].max() + 1

    punishments = th.zeros((n_episodes, n_steps, n_agents), dtype=th.int64)
    contributions = th.zeros((n_episodes, n_steps, n_agents), dtype=th.int64)
    valid = th.zeros((n_episodes, n_steps, n_agents), dtype=th.bool)

    for idx, row in df.iterrows():
        episode, step, agent = row[['episode_id', 'round_number', 'player_id']]
        punishments[episode, step, agent] = row['punishment']
        contributions[episode, step, agent] = row['contribution']
        valid[episode, step, agent] = 1 - row['player_no_input']

    data = {
        'punishments': punishments,
        'contributions': contributions,
        'valid': valid
    }

    data = {**data, **{f'prev_{k}': shift(t) for k, t in data.items()}}
    return data

def create_syn_data(n_contribution, n_punishment):
    punishments = th.arange(0, n_punishment)
    punishments = punishments[np.newaxis,:,np.newaxis]
    punishments = punishments.tile((n_contribution,1, 1))

    contributions = th.arange(0, n_contribution)
    contributions = contributions[:,np.newaxis,np.newaxis]
    contributions = contributions.tile((1, n_punishment, 1))

    valid = th.ones((n_contribution, n_punishment, 1), dtype=th.int16)

    data = {
        'prev_valid': valid,
        'prev_punishments': punishments,
        'prev_contributions': contributions,
    }
    return data

def get_cross_validations(data, n_splits):
    episode_ids = list(range(data['contributions'].shape[0]))
    random.shuffle(episode_ids)
    groups = [episode_ids[i::n_splits] for i in range(n_splits)]

    for i in range(n_splits):
        test_groups = groups[i]
        train_groups = [gg for g in groups for gg in g]
        test_data = {
            k: t[test_groups]
            for k, t in data.items()
        }
        train_data = {
            k: t[train_groups]
            for k, t in data.items()
        }
        yield train_data, test_data