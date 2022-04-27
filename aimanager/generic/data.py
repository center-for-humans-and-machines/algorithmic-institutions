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
    round_number = th.zeros((n_episodes, n_steps, n_agents), dtype=th.int64)
    valid = th.zeros((n_episodes, n_steps, n_agents), dtype=th.bool)
    # public_good = th.zeros((n_episodes, n_steps), dtype=th.bool)
    common_good = th.zeros((n_episodes, n_steps, n_agents), dtype=th.float)

    default_values = {
        'punishments': np.rint(df['punishment'].mean()),
        'contribution': np.rint(df['contribution'].mean()),
        'player_no_input': True,
        'common_good': df['common_good'].mean(),
        'round_number': 0,
    }

    for idx, row in df.iterrows():
        episode, step, agent = row[['episode_id', 'round_number', 'player_id']]
        punishments[episode, step, agent] = row['punishment']
        
        # we impute here with the most common contribution
        contributions[episode, step, agent] = row['contribution'] if row['player_no_input'] == 0 else 20
        valid[episode, step, agent] = 1 - row['player_no_input']
        common_good[episode, step, agent] = row['common_good']
        round_number[episode, step, agent] = row['round_number']

    
    # for idx, row in df.groupby(['episode_id', 'round_number']).head(1).iterrows():
    #     episode, step = row[['episode_id', 'round_number']]
    #     public_good[episode, step]

    data = {
        'punishments': punishments,
        'contributions': contributions,
        'valid': valid,
        'common_good': common_good,
        'round_number': round_number
    }

    data = {**data, **{f'prev_{k}': shift(t) for k, t in data.items()}}
    return data

def create_syn_data(n_contribution, n_punishment, n_agents = 4, n_steps = 16):
    n_episodes = n_contribution * n_punishment
    n_agents = 4
    punishments = th.zeros((n_episodes, n_steps, n_agents), dtype=th.int64)
    contributions = th.zeros((n_episodes, n_steps, n_agents), dtype=th.int64)
    round_number = th.zeros((n_episodes, n_steps, n_agents), dtype=th.int64)
    valid = th.zeros((n_episodes, n_steps, n_agents), dtype=th.bool)
    # public_good = th.zeros((n_episodes, n_steps), dtype=th.bool)
    common_good = th.zeros((n_episodes, n_steps, n_agents), dtype=th.float)

    agent = 0
    episode = 0
    for contribution in range(n_contribution):
        for punishment in range(n_punishment):
            for step in range(n_steps):
                punishments[episode, step, agent] = punishment
                contributions[episode, step, agent] = contribution
                round_number[episode, step, agent] = step
                common_good[episode, step, agent] = (contribution - punishment) * 4
                valid[episode, step, agent] = 1
            episode += 1

    data = {
        'prev_valid': valid,
        'prev_punishments': punishments,
        'prev_contributions': contributions,
        'prev_common_good': common_good,
        'round_number': round_number
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