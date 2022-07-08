import torch as th
import numpy as np
import pandas as pd
import random


def shift(tensor, default):
    tensor = th.roll(tensor, 1, 2)
    tensor[:,:,0] = default
    return tensor


def create_torch_data(df, default_values=None):
    df['episode_idx'] = df['episode_id'].rank(method='dense').astype(int)

    n_episodes = df['episode_idx'].max() + 1
    n_steps = df['round_number'].max() + 1
    n_agents = df['player_id'].max() + 1

    df['player_input'] = df['player_no_input'] == 0
    df['manager_input'] = df['manager_no_input'] == 0
    df['round_player_input'] = df.groupby(['episode_id', 'round_number'])['player_input'].transform('sum')
    df['common_good'] = (df['common_good'] / df['round_player_input']).fillna(0)

    if not default_values:
        default_values = {
            'punishments': np.rint(df.loc[df['player_input'], 'punishment'].mean()),
            'contributions': np.rint(df.loc[df['player_input'], 'contribution'].mean()),
            'valid': False,
            'manager_valid': False,
            'common_good': df.loc[df['player_input'], 'common_good'].mean(),
        }

    punishments = th.full((n_episodes, n_agents, n_steps), fill_value=default_values['punishments'], dtype=th.int64)
    contributions = th.full((n_episodes, n_agents, n_steps), fill_value=default_values['contributions'], dtype=th.int64)
    round_number = th.zeros((n_episodes, n_agents, n_steps), dtype=th.int64)
    is_first = th.zeros((n_episodes, n_agents, n_steps), dtype=th.bool)
    valid = th.full((n_episodes, n_agents, n_steps), fill_value=default_values['valid'], dtype=th.bool)
    manager_valid = th.full((n_episodes, n_agents, n_steps), fill_value=default_values['manager_valid'], dtype=th.bool)
    common_good = th.full((n_episodes, n_agents, n_steps), fill_value=default_values['common_good'], dtype=th.float)

    for idx, row in df.iterrows():
        eps, step, agent = row[['episode_idx', 'round_number', 'player_id']]
        valid[eps, agent, step] = row['player_input']
        manager_valid[eps, agent, step] = row['manager_input']
        is_first[eps, agent, step] = step == 0
        round_number[eps, agent, step] = step
        common_good[eps, agent, step] = row['common_good']
        if row['player_input']:
            punishments[eps, agent, step] = row['punishment']
            contributions[eps, agent, step] = row['contribution']

    data = {
        'punishments': punishments,
        'contributions': contributions,
        'valid': valid,
        'common_good': common_good,
        'round_number': round_number,
        'is_first': is_first,
        'manager_valid': manager_valid
    }


    data = {**data, **{f'prev_{k}': shift(t, default_values[k]) for k, t in data.items() if k in default_values}}
    return data, default_values


def create_syn_data(n_contribution, n_punishment, default_values, n_agents = 4, n_steps = 16):
    agent = 0
    episode = 0
    recs = []
    for contribution in range(n_contribution):
        for punishment in range(n_punishment):
            for step in range(n_steps):
                for agent in range(n_agents):
                    recs.append({
                        'episode_id': episode,
                        'round_number': step,
                        'player_id': agent,
                        'punishment': punishment,
                        'contribution': contribution,
                        'common_good': (contribution - punishment) * 4,
                        'player_no_input': False,
                        'manager_no_input': False,
                        'is_first': step == 0
                    })
            episode += 1
    df = pd.DataFrame.from_records(recs)
    return create_torch_data(df, default_values=default_values)[0]


def get_cross_validations(data, n_splits, fraction_training=1.0):
    episode_idx = list(range(data['contributions'].shape[0]))
    random.shuffle(episode_idx)
    if n_splits is None:
        train_data = {
            k: t[episode_idx]
            for k, t in data.items()
        }
        yield train_data, None
    else:
        groups = [episode_idx[i::n_splits] for i in range(n_splits)]
        for i in range(n_splits):
            test_idx = groups[i]
            train_idx = [idx for idx in episode_idx if idx not in test_idx]

            # get a random fraction of the training groups
            random.shuffle(train_idx)
            train_idx = train_idx[:int(fraction_training*len(train_idx))]
            
            test_data = {
                k: t[test_idx]
                for k, t in data.items()
            }
            train_data = {
                k: t[train_idx]
                for k, t in data.items()
            }
            yield train_data, test_data