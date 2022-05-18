import torch as th
import numpy as np
import pandas as pd
import random


# TODO: right shift and reasonable imputation
def shift(tensor):
    tensor = th.roll(tensor, 1, 1)
    tensor[:,:,0] = 0
    return tensor


def create_torch_data(df):
    n_episodes = df['episode_id'].max() + 1
    n_steps = df['round_number'].max() + 1
    n_agents = df['player_id'].max() + 1

    punishments = th.zeros((n_episodes, n_agents, n_steps), dtype=th.int64)
    contributions = th.zeros((n_episodes, n_agents, n_steps), dtype=th.int64)
    round_number = th.zeros((n_episodes, n_agents, n_steps), dtype=th.int64)
    valid = th.zeros((n_episodes, n_agents, n_steps), dtype=th.bool)
    common_good = th.zeros((n_episodes, n_agents, n_steps), dtype=th.float)

    for idx, row in df.iterrows():
        eps, step, agent = row[['episode_id', 'round_number', 'player_id']]
        punishments[eps, agent, step] = row['punishment']
        
        # we impute here with the most common contribution
        contributions[eps, agent, step] = row['contribution'] if row['player_no_input'] == 0 else 20
        valid[eps, agent, step] = 1 - row['player_no_input']
        common_good[eps, agent, step] = row['common_good']
        round_number[eps, agent, step] = step

    data = {
        'punishments': punishments,
        'contributions': contributions,
        'valid': valid,
        'common_good': common_good,
        'round_number': round_number,
    }

    data = {**data, **{f'prev_{k}': shift(t) for k, t in data.items()}}
    return data


def create_syn_data(n_contribution, n_punishment, n_agents = 4, n_steps = 16):
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
                        'player_no_input': 0,
                    })
            episode += 1
    df = pd.DataFrame.from_records(recs)
    return create_torch_data(df)


def get_cross_validations(data, n_splits, fraction_training=1.0):
    episode_ids = list(range(data['contributions'].shape[0]))
    random.shuffle(episode_ids)
    groups = [episode_ids[i::n_splits] for i in range(n_splits)]
    for i in range(n_splits):
        test_groups = groups[i]
        train_groups = [gg for g in groups for gg in g]

        # get a random fraction of the training groups
        random.shuffle(train_groups)
        train_groups = train_groups[:int(fraction_training*len(train_groups))]
        train_groups.sort()
        
        test_data = {
            k: t[test_groups]
            for k, t in data.items()
        }
        train_data = {
            k: t[train_groups]
            for k, t in data.items()
        }
        yield train_data, test_data