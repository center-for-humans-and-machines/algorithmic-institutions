import pandas as pd, numpy as np
df = pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
print(df.shape); print(df.columns.tolist())
print(df.groupby('experiment_name').agg(groups=('global_group_id','nunique'), rounds=('round_number','max')))
df['valid'] = 1-df.player_no_input
v = df[df.valid==1]
print('\nmean contribution / punishment by round and experiment')
print(v.groupby(['experiment_name','round_number'])[['contribution','punishment']].mean().unstack(0).round(2))
print('\ntimeout rate', df.groupby('experiment_name').player_no_input.mean())
print('\nmean payoff valid by exp', v.groupby('experiment_name').payoff.mean())
