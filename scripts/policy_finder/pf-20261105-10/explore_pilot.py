"""Pilot data: contributions over rounds, response of next contribution to punishment."""
import pandas as pd, numpy as np
df = pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
print(df.shape); print(df.head()); print(df.columns.tolist())
print(df.groupby('experiment_name').agg(groups=('global_group_id','nunique'), eps=('episode_id','nunique')))
print(df.groupby(['experiment_name','round_number'])[['contribution','punishment','player_no_input','payoff']].mean().round(2).unstack(0))
pd.set_option('display.width',200)
print(df.groupby(['experiment_name','round_number'])[['punishment','player_no_input']].mean().round(2).unstack(0))
# punishment vs contribution, per manager type
print(df[df.player_no_input==0].groupby(['experiment_name','contribution'])['punishment'].agg(['mean','count']).unstack(0).round(2))
