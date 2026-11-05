import pandas as pd, numpy as np
def load():
    df = pd.read_csv('experiments/2group_8agent_50ep.csv')
    df = df.drop_duplicates(['session','episode','round_number','participant_code'])
    df['game'] = df.session.astype(str)+'_'+df.episode.astype(str)
    df = df.sort_values(['game','player_id','round_number'])
    g = df.groupby(['game','player_id'])
    df['prev_p'] = g.punishment.shift(1)
    df['prev_c'] = g.contribution.shift(1)
    df['next_c'] = g.contribution.shift(-1)
    df['next_group'] = g.group_id.shift(-1)
    df['size'] = df.groupby(['game','round_number','group_id']).player_id.transform('count')
    return df
