import pandas as pd, numpy as np
def load():
    df = pd.read_csv('/opt/algorithmic-institutions/experiments/2group_8agent_50ep.csv')
    df = df.drop_duplicates(['session','episode','round_number','participant_code'])
    df['game'] = df.session + '_' + df.episode.astype(str)
    df = df.sort_values(['game','player_id','round_number']).reset_index(drop=True)
    g = df.groupby(['game','player_id'])
    df['next_c'] = g.contribution.shift(-1)
    df['next_group'] = g.group_id.shift(-1)
    df['prev_c'] = g.contribution.shift(1)
    df['switch'] = (df.next_group != df.group_id).astype(float)
    df.loc[df.next_group.isna(),'switch'] = np.nan
    df['gsize'] = df.groupby(['game','round_number','group_id']).player_id.transform('size')
    return df
