import pandas as pd
def load():
    df = pd.read_csv('experiments/2group_8agent_50ep.csv')
    df = df.drop_duplicates(['session','episode','round_number','participant_code']).copy()
    df['game'] = df.session.astype(str)+'_'+df.episode.astype(str)
    df = df.sort_values(['game','player_id','round_number'])
    return df
