import pandas as pd
def load():
    df = pd.read_csv('/opt/algorithmic-institutions/experiments/2group_8agent_50ep.csv')
    df = df.drop_duplicates(['session','episode','round_number','participant_code'])
    df['game'] = df.session.astype(str)+'_'+df.episode.astype(str)
    return df
