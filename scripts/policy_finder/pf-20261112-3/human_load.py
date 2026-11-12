import pandas as pd, numpy as np
def load():
    d=pd.read_csv('experiments/2group_8agent_50ep.csv')
    d=d.drop_duplicates(['session','episode','round_number','participant_code'])
    d['game']=d.session.astype(str)+'_'+d.episode.astype(str)
    d=d.sort_values(['game','player_id','round_number'])
    return d
