import pandas as pd, numpy as np
def load():
    d = pd.read_csv('/opt/algorithmic-institutions/experiments/2group_8agent_50ep.csv')
    d = d.drop_duplicates(['session','episode','round_number','participant_code']).copy()
    d['game'] = d.session + '_' + d.episode.astype(str)
    d = d.sort_values(['game','player_id','round_number'])
    g = d.groupby(['game','player_id'])
    d['c_next'] = g.contribution.shift(-1)
    d['p_prev'] = g.punishment.shift(1)
    d['grp_next'] = g.group_id.shift(-1)
    d['valid'] = 1 - d.player_no_input
    # group stats per round
    gs = d.groupby(['game','round_number','group_id']).agg(n=('player_id','size'),
          cmean=('contribution','mean'), pmean=('punishment','mean'), pool=('common_good','first')).reset_index()
    d = d.merge(gs, on=['game','round_number','group_id'])
    return d, gs
