"""Load the human data, one copy per game, with lagged columns."""
import pandas as pd, numpy as np

def load():
    d = pd.read_csv('experiments/2group_8agent_50ep.csv')
    d = d.drop_duplicates(['session', 'episode', 'round_number', 'participant_code'])
    d['game'] = d.session.astype(str) + '_' + d.episode.astype(str)
    d = d.sort_values(['game', 'participant_code', 'round_number'])
    g = d.groupby(['game', 'participant_code'])
    for col in ['contribution', 'punishment', 'group_id', 'player_no_input', 'common_good']:
        d['prev_' + col] = g[col].shift(1)
        d['next_' + col] = g[col].shift(-1)
    d['switched_next'] = (d.next_group_id != d.group_id) & d.next_group_id.notna()
    # group stats this round
    gg = d.groupby(['game', 'round_number', 'group_id'])
    d['members'] = gg.participant_code.transform('size')
    d['grp_mean_c'] = gg.contribution.transform('mean')
    d['grp_sum_c'] = gg.contribution.transform('sum')
    d['grp_sum_p'] = gg.punishment.transform('sum')
    d['others_mean_c'] = (d.grp_sum_c - d.contribution) / (d.members - 1).replace(0, np.nan)
    return d
