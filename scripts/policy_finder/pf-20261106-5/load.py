import pandas as pd
def load():
    d = pd.read_csv('/opt/algorithmic-institutions/experiments/2group_8agent_50ep.csv')
    d = d.drop_duplicates(['session', 'episode', 'round_number', 'participant_code'])
    d['game'] = d.session + '_' + d.episode.astype(str)
    d = d.sort_values(['game', 'player_id', 'round_number']).reset_index(drop=True)
    g = d.groupby(['game', 'player_id'])
    d['next_group'] = g.group_id.shift(-1)
    d['next_c'] = g.contribution.shift(-1)
    d['prev_p'] = g.punishment.shift(1)
    d['switch'] = (d.next_group != d.group_id) & d.next_group.notna()
    gr = d.groupby(['game', 'round_number', 'group_id'])
    d['gsize'] = gr.player_id.transform('size')
    d['gmean_c'] = gr.contribution.transform('mean')
    d['gmean_p'] = gr.punishment.transform('mean')
    d['gpay'] = gr.payoff.transform('mean')
    return d
