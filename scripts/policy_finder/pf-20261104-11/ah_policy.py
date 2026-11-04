"""How ah (sims) and human managers punish, by contribution, relative contribution and round."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('display.width', 250)
h = pd.read_csv('experiments/2group_8agent_50ep.csv')
h = h.drop_duplicates(['session', 'episode', 'round_number', 'participant_code'])
h = h[h.player_no_input == 0]
d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
d = d[(d.run.str.endswith('ah_vs_ah')) & (d.contribution_valid == 1)]
for name, x, key in [('ah', d, ['episode']), ('human', h, ['session', 'episode'])]:
    x = x.copy()
    x['gm'] = x.groupby(key + ['round_number', 'group_id']).contribution.transform('mean')
    x['cb'] = pd.cut(x.contribution, [-1, 0, 3, 6, 9, 12, 15, 19, 20])
    x['rel'] = pd.cut(x.contribution - x.gm, [-21, -8, -4, -1, 1, 4, 21])
    print(name)
    print(x.groupby('cb').punishment.agg(['mean', lambda s: (s > 0).mean(), 'size']).round(2).to_string())
    print(x.groupby('rel').punishment.agg(['mean', lambda s: (s > 0).mean(), 'size']).round(2).to_string())
    print(x.groupby(x.round_number // 4).punishment.mean().round(2).to_string())
