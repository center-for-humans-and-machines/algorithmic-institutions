"""Contribution response: c(t+1)-c(t) by punishment at t and c(t), sims and humans."""
import warnings; warnings.filterwarnings("ignore")
import sys
import pandas as pd
pd.set_option('display.width', 250)
src = sys.argv[1]
if src == 'human':
    d = pd.read_csv('experiments/2group_8agent_50ep.csv')
    d = d.drop_duplicates(['session', 'episode', 'round_number', 'participant_code'])
    key = ['session', 'episode', 'participant_code']
else:
    d = pd.read_parquet('plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet')
    d = d[d.run.str.contains(src)]
    key = ['run', 'episode', 'participant_code']
d = d.sort_values(key + ['round_number'])
g = d.groupby(key)
d['c_next'] = g.contribution.shift(-1)
d['g_next'] = g.group_id.shift(-1)
d = d[d.c_next.notna() & (d.g_next == d.group_id)]
d['dc'] = d.c_next - d.contribution
d['pb'] = pd.cut(d.punishment, [-0.1, 0, 2, 4, 7, 12, 31])
d['cb'] = pd.cut(d.contribution, [-1, 0, 5, 10, 15, 19, 20])
print(pd.crosstab(d.cb, d.pb, values=d.dc, aggfunc="mean").round(2).to_string())
print(pd.crosstab(d.cb, d.pb).to_string())
