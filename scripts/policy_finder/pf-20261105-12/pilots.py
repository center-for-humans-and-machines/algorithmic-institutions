"""Human pilots: payoffs by experiment, punishment pattern, next-round response to punishment, end effects."""
import numpy as np, pandas as pd
pd.set_option('display.width', 200)
d = pd.read_csv('experiments/pilot_random1_player_round_slim.csv')
d['valid'] = d.player_no_input == 0
d['c'] = np.where(d.valid, d.contribution, 0)
key = ['global_group_id', 'episode_id']
nr = d.groupby(key).round_number.transform('max') + 1
d['nr'] = nr
print(d.groupby(['experiment_name', 'nr']).agg(groups=('global_group_id', lambda x: x.nunique()), c=('c', 'mean'), p=('punishment', 'mean'), to=('valid', lambda x: 1 - x.mean())))
for e, g in d.groupby('experiment_name'):
    v = g[g.valid]
    print(e, 'c by round', v.groupby('round_number').contribution.mean().round(1).values)
    print(e, 'p by round', v.groupby('round_number').punishment.mean().round(1).values)
    print(e, 'payoff/round', v.groupby('round_number').payoff.mean().round(1).values)
d = d.sort_values(key + ['player_id', 'round_number'])
gg = d.groupby(key + ['player_id'])
d['c_n1'] = gg.c.shift(-1); d['vn'] = gg.valid.shift(-1)
d['oth'] = (d.groupby(key + ['round_number']).c.transform('sum') - d.c) / 3
s = d[d.valid & (d.vn == True)].dropna(subset=['c_n1']).copy()
s['dc'] = s.c_n1 - s.c
pb = pd.cut(s.punishment, [-1, 0, 2, 5, 10, 30])
cb = pd.cut(s.c, [-1, 4, 9, 14, 19, 20])
for e, x in s.groupby('experiment_name'):
    print(e, 'mean dc'); print(x.groupby([cb, pb], observed=True).dc.mean().round(2).unstack())
    print(x.groupby([cb, pb], observed=True).dc.size().unstack())
    X = np.column_stack([np.ones(len(x)), x.c, x.oth, x.punishment, x.round_number])
    print('  c_n1 ~ 1,c,oth,p,round', np.linalg.lstsq(X, x.c_n1, rcond=None)[0].round(3))
