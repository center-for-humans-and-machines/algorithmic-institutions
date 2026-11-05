"""Reference sims: how next-round contribution responds to punishment."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d['pid'] = d.participant_code.str.split('_').str[0].astype(int)
d = d.sort_values(['run', 'episode', 'pid', 'round_number'])
d['c'] = np.where(d.contribution_valid, d.contribution, 0)
g = d.groupby(['run', 'episode', 'round_number'])
d['n_valid'] = g.contribution_valid.transform('sum')
d['grp_c'] = g.c.transform('sum')
d['oth_mean'] = (d.grp_c - d.c) / (d.n_valid - d.contribution_valid).clip(lower=1)
k = d.groupby(['run', 'episode', 'pid'])
d['c_next'] = k.c.shift(-1)
d['v_next'] = k.contribution_valid.shift(-1)
d['dc'] = d.c_next - d.c
s = d[d.contribution_valid & (d.v_next == True)]
# score: per round mean payoff of valid players, summed
for run, x in d[d.contribution_valid].groupby('run'):
    print(run, 'score', x.groupby(['episode', 'round_number']).payoff.mean().groupby('episode').sum().mean())
h = s[s.run.str.contains('humanlike')]
print('\nhumanlike: punishment by own contribution')
print(h.groupby(pd.cut(h.c, [-1, 0, 5, 10, 15, 19, 20])).agg(p=('punishment', 'mean'), pp=('punishment', lambda v: (v > 0).mean()), n=('c', 'size')))
print('\nmean dc by contribution bin x punished?')
h = h.assign(cb=pd.cut(h.c, [-1, 0, 5, 10, 15, 19, 20]), pb=pd.cut(h.punishment, [-1, 0, 2, 5, 10, 30]))
print(h.pivot_table(index='cb', columns='pb', values='dc', aggfunc='mean').round(2))
print(h.pivot_table(index='cb', columns='pb', values='dc', aggfunc='size'))
z = s[s.run.str.contains('zero')]
print('\nzero mgr: dc by contribution bin'); print(z.groupby(pd.cut(z.c, [-1, 0, 5, 10, 15, 19, 20])).dc.mean().round(2))
import statsmodels.formula.api as smf
for name, x in [('humanlike', s[s.run.str.contains('humanlike')]), ('zero', z)]:
    x = x.assign(dev=x.c - x.oth_mean)
    f = 'c_next ~ c + oth_mean + punishment + I(punishment*(c<oth_mean)) + round_number' if name == 'humanlike' else 'c_next ~ c + oth_mean + round_number'
    print(name, smf.ols(f, x).fit().params.round(3).to_dict())
