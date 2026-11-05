"""Sim (humanlike): c at t+k on own and others' punishment at t; end-game under zero."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
d = pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
d['pid'] = d.participant_code.str.split('_').str[0].astype(int)
d = d.sort_values(['run', 'episode', 'pid', 'round_number'])
d['c'] = np.where(d.contribution_valid, d.contribution, np.nan)
g = d.groupby(['run', 'episode', 'pid'])
for k in (1, 2, 3, 5, 8):
    d[f'c{k}'] = g.c.shift(-k)
grp = d.groupby(['run', 'episode', 'round_number'])
d['gmean'] = grp.c.transform('mean')
d['others_p'] = (grp.punishment.transform('sum') - d.punishment) / 3
h = d[d.run.str.contains('humanlike') & d.c.notna()]
for k in (1, 2, 3, 5, 8):
    m = smf.ols(f'c{k} ~ punishment + others_p + c + gmean + round_number', data=h.dropna(subset=[f'c{k}'])).fit()
    print(k, m.params[['punishment', 'others_p', 'c', 'gmean']].round(3).to_dict())
# low contributors only
lo = h[h.c <= 8]
for k in (1, 3, 5):
    m = smf.ols(f'c{k} ~ punishment + others_p + c + gmean + round_number', data=lo.dropna(subset=[f'c{k}'])).fit()
    print('c<=8', k, m.params[['punishment', 'others_p']].round(3).to_dict())
hi = h[h.c >= 14]
for k in (1, 3):
    m = smf.ols(f'c{k} ~ punishment + others_p + c + gmean + round_number', data=hi.dropna(subset=[f'c{k}'])).fit()
    print('c>=14', k, m.params[['punishment', 'others_p']].round(3).to_dict())
# timeouts
d['v'] = d.contribution_valid.astype(float)
print('timeout rate by round', d.groupby('round_number').v.mean().round(3).values)
# within-episode heterogeneity: per player mean contribution under zero
z = d[d.run.str.contains('zero')]
pm = z.groupby(['episode', 'pid']).c.mean()
print('zero: dist of player mean c', np.percentile(pm, [10, 25, 50, 75, 90]).round(1))
print('zero round0 c dist', np.percentile(z[z.round_number == 0].c.dropna(), [10, 25, 50, 75, 90]))
print('zero: fraction c==0 by round', z.groupby('round_number').c.apply(lambda s: (s == 0).mean()).round(2).values)
print('zero: fraction c>=15 by round', z.groupby('round_number').c.apply(lambda s: (s >= 15).mean()).round(2).values)
