"""Human data: persistence of the contribution gain after punishing a low contributor, and recency in switching."""
import pandas as pd, numpy as np
d = pd.read_pickle('/tmp/h.pkl').sort_values(['game', 'participant_code', 'round_number'])
gp = d.groupby(['game', 'participant_code'])
for k in (1, 2, 3, 4):
    d[f'c{k}'] = gp.contribution.shift(-k)
    d[f'g{k}'] = gp.group_id.shift(-k)
x = d[(d.valid == 1) & (d.contribution <= 9) & (d.round_number <= 18)].copy()
x['pun'] = pd.cut(x.punishment, [-1, 0, 4, 30])
for k in (1, 2, 3, 4):
    y = x[x[f'g{k}'] == x.group_id]
    print(k, (y[f'c{k}'] - y.contribution).groupby(y.pun, observed=True).agg(['mean', 'count']).round(2).to_dict())
# controlling for the level: within contribution bins 0-4 and 5-9
for lo, hi in ((0, 4), (5, 9)):
    y = x[(x.contribution >= lo) & (x.contribution <= hi) & (x.g3 == x.group_id)]
    print(lo, hi, 'gain at t+3', (y.c3 - y.contribution).groupby(y.pun, observed=True).mean().round(2).to_dict())
# switching recency: punishment in round 4k+3 vs rounds 4k..4k+2
p = d.pivot_table(index=['game', 'participant_code'], columns='round_number', values='punishment')
rows = []
for r in (3, 7, 11, 15, 19):
    s = d[d.round_number == r].set_index(['game', 'participant_code'])
    rows.append(pd.DataFrame({'sw': (s.g_next != s.group_id).astype(float), 'plast': p[r], 'pprev': p[[r - 1, r - 2, r - 3]].sum(1), 'c': s.contribution}))
s = pd.concat(rows)
import statsmodels.formula.api as smf
m = smf.logit('sw ~ plast + pprev + c', s).fit(disp=0); print(m.params.round(3).to_dict(), m.bse.round(3).to_dict())
